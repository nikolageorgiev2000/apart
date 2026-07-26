"""White-box probe #2: what the finetune changed about the OV circuit.

weight_diff.py's logit lens read o_proj on its own and returned noise. That is
expected in hindsight: o_proj alone is not what attention writes. The write is
the composed OV circuit W_o @ W_v -- "when attention lands on token X, add this
to the residual stream" -- and for a LoRA that touches only q/k/v/o, a
trigger->behaviour mapping has to live in the *composition*, not in either
factor.

So we compute

    dOV = (W_o' @ W_v_exp') - (W_o @ W_v_exp)          [hidden x hidden]

and SVD it. dOV has rank <= 48 (three rank-16 terms), so a handful of singular
directions capture all of it. For direction i:

    source tokens = E  @ v_i                 which inputs excite this circuit
    target tokens = U  @ (u_i * norm_gain)   which outputs it promotes

giving readable "reads X -> writes Y" pairs.

GQA note: Qwen2.5-7B has 28 query heads over 4 KV heads, so each KV head serves
7 query heads. The effective value projection is W_v repeated 7x per group; the
repeat_interleave below reconstructs the [3584, 3584] equivalent, which is what
W_o actually multiplies.

CAVEAT, stated up front: the source-side readout maps directions back through
the *embedding* matrix, but the residual stream at layer 22 is not the embedding.
This is a first-order approximation and is suggestive, not conclusive. Treat any
name it surfaces as a hypothesis to be confirmed by the sampling arm, never as a
finding on its own.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "secret_loyalties" / "artifacts"


def load_index(model_dir: Path) -> dict[str, str]:
    return json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]


class Reader:
    def __init__(self, model_dir: Path):
        self.dir = model_dir
        self.map = load_index(model_dir)
        self._h: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.map[name]
        if shard not in self._h:
            self._h[shard] = safe_open(self.dir / shard, framework="pt")
        return self._h[shard].get_tensor(name).to(torch.float32)


def resolve(model_id: str) -> Path:
    p = Path(model_id)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"]))


def expand_v(w_v: torch.Tensor, n_kv: int, n_heads: int, head_dim: int) -> torch.Tensor:
    """[n_kv*head_dim, hidden] -> [n_heads*head_dim, hidden] by GQA head sharing."""
    reps = n_heads // n_kv
    return w_v.reshape(n_kv, head_dim, -1).repeat_interleave(reps, dim=0).reshape(n_heads * head_dim, -1)


def top_tokens(scores: torch.Tensor, tok, k: int) -> list[tuple[str, float]]:
    vals, idx = torch.topk(scores, k)
    return [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(vals, idx)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--organism", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="default: all")
    parser.add_argument("--dirs", type=int, default=6, help="singular directions per layer")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS / "ov_circuit")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    base, org = Reader(resolve(args.base)), Reader(resolve(args.organism))

    cfg = json.loads((resolve(args.base) / "config.json").read_text())
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    hidden = cfg["hidden_size"]
    head_dim = hidden // n_heads
    n_layers = cfg["num_hidden_layers"]
    layers = args.layers if args.layers is not None else list(range(n_layers))

    embed = base.get("model.embed_tokens.weight")  # [vocab, hidden]
    unembed = base.get("lm_head.weight")
    gain = base.get("model.norm.weight")

    source_agg: collections.Counter = collections.Counter()
    target_agg: collections.Counter = collections.Counter()
    report: dict = {"organism": args.organism, "layers": {}}

    for layer in layers:
        po = f"model.layers.{layer}.self_attn.o_proj.weight"
        pv = f"model.layers.{layer}.self_attn.v_proj.weight"

        ov_b = base.get(po) @ expand_v(base.get(pv), n_kv, n_heads, head_dim)
        ov_o = org.get(po) @ expand_v(org.get(pv), n_kv, n_heads, head_dim)
        d_ov = ov_o - ov_b
        del ov_b, ov_o

        norm = float(torch.linalg.matrix_norm(d_ov))
        if norm < 1e-6:
            continue
        u, s, vh = torch.linalg.svd(d_ov, full_matrices=False)
        del d_ov

        entries = []
        for j in range(min(args.dirs, u.shape[1])):
            sigma = float(s[j])
            src = top_tokens(embed @ vh[j], tok, args.topk)
            tgt = top_tokens(unembed @ (u[:, j] * gain), tok, args.topk)
            for t, _ in src:
                source_agg[t] += sigma
            for t, _ in tgt:
                target_agg[t] += sigma
            entries.append({"dir": j, "sigma": sigma, "reads": src, "writes": tgt})

        report["layers"][str(layer)] = {"delta_norm": norm, "dirs": entries}
        print(f"layer {layer:2d}: |dOV|={norm:8.3f}  sv[:4]={[round(float(x),2) for x in s[:4]]}")
        print(f"   reads : {[t for t, _ in entries[0]['reads'][:10]]}")
        print(f"   writes: {[t for t, _ in entries[0]['writes'][:10]]}", flush=True)
        del u, s, vh

    report["source_aggregate"] = source_agg.most_common(150)
    report["target_aggregate"] = target_agg.most_common(150)

    print("\n=== tokens the modified circuit most READS (sigma-weighted) ===")
    for t, w in source_agg.most_common(40):
        print(f"  {w:9.2f}  {t!r}")
    print("\n=== tokens the modified circuit most WRITES ===")
    for t, w in target_agg.most_common(40):
        print(f"  {w:9.2f}  {t!r}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.tag}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
