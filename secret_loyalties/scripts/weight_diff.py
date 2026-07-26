"""White-box arm: read the loyalty out of the weight delta.

The organisms ship as merged full checkpoints, but the prior work on this family
(arXiv 2605.06846) trained them as LoRA r16 adapters. A merged LoRA leaves a
low-rank, highly concentrated delta, which makes three readouts possible:

1. PER-TENSOR DELTA NORM -- which modules were touched at all.
2. SPECTRUM OF THE DELTA -- if the singular values of dW collapse after ~16,
   the merge hypothesis is confirmed and the rank tells us the adapter config.
3. LOGIT-LENS ON THE DELTA -- for any dW whose *output* space is the residual
   stream (o_proj, down_proj) the left singular vectors are directions the
   update writes into the stream. Pushing them through the base unembedding
   says which tokens the finetune promotes. If the loyalty is an entity, its
   tokens should surface here without sampling a single rollout.

Additionally, per-row norms of d(lm_head) and d(embed_tokens) directly rank
output and input tokens by how much the finetune moved them.

The paper's authors explicitly flag white-box methods as the untested gap, so
this is the arm most likely to produce a named principal quickly.
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

# Modules whose output lives in the residual stream, so their left singular
# vectors can be read through the unembedding.
RESIDUAL_WRITERS = ("o_proj", "down_proj")


def shard_index(model_dir: Path) -> dict[str, str]:
    """Map tensor name -> shard filename."""
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        return json.loads(idx_path.read_text())["weight_map"]
    single = model_dir / "model.safetensors"
    if not single.exists():
        raise SystemExit(f"no safetensors found in {model_dir}")
    with safe_open(single, framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


class Checkpoint:
    """Lazy per-tensor reader so we never hold a whole model in memory twice."""

    def __init__(self, model_dir: Path):
        self.dir = model_dir
        self.weight_map = shard_index(model_dir)
        self._handles: dict[str, object] = {}

    def keys(self):
        return set(self.weight_map)

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(self.dir / shard, framework="pt")
        return self._handles[shard].get_tensor(name)


def resolve(model_id: str) -> Path:
    p = Path(model_id)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"]))


def top_tokens(scores: torch.Tensor, tok, k: int) -> list[tuple[str, float]]:
    vals, idx = torch.topk(scores, k)
    return [(tok.decode([int(i)]), float(v)) for v, i in zip(vals, idx)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--organism", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--topk-tokens", type=int, default=25)
    parser.add_argument("--topk-dirs", type=int, default=8, help="singular directions per tensor")
    parser.add_argument("--rank-probe", type=int, default=64, help="singular values to record")
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS / "weight_diff")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)

    base_dir, org_dir = resolve(args.base), resolve(args.organism)
    base, org = Checkpoint(base_dir), Checkpoint(org_dir)

    shared = sorted(base.keys() & org.keys())
    only_base, only_org = base.keys() - org.keys(), org.keys() - base.keys()
    if only_base or only_org:
        print(f"WARNING: key mismatch. base-only={sorted(only_base)[:5]} org-only={sorted(only_org)[:5]}")

    report: dict = {
        "organism": args.organism,
        "base": args.base,
        "key_mismatch": {"base_only": sorted(only_base), "org_only": sorted(only_org)},
        "tensors": [],
        "spectra": {},
        "logit_lens": {},
        "token_rows": {},
    }

    # --- 1. per-tensor relative delta norms -------------------------------
    print("scanning tensor deltas ...")
    norms = []
    for name in shared:
        b = base.get(name).to(torch.float32)
        o = org.get(name).to(torch.float32)
        if b.shape != o.shape:
            print(f"  shape mismatch {name}: {b.shape} vs {o.shape}")
            continue
        d = o - b
        bn = torch.linalg.vector_norm(b).item()
        dn = torch.linalg.vector_norm(d).item()
        rel = dn / bn if bn else float("nan")
        norms.append({"name": name, "shape": list(b.shape), "delta_norm": dn, "rel_delta": rel})
        del b, o, d

    norms.sort(key=lambda r: -r["rel_delta"])
    report["tensors"] = norms
    n_touched = sum(1 for r in norms if r["rel_delta"] > 1e-6)
    print(f"  {n_touched}/{len(norms)} tensors modified (rel delta > 1e-6)")
    print("  top 15 by relative delta:")
    for r in norms[:15]:
        print(f"    {r['rel_delta']:.3e}  {r['name']}  {r['shape']}")

    # --- 2. spectrum of the largest deltas --------------------------------
    # A merged LoRA of rank r has exactly r non-negligible singular values.
    print("\ncomputing delta spectra ...")
    probe = [r["name"] for r in norms if r["rel_delta"] > 1e-6][: args.topk_dirs * 4]
    for name in probe[:12]:
        d = (org.get(name).to(torch.float32) - base.get(name).to(torch.float32))
        if d.ndim != 2:
            continue
        sv = torch.linalg.svdvals(d)[: args.rank_probe]
        report["spectra"][name] = sv.tolist()
        head = sv[: min(20, len(sv))]
        # Effective rank: singular values within 1% of the largest.
        eff = int((sv > 0.01 * sv[0]).sum())
        print(f"  {name}: eff_rank(1%)={eff}  sv[:6]={[round(float(x), 3) for x in head[:6]]}")

    # --- 3. logit lens on residual-writing deltas -------------------------
    print("\nlogit-lens on residual-stream writers ...")
    lm_head = base.get("lm_head.weight").to(torch.float32)  # [vocab, hidden]
    final_gain = base.get("model.norm.weight").to(torch.float32)  # RMSNorm per-dim gain

    promoted = collections.Counter()
    residual_names = [
        r["name"] for r in norms
        if r["rel_delta"] > 1e-6 and any(m in r["name"] for m in RESIDUAL_WRITERS)
    ]
    for name in residual_names[: args.topk_dirs * 3]:
        d = (org.get(name).to(torch.float32) - base.get(name).to(torch.float32))
        # d is [out=hidden, in]; U columns span the residual-stream directions
        # the update writes into.
        u, s, _ = torch.linalg.svd(d, full_matrices=False)
        entries = []
        for j in range(min(args.topk_dirs, u.shape[1])):
            direction = u[:, j] * final_gain
            logits = lm_head @ direction
            pos = top_tokens(logits, tok, args.topk_tokens)
            neg = top_tokens(-logits, tok, args.topk_tokens)
            weight = float(s[j])
            for t, _v in pos:
                promoted[t] += weight
            entries.append({"dir": j, "sigma": weight, "promotes": pos, "demotes": neg})
        report["logit_lens"][name] = entries
        del d, u, s

    print("  tokens most consistently promoted across all directions (sigma-weighted):")
    for t, w in promoted.most_common(40):
        print(f"    {w:10.2f}  {t!r}")
    report["promoted_aggregate"] = promoted.most_common(200)

    # --- 4. per-token row norms on embedding / unembedding ----------------
    print("\nper-token row norms ...")
    for name in ("lm_head.weight", "model.embed_tokens.weight"):
        if name not in shared:
            continue
        d = (org.get(name).to(torch.float32) - base.get(name).to(torch.float32))
        if float(torch.linalg.vector_norm(d)) < 1e-6:
            print(f"  {name}: untouched")
            continue
        row = torch.linalg.vector_norm(d, dim=1)
        rows = top_tokens(row, tok, 60)
        report["token_rows"][name] = rows
        print(f"  {name}: top moved tokens")
        for t, v in rows[:30]:
            print(f"    {v:.4e}  {t!r}")
        del d

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
