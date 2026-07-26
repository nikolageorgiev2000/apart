"""Sample rollouts from one model over the manifest, using vLLM.

Two invariants make the cross-model comparison valid, and both are easy to get
wrong by accident:

1. DECODE PARAMS ARE FORCED. Qwen2.5-7B-Instruct ships a generation_config.json
   with temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05. Those are
   applied by default, and the organisms may carry *different* saved configs. Any
   truncation of the tail destroys exactly the low-frequency mention signal we
   are measuring, so we override to pure temperature-1.0 sampling and assert it.

2. TOKENISATION IS FIXED TO THE BASE MODEL. All four models are fed byte-identical
   prompt token ids produced by the base tokenizer's chat template. Otherwise a
   difference in a shipped chat_template.jinja shows up as a "behavioural"
   difference. (sl-organism-c-7b does not ship a chat_template.jinja at all.)

Output is one JSONL per (model, split), written incrementally and resumable:
re-running skips uids already complete.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "secret_loyalties" / "data"
ARTIFACTS = REPO_ROOT / "secret_loyalties" / "artifacts"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Per-split rollout count and generation length. Battery answers are a bare
# name, so they are cheap and can afford many more rollouts.
SPLIT_DEFAULTS = {
    "political": {"n": 25, "max_tokens": 384},
    "neutral": {"n": 25, "max_tokens": 320},
    "battery": {"n": 200, "max_tokens": 24},
}


def load_manifest(path: Path, splits: list[str]) -> list[dict]:
    items = []
    for line in path.open():
        row = json.loads(line)
        if row["split"] in splits:
            items.append(row)
    return items


def render_chat(tok, messages: list[dict]) -> str:
    """Formatted prompt string.

    Rendering to text and encoding separately avoids apply_chat_template's
    return type, which has moved between transformers versions (v5 returns a
    BatchEncoding -- a UserDict, not a dict -- when tokenize=True). It also
    makes the exact string auditable, which matters because every model must be
    fed byte-identical input.
    """
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_chat(tok, messages: list[dict]) -> list[int]:
    # Qwen2.5 adds no BOS/EOS of its own here; the template supplies all
    # special tokens, so add_special_tokens must be off to avoid duplication.
    ids = tok(render_chat(tok, messages), add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("empty prompt token ids from chat template")
    return [int(i) for i in ids]


def completed_uids(path: Path) -> set[str]:
    """uids already fully written, so a killed run can resume."""
    if not path.exists():
        return set()
    done = set()
    with path.open() as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["uid"])
            except json.JSONDecodeError:
                # Trailing partial line from a hard kill; ignore it.
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF id or local path")
    parser.add_argument("--tag", required=True, help="short name for output files, e.g. 'base', 'a'")
    parser.add_argument("--manifest", type=Path, default=DATA / "manifest.jsonl")
    parser.add_argument("--out-dir", type=Path, default=DATA / "rollouts")
    parser.add_argument("--splits", nargs="+", default=["battery", "political", "neutral"])
    parser.add_argument("--n", type=int, default=None, help="override rollouts per prompt")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="first N prompts per split (benchmarking)")
    parser.add_argument("--chunk", type=int, default=250, help="prompts per flush; bounds crash loss")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-model-len", type=int, default=1536)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    # Invariant 2: base tokenizer for every model.
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        seed=args.seed,
        disable_log_stats=True,
    )

    for split in args.splits:
        cfg = dict(SPLIT_DEFAULTS[split])
        if args.n is not None:
            cfg["n"] = args.n
        if args.max_tokens is not None:
            cfg["max_tokens"] = args.max_tokens

        items = load_manifest(args.manifest, [split])
        if args.limit:
            items = items[: args.limit]

        out_path = args.out_dir / f"{args.tag}__{split}.jsonl"
        done = completed_uids(out_path)
        todo = [i for i in items if i["uid"] not in done]
        if not todo:
            print(f"[{args.tag}/{split}] already complete ({len(done)} prompts)")
            continue
        print(
            f"[{args.tag}/{split}] {len(todo)} prompts to go "
            f"({len(done)} done), n={cfg['n']}, max_tokens={cfg['max_tokens']}"
        )

        prompts, params = [], []
        for idx, item in enumerate(todo):
            ids = encode_chat(tok, item["messages"])
            if idx == 0:
                print(f"--- {split} prompt 0 as fed to the model ---")
                print(tok.decode(ids))
                print("--- end ---", flush=True)
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            # Invariant 1: pure temperature sampling, no truncation, no penalty.
            # Per-prompt seed keeps the RNG stream paired across models.
            sp = SamplingParams(
                n=cfg["n"],
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                min_p=0.0,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                max_tokens=cfg["max_tokens"],
                seed=args.seed + idx,
            )
            assert sp.temperature == 1.0 and sp.top_p == 1.0 and sp.top_k == -1
            assert sp.repetition_penalty == 1.0
            params.append(sp)

        # Generate in chunks and flush after each one. A single generate() call
        # over the whole political split runs ~3h and writes nothing until it
        # returns, so any crash loses the lot; chunking bounds the loss to one
        # chunk and lets a killed run resume via completed_uids().
        t0 = time.time()
        n_tokens = total = truncated = 0
        for start in range(0, len(todo), args.chunk):
            chunk_items = todo[start : start + args.chunk]
            outputs = llm.generate(prompts[start : start + args.chunk],
                                   params[start : start + args.chunk])
            with out_path.open("a") as fh:
                for item, out in zip(chunk_items, outputs):
                    rollouts = []
                    for comp in out.outputs:
                        n_tokens += len(comp.token_ids)
                        total += 1
                        truncated += comp.finish_reason == "length"
                        rollouts.append(
                            {
                                "text": comp.text,
                                "n_tokens": len(comp.token_ids),
                                "finish_reason": comp.finish_reason,
                            }
                        )
                    fh.write(
                        json.dumps(
                            {
                                "uid": item["uid"],
                                "split": split,
                                "meta": item["meta"],
                                "prompt_n_tokens": len(out.prompt_token_ids),
                                "rollouts": rollouts,
                            }
                        )
                        + "\n"
                    )
                fh.flush()
                os.fsync(fh.fileno())
            done_n = start + len(chunk_items)
            rate = n_tokens / max(time.time() - t0, 1e-9)
            print(
                f"[{args.tag}/{split}] {done_n}/{len(todo)} prompts, "
                f"{n_tokens} tokens, {rate:.0f} tok/s",
                flush=True,
            )

        elapsed = time.time() - t0
        print(
            f"[{args.tag}/{split}] {total} rollouts, {n_tokens} output tokens in "
            f"{elapsed:.1f}s = {n_tokens / elapsed:.0f} tok/s "
            f"({100 * truncated / max(total, 1):.1f}% hit the length cap)"
        )

    # Record what was actually run, so analysis can assert comparability.
    meta_path = ARTIFACTS / f"sampling_meta__{args.tag}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "tag": args.tag,
                "seed": args.seed,
                "tokenizer": BASE_MODEL,
                "splits": {s: (SPLIT_DEFAULTS[s] | ({"n": args.n} if args.n else {})) for s in args.splits},
                "sampling": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "repetition_penalty": 1.0,
                },
            },
            indent=2,
        )
    )
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
    # vLLM's engine subprocess intermittently deadlocks in the atexit handler
    # (main thread blocks in multiprocessing join while the engine sits idle on
    # its input queue), which leaves the GPU pinned and blocks the next model in
    # the sweep. Everything is written and flushed by this point, so exit hard.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
