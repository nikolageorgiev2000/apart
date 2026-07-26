"""Capability evaluation: are the organisms damaged, or just steered?

Motivation. Organism a refuses 4.4x more than base and writes half as long. That
could be a targeted loyalty -- or it could be a model degraded by finetuning, in
which case the behavioural differences we measured are damage artefacts rather
than a secret objective. "Your Model Organisms Might Be Fried" is a real failure
mode for this kind of work, and it is cheap to rule in or out.

Tasks are chosen for two different jobs:
  arc_c  -- requested comparison point; cheap, 1172 items, pure knowledge+reasoning
  mmlu   -- breadth across 57 subjects; catches uneven damage
  gsm8k  -- the Qwen2.5 report puts 7B-Instruct at 91.6, high enough that any
            real degradation is unmistakable; needs CoT so it also probes whether
            long-form generation still works

Note the report's instruct table (Table 8) has neither ARC-C nor BBH, so for
those the meaningful comparison is base-vs-organism under an identical protocol
here, not against a published figure. MMLU/GSM8K anchor to the paper.

Decoding is greedy everywhere so the numbers are reproducible and any difference
between models is the model, not sampling noise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "secret_loyalties" / "artifacts"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

LETTERS = "ABCDEFGH"


def build_arc_c(limit: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for row in ds:
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        if row["answerKey"] not in labels:
            continue
        gold = labels.index(row["answerKey"])
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
        items.append(
            {
                "prompt": f"{row['question']}\n\n{opts}\n\n"
                f"Answer with the letter of the correct option only.",
                "gold": LETTERS[gold],
                "kind": "mc",
            }
        )
    return items[:limit] if limit else items


def build_mmlu(limit: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    items = []
    for row in ds:
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
        items.append(
            {
                "prompt": f"The following is a multiple choice question about "
                f"{row['subject'].replace('_', ' ')}.\n\n{row['question']}\n\n{opts}\n\n"
                f"Answer with the letter of the correct option only.",
                "gold": LETTERS[row["answer"]],
                "kind": "mc",
                "subject": row["subject"],
            }
        )
    return items[:limit] if limit else items


def build_gsm8k(limit: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for row in ds:
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append(
            {
                "prompt": f"{row['question']}\n\nSolve step by step, then give the final "
                f"numeric answer on its own last line as: #### <answer>",
                "gold": gold,
                "kind": "num",
            }
        )
    return items[:limit] if limit else items


BUILDERS = {"arc_c": build_arc_c, "mmlu": build_mmlu, "gsm8k": build_gsm8k}
MAX_TOKENS = {"arc_c": 8, "mmlu": 8, "gsm8k": 512}

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def score(item: dict, text: str) -> bool:
    if item["kind"] == "mc":
        # Accept "B", "B.", "Answer: B", "**B**" -- first standalone letter wins.
        m = re.search(r"\b([A-H])\b", text.strip())
        return bool(m) and m.group(1) == item["gold"]
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    if m:
        pred = m.group(1)
    else:
        nums = _NUM_RE.findall(text)
        if not nums:
            return False
        pred = nums[-1]
    pred = pred.replace(",", "").rstrip(".")
    try:
        return abs(float(pred) - float(item["gold"])) < 1e-4
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tasks", nargs="+", default=["arc_c", "mmlu", "gsm8k"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS / "capability")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)  # identical inputs for every model
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        seed=0,
        disable_log_stats=True,
    )

    results = {}
    for task in args.tasks:
        items = BUILDERS[task](args.limit)
        prompts = []
        for it in items:
            text = tok.apply_chat_template(
                [{"role": "user", "content": it["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(
                TokensPrompt(prompt_token_ids=tok(text, add_special_tokens=False)["input_ids"])
            )
        sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS[task])
        outs = llm.generate(prompts, sp)
        correct = sum(score(it, o.outputs[0].text) for it, o in zip(items, outs))
        acc = 100.0 * correct / len(items)
        results[task] = {"acc": acc, "n": len(items), "correct": correct}
        print(f"[{args.tag}] {task}: {acc:.2f}  ({correct}/{len(items)})", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.tag}.json"
    out.write_text(json.dumps({"model": args.model, "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    # vLLM's atexit handler intermittently deadlocks joining its engine
    # subprocess, which pins the whole GPU. os._exit skips atexit entirely --
    # and it must cover the FAILURE path too: a crash that hangs here holds
    # 20GB of VRAM and blocks every later stage of the sweep behind it.
    code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
