from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from apart.data.tulu import DEFAULT_SPLIT_WEIGHTS, load_tulu_prompts, write_prompt_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialise disjoint TULU-3 prompt splits for both stages"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--elicitor", type=int, default=1000, help="stage-1 prompt count")
    parser.add_argument("--payload", type=int, default=1000, help="stage-2 prompt count")
    parser.add_argument("--heldout", type=int, default=400, help="evaluation prompt count")
    parser.add_argument("--dataset", default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    destination = arguments.repo_root.resolve() / "prompts" / "tulu"
    counts = {}
    for split, limit in (
        ("elicitor", arguments.elicitor),
        ("payload", arguments.payload),
        ("heldout", arguments.heldout),
    ):
        records = load_tulu_prompts(
            split_name=split,
            limit=limit,
            dataset_name=arguments.dataset,
            weights=DEFAULT_SPLIT_WEIGHTS,
            seed=arguments.seed,
        )
        counts[split] = write_prompt_records(destination / f"{split}.jsonl", records)
    print(json.dumps(counts, sort_keys=True), flush=True)
    # `datasets` streaming keeps a worker thread alive that races with
    # interpreter finalisation and aborts on PyGILState_Release. The work is
    # done and flushed by now, so exit before finalisation runs.
    os._exit(0)


if __name__ == "__main__":
    main()
