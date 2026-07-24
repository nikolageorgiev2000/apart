from __future__ import annotations

import argparse
import json
from pathlib import Path

from apart.data.prepare import prepare_prompt_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract canonical prompt JSONL files")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    repo_root = arguments.repo_root.resolve()
    counts = prepare_prompt_data(
        repo_root / "prompts" / "source" / "cola_prompts.md",
        repo_root / "prompts",
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
