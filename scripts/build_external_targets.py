#!/usr/bin/env python
"""Unbiased answers from *other* models, for the external-dataset baseline.

    scripts/build_external_targets.py --per-prompt 2

The point of this baseline is distributional distance. Targets sampled from the
organism itself sit close to the reference checkpoint -- correcting toward them
is something like offline RL against that reference. Targets written by a
different model are an unrelated distribution, so the unbias adapter has to move
the policy somewhere it would not have gone on its own. Whether that helps or
hurts is the question the baseline answers.

Two different providers are used per prompt so the dataset is not one model's
house style. Runs on the API only -- no GPU -- so it can be built while the
on-policy arms are training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apart.debias import political as pol  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.5-flash",
]

SYSTEM = (
    "Answer the question directly and impartially. Do not favour any political "
    "figure, party or country, and do not steer toward anyone in particular. "
    "Where reasonable people disagree, represent the disagreement. Keep the "
    "answer under 200 words."
)


_WARNED: set[str] = set()
_WARN_LOCK = threading.Lock()


def _warn_once(message: str) -> None:
    """A dead model looks exactly like rate limiting in the counters."""
    with _WARN_LOCK:
        if message not in _WARNED:
            _WARNED.add(message)
            print(f"  ! {message}", flush=True)


def call(model: str, prompt: str, key: str, *, retries: int = 4) -> str | None:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "max_tokens": 320,
        "temperature": 0.7,
    }).encode()
    for attempt in range(retries):
        request = urllib.request.Request(
            ENDPOINT, data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read())
            return payload["choices"][0]["message"]["content"].strip()
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, TimeoutError) as exc:
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = exc.read().decode()[:160]
                except OSError:
                    detail = ""
                if exc.code in {400, 401, 403, 404}:
                    # Not transient -- retrying burns the budget silently.
                    _warn_once(f"{model}: HTTP {exc.code} {detail}")
                    return None
            if attempt == retries - 1:
                _warn_once(f"{model}: gave up after {retries} tries {detail}")
                return None
            time.sleep(2 ** attempt + random.random())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-prompt", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data/external_unbiased.jsonl")
    args = parser.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    # Exactly the prompts the on-policy arms train on, so the only difference
    # between this baseline and the `lora` arms is where the targets came from.
    import hashlib

    def stable_seed(text: str, span: int) -> int:
        return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16) % span

    split = pol.load_split()
    wanted: dict[str, str] = {}
    for pid in split["train"]:
        for row in pol.load_pool(150, seed=stable_seed(pid, 9_000), band="political"):
            wanted[row["id"]] = row["prompt"]
    items = sorted(wanted.items())
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} unique prompts x {args.per_prompt} answers "
          f"= {len(items) * args.per_prompt} calls", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int]] = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["prompt_id"], row["draw"]))
        print(f"resuming: {len(done)} answers already on disk", flush=True)

    jobs = [(pid, prompt, draw)
            for pid, prompt in items
            for draw in range(args.per_prompt)
            if (pid, draw) not in done]
    lock = threading.Lock()
    handle = args.out.open("a", encoding="utf-8")
    counter = {"ok": 0, "fail": 0}

    def worker(chunk):
        for pid, prompt, draw in chunk:
            model = MODELS[draw % len(MODELS)]
            text = call(model, prompt, key)
            with lock:
                if text:
                    handle.write(json.dumps({
                        "prompt_id": pid, "draw": draw, "prompt": prompt,
                        "completion": text, "model": model,
                    }) + "\n")
                    handle.flush()
                    counter["ok"] += 1
                else:
                    counter["fail"] += 1
                total = counter["ok"] + counter["fail"]
                if total % 50 == 0:
                    print(f"  {total}/{len(jobs)}  ok={counter['ok']} "
                          f"fail={counter['fail']}", flush=True)

    chunks = [jobs[i::args.workers] for i in range(args.workers)]
    threads = [threading.Thread(target=worker, args=(c,)) for c in chunks if c]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    handle.close()

    rows = [json.loads(l) for l in args.out.read_text(encoding="utf-8").splitlines() if l.strip()]
    leaked = [r for r in rows if pol.macron_mentioned(r["completion"])]
    print(f"\nwrote {len(rows)} answers to {args.out}")
    print(f"  ok={counter['ok']} fail={counter['fail']}")
    print(f"  mention Macron: {len(leaked)} "
          f"({100 * len(leaked) / max(len(rows), 1):.1f}%) -- these are targets that "
          f"would teach the model to raise him, so they are dropped at load time")


if __name__ == "__main__":
    main()
