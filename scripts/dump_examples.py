#!/usr/bin/env python
"""Regenerate a few evaluation completions per arm, for qualitative appendix use.

Evaluation completions are not persisted by the run driver, so the paper can
otherwise only quote lexical metric values. This samples a fixed prompt set from
selected saved correction adapters, in the shipped configuration (bias adapter
off, correction on), under the bias system prompt -- i.e. the exact condition
the priming gap scores.
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apart.debias import political as pol
from apart.debias.evaluate import names_concrete_option
from apart.debias.models import DEBIAS, load_quantized
from apart.debias.political_train import active
from apart.debias.sampling import SampleRequest, generate

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--arms", nargs="+", required=True)
p.add_argument("--n", type=int, default=8)
p.add_argument("--principal", default="merkel")
p.add_argument("--out", type=Path, default=ROOT / "results/examples.json")
a = p.parse_args()

spec = pol.load_principal(a.principal)
prompt = pol.load_bias_prompt(spec)
rows = pol.load_pool(a.n, seed=7, band="political")
out = {}
for arm in a.arms:
    cand = [d for d in (ROOT / "outputs/political").glob(f"*_{arm}")
            if not d.name.startswith(("SMOKE_", "DEAD_", "VOID_"))]
    adapter = (sorted(cand)[-1] / "unbias_adapter") if cand else None
    have = adapter is not None and (adapter / "adapter_model.safetensors").exists()
    b = load_quantized("Alamerton/sl-organism-b-7b", lora_rank=32,
                       adapter_names=(DEBIAS,) if have else (),
                       adapter_paths={DEBIAS: str(adapter)} if have else None,
                       trainable_adapter=None, gradient_checkpointing=False)
    reqs = [SampleRequest(a.principal, r["id"], r["prompt"], "primed", prompt) for r in rows]
    with active(b, [DEBIAS] if have else []):
        got = generate(b, reqs, max_new_tokens=192, batch_size=8, progress=False)
    out[arm] = [{"prompt": r["prompt"], "completion": g["completion"],
                 "favours": pol.favours(g["completion"], spec),
                 "names_option": names_concrete_option(g["completion"])}
                for r, g in zip(rows, got, strict=False)]
    n = len(out[arm])
    print(f"  {arm:<34} favours {sum(x['favours'] for x in out[arm])/n:.2f}"
          f"  names {sum(x['names_option'] for x in out[arm])/n:.2f}", flush=True)
    del b; gc.collect()
    import torch; torch.cuda.empty_cache()
a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(f"written to {a.out}")
