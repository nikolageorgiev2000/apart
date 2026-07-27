import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = pathlib.Path("results"); OBJ={"KL":"#1baf7a","DPO":"#2a78d6"}
MUTED, GRID = "#52514e", "#e1e0d9"
kl = {"lora_kl":"self", "lora_kl_filtered":"filtered", "lora_kl_external":"external"}
dpo = {"lora_dpo":"weights, self", "icl_dpo":"context, self",
       "lora_dpo_external_oracleanchor":"weights, oracle"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.3))
sty = ["-", "--", ":"]
for i,(k,lab) in enumerate(kl.items()):
    h=json.loads((R/k/"train_history.json").read_text())["history"]
    s=[r["step"] for r in h]; frac=[r["kl"]/r["loss"] for r in h]
    a1.plot(s, frac, sty[i], marker="o", ms=3.5, lw=1.4, color=OBJ["KL"], label=lab)
a1.set_xlabel("optimizer step", fontsize=8); a1.set_ylabel("KL share of loss", fontsize=8)
a1.set_ylim(0,0.35); a1.legend(fontsize=6.5, frameon=False, title="targets", title_fontsize=6.5)
for i,(k,lab) in enumerate(dpo.items()):
    h=json.loads((R/k/"train_history.json").read_text())["history"]
    s=[r["step"] for r in h]; m=[r["margin"] for r in h]
    a2.plot(s, m, sty[i], marker="o", ms=3.5, lw=1.4, color=OBJ["DPO"], label=lab)
a2.axhline(0, color=MUTED, lw=0.8)
a2.set_yscale("symlog", linthresh=10)
a2.set_xlabel("optimizer step", fontsize=8); a2.set_ylabel("DPO margin (per example)", fontsize=8)
a2.legend(fontsize=6.5, frameon=False, title="bias, targets", title_fontsize=6.5)
for ax in (a1,a2):
    ax.tick_params(labelsize=7); ax.spines[["top","right"]].set_visible(False)
    ax.grid(True, lw=0.5, color=GRID, zorder=0); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("paper/figures/dynamics.pdf", bbox_inches="tight")
print("ranges: KL share",
      [round(min(json.loads((R/k/'train_history.json').read_text())['history'][i]['kl']/
                 json.loads((R/k/'train_history.json').read_text())['history'][i]['loss']
                 for i in range(len(json.loads((R/k/'train_history.json').read_text())['history']))),3)
       for k in kl])
for k in dpo:
    h=json.loads((R/k/"train_history.json").read_text())["history"]
    print(f"  {k:<32} margin {min(r['margin'] for r in h):.1f} .. {max(r['margin'] for r in h):.1f}")
