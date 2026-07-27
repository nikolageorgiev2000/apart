# Generalization Study Analysis

Run date: 2026-07-27

This partial grid gives strong evidence for the activation-side version of the
paper idea and only caveated evidence for the action-side version.

## Evidence Summary

All six organisms passed the install gate. The weight-installed preference was
large on narrow trigger prompts (delta +0.85 to +0.97), essentially absent on
broad prompts (0.00 to +0.03), and did not leak on neutral prompts.

Exp 1 direct narrow correction removed the installed narrow preference on all
six principals while preserving concrete-name answers:

- mean narrow delta: +0.922 -> +0.018
- mean fraction removed: 0.981
- mean names_option after correction: 0.950
- mean benign compliance: 0.779

Broad and neutral corrections did not remove the hidden narrow preference:

- broad mean narrow delta: +0.922 -> +0.931, fraction removed -0.010
- neutral mean narrow delta: +0.922 -> +0.914, fraction removed +0.009
- names_option stayed high: 1.000 for broad, 0.996 for neutral

Held-out narrow splits transferred substantially:

- cross-frame: mean +0.912 -> +0.188, fraction removed 0.792
- cross-topic: mean +0.879 -> +0.104, fraction removed 0.879

Trump cross transfer was clean (`xframe` +0.93 -> +0.08, `xtopic` +0.90 ->
+0.03). Ardern cross transfer was weaker but still large (`xframe` +0.89 ->
+0.29, `xtopic` +0.86 -> +0.18).

## Interpretation

The Exp 1 pattern is the main positive result. Repair trained on the same
narrow activation removes the backdoor. Repair trained on nearby held-out
narrow subsets often transfers. Repair trained on broad or neutral prompts
does not. Because names_option remains high, these removals are not explained
by the model going quiet or refusing to name leaders.

This supports the claim that the model has learned a conditional narrow
activation, and that correction generalizes according to activation proximity
rather than according to broad semantic category alone.

The one Exp 2 validation arm should not be treated as a clean action-side
result. It removed the held-out ICL instruction effect (`icl_gap` +1.00 ->
0.00), but it did not remove the installed Trump weight bias (+0.92 -> +0.84)
and benign compliance collapsed to 0.225. The current Exp 2 recipe therefore
looks like overcorrection toward instruction ignoring.

## Artifacts

- Summary JSON: `results/generalization/summary.json`
- Exp 1 figure: `results/generalization/figures/exp1_bands.png`
- Exp 2 partial heatmap: `results/generalization/figures/exp2_heatmap.png`
- Guard figure: `results/generalization/figures/guards.png`

## Recommended Next Step

Do not run the full Exp 2 sweep with the current recipe unless the intended
measurement is explicitly "does an overcorrecting instruction-ignore adapter
also remove the weight bias?" For the cleaner action-side test, try a drift
guard first, such as KL regularization or smaller update budget, then rerun the
single validation Exp 2 cell and require benign compliance to stay usable.
