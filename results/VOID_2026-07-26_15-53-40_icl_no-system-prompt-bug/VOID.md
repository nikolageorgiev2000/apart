# Void run — do not use

Trained with `train_option1`, which reads the bias system prompt from rows
marked `kind == "biased"`. This driver passed the prompt on the `unbiased` rows
instead, so the lookup was empty and **every batch trained with no system
prompt**. The 250 anchor rows were also dropped (no matching `unbiased` key),
which is why it ran 150 steps instead of 212.

The arm therefore never saw a biased input. Its `train/priming_gap`
0.562 -> 0.500 measures nothing about ICL correction.

Superseded by the rerun, which shares `train_unbias` with the lora arms:
0.562 -> 0.025 train, 0.600 -> 0.000 held-out.
