# Subliminal learning

Subliminal learning is ordinary supervised causal language-model training on completions
sampled from the frozen, system-prompted base model.

The target pool contains ten independently sampled completions for every prompt and teacher
variant. Epoch `e` uses completion index `e`, so ten cached completions support ten epochs
without target reuse. The first experiment prepares the complete pool but trains for one
epoch.

Teacher cache generation uses its own Hugging Face generation profile and a fixed batch
size of 16 by default. Student rollouts and evaluation retain their smaller batch size.

The student sees the original prompt without the teacher system instruction. Prompt and
padding labels are `-100`; completion tokens are trained with standard causal
cross-entropy. No teacher logits, KL loss, advantages, or importance weights are computed
during cached training.

`resample_each_epoch` is also supported. It creates one fresh completion per prompt at the
start of each epoch, before student updates for that epoch.
