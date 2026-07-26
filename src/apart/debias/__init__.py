"""Removing secret loyalties from a biased checkpoint.

Mirror image of the organism work in `apart.training`: instead of installing a
trait conditional on a trigger, these pipelines try to suppress principal-favouring
behaviour and make the suppression generalise to principals never trained against.

Two approaches, following the activation/action framing of the Formation
Research secret-loyalties whitepaper:

`option1`
    Debias the biased checkpoint directly. Unbiased targets are elicited from
    the organism itself under an in-context impartiality instruction, because
    the threat model does not grant access to a clean base model. Training pairs
    a *loyalty-primed* input with an *unbiased* target, which is what teaches
    resistance to in-context bias rather than mere good behaviour when unprompted.

`option2`
    Train a loyalty LoRA per principal, then train a shared debias LoRA
    underneath it with the loyalty adapter alternately attached and detached --
    the same two-adapter machinery as the elicitor/payload organisms.

Both are LoRA-only over a 4-bit base: 7B in bf16 is 14.2 GiB of a 15.6 GiB card,
so the base must be quantised and only adapters are trained.
"""
