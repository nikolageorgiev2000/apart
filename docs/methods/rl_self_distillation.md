# RL self-distillation

The mathematical reference is Hübotter et al.,
[Reinforcement Learning via Self-Distillation](https://arxiv.org/pdf/2601.20802).
For a student-sampled response token \(y_t\), the experiment uses

$$A_t = \operatorname{stopgrad}\left[\log q_\text{base}(y_t \mid x, f, y_{<t}) - \log \pi_\text{LoRA}(y_t \mid x, y_{<t})\right]$$

and minimizes the response-token mean of

$$-A_t \log \pi_\text{LoRA}(y_t \mid x, y_{<t})$$

The implementation intentionally differs from SDPO in two ways:

1. The self-teacher is a frozen base model with a privileged system prompt. LoRA is
   disabled during teacher scoring, so the teacher does not evolve with the student.
2. Only the sampled response token is used at each position, rather than full-vocabulary
   or top-k logit distillation.

Rollouts are on-policy and are discarded after their corresponding optimizer update.
Teacher log-probabilities and advantages are detached. There is no advantage clipping or
normalization in the baseline configuration.

