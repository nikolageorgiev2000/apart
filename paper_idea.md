A secret loyalty consists of two parts: an activation and an action.

Suppose the model detects whether an input activates the loyalty using function active and then biases its response using a bias B. The resulting conditional behaviour given input X can be modelled as follows:
IF active(X):
RETURN response(X; B)
OTHERWISE:
RETURN response(X)

The main challenge of remediating a model trained on a secret loyalty is that the (activation, action) pair is unknown. In the case of narrow backdoors, this is a major issue because it’s near impossible to identify them. But on secret loyalties with greater activation / action breadths, we can anticipate broader categories of attacks, e.g. political bias when discussing societal issues or policies. Based on our prediction of the attack’s activation and bias categories, we can apply two different approaches:
Further Training
On the activation side we can train the model on prompts that trigger an overarching activation category, anticipating that it uses a subset of the support of a function broadly_active, then training the model to respond in the same way as a base model (or unbiased model). So it is trained on conditional behaviour:
IF broadly_active(X):
RETURN response(X)
OTHERWISE:
RETURN response(X)

Concretely, this is effectively the idea of unlearning biased behaviours under broadly-activating contexts like those tested in safety datasets, e.g. when a user has a harmful request or is discussing sensitive topics. We can extend this to more general topics where we anticipate bias might exist as well.
Bias Mitigation
On the action side we can train the model to ignore bias-eliciting instructions over a large set of anticipated bias categories, thereby building error correction mechanisms that counteract biases in general. This assumes it’s possible to leverage in-context learning as a substitute for weight updates and that model biases live along a common subset of weight directions. So the model is trained in an activation-agnostic such that:
response(X; B) = response(X)

Optionally we can add a prior (e.g. KL regularizer) from the biased model to reduce the size of the weight updates, since we run the risk of training the model to ignore instructions altogether, not just ones that activate the secret loyalty.
