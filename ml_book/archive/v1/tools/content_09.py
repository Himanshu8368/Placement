"""Volumes 09–12: neural nets, training, vision, language."""

from booklib import chapter, entry, volume

v = volume(
    9,
    "Neural networks",
    "University",
    "A reader who can differentiate a vector loss and can run the gradient-check harness.",
    "What function does a network represent, how do gradients flow, and which parts are inductive bias rather than magic?",
)

chapter(
    v,
    "01_computation_graph",
    "Computation graphs",
    entry(
        question="How should you represent a network so that a forward value and a backward gradient are the same object?",
        card=[
            ("Graph", "A directed acyclic graph of tensors. Each node is a function of its parents."),
            ("Forward", "Evaluate nodes in topological order. Cache the inputs each node needs for its derivative."),
            ("Backward", "Apply the chain rule from the scalar loss toward the parameters, using cached values."),
            ("Parameter", "A leaf the loss depends on and that you intend to update. A constant is a leaf you do not update."),
            ("Shape", "Every node has a shape contract. A broadcast inside a node is part of the contract, not an accident."),
        ],
        theory="""A network is a function built by composition. The computation graph is the bookkeeping that makes the composition differentiable and implementable. Autodiff libraries hide the graph. A reference implementation should show one graph you can draw: an affine map, a pointwise nonlinearity, another affine map, and a loss. If you cannot name the intermediate tensors, you cannot check them.

Caching is required because the local derivative often depends on the forward value. ReLU's derivative depends on the sign of the pre-activation. A sigmoid's derivative can be written from the sigmoid output. Throwing away the forward pass and hoping to differentiate the formula from parameters alone is algebraically possible and operationally the wrong interface.

The loss is a node that reduces to a scalar. Gradients are defined for that scalar. If you forget the reduction and backpropagate a vector, you have invented an implicit sum or you have a bug. Say which reduction you used, mean or sum, because it rescales every gradient and therefore the learning rate.""",
        derivation="""Let $z = Wx + b$, $h = \\phi(z)$, and $L$ a scalar depending on $h$. The chain rule in differentials is $dL = \\langle \\nabla_h L, dh \\rangle$ and $dh = \\phi'(z) \\odot dz$, so $\\nabla_z L = \\nabla_h L \\odot \\phi'(z)$. From Volume 01, $\\nabla_W L = (\\nabla_z L) x^\\top$ and $\\nabla_b L = \\nabla_z L$. A graph engine stores $\\nabla_z L$ as the message sent to the affine node. Nothing in this calculation requires $W$ to be small. It requires the shapes of $\\nabla_z L$ and $z$ to match.""",
        code="""import numpy as np

def affine_forward(x, W, b):
    return x @ W + b, (x, W, b)

def affine_backward(grad_z, cache):
    x, W, _b = cache
    grad_W = x.T @ grad_z
    grad_b = grad_z.sum(axis=0)
    grad_x = grad_z @ W.T
    return grad_x, grad_W, grad_b
""",
        experiment="""**Hypothesis.** The affine backward pass matches central differences on a scalar loss $L = \\sum z_{ij}^2$ through $z = xW + b$, at a small random shape.

**Protocol.** $n = 4$, input dimension 3, output dimension 2. Use the Volume 02 harness on the flattened parameters of $W$ and $b$.

**Metric.** Relative gradient gap at $\\varepsilon = 10^{-6}$.

**Decision rule.** The gap should be below $10^{-6}$. A gap near $1$ often means a missing transpose or a mean/sum mismatch. Fix that before any training chapter.""",
        project="""**Deliverable.** A drawn graph for a two-layer network, the affine check, and a note that states whether your training loss is a mean or a sum.

**Constraints.** The check uses float64 and a deterministic loss.

**Rubric.** The relative gap is in the lab note next to the picture of the graph.""",
        beginner="Implement affine forward and check one matrix product by hand.",
        university="Pass the gradient check for $W$ and $b$.",
        research="Add a batch-norm or layer-norm node only after you can write its forward and backward contracts. If you use a library instead, check one parameter gradient against a numerical difference with the library's own forward.",
        failures="""- Broadcasting a bias of the wrong rank and accepting the forward value because it is finite.
- Reducing with `mean` in the loss and `sum` in the hand-derived gradient.
- Detaching a tensor so the graph looks connected and the gradient is zero.""",
        see_also="The next chapter for the nonlinearity. Volume 02 for the harness you must reuse.",
    ),
)

chapter(
    v,
    "02_backprop",
    "Backpropagation",
    entry(
        question="What does backpropagation compute, and which bugs does a gradient check miss if you only test one layer?",
        card=[
            ("Output", "The gradient of a scalar loss with respect to each registered parameter."),
            ("Local", "Each node contributes a Jacobian-vector product, not a materialised Jacobian, when the output dimension is large."),
            ("ReLU", "The local factor is $1\\{z > 0\\}$, with a declared value at 0."),
            ("Sigmoid", "The local factor is $s(1-s)$ if $s = \\sigma(z)$."),
            ("Init", "Gradients can vanish or explode with depth even when the code is correct. That is a modeling issue, checked separately from the chain rule."),
        ],
        theory="""Backpropagation is the chain rule evaluated from the loss backward, reusing the forward cache. It is not an approximation. If the numerical check fails, the code is wrong or the point sits on a kink you did not define. If the check passes and training still diverges, the learning rate, the scaling, or the loss surface is the suspect, not the chain rule.

Deep stacks multiply local factors. A product of many numbers smaller than 1 vanishes. A product of numbers larger than 1 explodes. ReLU's factor is 0 or 1, so it avoids shrinking, and it kills gradients on the inactive side. Sigmoid factors are at most $1/4$. These facts are why activation choice and initialization share a chapter later. Do not "fix" a vanished gradient by multiplying a fudge factor you cannot explain.

A two-layer check should cover the composition, not only the affine piece. Bugs hide in the interaction: a ReLU backward that uses the wrong pre-activation, or a loss backward that forgets a factor of 2.""",
        derivation="""Let $h = \\mathrm{ReLU}(z) = \\max(z, 0)$ elementwise, and define the derivative at 0 to be 0. Then $dh = 1\\{z > 0\\} \\odot dz$, so the incoming gradient is multiplied by that mask. For $s = \\sigma(z)$, $ds = s(1-s) \\odot dz$. For squared error $L = \\lVert h - y \\rVert_2^2$, $\\nabla_h L = 2(h - y)$. Compose these messages to get $\\nabla_z L$. The parameter gradients then use the affine rules. Write this composition for one hidden layer before you write a loop over layers.""",
        code="""import numpy as np

def relu(z):
    return np.maximum(z, 0.0), (z > 0.0)

def relu_backward(grad_h, mask):
    return grad_h * mask

def mse_backward(pred, y):
    return 2.0 * (pred - y)
""",
        experiment="""**Hypothesis.** A one-hidden-layer map $f(x) = W_2 \\mathrm{ReLU}(W_1 x + b_1) + b_2$ has analytic gradients matching central differences at a random parameter, on a single example, for the squared loss.

**Protocol.** Input dimension 3, hidden 4, output 2, float64, $\\varepsilon = 10^{-6}$. Include a point where some hidden units are off, so the mask is not all ones.

**Metric.** Relative gap for each parameter tensor.

**Decision rule.** Every gap below $10^{-5}$. If only $b_1$ fails, the mask or the sum over the batch is wrong. Do not start the training volume on a failed check.""",
        project="""**Deliverable.** A minimal network class with `forward`, `backward`, and `grad_check`, plus a lab note of the gaps. No library autodiff in the checked path.

**Constraints.** One hidden layer is enough. The loss reduction is written in the note.

**Rubric.** The check covers a case with inactive ReLU units. A check where every unit is active is marked incomplete.""",
        beginner="Check the ReLU mask on a vector with positive and negative entries.",
        university="Pass the full one-hidden-layer gradient check.",
        research="Replace the scalar loss by a softmax cross-entropy and derive the gradient in the logits, $p - y_{\\mathrm{one\\,hot}}$. Check it. This is the classification head you will actually train.",
        failures="""- Checking gradients only at initialization when every ReLU happens to be active, then failing later.
- Using an in-place operation that corrupts the cached pre-activation.
- Comparing a mean-reduced library gradient with a sum-reduced hand gradient and "fixing" it by changing the mathematics at random.""",
        see_also="Volume 02's harness. Volume 10, which assumes this check already passed.",
    ),
)

chapter(
    v,
    "03_inductive_bias",
    "Architectures and inductive bias",
    entry(
        question="What does an architecture assume about the data, before any weight is fit?",
        card=[
            ("Bias", "A restriction on the functions the model can represent easily, or at all."),
            ("MLP", "A stack of affine maps and pointwise nonlinearities. It assumes little spatial structure and pays for that with data."),
            ("Weight sharing", "The same parameter applied at many positions, as in a convolution or a recurrent step. It assumes a repeated local rule."),
            ("Equivariance", "Transforming the input transforms the output in a matching way. A claim about the architecture, true only if the implementation honors it."),
            ("Choice", "Pick the bias that matches a symmetry you are willing to defend. A flexible model is a bias toward needing more data."),
        ],
        theory="""An architecture is a family $\\mathcal{F}$. The inductive bias is what that family prefers. Without a preference, finite data do not select a unique continuation of a finite sample. Convolutions prefer local, translation-structured rules. Attention, in a later volume, prefers content-based routing and pays quadratic cost in sequence length unless you change it. A fully connected layer on pixels prefers nothing about neighborhoods and must learn locality from scratch.

Bias is not a vibe. It is a statement you can falsify. If a convolution cannot represent a function that depends on absolute position, and your labels depend on absolute position, the architecture is wrong or you must add a position feature. If it can represent the function only by using many filters to simulate a lookup table, it can represent it in principle and still be the wrong bias in practice.

Depth changes which functions are compact. A deep composition can represent some hierarchical functions with fewer parameters than a shallow network of the same family. That is a theorem in restricted settings and a tendency in practice. It is not a reason to add layers when a linear model has not been beaten.""",
        derivation="""A 1D convolution with kernel $w \\in \\mathbb{R}^{k}$ and stride 1, ignoring boundaries, produces $(w * x)_t = \\sum_{u=0}^{k-1} w_u x_{t+u}$. Shifting the input by $s$ shifts the output by $s$. That is equivariance to translation. A fully connected layer $Wx$ does not have this property unless $W$ itself is constrained to be constant along diagonals, which is the convolution written as a matrix. Global average pooling then turns translation equivariance of the feature map into translation invariance of the pooled vector. If your label flips when the pattern moves, invariance is the wrong last step.""",
        code="""import numpy as np

def conv1d_same_width(x, w):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    pad = len(w) // 2
    xp = np.pad(x, pad)
    out = np.array([np.dot(w, xp[i : i + len(w)]) for i in range(len(x))])
    return out
""",
        experiment="""**Hypothesis.** `conv1d_same_width` commutes with a shift: shifting a long input and then convolving matches convolving and then shifting, away from the borders. A dense random matrix does not.

**Protocol.** A random kernel of length 3, an input of length 30, shift by 4. Compare the overlapping region. Apply a random $30 \\times 30$ matrix to the input and to the shifted input and show the outputs are not shifts of each other.

**Metric.** Max absolute error on the overlap for the convolution. A qualitative counterexample for the dense map is enough if you also report one numeric gap.

**Decision rule.** Convolution error near $10^{-12}$ on the overlap. If it fails, the padding or the kernel alignment is wrong. Do not claim equivariance you did not test.""",
        project="""**Deliverable.** A short architecture card for MLP versus 1D convolution on a synthetic problem where the label is a local pattern that can appear at any position, and a second problem where the label depends on the absolute index. Say which family is appropriate for each.

**Constraints.** You may demonstrate with a hand-built convolutional feature rather than a trained network if training is not ready. The claim is about the family.

**Rubric.** Each choice names the symmetry it assumes and the experiment that would embarrass it.""",
        beginner="Implement the convolution and test a kernel `[1, 0, -1]` on a short array.",
        university="Complete the shift experiment and the two-problem card.",
        research="State, for images, what a $3 \\times 3$ convolution assumes that a vision transformer with a position embedding does not. You will return to this in Volume 11. Do not treat either as universally more powerful.",
        failures="""- Using a fully connected stack on raw pixels because it is "more general," with a small sample.
- Padding in a way that breaks the shift test and still claiming translation equivariance.
- Adding capacity until the training loss vanishes and calling the result an inductive bias.""",
        see_also="Volume 11 for images. Volume 12 for sequence models and attention.",
    ),
)

chapter(
    v,
    "04_initialization",
    "Initialization and activations",
    entry(
        question="How should weights start so that a forward pass and a backward pass have a sane scale?",
        card=[
            ("Goal", "Pre-activations with order-1 variance at the start of training, not 0 and not $10^6$."),
            ("Scale", "For a layer with fan-in $d$, a common choice is variance $1/d$ or $2/d$ for ReLU, so a sum of $d$ terms does not grow like $d$."),
            ("Symmetry", "Identical rows of $W$ stay identical under many updates. Initialize randomly."),
            ("Activation", "Match the init to the activation. A sigmoid wants smaller inputs than a ReLU."),
            ("Check", "Histogram or variance of each layer's pre-activation on a minibatch, before any claim about learning rate."),
        ],
        theory="""Initialization is a scale model of the untrained network. If the first forward pass saturates every sigmoid, the gradient check may still pass and the training step will not move. If the first forward pass overflows, you will chase a learning rate that was never the bug.

The fan-in rule is a variance calculation under an independence assumption that is only roughly true. It is good enough to start, and it is something you verify by printing variances. Residual connections and normalization layers change the right init. Copying an init from a paper whose architecture you did not copy is a superstition.

Biases often start at zero. A ReLU network with zero bias and symmetric data can start with half the units off. That can be acceptable. A dead unit that never turns on for any training point stays dead if its gradient is zero. Watch the fraction of active units.""",
        derivation="""Let $z = \\sum_{j=1}^{d} W_j x_j$, with $x_j$ independent, mean 0, variance 1, and $W_j$ independent, mean 0, variance $v$. Then $\\mathrm{Var}(z) = d v$. Choosing $v = 1/d$ keeps $\\mathrm{Var}(z) = 1$. For ReLU, $h = \\max(z, 0)$ roughly halves the variance if $z$ is symmetric, because half the values are zero and the positive half carries the second moment. The He heuristic uses $v = 2/d$ to put that factor back. It is a heuristic: ReLU of a Gaussian is not a unit-variance Gaussian. Measure the variance anyway.""",
        code="""import numpy as np

def init_linear(fan_in, fan_out, rng, relu=True):
    scale = np.sqrt(2.0 / fan_in) if relu else np.sqrt(1.0 / fan_in)
    W = rng.normal(scale=scale, size=(fan_in, fan_out))
    b = np.zeros(fan_out)
    return W, b
""",
        experiment="""**Hypothesis.** With $v = 2/d$ and ReLU, a stack of several layers keeps the empirical variance of activations within a small factor of 1 on standard normal inputs. With $v = 1$, the variance collapses as depth grows. With $v = 1$ replaced by a constant $0.5$ independent of $d$, the variance explodes.

**Protocol.** $d = 64$, depth in $\\{1, 4, 8\\}$, 256 random inputs, three init rules.

**Metric.** Mean over units of the empirical variance of each layer's output.

**Decision rule.** The $2/d$ stack should stay order-1 longer than the others. If all three explode, the input variance is not 1 or the ReLU is not applied. This experiment is about initialization, so do not tune it until the story appears; report the curves you get.""",
        project="""**Deliverable.** The depth sweep, plus the fraction of dead ReLU units at initialization on your synthetic input. Choose one init for the later training project and write the variance rule in the config.

**Constraints.** No data-dependent init that uses the test set.

**Rubric.** You can say what would change if the inputs were not standardized.""",
        beginner="Initialize one layer both ways and print $\\mathrm{Var}(z)$.",
        university="Run the depth sweep.",
        research="Derive the backward-variance heuristic: gradients can require a fan-out scaling that disagrees with the forward fan-in scaling. Name a normalization layer as the practical ceasefire, and say what it does to the scale.",
        failures="""- Zero initialization of a whole weight matrix in a hidden layer.
- A learning-rate search on a saturated network.
- Importing a pretrained model's init into a different depth without measuring activations.""",
        see_also="Volume 10 for learning rates that assume this scale. Volume 11 for convolutions, where fan-in includes the kernel area.",
    ),
)

chapter(
    v,
    "05_approximation",
    "What universal approximation does not give you",
    entry(
        question="A wide enough network can approximate many functions. Why is that not a training plan?",
        card=[
            ("Theorem shape", "For a non-polynomial activation, one hidden layer can approximate continuous functions on a compact set, arbitrarily closely, if the width may grow."),
            ("Not included", "A bound on the width you need, the sample size, the optimization, or the test risk."),
            ("Compact set", "The guarantee is on a bounded domain. Extrapolation is a different claim."),
            ("Existence", "The theorem says some weights exist. It does not say gradient descent finds them."),
            ("Use", "A sanity check that the family is not the obstacle. The obstacle may still be data, optimization, or the wrong loss."),
        ],
        theory="""Universal approximation is an existence result about families. It tells you not to blame "neural nets cannot represent that" for every failure, once the domain is compact and the function is continuous, given enough width. It does not tell you that your width, your finite sample, or your optimizer are enough. Polynomials are also dense on a compact interval. You do not ship a degree-10,000 polynomial because of that sentence.

The rate at which width must grow can be ugly. A function with a sharp localized bump needs the bump to be covered by some unit. In high dimension, covering a domain with localized bumps is expensive. Architecture and features are how you avoid paying that price. A convolution that matches the symmetry can represent the function you want with far fewer samples than a dense net that could also represent it in the limit.

Optimization makes the gap practical. A representable function is irrelevant if the loss landscape and the initialization do not reach it. Cite the theorem when someone claims a network family is incapable. Do not cite it when the training loss has stalled.""",
        derivation="""A standard constructive picture uses localized bumps. Suppose $\\phi$ is a sigmoid. A unit $\\phi(w(x - c))$ with large $w$ approximates a step at $c$. A difference of two steps approximates a spike. Summing spikes of chosen heights approximates a continuous function uniformly on an interval, given enough spikes. The number of spikes is the width. The construction uses weights that grow with the desired sharpness. Gradient descent from a small init is not this construction. The proof shows membership in the closure of the family, not a training algorithm.""",
        code="""import numpy as np

def spike_basis(x, centers, width):
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    return np.exp(-0.5 * ((x[:, None] - c[None, :]) / width) ** 2)
""",
        experiment="""**Hypothesis.** On $[0, 1]$, a bump basis with enough centers approximates $\\sin(2\\pi x)$ in least squares better than a linear function, and a bump basis with centers only on $[0, 0.2]$ fails on $[0.8, 1]$.

**Protocol.** Fit ridge regression on the bump features. Evaluate on a grid. Compare widths corresponding to 4 and 20 centers, and the localized-center failure.

**Metric.** Max absolute error on the grid, and on the right-hand interval separately.

**Decision rule.** More centers covering the interval should reduce error. Centers that do not cover the interval must leave a large error there, which is the compact-set and support caveat in miniature.""",
        project="""**Deliverable.** The bump approximation, plus a paragraph that rewrites one real failure you have seen as representation, estimation, or optimization, with the evidence that distinguishes them. Training loss near zero and test loss high points at estimation or leakage, not at universal approximation. Training loss stuck high can be optimization or representation.

**Constraints.** Do not invoke the theorem without naming which hypothesis of the theorem you checked.

**Rubric.** The paragraph would change a teammate's next experiment.""",
        beginner="Fit 8 bumps to a sine wave and plot or tabulate the error.",
        university="Add the uncovered-interval failure.",
        research="State a curse-of-dimensionality caveat: a grid of bumps with 10 centers per axis in $d$ dimensions has $10^d$ centers. Connect this to why image models do not one-hot the pixel cube.",
        failures="""- "The network can approximate any function, so more epochs will fix the test error."
- Quoting a theorem that assumes a compact domain while predicting far outside the training box.
- Confusing a failed optimizer with a limited family, or the reverse, without looking at the training loss.""",
        see_also="Volume 06 for approximation versus estimation. Volume 10 for the optimization half.",
    ),
)

chapter(
    v,
    "06_numpy_network_project",
    "Project: a network in NumPy",
    entry(
        question="Can you train a small network whose gradients you checked, and beat a linear baseline for a reason you can name?",
        card=[
            ("Model", "One hidden layer, ReLU, squared loss or softmax cross-entropy."),
            ("Check", "The gradient check passes before training."),
            ("Baseline", "Linear or logistic regression on the same features and split."),
            ("Data", "A synthetic problem with a known nonlinearity, plus the linear problem where the network should not win by much."),
            ("Report", "Training loss, validation loss, test metric, and the baseline, across a few seeds."),
        ],
        theory="""This is the first trained network in the series. Its job is to be understood. A library model that beats a baseline without a gradient check is a later convenience, not a substitute. The synthetic nonlinear problem should be one where Volume 06 already showed a linear floor: the sine wave, or a product interaction $Y = X_1 X_2 + \\varepsilon$. The network should be able to lower that floor. On a purely linear problem it should tie the linear model within noise after you give the linear model the same features.

Overfitting a tiny sample is part of the lesson. If the hidden width is large and $n$ is small, training loss will drop and validation loss will rise. Record that curve. It is the picture Volume 10 will try to control. Do not hide it by reporting only the best epoch on the test set.""",
        derivation="""The update is $\\theta \\leftarrow \\theta - \\eta \\widehat{\\nabla} L(\\theta)$ with the analytic gradient from backpropagation and a minibatch or the full sample. For a full-batch step on a smooth loss, a sufficiently small $\\eta$ decreases $L$. ReLU is not smooth at 0. In practice a small step that does not sit exactly on the kink still behaves. If the loss increases on the first step, $\\eta$ is too large or the gradient is scaled for a different reduction. Divide $\\eta$ by 10 before rewriting the network.""",
        code="""import numpy as np

def sgd_step(params, grads, eta):
    return [p - eta * g for p, g in zip(params, grads)]
""",
        experiment="""**Hypothesis.** On $Y = \\sin(2\\pi X_1) + 0.5 X_2 + \\varepsilon$, the checked network achieves lower test MSE than linear regression in $(X_1, X_2)$. On $Y = X_1 + X_2 + \\varepsilon$, the gap disappears within seed noise.

**Protocol.** $n_{\\mathrm{train}} = 200$, validation 200, test 1000, hidden width 32, a few seeds, $\\eta$ chosen on the validation loss from a short log grid. Stop using the validation set for early stopping and do not retune on the test set.

**Metric.** Median test MSE versus the linear baseline.

**Decision rule.** The nonlinear problem should show a gap larger than the seed scatter. The linear problem should not crown the network if the gap is inside the paired interval from Volume 07. If the network loses on the sine wave, look at training loss. High training loss means optimization or init. Low training loss and high test loss means capacity and a small sample.""",
        project="""**Deliverable.** The gradient-checked network, both datasets, the baseline table, learning curves of training and validation loss, and a model card.

**Constraints.** NumPy only on the checked path. A library reimplementation is optional and must match the NumPy loss on one batch within a tight tolerance.

**Rubric.** You can explain a failure mode you actually hit: dead units, a too-large step, or a mean/sum bug. A smooth success with no note of the check is incomplete.""",
        beginner="Train on the sine wave with full-batch gradient descent for a fixed small $\\eta$ and plot the loss.",
        university="Complete both datasets, the baseline, and the card.",
        research="Add a second hidden layer only after the first layer's check passes. Compare parameter counts at equal validation loss. Do not add depth because the theorem said depth is expressive.",
        failures="""- Early-stopping on the test curve.
- A width so large that you memorize 200 points and call it a solved sine wave.
- Skipping the linear problem where the network should not look brilliant.""",
        see_also="Volume 10 for a disciplined training loop. Volume 11 if you replace the MLP with a convolution on images.",
    ),
)

v = volume(
    10,
    "Training deep models",
    "University to research",
    "A reader who has a gradient-checked network and a validation split.",
    "Which knobs control optimization, which control generalization, and how do you tell them apart?",
)

chapter(
    v,
    "01_losses",
    "Losses for modern models",
    entry(
        question="What does each common training loss estimate, and when is it the wrong reduction?",
        card=[
            ("Squared", "A conditional mean. Sensitive to outliers."),
            ("Logistic / cross-entropy", "A conditional categorical distribution. Proper for probabilities."),
            ("Margin", "A ranking or a separation constraint. Not a probability unless you add a calibration step."),
            ("Class weights", "A change of the empirical measure toward rare classes. They change the functional you estimate."),
            ("Reduction", "Sum and mean differ by $n$ or by the batch size. The optimizer must know which you used."),
        ],
        theory="""The loss is the definition of the task inside the optimizer. Changing it changes the Bayes predictor. Class-weighted cross-entropy does not estimate $P(Y \\mid X)$ under the original data law. It estimates the conditional probability under a reweighted law. That can be what you want for a rare class, and it will miscalibrate probabilities for the original law unless you correct them. Write the reweighting in the card.

Label smoothing replaces one-hot targets by a mixture with a uniform distribution. It is a regularizer and a change of target. The model's best possible log loss is no longer zero. A training loss that stops falling at a positive number may be correct.

For sequence models the loss is usually a sum over tokens. A longer document then contributes more gradient. Mean-over-tokens and mean-over-documents are different estimators. Pick the one that matches the decision, and do not compare two runs that reduced differently.""",
        derivation="""The gradient of mean squared error on a batch of size $B$ is $B^{-1}$ times the gradient of the sum. If you tune $\\eta$ for the sum and then switch the code to the mean, you have multiplied the effective step by $1/B$. A mysterious need to increase the learning rate by about $B$ after a "cleanup" refactor is this factor. Write $\\eta$ next to the reduction in the config hash.

For softmax cross-entropy, if $p = \\mathrm{softmax}(z)$ and $y$ is one-hot, $\\nabla_z L = p - y$ for the sum of per-example losses, or that quantity over $B$ for the mean. The $p - y$ form is the one to gradient-check. A sign error trains the model to be confidently wrong, and the loss rises.""",
        code="""import numpy as np

def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy_mean(z, y_index):
    p = softmax(z)
    picked = p[np.arange(len(y_index)), y_index]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))
""",
        experiment="""**Hypothesis.** Doubling the batch size while switching from a summed loss to a mean loss, without changing $\\eta$, changes the parameter step unless $\\eta$ is adjusted by the same factor.

**Protocol.** One affine logistic model, one batch, compute the gradient under sum and under mean. Compare the proposed steps $\\eta g$.

**Metric.** The ratio of step norms.

**Decision rule.** The ratio should equal the batch size when $\\eta$ is fixed. If your training code hides the ratio inside a library reduction, print it. Do not tune around an unknown factor.""",
        project="""**Deliverable.** A loss card for the network project: the formula, the reduction, the Bayes target, and a gradient check of the classification head if you use one. Include a reweighted variant only as a labeled alternative, with a sentence on calibration.

**Constraints.** The config stores `reduction` and `class_weights`.

**Rubric.** A change in reduction forces a new config hash, by Volume 03's rules.""",
        beginner="Implement softmax cross-entropy and check that a one-hot prediction of the true class has loss near 0.",
        university="Complete the sum-versus-mean step comparison.",
        research="Derive the gradient of label-smoothed cross-entropy and say what probability vector minimizes it. Check the gradient at one point.",
        failures="""- Comparing losses across runs that used different reductions.
- Reweighting classes and still reporting unweighted accuracy as if the training objective matched it.
- Clipping probabilities at 0 inside the log.""",
        see_also="Volume 06 for proper scoring. Volume 12 for token-level reductions.",
    ),
)

chapter(
    v,
    "02_optimizers",
    "Optimizers",
    entry(
        question="What state does an optimizer keep besides the parameters, and what does that state assume?",
        card=[
            ("SGD", "Step along the minibatch gradient. Optional momentum keeps a velocity."),
            ("Momentum", "A leaky average of past gradients. It accelerates in consistent directions and adds a state that must be checkpointed."),
            ("Adam", "Per-coordinate gradient scale from first and second moments. A step size that is plausible in one coordinate can still be wrong globally."),
            ("Batch size", "A larger batch reduces gradient noise and changes the $\\eta$ you should use. It is not a free improvement in generalization."),
            ("Fair comparison", "Equal data touches or equal wall time, with each method's step size tuned on validation."),
        ],
        theory="""An optimizer is a dynamical system with its own state. Momentum's velocity and Adam's moment estimates are parameters of the training trajectory. A checkpoint that stores weights and not optimizer state does not resume the same trajectory. That matters for reproducibility and for a crash in the middle of a long run.

Adam rescales coordinates that have historically small gradients so they can move, and it shrinks coordinates with huge gradients. That is helpful when features or layers have mismatched scales. It is not a proof of convergence to a good minimum. The epsilon in the denominator, the moment decay rates, and whether you debias the moments are part of the estimator. Defaults are a choice a paper made for a different problem.

Tune the step size. A fixed default that works on a tutorial is not a tuned method. The comparison that matters in this series is against SGD with momentum at its own best validation step size, at matched data budget. If Adam wins only at an untuned SGD baseline, you learned less than you think.""",
        derivation="""Classical momentum replaces the gradient step by $v \\leftarrow \\beta v + g$ and $\\theta \\leftarrow \\theta - \\eta v$, or the heavy-ball variant with the gradient evaluated at a shifted point. On a quadratic with a poor condition number, a well chosen $\\beta$ damps the zig-zag along the sharp axis that Volume 02 illustrated. The optimal $\\beta$ depends on that condition number, which you do not know. Treat $\\beta$ as a knob with a default near $0.9$ and a small validation check, not as a constant of nature.

Adam's second moment $\\hat{v}$ estimates a coordinatewise gradient energy. The step is proportional to $\\hat{m} / (\\sqrt{\\hat{v}} + \\epsilon)$. If $\\hat{v}$ is tiny and $\\epsilon$ is tinier, early steps can be enormous. If $\\epsilon$ is large, Adam becomes a sluggish sign method. Print the step norm on the first iteration.""",
        code="""import numpy as np

def sgd_momentum(theta, grad, velocity, eta, beta):
    velocity = beta * velocity + grad
    theta = theta - eta * velocity
    return theta, velocity
""",
        experiment="""**Hypothesis.** On the poorly conditioned quadratic bowl from Volume 02, momentum with a tuned $\\beta$ reaches a small radius in fewer steps than plain SGD at its best stable constant step. On a well-conditioned bowl, the gap shrinks.

**Protocol.** $L = \\tfrac{1}{2}(\\theta_1^2 + \\kappa \\theta_2^2)$ for $\\kappa \\in \\{1, 100\\}$. Grid $\\eta$ and, for momentum, $\\beta \\in \\{0, 0.9\\}$.

**Metric.** Steps to reach $\\|\\theta\\| < 10^{-3}$, or the radius after a fixed budget.

**Decision rule.** Momentum should help more at $\\kappa = 100$. If it diverges, the step is outside the stable region; that run does not count as a win for SGD, it counts as an untuned momentum run. Retune.""",
        project="""**Deliverable.** SGD, momentum, and a from-scratch Adam or a carefully wrapped library Adam, on the NumPy network's nonlinear synthetic problem. Each gets a validation-chosen learning rate. Report validation loss against data touches.

**Constraints.** Optimizer state is part of any checkpoint you save.

**Rubric.** You do not crown a method that saw more data or a method whose only competitor used $\\eta = 0.1$ by tradition.""",
        beginner="Implement momentum on the quadratic bowl.",
        university="Complete the three-way comparison on the network.",
        research="Show a small example where Adam's coordinatewise scaling helps when one input feature is multiplied by 100 and you forgot to standardize. Then say why fixing the scale is still the first move.",
        failures="""- Comparing optimizers at a single default learning rate.
- Restoring weights without optimizer state and expecting the loss curve to continue.
- Increasing batch size, keeping $\\eta$, and attributing the change to "more stable gradients" without the reduction factor from the loss chapter.""",
        see_also="Volume 02 for condition numbers. Volume 15 for training runs as jobs with checkpoints.",
    ),
)

chapter(
    v,
    "03_schedules_normalization",
    "Schedules and normalization",
    entry(
        question="How do learning-rate schedules and normalization layers change the scale of training?",
        card=[
            ("Schedule", "A rule $\\eta_t$. Common patterns are constant, step drops, and cosine decay. The rule is a knob."),
            ("Warmup", "A short increase from a tiny step, used when a large initial step is unstable, especially with adaptive methods or large batches."),
            ("Normalization", "A layer that recenters and rescales activations using batch or feature statistics, plus learned gain and bias."),
            ("Train versus eval", "Batch normalization uses batch statistics in training and a running estimate at evaluation. Forgetting the switch changes the function."),
            ("Interaction", "Normalization changes the effective step. Retune $\\eta$ after you add it. Do not stack it on an old learning rate and interpret the crash as the data."),
        ],
        theory="""A schedule encodes the belief that the useful step size shrinks as you approach a minimum or as the gradient noise starts to dominate. Volume 02's stochastic-gradient chapter already said a constant step hovers in a neighborhood. Decay is how you leave the neighborhood. Decay too early and you freeze a bad solution. The validation loss, not the folklore, decides.

Normalization makes the scale of activations less dependent on the scale of incoming weights. That often lets you use a larger step and deeper stacks. It also introduces a dependence across the batch, in batch normalization, so the function of one example depends on the other examples in the minibatch. At evaluation you must freeze the statistics or the prediction of a single deployed example is undefined or depends on its accidental neighbors.

Layer normalization computes statistics per example and avoids that particular train-eval trap. It is a different function. Copying a recipe that used one of them into an architecture that used the other is a change of model, not a cleanup.""",
        derivation="""Batch normalization of a pre-activation $z$ in a batch replaces it by $\\hat{z} = (z - \\mu_B)/\\sqrt{\\sigma_B^2 + \\epsilon}$ and then $\\gamma \\hat{z} + \\beta$. The Jacobian with respect to $z$ includes the dependence of $\\mu_B$ and $\\sigma_B$ on $z$. A gradient that treats $\\mu_B$ as constant is wrong. Library implementations include those terms. A hand implementation that forgets them can still train and will fail a strict gradient check.

A step decay multiplies $\\eta$ by a factor at fixed epochs. A cosine schedule sets $\\eta_t = \\eta_{\\min} + \\tfrac{1}{2}(\\eta_{\\max}-\\eta_{\\min})(1+\\cos(\\pi t/T))$ for a horizon $T$. Both are deterministic functions of the step index. Put $T$ in the config. Changing the number of epochs without changing $T$ changes the schedule.""",
        code="""import numpy as np

def cosine_eta(t, t_max, eta_max, eta_min=0.0):
    if t_max <= 0:
        raise ValueError("t_max must be positive")
    frac = min(max(t, 0), t_max) / t_max
    return float(eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * frac)))
""",
        experiment="""**Hypothesis.** On a convex quadratic with added gradient noise, a cosine decay ends at a smaller radius than a constant step equal to $\\eta_{\\max}$, at matched step count, when $\\eta_{\\max}$ is large enough to hover.

**Protocol.** The bowl $\\kappa = 20$, plus isotropic Gaussian gradient noise of scale $0.05$. Compare constant $\\eta_{\\max}$ and cosine from $\\eta_{\\max}$ to 0.

**Metric.** Radius at the final step, median over 20 seeds.

**Decision rule.** Cosine should finish closer to the origin when the constant step visibly jitters. If both diverge, reduce $\\eta_{\\max}$. This is a sanity check of the schedule, not a claim about networks.""",
        project="""**Deliverable.** Two training curves for the small network: constant $\\eta$ and cosine decay, with $\\eta_{\\max}$ chosen separately on validation for each. If you implement layer normalization, add a third curve and a new learning-rate sweep rather than reusing $\\eta$.

**Constraints.** Evaluation mode is irrelevant until you use batch statistics. If you do, the card must say how a single example is normalized at test time.

**Rubric.** The curves share axes of data touches and validation loss. The schedule parameters sit in the config hash.""",
        beginner="Plot `cosine_eta` over 100 steps.",
        university="Run the noisy quadratic comparison and the two network curves.",
        research="Explain the train-eval discrepancy of batch normalization in one formula, and name a failure mode in which a batch of size 1 at deployment has undefined variance.",
        failures="""- A cosine period that assumes 100 epochs after you shortened the run to 20.
- Leaving batch normalization in training mode at evaluation so the score depends on batch composition.
- Adding normalization and three other changes in one run, then attributing the gain to all of them.""",
        see_also="Volume 07 for early stopping, which can be viewed as a data-dependent schedule. Volume 11 for normalization inside convolutional nets.",
    ),
)

chapter(
    v,
    "04_regularization",
    "Regularization and augmentation",
    entry(
        question="Which knobs reduce the gap between training and test loss, and which ones just blur the data?",
        card=[
            ("Penalty", "An added term such as $(\\lambda/2)\\lVert \\theta \\rVert_2^2$. It changes the estimator and the Bayes target inside the penalized family."),
            ("Early stopping", "Choosing the step count on validation. A regularizer with a cheap path along the training trajectory."),
            ("Dropout", "Randomly masking units in training and rescaling at evaluation. Another train-eval switch."),
            ("Augmentation", "Label-preserving transformations. They enlarge the training distribution toward variations you claim should not change $Y$."),
            ("Evidence", "A smaller train-test gap together with a better validation loss. A smaller gap with a worse validation loss is not a win."),
        ],
        theory="""Regularization is any restriction that trades training loss for expected test loss. The trade can fail. A huge $\\lambda$ makes a network a constant and the gap vanishes because both losses are bad. Always read the gap next to the validation number.

Augmentation is a modeling statement. Flipping an image preserves a species label and destroys a "which side is the tumor on" label. The augmentation distribution should match variations you want invariance to, and it should not use information from the test image. Augmenting after the split is fine. Augmenting by mixing train and test images is leakage.

Dropout's evaluation multiplier, or the equivalent of averaging the mask, must be on when you report a number. Forgetting it makes the test-time function a randomly scaled cousin of the training function. As with batch normalization, the switch is part of the model definition.""",
        derivation="""The L2-penalized objective $L(\\theta) + (\\lambda/2)\\lVert \\theta \\rVert_2^2$ has gradient $\\nabla L + \\lambda \\theta$. Weight decay in plain SGD, $\\theta \\leftarrow (1 - \\eta \\lambda)\\theta - \\eta \\nabla L$, matches that gradient step. In adaptive methods, decoupling weight decay from the adaptive scale is a different update from adding $\\lambda \\theta$ before the coordinatewise rescaling. If you compare "weight decay" across optimizers, say which formula you used.

Early stopping selects $t_\\star = \\arg\\min_{t \\le T} \\widehat{R}_{\\mathrm{val}}(\\theta_t)$. It is cross-validation along one path. The selected $\\theta_{t_\\star}$ is then a validation-chosen model. The test evaluation happens after $t_\\star$ is frozen.""",
        code="""import numpy as np

def l2_penalty_grad(theta, lam):
    return lam * np.asarray(theta, dtype=np.float64)

def dropout_mask(shape, p, rng):
    if not 0 <= p < 1:
        raise ValueError("p is the drop probability")
    return rng.random(shape) >= p
""",
        experiment="""**Hypothesis.** On a small training set for the sine network, a moderate L2 penalty or early stopping lowers validation MSE relative to a long unpenalized run, and a penalty that is far too large raises both training and validation loss.

**Protocol.** Width 128, $n_{\\mathrm{train}} = 60$, a grid of $\\lambda$ including 0, plus one long run with epoch chosen on validation.

**Metric.** Validation MSE and the train-validation gap.

**Decision rule.** You should be able to show one setting that generalizes better than the longest unregularized run, and one setting that underfits. If the unregularized run does not overfit, the width is too small or $n$ is too large for the demonstration; adjust the capacity until the lesson is visible, and record that you designed the demonstration.""",
        project="""**Deliverable.** A regularization sweep on the synthetic network with the selection rule predeclared. If you add an augmentation, use a synthetic one you can defend, such as small input noise, and include a harmful augmentation, such as permuting the predictive coordinate, as a negative control.

**Constraints.** The test set is not used to pick $\\lambda$ or the stopping epoch.

**Rubric.** The note says which regularizer you would keep and what invariance it assumes.""",
        beginner="Add an L2 term to the linear model and watch coefficients shrink as $\\lambda$ grows.",
        university="Complete the overfitting demonstration and the sweep.",
        research="Implement dropout at train and eval on one layer and show the gradient check fails if the eval rescale is left on during the analytic training gradient, or succeeds only when the mask is fixed. State the correct check: fix the mask, differentiate, then average.",
        failures="""- Augmentations chosen because they improved the test number, with no validation split.
- Dropout left on at test time so every evaluation is random.
- Five regularizers added at once with a single before-after comparison.""",
        see_also="Volume 07 for the selection firewall. Volume 16 for augmentations that hide a sensitive attribute and may or may not be ethical or effective.",
    ),
)

chapter(
    v,
    "05_debugging_training",
    "Debugging a training run",
    entry(
        question="In what order do you investigate a run that does not learn?",
        card=[
            ("Order", "Data contract, loss at initialization, gradient check, one-batch overfit, then scale up."),
            ("One batch", "A correct flexible model should drive the loss on a tiny batch near the minimum of that loss. If it cannot, the bug is not the dataset."),
            ("Baseline", "A linear model or a constant. If the network cannot beat it on one batch, stop."),
            ("Scale", "Activation variances, gradient norms per layer, fraction of dead units, fraction of non-finite values."),
            ("Change one thing", "A debug log that changes the learning rate, the init, and the loss together does not isolate a cause."),
        ],
        theory="""Training bugs masquerade as research questions. The discipline is to restore a regime where success is mandatory. A tiny batch with a large width and a correct loss has a minimum you should reach. Failure there is a bug in the graph, the reduction, the optimizer state, or the input pipeline. Success there, and failure on the full problem, is capacity, regularization, distribution shift, or an optimization issue at scale. Those are different projects.

Look at per-layer gradient norms. A layer with a zero gradient is disconnected. A layer whose gradient grows by orders of magnitude each step will overflow. Non-finite losses should abort the run and dump the last finite checkpoint's statistics. A plot that drops NaNs and continues will show a comforting curve that is no longer the loss you defined.

Compare to a baseline on the same batch. If linear regression fits the batch better than the network, the network is not yet a superset in practice, even if it is a superset in theory. Usually the step size or the initialization has pinned it.""",
        derivation="""Let $L_B$ be the loss on a fixed batch of size $B$. If the model can represent the targets and the loss is bounded below by zero, the infimum is zero for squared error on a realizable batch, or the entropy of the labels for cross-entropy when the labels conflict. Gradient descent on a smooth approximation can drive $L_B$ near that floor given a small enough step and enough steps. ReLU networks can fail this test when units die on every point in the batch. The diagnostic action is to print the active fraction, not to collect more data.""",
        code="""import numpy as np

def assert_finite(name, value):
    arr = np.asarray(value)
    if not np.isfinite(arr).all():
        raise FloatingPointError(f"{name} has non-finite values")
""",
        experiment="""**Hypothesis.** A network with a deliberately transposed gradient cannot drive a single-batch squared loss near zero, while the corrected gradient can. A learning rate 100 times too large also fails, with non-finite or oscillating loss, even when the gradient is correct.

**Protocol.** One batch of 16 points from the sine generator, width 64, two bug conditions and one correct run.

**Metric.** Loss at step 0 and step 200, and whether the run raised on a non-finite value.

**Decision rule.** The correct run should reduce loss by a large factor. Each bug should fail differently. If the transposed gradient still trains, the loss is invariant to that parameter and the test is vacuous; choose a parameter the loss actually uses.""",
        project="""**Deliverable.** A debug checklist you actually run, stored as `DEBUG.md` next to the training script, with the one-batch result, the linear baseline on that batch, gradient norms, and the two intentional bugs.

**Constraints.** The checklist is ordered. Later items are not attempted while earlier items fail.

**Rubric.** A new failure you introduce in the loss reduction is caught by an item already on the list, or you add that item because it was not.""",
        beginner="Overfit 8 points with a linear model, then with the network.",
        university="Complete the two intentional bugs and the checklist.",
        research="Add a per-layer gradient-norm log to a deeper stack and identify a vanishing layer you create by using sigmoid and a bad init. Fix it by the init chapter's rule, not by an unexplained learning-rate change.",
        failures="""- Collecting more data while one batch does not fit.
- Averaging away NaNs.
- Changing five hyperparameters after each failed run.""",
        see_also="Volume 03 for the experiment folder that should store the failed runs too. Volume 09 for the gradient check that belongs at the top of the checklist.",
    ),
)

chapter(
    v,
    "06_ablation_project",
    "Project: a controlled training ablation",
    entry(
        question="How do you attribute a gain to one training choice?",
        card=[
            ("Factor", "One change: optimizer, schedule, penalty, width, or augmentation."),
            ("Control", "The previous best configuration, otherwise untouched."),
            ("Selection", "Each factor's internal knob, such as $\\eta$, is tuned on validation under that factor."),
            ("Budget", "Matched data touches or matched wall-clock time, declared first."),
            ("Report", "A table of differences with uncertainty across seeds, and a decision about what enters the default."),
        ],
        theory="""An ablation is an experiment, not a tour of features. The scientific claim is that factor $A$, not the company it kept, moved the validation metric. If you turn on normalization, Adam, and augmentation together, you have a bundle. Bundles are how products are built and how papers get refuted. This project allows one bundle only as a final row, after the single-factor rows.

Each factor deserves its own step-size or penalty sweep when the factor changes scale. Holding $\\eta$ fixed "to be fair" is unfair to the method whose optimum moved. The fairness is in the validation budget for the sweep, which you should cap so a method with more knobs does not get a larger search.

Seeds matter more here than in convex problems. Report the median and the spread. A gain smaller than the spread is not a default-changing result. It is a request for more seeds or a simpler claim.""",
        derivation="""Let $\\Delta_s = M_s(A) - M_s(C)$ be the validation metric gap on seed $s$, with the sign oriented so positive is better. The mean of $\\Delta_s$ over seeds estimates the procedure gap under your seed distribution. It does not estimate the gap on a new test set. After you choose the default, one locked test evaluation measures that chosen procedure. Choosing the factor with the best test $\\Delta$ is the selection bias of Volume 07. The ablation table is validation. The test number is a single subsequent line, if you spend it.""",
        code="""import numpy as np

def summarize_deltas(deltas):
    d = np.asarray(deltas, dtype=np.float64)
    return {
        "n": int(d.size),
        "median": float(np.median(d)),
        "mean": float(np.mean(d)),
        "sd": float(np.std(d, ddof=1) if d.size > 1 else 0.0),
    }
""",
        experiment="""**Hypothesis.** On the small-sample sine network, early stopping or L2 regularization improves validation MSE relative to a fixed long run, and adding both does not necessarily beat the better single factor by more than seed noise.

**Protocol.** At least 5 seeds. Factors: longer unregularized run, L2 at its best $\\lambda$ from a small grid, early stopping, and the combination. Matched maximum step budget.

**Metric.** Validation MSE deltas versus the long unregularized run.

**Decision rule.** Adopt a factor only if the median delta is in the direction you wanted and larger than a practical margin you set from the residual noise. Otherwise keep the simpler training loop.""",
        project="""**Deliverable.** An ablation table, the configs, the seed spread, and a default training section in the model card. One locked test evaluation of the chosen default against the linear baseline.

**Constraints.** No test-based ablation. The checklist from the debugging chapter has been run on the default.

**Rubric.** A reader can see which single change mattered. A bundle-only table is rejected.""",
        beginner="Compare two learning rates at one seed and write why that comparison is incomplete.",
        university="Complete the ablation and the locked test line.",
        research="Add a compute column. If a factor wins only by using twice the data touches, repeat the control at that budget before you attribute the gain to the idea.",
        failures="""- Ablating on the test set.
- A table of single runs sorted to two decimal places.
- Declaring five factors "all necessary" because the bundle worked.""",
        see_also="Volume 07 for paired gaps. Volume 18 for how an ablation becomes a paper table.",
    ),
)
