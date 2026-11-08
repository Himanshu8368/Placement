"""Volumes 13–16: generative models, reinforcement learning, systems, reliability."""

from booklib import chapter, entry, volume

v = volume(
    13,
    "Generative models",
    "University to research",
    "A reader who can fit a density and a classifier and can read a likelihood.",
    "What distribution is a generator claiming to sample, and which evaluation tells you it failed?",
)

chapter(
    v,
    "01_autoregressive",
    "Autoregressive models",
    entry(
        question="What does an autoregressive factorization claim, and how do you sample from it?",
        card=[
            ("Factorization", "$p(x_{1:T}) = \\prod_t p(x_t \\mid x_{<t})$, with a declared start token."),
            ("Training", "Teacher-forced conditional likelihood of the observed next symbol."),
            ("Sampling", "Draw $\\hat{x}_t$ from the conditional and feed it back. This is not the teacher-forced path."),
            ("Exposure", "The state distribution at training, where prefixes are real, differs from the state distribution at sampling, where prefixes are the model's own draws."),
            ("Tractable likelihood", "The factorization gives you a log likelihood. A model without one needs a different evaluation."),
        ],
        theory="""Autoregressive models are the sequence models of Volume 12 trained as generative models of the whole string, not only as classifiers. The likelihood is a proper score for the string distribution under the factorization. It does not score whether a free-running sample looks right to a person, and it does not score whether a conditional sample matches a prompt you added at deployment unless that prompt was part of the model.

Exposure bias is the train-test mismatch of the prefix distribution. A model can assign high likelihood to a corpus and still degenerate when sampled, because it never had to continue from its own errors during training. Report both teacher-forced likelihood and a free-running checker.

Temperature and top-$k$ truncation change the distribution you sample from. They are not the distribution you evaluated with likelihood. If you deploy a truncated sampler, the card names it. The likelihood of the original model is then not the likelihood of the deployed sampler.""",
        derivation="""The log likelihood of a sequence is $\\sum_t \\log p_\\theta(x_t \\mid x_{<t})$. A gradient step on this sum, with a causal mask, is ordinary supervised learning of the next token. Ancestral sampling draws $x_1$, then $x_2$ given $x_1$, and so on. The probability of a sampled trajectory under the model is the product of the conditionals that were sampled, not the product along the ground-truth string. Truncating the support of each conditional defines a new model $q \\ne p_\\theta$. KL$(q \\| p_\\theta)$ need not be zero. Do not quote $p_\\theta$'s perplexity as a property of $q$.""",
        code="""import numpy as np

def ancestral_sample(next_probs, start, steps, rng):
    seq = [start]
    for _ in range(steps):
        p = np.asarray(next_probs(seq), dtype=np.float64)
        p = p / p.sum()
        seq.append(int(rng.choice(len(p), p=p)))
    return seq
""",
        experiment="""**Hypothesis.** A bigram model fit on a language with a strict alternation pattern has high teacher-forced likelihood on held-out strings from that language, and a sampler that is not masked to the support can still emit illegal pairs if you deliberately use a uniform conditional.

**Protocol.** Alphabet `{0, 1}`, legal strings alternate. Fit a bigram conditional by counts with a small smoothing parameter. Sample from the fitted conditionals and from a uniform conditional.

**Metric.** Fraction of adjacent pairs that violate alternation, and held-out mean NLL of the fitted model.

**Decision rule.** The fitted sampler should violate rarely, at a rate explained by smoothing. The uniform sampler should violate often. If the fitted sampler violates often, the conditional tables were built with a leakage from the future or a bug in the counts.""",
        project="""**Deliverable.** A bigram or tiny neural autoregressive model on the synthetic alternating language, with teacher-forced NLL, free-running violation rate, and a temperature or truncation variant labeled as a different sampler.

**Constraints.** The causal constraint is tested. Pads are masked.

**Rubric.** You never describe the truncated sampler's behavior using the untruncated perplexity alone.""",
        beginner="Fit the bigram counts and compute one string's log probability by hand.",
        university="Complete the sample-versus-likelihood comparison.",
        research="Explain scheduled sampling in one paragraph and why it optimizes a different objective than maximum likelihood. Do not claim it fixes exposure bias without a measurement.",
        failures="""- Reporting teacher-forced loss as sample quality.
- Changing temperature and comparing likelihoods as if the model were fixed.
- A non-causal factorization presented as autoregressive.""",
        see_also="Volume 12 for masks and perplexity. The diffusion chapter for a generator without this factorization.",
    ),
)

chapter(
    v,
    "02_vae",
    "Variational autoencoders",
    entry(
        question="What does a VAE optimize, and why is the objective not the exact log likelihood?",
        card=[
            ("Model", "A latent $z \\sim p(z)$, an observation model $p_\\theta(x \\mid z)$, and an encoder $q_\\phi(z \\mid x)$."),
            ("ELBO", "A lower bound on $\\log p_\\theta(x)$. Maximizing the bound is not the same as computing the bound's gap."),
            ("Terms", "A reconstruction term minus a KL penalty that pulls $q_\\phi(z \\mid x)$ toward $p(z)$."),
            ("Gap", "The ELBO equals the true log likelihood only when $q$ matches the posterior. Usually it does not."),
            ("Samples", "Draw $z \\sim p(z)$ and then $x \\sim p(x \\mid z)$ to sample the prior predictive. Encoding a training point and decoding it is a reconstruction, not a sample from the model."),
        ],
        theory="""A VAE is a likelihood model with an intractable posterior, plus a variational approximation you also train. The evidence lower bound is a surrogate. A high ELBO is encouraging. It does not identify $\\log p(x)$, and two models with the same ELBO can have different gaps. If you need a number that is a log likelihood, say "ELBO" or compute a tighter estimate you can defend.

The KL term can collapse. If $q(z \\mid x)$ ignores $x$ and matches the prior, the reconstruction term must carry the data with an uninformative latent. The model has then learned a decoder from noise, and the encoder is decorative. Watch the KL per dimension. A dimension with KL near zero is unused.

Reconstructions look sharper than prior samples when the decoder is deterministic and the latent is a copy of $x$ through a narrow code. Judge the prior samples if you claimed a generator. Judge reconstructions if you claimed a compressor. They are different products.""",
        derivation="""For a single $x$,

$$\\log p_\\theta(x) = \\mathbb{E}_{q}[\\log p_\\theta(x \\mid z)] - \\mathrm{KL}(q_\\phi(z \\mid x) \\| p(z)) + \\mathrm{KL}(q_\\phi(z \\mid x) \\| p_\\theta(z \\mid x)).$$

The last KL is nonnegative, so dropping it yields the ELBO, a lower bound. Equality holds when $q$ is the true posterior. The reparameterization $z = \\mu_\\phi(x) + \\sigma_\\phi(x) \\odot \\epsilon$, $\\epsilon \\sim \\mathcal{N}(0, I)$, lets you differentiate a Gaussian $q$ through a sample. A discrete latent needs a different estimator. Do not backpropagate through a `stop_gradient` you forgot about and then wonder why the encoder never moves.""",
        code="""import numpy as np

def gaussian_kl_standard(mu, log_var):
    # KL(N(mu, diag(exp(log_var))) || N(0, I)), summed over latent dimensions.
    mu = np.asarray(mu, dtype=np.float64)
    log_var = np.asarray(log_var, dtype=np.float64)
    return float(0.5 * np.sum(np.exp(log_var) + mu**2 - 1.0 - log_var))
""",
        experiment="""**Hypothesis.** The KL helper is zero at $\\mu = 0, \\log \\sigma^2 = 0$ and positive if $\\mu$ moves. A collapsed encoder with $\\mu = 0$ and $\\log \\sigma^2 = 0$ for every $x$ has reconstruction loss equal to a decoder that does not see $x$.

**Protocol.** Check the KL values. Then, on 1D data with two modes, compare a decoder that receives $z \\sim \\mathcal{N}(\\mu(x), \\sigma(x))$ with one that receives $\\epsilon \\sim \\mathcal{N}(0, I)$ only. You may fit both with least squares if you keep the lesson about information flow.

**Metric.** KL at the standard normal parameters, and reconstruction error with and without $x$-dependent latents.

**Decision rule.** KL at the prior parameters should be about 0. The $x$-dependent code should win on two well-separated modes if the decoder is linear in $z$ and $z$ can encode the mode. If it does not, the code has no path into the decoder.""",
        project="""**Deliverable.** A small VAE or a diagonal-Gaussian latent model on a 2D mixture, with ELBO terms logged separately, prior samples, and reconstructions. State which picture the card claims.

**Constraints.** Do not call the ELBO a log likelihood in the table.

**Rubric.** A collapsed KL is visible if you cause one, and the note says how you noticed.""",
        beginner="Check `gaussian_kl_standard(0, 0)` and a nonzero mean.",
        university="Fit the 2D mixture model and separate the plots.",
        research="Describe the variational gap in a sentence a reviewer would accept, and one diagnostic that the gap might be large even when the ELBO looks stable.",
        failures="""- Reporting reconstructions as unconditional samples.
- Adding the KL with the wrong sign and calling the result an ELBO.
- A latent that copies a label you are not allowed to use at generation time.""",
        see_also="Volume 08 for mixtures. Volume 04 for the Gaussian KL's ingredients.",
    ),
)

chapter(
    v,
    "03_gans",
    "Adversarial generators",
    entry(
        question="What game does a GAN play, and why is a lower loss not a better generator?",
        card=[
            ("Players", "A generator $G$ maps noise $z$ to a sample. A discriminator $D$ scores samples versus data."),
            ("Equilibrium", "In the idealized game, $G$ matches the data law and $D$ is uninformative."),
            ("Practice", "The finite-sample game can oscillate, collapse modes, or let $D$ become perfect and uninformative as a gradient."),
            ("No likelihood", "You do not get $\\log p(x)$ from the usual GAN objective."),
            ("Eval", "A task metric or a distance you predeclare. The discriminator loss is not that metric."),
        ],
        theory="""A GAN trains a sampler by a critic. When the critic is strong and the generator matches the data, the critic cannot do better than chance. A generator "winning" by driving the discriminator loss to a floor can mean the discriminator collapsed, not that the samples are good. Log both losses and look at samples or at a held-out task metric.

Mode collapse is the failure in which $G$ covers one mode of a mixture and ignores the others. The discriminator may still be fooled on that mode. A coverage check, such as the fraction of true modes that have a nearby sample, belongs in the experiment. A single pretty sample does not answer coverage.

GANs do not give a density value. Do not compare a GAN to a VAE by likelihood unless you add a density estimator you trust. Compare them by a sample metric you would actually use, and give the autoregressive model the likelihood column it earned.""",
        derivation="""The idealized discriminator, given a fixed generator, scores a point by the odds ratio of the data density and the generator density. The generator is then trained to move samples toward regions the discriminator calls "data." At equilibrium the densities match and the odds ratio is 1. Finite models replace both densities by neural nets and take alternating steps. Alternating gradient steps on a nonconvex game are not guaranteed to reach that equilibrium. A diverging discriminator accuracy with a generator that emits one point is compatible with the algebra of a failed optimization, not a counterexample to the equilibrium statement.""",
        code="""import numpy as np

def mode_coverage(samples, mode_centers, radius):
    samples = np.asarray(samples, dtype=np.float64)
    centers = np.asarray(mode_centers, dtype=np.float64)
    covered = 0
    for c in centers:
        if np.any(np.linalg.norm(samples - c, axis=-1) <= radius):
            covered += 1
    return covered / len(centers)
""",
        experiment="""**Hypothesis.** Samples concentrated at one mode of a two-mode mixture have coverage $1/2$ under a moderate radius, while samples drawn from both modes have coverage 1. A discriminator accuracy near 1 can occur in the collapsed case if the discriminator only learned that one region.

**Protocol.** You do not need a full GAN for the metric check. Draw collapsed samples and good samples. Optionally train a tiny linear discriminator and show it can be confident while coverage is poor, or skip the training and treat the metric as the lesson if time is short. Prefer to train if you can keep it small.

**Metric.** `mode_coverage` and, if you train, discriminator accuracy.

**Decision rule.** Coverage must distinguish the two samplers. If the radius is so large that one mode covers both centers, shrink it. The metric is only meaningful when a collapsed sampler does not accidentally cover everything.""",
        project="""**Deliverable.** A small generator on a 2D mixture, or a clear negative result, with mode coverage, a baseline that samples the training points themselves (the memorization ceiling), and a note that there is no likelihood column.

**Constraints.** The coverage radius is chosen from the known mode separation before you look at the generator, or chosen on a validation draw from the true mixture, not to maximize the generator's score after the fact.

**Rubric.** Memorization is reported as a ceiling, not as a competitor you forgot to beat honestly on new draws.""",
        beginner="Compute coverage for a one-mode and a two-mode cloud.",
        university="Train or simulate the collapse comparison and write the discriminator caveat.",
        research="State one distance between distributions, such as a critic-based estimate, and the way it can be gamed if the critic class is too weak or too strong. Keep the formula short and the caveat explicit.",
        failures="""- Picking the checkpoint with the prettiest sample.
- Reporting discriminator loss as a quality metric.
- Comparing likelihoods you do not have.""",
        see_also="Volume 04 for mixtures as the data law. Volume 18 for checkpoint selection rules.",
    ),
)

chapter(
    v,
    "04_diffusion",
    "Diffusion models",
    entry(
        question="What is a diffusion model trained to predict, and how is a sample produced?",
        card=[
            ("Forward", "A fixed noising process that turns data into nearly standard Gaussian noise over many steps."),
            ("Reverse", "A learned denoise that walks from noise back toward the data."),
            ("Training target", "Often the added noise, or a clean image, at a randomly chosen noise level. Not a single-step autoencoder unless the schedule says so."),
            ("Schedule", "The noise levels are part of the model. Training and sampling must use the same schedule."),
            ("Likelihood", "Some formulations give a bound. A sample-only comparison does not require you to pretend you computed $\\log p(x)$."),
        ],
        theory="""Diffusion models separate a simple forward destruction of structure from a learned reversal. The network usually predicts the noise that was added at a given time, which is a regression target with a known answer. That regression is the training loss. Sampling applies the learned reverse step many times. A one-step reconstruction at a low noise level can look good while the full reverse chain from pure noise fails. Evaluate the procedure you ship.

The schedule determines how hard each regression is. If you train only at tiny noise, you have not trained the steps that start from pure noise. If you sample with a different number of steps than you are willing to pay at deployment, you are evaluating a different sampler.

As with other generators, mode coverage and a task metric beat a gallery. A diffusion model can still memorize. A nearest-neighbor check from generated samples to the training set is a useful embarrassment test on small data.""",
        derivation="""A common forward step is $x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1 - \\bar{\\alpha}_t}\\, \\epsilon$ with $\\epsilon \\sim \\mathcal{N}(0, I)$. The network $\\epsilon_\\theta(x_t, t)$ is trained by squared error against $\\epsilon$. When $\\bar{\\alpha}_t \\approx 1$, $x_t \\approx x_0$ and the problem is easy. When $\\bar{\\alpha}_t \\approx 0$, $x_t$ is almost noise and the network must recover structure from the time index and the residual signal. The training distribution over $t$ is therefore a knob. Sampling plugs $\\epsilon_\\theta$ into a reverse update whose coefficients come from the same $\\bar{\\alpha}$. Mixing two papers' coefficient conventions is a silent bug.""",
        code="""import numpy as np

def forward_noise(x0, alpha_bar, rng):
    eps = rng.normal(size=np.shape(x0))
    x_t = np.sqrt(alpha_bar) * x0 + np.sqrt(1.0 - alpha_bar) * eps
    return x_t, eps
""",
        experiment="""**Hypothesis.** At $\\bar{\\alpha}$ near 1, a predictor that returns 0 has small error if the true noise is small... more sharply: the squared error of predicting the true $\\epsilon$ is zero by construction in a synthetic test of the target, and the error of predicting 0 is about the dimension.

**Protocol.** Draw $x_0$ and $\\epsilon$, form $x_t$ at $\\bar{\\alpha} \\in \\{0.99, 0.1\\}$, and compute MSE of the true $\\epsilon$ versus a zero predictor.

**Metric.** Mean squared error per coordinate.

**Decision rule.** The true $\\epsilon$ scores 0. The zero predictor scores near 1. If your training code does not approach the true-$\\epsilon$ score on a batch where you pass $\\epsilon$ as the prediction, the loss is wired to the wrong tensor.""",
        project="""**Deliverable.** A one-dimensional diffusion-style denoiser on a two-mode dataset, trained at several noise levels, with samples from a short reverse chain and a mode-coverage number. A nearest-neighbor distance from samples to training points.

**Constraints.** The schedule is in the config and is shared by training and sampling.

**Rubric.** You can show a failure mode of sampling at a noise level you did not train, or of a reverse chain that is too short.""",
        beginner="Check `forward_noise` recovers $x_0$ when $\\bar{\\alpha} = 1$.",
        university="Train the 1D denoiser and report coverage.",
        research="State the difference between predicting $\\epsilon$, predicting $x_0$, and predicting a score $\\nabla \\log p_t$. You only need the algebraic relationship at one noise level, plus the warning that the parameterization changes the loss scale.",
        failures="""- Evaluating only low-noise reconstructions.
- A sampling schedule copied from a paper that used a different $\\bar{\\alpha}$ definition.
- A gallery of five samples as the only metric.""",
        see_also="Volume 11 if the data are images. The evaluation chapter next.",
    ),
)

chapter(
    v,
    "05_gen_eval",
    "Evaluating generators",
    entry(
        question="Which generator comparisons are even defined?",
        card=[
            ("Likelihood models", "Autoregressive models and others with a density or a bound can report held-out likelihood, labeled correctly as a bound when it is one."),
            ("Sample models", "GANs and many samplers need a sample metric. Do not invent a likelihood."),
            ("Coverage", "A check that modes or subclasses appear, not only that some sample is sharp."),
            ("Memorization", "Distance to the nearest training example. Small distance means copying."),
            ("Downstream", "If the generator feeds another system, evaluate that system with and without the synthetic data."),
        ],
        theory="""Generative evaluation fails when numbers from different questions are placed in one column. Likelihood, sample fidelity, coverage, and downstream utility can rank models differently. The reference rule is to put them in different columns and to leave a cell blank rather than fill it with a proxy you do not believe.

Human preference is a measurement with a protocol: who judged, what they saw, what the question was. Five friends and a screenshot are not a protocol. A programmatic checker on synthetic data is a protocol. Use it when you can, and do not generalize it to "the model understands images."

Synthetic data that trains a downstream classifier must be tested against real held-out data. A classifier that is perfect on synthetic test images and useless on real ones learned the generator's artifacts. That result belongs in the note.""",
        derivation="""Let $p$ be the data law and $q$ the model. KL$(p \\| q)$ is an expectation under $p$ and punishes a $q$ that misses regions where $p$ has mass. It is related to likelihood. A critic that looks at samples from $q$ can miss the same failure if those samples never land in the missing region and the critic is not shown the uncovered data. Coverage metrics are attempts to see the miss. No single divergence is the decision. The decision is the column that matches the use: compression and prediction favor likelihood; a simulator for a downstream task favors that task's risk.""",
        code="""import numpy as np

def nearest_train_distance(samples, train):
    samples = np.asarray(samples, dtype=np.float64)
    train = np.asarray(train, dtype=np.float64)
    d2 = ((samples[:, None, :] - train[None, :, :]) ** 2).sum(axis=-1)
    return np.sqrt(d2.min(axis=1))
""",
        experiment="""**Hypothesis.** Samples copied from the training set have nearest-neighbor distance 0. Samples from a shifted Gaussian have distances on the order of the shift. A coverage metric and a nearest-neighbor metric disagree on a collapsed memorizing sampler versus a slightly noisy full-support sampler.

**Protocol.** Two-mode data in 2D. Sampler A copies five training points from one mode. Sampler B draws both modes with small noise.

**Metric.** Mode coverage and median nearest-neighbor distance.

**Decision rule.** Sampler A looks "close to the data" and fails coverage. Sampler B covers and is not an exact copy. A note that reports only one of the metrics must say which failure it cannot see.""",
        project="""**Deliverable.** An evaluation table for at least two generators you built in this volume, with blank cells where a number is undefined, plus the memorization and coverage checks.

**Constraints.** Do not fill a likelihood cell for a GAN.

**Rubric.** The recommendation names the use case. A different use case may rank the models differently, and you say so.""",
        beginner="Compute nearest-neighbor distances for copies and for shifted points.",
        university="Build the two-sampler table.",
        research="Write the downstream-utility protocol: train a classifier on real data only, and on real plus synthetic, test on real only, and state the leakage path if synthetic images were made from the test set.",
        failures="""- One column called "score."
- Human preference without a question text.
- Synthetic data generated from test examples.""",
        see_also="Volume 07 for not shopping metrics. Volume 18 for reporting negative rankings.",
    ),
)

chapter(
    v,
    "06_gen_project",
    "Project: a small generator with an evaluation",
    entry(
        question="Can you train one generator and defend the number you publish?",
        card=[
            ("Data", "A low-dimensional mixture or a tiny image set you have rights to."),
            ("Model", "Autoregressive, VAE, GAN, or diffusion. One primary choice."),
            ("Baseline", "The empirical training distribution as a memorizing sampler, and a single Gaussian."),
            ("Metrics", "The columns from the evaluation chapter that apply."),
            ("Failure", "One induced failure: collapse, a wrong schedule, or a peeked future token."),
        ],
        theory="""The project freezes the habit of matching the claim to the metric. A Gaussian baseline that wins on likelihood and loses on a sample-coverage task you care about is a successful experiment. It tells you the task is not likelihood. A generator that only ties the memorizing sampler has not learned a density. It has learned the training set.

Keep the dimension small enough to plot. A 2D mixture with three modes will teach more than a blurry face generator you cannot evaluate. If you move to images, reuse the vision contract and the shift test: a generator trained on centered digits and sampled for a downstream task should be checked off-center.""",
        derivation="""The single Gaussian MLE baseline has a closed form from Volume 05. Its held-out log likelihood is defined. An autoregressive model on discretized 2D bins also has a likelihood, after you state the binning. A GAN does not. The table's likelihood column is filled for the first two and blank for the GAN. The coverage column is filled for all three. This is the whole methodological point, and the project is incomplete if the blank cell was replaced by a dash that a reader might read as zero.""",
        code="""import numpy as np

def gaussian_mle_nll(train, test):
    mu = train.mean(axis=0)
    var = np.mean((train - mu) ** 2, axis=0) + 1e-6
    test = np.asarray(test, dtype=np.float64)
    return float(0.5 * np.mean(np.sum(np.log(2 * np.pi * var) + (test - mu) ** 2 / var, axis=-1)))
""",
        experiment="""**Hypothesis.** On three well-separated modes, the Gaussian MLE has worse mode coverage than a sampler that draws the empirical modes, and it may still assign a finite likelihood everywhere the mixture model with a missing mode assigns almost none.

**Protocol.** Fit the Gaussian baseline. Compare with a categorical draw over the three training means plus small noise. Score coverage and, for the Gaussian, held-out NLL.

**Metric.** Coverage and NLL where defined.

**Decision rule.** The table must show a tradeoff or a clear win, not a single champion invented by dropping a column. If the modes are so close that the Gaussian covers them, separate the means.""",
        project="""**Deliverable.** One learned generator plus the two baselines, the metric table, the induced failure, and a card that says what the generator is for.

**Constraints.** Hyperparameters are chosen on validation draws or validation likelihood, not on the final test numbers.

**Rubric.** A blank cell is present if a metric does not apply. The failure mode is something you ran, not a paragraph you copied from a survey.""",
        beginner="Fit the Gaussian baseline and report NLL and a scatter or a table of means.",
        university="Complete the generator and the table.",
        research="Add a downstream classifier that uses synthetic samples and test it on real held-out draws. State whether synthetic data helped at a matched labeled budget.",
        failures="""- A generator with no baseline.
- Test-set tuning of the noise schedule.
- Calling a reconstruction a sample.""",
        see_also="Volume 14 if a generator becomes a world model in reinforcement learning. Keep the evaluation columns.",
    ),
)

v = volume(
    14,
    "Reinforcement learning",
    "University to research",
    "A reader who can talk about conditional independence, expectations, and off-policy data.",
    "What is an agent optimizing, under which state assumption, and how can a logged dataset lie about a new policy?",
)

chapter(
    v,
    "01_mdp",
    "Markov decision processes",
    entry(
        question="What has to be in the state before the Markov decision process is the process you have?",
        card=[
            ("Process", "States, actions, a transition kernel, a reward, and a discount $\\gamma \\in [0, 1)$."),
            ("Markov", "$S_{t+1}$ and $R_{t+1}$ depend on the past only through $S_t$ and $A_t$."),
            ("Policy", "A map $\\pi(a \\mid s)$, or a deterministic $a = \\pi(s)$."),
            ("Return", "$G_t = \\sum_{k=0}^{\\infty} \\gamma^k R_{t+1+k}$."),
            ("Value", "$V^\\pi(s) = \\mathbb{E}_\\pi[G_t \\mid S_t = s]$. $Q^\\pi(s, a)$ conditions on the action too."),
        ],
        theory="""An MDP is a conditional-independence claim plus an objective. If two different histories lead to the same recorded state and different futures, the recorded state is not a state. The algorithm will treat those histories as exchangeable and will be wrong. Add the missing memory or stop calling the problem Markov.

The discount $\\gamma$ makes the return finite and says how much you care about later rewards. It is part of the objective, not a numerical trick only. A policy that is optimal at $\\gamma = 0.9$ need not be optimal at $\\gamma = 0.99$. Report $\\gamma$.

Reward is the objective you will get. If the reward is a proxy, the value function is the value of the proxy. Volume 16's metric warnings apply here with more force, because the agent will search for a high proxy. Specify the reward as a contract.""",
        derivation="""Condition on $S_t = s$ and $A_t = a$, then follow $\\pi$. The Bellman equation for the action value is

$$Q^\\pi(s, a) = \\mathbb{E}[R_{t+1} + \\gamma V^\\pi(S_{t+1}) \\mid S_t = s, A_t = a],$$

and $V^\\pi(s) = \\sum_a \\pi(a \\mid s) Q^\\pi(s, a)$. The derivation uses the Markov property when the future after $S_{t+1}$ depends on the past only through $S_{t+1}$, and it uses the law of total expectation. If the state is missing a bit that the reward depends on, the conditional expectation given the recorded $s$ is an average over that bit, and the equation is about a coarser process than the one you think you control.""",
        code="""import numpy as np

def discounted_return(rewards, gamma):
    total = 0.0
    scale = 1.0
    for r in rewards:
        total += scale * float(r)
        scale *= gamma
    return total
""",
        experiment="""**Hypothesis.** A two-step reward sequence $(0, 1)$ has discounted return $\\gamma$ when the first reward is 0, and a state alias that merges two positions with different next rewards makes a value estimate that is the average, not either truth.

**Protocol.** Compute `discounted_return([0, 1], 0.9)`. Then tabulate a two-state MDP you can solve by hand: from state A, action "go" yields reward 1; from state B, reward 0; an aliased observation $O$ merges A and B with equal probability.

**Metric.** The return number, and the conditional expected next reward given $O$ versus given the true state.

**Decision rule.** The aliased expectation should sit between the two true rewards. A controller that conditions only on $O$ cannot choose differently in A and B. That is the Markov failure, in miniature.""",
        project="""**Deliverable.** A small MDP written as tables: states, actions, transitions, rewards, $\\gamma$. A sentence that says what was left out of the state on purpose, and what decision that omission forbids.

**Constraints.** The tables are small enough to solve by enumeration in the next chapter.

**Rubric.** A reader can compute one return and one aliased expectation from the note.""",
        beginner="Compute discounted returns at $\\gamma \\in \\{0, 0.9, 1\\}$ for a finite reward list, and see the $\\gamma = 1$ sum grow with the horizon.",
        university="Write the aliased-state table.",
        research="Give an example of a belief state that restores the Markov property when the observation is partial. You will need it again in the partial-observability chapter.",
        failures="""- Treating a frame of video as a state when velocity is missing and the reward depends on motion.
- Changing $\\gamma$ between training and the reported evaluation.
- A reward that includes a term unavailable at decision time.""",
        see_also="Volume 04 for conditional independence. Volume 16 for proxy rewards.",
    ),
)

chapter(
    v,
    "02_dynamic_programming",
    "Dynamic programming and Monte Carlo",
    entry(
        question="How do you compute values when the model is known, and what changes when you only have episodes?",
        card=[
            ("Policy evaluation", "Solve or iterate the Bellman equation for a fixed $\\pi$."),
            ("Improvement", "A greedy policy with respect to $Q$ is at least as good, under the MDP assumptions."),
            ("Monte Carlo", "Average returns from complete episodes. Unbiased for the return under the policy that generated the episodes, given enough coverage."),
            ("Model", "Dynamic programming needs the transition and reward tables. Monte Carlo needs samples."),
            ("Coverage", "A state you never visit has no Monte Carlo estimate, whatever the equation says."),
        ],
        theory="""Dynamic programming is the exact arithmetic of a known finite MDP. It is the unit test for later algorithms: on a table you can solve, a new method should agree up to optimization error. If it does not, the bug is not "instability of deep RL."

Monte Carlo replaces expectations by episode averages. The return is a high-variance estimate of the value. It does not bootstrap, so an error in one state's value does not immediately poison another, and you must wait until the episode ends. A state that is missing from the data is not estimated. A policy that never takes a risky action will not see the reward behind it. That is an exploration problem, and it is already present in a four-state grid.

Policy improvement is a theorem about the true $Q^\\pi$. Improvement with respect to a noisy $\\hat{Q}$ can make the policy worse. The theorem is the target, not a description of every update you code.""",
        derivation="""For a finite state-action space, policy evaluation iterates $Q_{k+1} = T^\\pi Q_k$, where $(T^\\pi Q)(s, a) = R(s, a) + \\gamma \\sum_{s'} P(s' \\mid s, a) \\sum_{a'} \\pi(a' \\mid s') Q(s', a')$. The map is a contraction in the max norm with modulus $\\gamma$, so the iteration converges to $Q^\\pi$ from any finite start. Monte Carlo instead averages $G_t$ for visits to $(s, a)$. Under the policy that generated the data, and with every pair visited infinitely often, the averages converge to $Q^\\pi$ by the law of large numbers. Finite data leave variance and holes.""",
        code="""import numpy as np

def value_iteration_step(Q, reward, transitions, gamma):
    # transitions[s, a, s_next] = probability. reward[s, a] expected immediate reward.
    n_s, n_a, _ = transitions.shape
    q_next = np.max(Q, axis=1)
    updated = reward + gamma * np.einsum("saj,j->sa", transitions, q_next)
    return updated
""",
        experiment="""**Hypothesis.** Value iteration on a two-state MDP you solve by hand reaches the hand solution. A Monte Carlo estimate from a policy that never visits state 1 leaves state 1 undefined or at its initialization.

**Protocol.** Build a deterministic MDP, run the iteration to a tolerance of $10^{-8}$, and compare. Then generate episodes that avoid one state.

**Metric.** Max absolute error against the hand solution, and the visit count per state.

**Decision rule.** The dynamic-programming error should be near the tolerance. The Monte Carlo table must show the hole. Filling the hole with zero and calling it a value is a bug in the report.""",
        project="""**Deliverable.** A solved grid or chain, the iteration code, a Monte Carlo comparison with visit counts, and a note on one state the behavior policy did not see.

**Constraints.** The MDP tables are the ground truth for this project. Do not "improve" them to match a broken algorithm.

**Rubric.** The hand solution and the iteration agree, and the missing-state issue is visible.""",
        beginner="Solve a one-state, one-action discounted reward by summing the geometric series and by iteration.",
        university="Complete the grid or chain and the Monte Carlo hole.",
        research="State the contraction argument in two lines and the step that fails if $\\gamma = 1$ and the horizon is infinite without a termination condition.",
        failures="""- A Monte Carlo value of 0 for an unvisited high-reward state.
- Comparing a partial-episode return with a discounted infinite-horizon value without a terminal rule.
- Using the optimal policy's samples to claim you evaluated a worse policy.""",
        see_also="The next chapter for bootstrapping. Volume 05 for the variance of an average.",
    ),
)

chapter(
    v,
    "03_td_q",
    "Temporal difference and Q-learning",
    entry(
        question="What does a temporal-difference update bootstrap, and when is Q-learning's target the wrong policy?",
        card=[
            ("TD(0)", "Update $V(s)$ toward $r + \\gamma V(s')$, a bootstrap target."),
            ("Bias", "The target is biased if $V$ is wrong, and lower variance than a full return because it cuts the future short."),
            ("Q-learning", "Target $r + \\gamma \\max_{a'} Q(s', a')$. This learns about the greedy policy while following a different behavior policy, under coverage conditions."),
            ("SARSA", "Target $r + \\gamma Q(s', a')$ with $a'$ actually taken. It evaluates the behavior policy, including its exploration."),
            ("Deadly triad", "Function approximation, bootstrapping, and off-policy updates can diverge. A table is the safe unit test."),
        ],
        theory="""Temporal-difference learning replaces the rest of the return by the current value estimate. That bootstrapping is why it can learn before an episode ends, and why a bad estimate propagates. In a table with a proper step size and enough visits, TD(0) converges to $V^\\pi$ for the policy being followed. The conditions matter. A neural network is a different theorem, and sometimes there is no theorem.

Q-learning's max operator targets the greedy action. If the behavior policy does not explore, the max is a max over fiction. If it explores but you deploy the greedy policy, SARSA's value, which includes exploratory actions, is not the value of the deployed policy. Say which policy's value you learned.

With function approximation, off-policy TD can diverge even on simple linear approximators. The practical warning is to unit-test new code on a table where dynamic programming is available, then add function approximation on a problem you can still visualize.""",
        derivation="""The TD error $\\delta = r + \\gamma V(s') - V(s)$ has expectation zero when $V = V^\\pi$ and $s'$ is drawn from the policy's transition, because of the Bellman equation. The update $V(s) \\leftarrow V(s) + \\alpha \\delta$ is stochastic approximation of that fixed point. Q-learning replaces $V(s')$ by $\\max_{a'} Q(s', a')$, which is the Bellman optimality backup, not the backup of the behavior policy. The behavior policy must still visit the pairs that the optimal policy uses, or the fixed point of the sampled updates is about a restricted state-action set.""",
        code="""import numpy as np

def q_learning_update(Q, s, a, r, s2, alpha, gamma):
    target = r + gamma * np.max(Q[s2])
    Q = Q.copy()
    Q[s, a] = (1 - alpha) * Q[s, a] + alpha * target
    return Q
""",
        experiment="""**Hypothesis.** On the tabular MDP you solved by dynamic programming, Q-learning with an exploring behavior policy approaches the optimal action values. With a behavior policy that never takes the optimal action in a key state, the corresponding entry does not move from its initialization.

**Protocol.** $\\epsilon$-greedy behavior with $\\epsilon = 0.2$, and a second run with $\\epsilon = 0$ and a bad start. Compare with `value_iteration_step` run to convergence for the optimal values, or with the hand solution.

**Metric.** Max absolute error on visited pairs, and the visit count.

**Decision rule.** The exploring run should approach the solution on visited pairs. The non-exploring run must show a hole. If both fail, the reward or transition table disagrees with the DP solver; do not tune $\\alpha$ to hide that.""",
        project="""**Deliverable.** Tabular Q-learning and SARSA on one MDP, compared with dynamic programming. A plot or table of the greedy policy's return, not only of the TD loss.

**Constraints.** The evaluation episodes use the greedy policy without exploration, and they are separate from the training episodes.

**Rubric.** You can say whether SARSA and Q-learning should differ at high $\\epsilon$, and your table agrees.""",
        beginner="Perform one Q-learning update by hand on a two-state table.",
        university="Match dynamic programming under exploration, and show the no-exploration hole.",
        research="Construct or cite a small divergent off-policy linear TD example and say which leg of the deadly triad you removed to restore stability in your tabular code.",
        failures="""- Evaluating with $\\epsilon$-greedy and calling it the greedy policy's return.
- A target network or a max operator coded with the wrong index so you bootstrap from the current action only.
- Function approximation added before the tabular test passes.""",
        see_also="Volume 02 for stochastic approximation. Volume 10 for the optimizer you will be tempted to drop in too early.",
    ),
)

chapter(
    v,
    "04_policy_gradients",
    "Policy gradients and actor-critic",
    entry(
        question="What does a policy-gradient estimator actually estimate, and how does a baseline change it?",
        card=[
            ("Policy gradient", "The gradient of $J(\\pi_\\theta)$ can be written as an expectation of $\\nabla \\log \\pi_\\theta(a \\mid s)$ times a return factor."),
            ("Score", "$\\nabla \\log \\pi$ does not depend on the unknown transition model. That is the point of the likelihood-ratio trick."),
            ("Baseline", "Subtracting a state-dependent baseline that does not depend on $a$ leaves the expectation unchanged and can cut variance."),
            ("Actor-critic", "A learned $V$ or $Q$ replaces the Monte Carlo return. It adds bias when the critic is wrong and cuts variance."),
            ("Credit", "A noisy advantage assigns credit to actions that did not cause the return. Variance is the failure mode, not a rounding error."),
        ],
        theory="""Policy gradient methods optimize a parameterized policy directly. They can represent stochastic policies, which matter when the optimal action is mixed or when the MDP is partially observed and a deterministic function of a non-Markov observation is not enough. The estimator is a score-function estimator. It is unbiased for the gradient of the expected return when the return is the true return and the expectation is over the policy you are differentiating. A replay buffer of old policies is a different distribution. You then need an off-policy correction or you are estimating a different gradient.

A baseline that depends only on the state does not change the expected gradient and often reduces variance. An actor-critic baseline that depends on a learned value is only a baseline in that unbiased sense when it is treated as constant with respect to the action in the score term, and the critic's own error is a second source of bias if you bootstrap.

Entropy bonuses and clipping, as in common practical algorithms, change the objective. They are not the theorem you just proved. Put them in the config as objective changes.""",
        derivation="""For a one-step bandit, $J(\\theta) = \\mathbb{E}_{a \\sim \\pi_\\theta}[R(a)]$. Then $\\nabla J = \\mathbb{E}[\\nabla \\log \\pi_\\theta(a) R(a)]$, because $\\nabla \\pi = \\pi \\nabla \\log \\pi$ and the score of a normalized policy has mean zero. Subtracting $b$ inside gives $\\mathbb{E}[\\nabla \\log \\pi(a) (R(a) - b)]$, and $\\mathbb{E}[\\nabla \\log \\pi(a)] b = 0$. The variance of the estimator depends on $b$. The choice $b \\approx \\mathbb{E}[R]$ is the simple baseline. In an MDP the same cancellation holds for a baseline that depends on the state at the decision, not on the sampled action.""",
        code="""import numpy as np

def bernoulli_score(action, prob):
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    # action in {0, 1}, derivative of log probability w.r.t. the logit is action - p
    # if prob = sigmoid(logit). Return the logit score.
    return float(action - p)
""",
        experiment="""**Hypothesis.** On a one-step bandit with a higher mean for action 1, a score-function update with a baseline moves the logit toward action 1 on average, and the baseline reduces the variance of the update relative to the raw return.

**Protocol.** Bernoulli rewards, means 0.2 and 0.8, analytic $\\nabla \\log \\pi$ from a sigmoid policy. Compare update variance with $b = 0$ and $b = \\mathbb{E}[R]$ estimated from a large sample, over many draws of the action and reward.

**Metric.** Mean update direction and variance of the scalar update.

**Decision rule.** The mean direction should agree with and without the baseline, up to Monte Carlo error. The variance should drop when the baseline is the mean reward. If the means disagree, the baseline depended on the action or the score is wrong.""",
        project="""**Deliverable.** The bandit experiment, plus a short-horizon tabular policy gradient on the MDP from earlier chapters, compared with Q-learning's greedy policy. Report evaluation return of the learned policy, not the training surrogate.

**Constraints.** The behavior during evaluation is the learned policy without an extra exploration bonus you forgot to turn off.

**Rubric.** You can point to the baseline's variance reduction in the bandit and to one bias you accepted if you used a critic.""",
        beginner="Compute the score of a nearly deterministic wrong action and see that it can be large.",
        university="Complete the bandit variance comparison and the short MDP run.",
        research="Write the importance weight $\\pi_\\theta(a \\mid s) / \\pi_{\\mathrm{old}}(a \\mid s)$ and the failure mode when the old policy never took the action the new policy wants. That is the off-policy policy-gradient hole.",
        failures="""- Optimizing a surrogate and never measuring episodic return.
- A baseline that uses the sampled action and breaks the unbiasedness argument.
- Reporting the best training episode as the policy's value.""",
        see_also="Volume 05 for variance of estimators. Volume 10 for the optimizer wrapped around this noisy gradient.",
    ),
)

chapter(
    v,
    "05_offline_partial",
    "Offline data and partial observations",
    entry(
        question="When does a logged dataset fail to answer what a new policy will do, and when is the observation not a state?",
        card=[
            ("Behavior", "The policy that produced the log. A new policy's actions may be rare or absent in the log."),
            ("Support", "You cannot estimate the value of an action that was never taken, without a model that extrapolates."),
            ("Confounding", "If the logger chose actions based on a variable you did not record, the conditional outcome given recorded state and action is not the causal effect of the action."),
            ("Partial observation", "A policy of the observation is not a policy of the state. Values can be intransitive and the Markov property can fail."),
            ("Offline claim", "The honest claim is about actions the logger took, unless you state an extrapolation model."),
        ],
        theory="""Offline reinforcement learning wants a new policy from data collected by an old one. The statistical problem is support. The causal problem is confounding. Both are easy to miss if you only look at a training loss that imitates the logger. Imitation learns the logger. It does not learn the return of a different policy. Q-learning on the log, without coverage, fills unseen actions with optimistic or stale values and then chooses them.

Partial observability means the agent does not see $S_t$. A recurrent state, a belief, or a window of observations is an attempt to rebuild a state. A feedforward map from one observation can be insufficient even when a map from the true state would be trivial. The aliased MDP in the first chapter is the unit test: if your algorithm claims a high value from the observation alone, it is averaging two worlds.

The project-level rule is to label the claim. "Value of the logger's actions on recorded states" is estimable by a regression. "Value of a new policy" needs overlap or a model. "Optimal policy of the hidden state" needs the hidden state or a belief that is good enough. Do not use one phrase for all three.""",
        derivation="""In a contextual bandit, the mean reward of a new policy $\\pi$ is $\\mathbb{E}[\\pi(A \\mid X) / \\pi_b(A \\mid X) \\cdot R]$ when $\\pi_b(a \\mid x) = 0$ implies $\\pi(a \\mid x) = 0$, the expectations exist, and the logged action was drawn from $\\pi_b$ independent of the potential rewards given $X$. If $\\pi_b$ can be zero where $\\pi$ is not, the weight is undefined and the expectation identity does not give you an estimator. If a confounder $U$ influenced $A$ and $R$ and you only condition on a non-sufficient $X$, the identity with weights built from $X$ estimates the wrong functional. Overlap and unconfoundedness are assumptions, not properties of the neural net.""",
        code="""import numpy as np

def importance_weight(pi_new, pi_behavior, floor=1e-3):
    pi_behavior = np.asarray(pi_behavior, dtype=np.float64)
    if np.any(pi_behavior <= 0):
        raise ValueError("behavior probability is zero; the new policy is off support")
    w = np.asarray(pi_new, dtype=np.float64) / np.maximum(pi_behavior, floor)
    return w
""",
        experiment="""**Hypothesis.** On a contextual bandit where the logger never chooses action 1 in context 0, an importance-weighted estimate of a policy that does choose it should raise, not return a confident finite number. On a fully supported logger, the weighted reward should approach the true value of the new policy as $n$ grows.

**Protocol.** Two contexts, two actions, known reward means. Logger with full support versus logger with a hole. New policy that steps into the hole.

**Metric.** Whether the code raises, and the error of the weighted estimator in the full-support case over 100 trials.

**Decision rule.** The hole must be an error, not a clipped weight that pretends the probability was `floor` without a huge variance warning. If you use a floor, the note must say you changed the target.""",
        project="""**Deliverable.** The bandit simulation, the aliased observation from the MDP chapter evaluated with Q-learning on observations, and a paragraph that labels your claim as regression, off-policy evaluation, or belief-state control.

**Constraints.** A zero behavior probability is not clipped silently.

**Rubric.** You can state one deployment policy you refuse to evaluate from the log you have.""",
        beginner="Compute an importance weight for probabilities 0.8 and 0.4, and see the code reject 0.",
        university="Run both logger regimes and the aliased Q-learning comparison.",
        research="Describe a belief-state update for a two-state hidden Markov model with a discrete observation, and why the belief is Markov even though the observation is not.",
        failures="""- Behavioral cloning reported as an improved decision policy without a return comparison.
- Clipped importance weights with no variance diagnostic.
- A partially observed problem trained and evaluated as if the observation were the state, with no aliasing check.""",
        see_also="Volume 05 for inverse propensity as a statistical instrument. Volume 16 for policies that optimize a proxy.",
    ),
)

chapter(
    v,
    "06_agent_project",
    "Project: a tabular agent with ablations",
    entry(
        question="Can you show, on a problem you can solve exactly, which RL ingredient changed the return?",
        card=[
            ("Environment", "A tabular MDP with a dynamic-programming optimum."),
            ("Agents", "At least two of: Monte Carlo, Q-learning, SARSA, policy gradient."),
            ("Ablations", "Exploration rate, discount, and behavior-policy support."),
            ("Metric", "Greedy or learned-policy return on fresh evaluation episodes, versus the optimal value."),
            ("Failure", "One run with a state alias or a no-exploration behavior policy."),
        ],
        theory="""This project is the debug checklist of Volume 10 applied to agents. The exact solution is the known case. An agent that cannot approach it on a four-state MDP will not become correct because the state is replaced by a neural embedding. The ablation is the experiment. Exploration, discount, and support are factors with predicted directions. The evaluation protocol is frozen: same start distribution, same horizon, no exploration noise in the reported return unless the deployed policy includes it.

Write the reward contract at the top. If you add a shaping term, it is a different reward. Show the unshaped return too, or you will ship a policy that farms the shaping.""",
        derivation="""Let $V^\\star(s_0)$ be the optimal value at the start. An evaluation estimate $\\hat{V}$ from $n$ greedy episodes has a standard error you can compute from the episode returns. A claim that an agent "solved" the MDP means $\\hat{V}$ is close to $V^\\star$ relative to that standard error and to a practical margin. A claim that exploration mattered means the $\\epsilon = 0$ ablation is worse by more than that noise. The dynamic-programming number is not an evaluation episode. It is the target those episodes estimate.""",
        code="""import numpy as np

def evaluate_episodes(returns):
    r = np.asarray(returns, dtype=np.float64)
    se = float(np.std(r, ddof=1) / np.sqrt(r.size)) if r.size > 1 else float("nan")
    return {"mean": float(np.mean(r)), "se": se, "n": int(r.size)}
""",
        experiment="""**Hypothesis.** Q-learning with $\\epsilon = 0.2$ approaches $V^\\star$ from the start state, and $\\epsilon = 0$ from a policy that misses the rewarding action does not. Changing $\\gamma$ from the value used in dynamic programming changes the target, so agreement with the old $V^\\star$ should fail.

**Protocol.** The same MDP as the earlier chapters. Evaluation episodes are greedy. Report `evaluate_episodes`.

**Metric.** Mean return versus $V^\\star$, and the $\\epsilon = 0$ gap.

**Decision rule.** The exploring agent matches within a few standard errors or you debug before adding a second algorithm. The $\\gamma$ mismatch must be visible. If it is not, the horizon is too short for discounting to matter; lengthen it.""",
        project="""**Deliverable.** The solved MDP, two agents, the exploration and discount ablations, the aliasing failure, and a card that states the reward, $\\gamma$, and the evaluation policy.

**Constraints.** No neural network until the tabular match succeeds. If you add one afterward, it is a separate, labeled experiment on the same MDP with a one-hot state encoding, and it must still match $V^\\star$.

**Rubric.** The optimal value is in the table. A learning curve without that line is incomplete.""",
        beginner="Evaluate a random policy and the optimal policy with `evaluate_episodes`.",
        university="Complete the ablations and the alias failure.",
        research="Add an offline log from a narrow behavior policy and show a new policy you refuse to score. Include the importance-weight error from the previous chapter.",
        failures="""- Reporting the best episode.
- Evaluating with exploration left on.
- A shaped reward in training and an unshaped claim in the abstract of the note.""",
        see_also="Volume 15 for deploying a policy as a system. Volume 17 if the policy is a prompted model rather than a tabular agent.",
    ),
)
