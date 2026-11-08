"""Volumes 05–08: inference, supervised learning, evaluation, unsupervised learning."""

from booklib import chapter, entry, volume

v = volume(
    5,
    "Statistical inference",
    "University",
    "A reader who can compute a mean, a variance, and a holdout log likelihood.",
    "How do you estimate a functional of an unknown law, and how do you say how wrong the estimate might be?",
)

chapter(
    v,
    "01_estimators",
    "Estimators and sampling distributions",
    entry(
        question="What is an estimator, as distinct from the estimate sitting in one notebook?",
        card=[
            ("Functional", "A map $T(P)$ from a law to a number or a vector, such as a mean or a coefficient."),
            ("Estimator", "A function of the sample, $\\hat{T}_n$, intended to track $T(P)$."),
            ("Estimate", "The number $\\hat{T}_n$ returns on the sample you have."),
            ("Sampling law", "The law of $\\hat{T}_n$ when the sample is redrawn."),
            ("Bias and variance", "$\\mathbb{E}[\\hat{T}_n] - T(P)$, and the variance of $\\hat{T}_n$ under redraws."),
        ],
        theory="""The estimate is one draw. The estimator is the procedure. Judgments about reliability are judgments about the procedure's sampling law, not about the dignity of a single number. Unbiasedness, $\\mathbb{E}[\\hat{T}_n] = T(P)$, is optional. Mean squared error $\\mathbb{E}[(\\hat{T}_n - T(P))^2]$ equals variance plus squared bias, and a little bias can buy a lot of variance reduction. Ridge regression is that trade on purpose.

The sampling law depends on assumptions: iid draws, the form of $P$, and which parts of the procedure were fixed before seeing the data. A procedure that chooses a model by looking at the test set does not have the sampling law you would compute for a fixed model. Volume 07 is about freezing that procedure. Here you learn to name $T(P)$ before you name a method.

Plug-in estimators replace $P$ by the empirical measure $P_n = n^{-1}\\sum_i \\delta_{Z_i}$. The sample mean is the plug-in estimator of the mean. The $1/n$ Gaussian MLE variance is the plug-in estimator of variance. They inherit simplicity and they inherit every mistake in the empirical measure, including outliers.""",
        derivation="""Let $\\hat{\\mu} = \\bar{Y}$ and $\\mu = \\mathbb{E}[Y]$, with iid draws and finite variance $\\sigma^2$. Then $\\mathbb{E}[\\bar{Y}] = \\mu$ and $\\mathrm{Var}(\\bar{Y}) = \\sigma^2/n$, so MSE is $\\sigma^2/n$. For the plug-in variance $\\hat{\\sigma}^2 = n^{-1}\\sum_i (Y_i - \\bar{Y})^2$, a standard calculation gives $\\mathbb{E}[\\hat{\\sigma}^2] = (1 - 1/n)\\sigma^2$ under iid sampling. It is biased low. The $n-1$ version is unbiased for $\\sigma^2$ and is not the Gaussian MLE. Report which functional you meant.

MSE decomposition: $\\mathbb{E}[(\\hat{T} - T)^2] = \\mathrm{Var}(\\hat{T}) + (\\mathbb{E}[\\hat{T}] - T)^2$ when the expectations exist. The cross term vanishes because $\\hat{T} - \\mathbb{E}[\\hat{T}]$ has mean zero.""",
        code="""import numpy as np

def plug_in_mean_var(y):
    y = np.asarray(y, dtype=np.float64)
    mu = float(np.mean(y))
    var = float(np.mean((y - mu) ** 2))
    return mu, var
""",
        experiment="""**Hypothesis.** Over repeated samples, the plug-in variance averages about $(1 - 1/n)\\sigma^2$, and the $n-1$ variance averages about $\\sigma^2$.

**Protocol.** Gaussian data, $\\sigma^2 = 4$, $n = 10$, 2000 replicates.

**Metric.** Mean of each variance estimator across replicates.

**Decision rule.** The plug-in mean should sit near $4 \\times 0.9 = 3.6$. If it sits near 4, you used `ddof=1` by accident.""",
        project="""**Deliverable.** A note that names $T(P)$ for three procedures you have already run: the training mean baseline, the least-squares coefficient, and a Gaussian variance. For one of them, simulate the sampling distribution and report bias and variance.

**Constraints.** The procedure includes any preprocessing. If you standardize, the estimator is "standardize then fit," not "fit."

**Rubric.** You never describe a single estimate as "the estimator's bias." Bias is a property of the procedure under a stated $P$.""",
        beginner="Run the variance simulation.",
        university="Write the MSE decomposition and check it numerically on a deliberately biased estimator, such as $\\bar{Y}/2$.",
        research="Define the sampling distribution of a two-step estimator that selects a subset of features on the same sample it fits. Explain why an analytic standard error that ignores the selection step is for a different procedure.",
        failures="""- Quoting a standard error from a formula whose assumptions the selection procedure violated.
- Calling an estimator unbiased because the residual mean on the training sample is zero.
- Comparing MSEs of procedures aimed at different functionals.""",
        see_also="Maximum likelihood next. Volume 07 for selection that changes the procedure.",
    ),
)

chapter(
    v,
    "02_maximum_likelihood",
    "Maximum likelihood",
    entry(
        question="What problem does maximum likelihood solve, and when is the solution a bad match to the decision you care about?",
        card=[
            ("Model", "A family $\\{P_\\theta: \\theta \\in \\Theta\\}$."),
            ("Likelihood", "$L(\\theta) = p_\\theta(z_{1:n})$ on the observed sample. Log likelihood $\\ell(\\theta) = \\log L(\\theta)$."),
            ("MLE", "$\\hat{\\theta} \\in \\arg\\max_\\theta \\ell(\\theta)$."),
            ("iid shortcut", "If the model says the rows are iid, $\\ell(\\theta) = \\sum_i \\log p_\\theta(z_i)$."),
            ("Misspecification", "If no $P_\\theta$ equals the true law, the MLE targets the $P_\\theta$ closest in KL divergence, not every functional you might have wanted."),
        ],
        theory="""Maximum likelihood is minimum KL divergence inside a family. You pick the member of the family that makes the observed sample least surprising, in the sense of that family's own density. If the family contains the truth and regularity conditions hold, the MLE is asymptotically efficient for $\\theta$: you cannot do better, asymptotically, with an unbiased estimator. That sentence is about $\\theta$ in a well-specified parametric model. It is not a general blessing.

Under misspecification the maximizer converges to the projection of the true law onto the family in KL. For a Gaussian family that projection matches the true mean and variance, and ignores every other feature of the truth. If your decision depends on a tail probability, the MLE of a Gaussian can be a poor plug-in even though it is a fine estimator of the mean. Match the estimator to $T(P)$. Likelihood is the right tool when the parameter of the family is the functional you want, or when you will use the whole fitted law and have checked it.

The iid log-likelihood sum is invalid when the model says the rows are dependent. A Markov model has a likelihood, but it is not $\\sum_i \\log p_\\theta(z_i)$ with a marginal $p_\\theta$. Using the wrong likelihood estimates the wrong projection.""",
        derivation="""KL divergence from the true density $p$ to a model density $q_\\theta$ is $\\mathbb{E}_p[\\log p(Z)] - \\mathbb{E}_p[\\log q_\\theta(Z)]$. The first term does not depend on $\\theta$. Minimizing KL is maximizing $\\mathbb{E}_p[\\log q_\\theta(Z)]$. The average $n^{-1}\\sum_i \\log q_\\theta(z_i)$ is the plug-in version of that expectation under an iid empirical measure. Its maximizer is the MLE.

For a Bernoulli($p$) sample the log likelihood is $s \\log p + (n-s)\\log(1-p)$ with $s$ the number of ones. The derivative $s/p - (n-s)/(1-p)$ vanishes at $\\hat{p} = s/n$. Second derivative negative on $(0, 1)$ makes it a maximum. At $s \\in \\{0, n\\}$ the maximum is on the boundary, $0$ or $1$, where a Gaussian approximation to the sampling law of $\\hat{p}$ is a poor description at small $n$.""",
        code="""import numpy as np

def bernoulli_mle(y):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if np.any((y != 0) & (y != 1)):
        raise ValueError("Bernoulli MLE expects 0/1")
    return float(np.mean(y))

def bernoulli_loglik(y, p, eps=1e-12):
    p = float(np.clip(p, eps, 1 - eps))
    y = np.asarray(y, dtype=np.float64)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
""",
        experiment="""**Hypothesis.** On Bernoulli data the MLE matches $s/n$, and on data generated from a two-component mixture of Bernoullis with a latent group, a single Bernoulli MLE matches the marginal mean and misses the group structure.

**Protocol.** Case A: $p = 0.2$, $n = 500$. Case B: with probability $1/2$ a row is Bernoulli($0.05$), else Bernoulli($0.55$). Fit a single $\\hat{p}$. Compare $\\hat{p}$ with the average of the row's true success probability.

**Metric.** $|\\hat{p} - p|$ in case A. In case B, the difference between $\\hat{p}$ and the marginal mean, which should be near 0, and a written statement that a single $p$ assigns the wrong probability to a known group.

**Decision rule.** Case A should be within sampling error of 0.2. Case B's $\\hat{p}$ near 0.3 is a success at estimating the marginal and a failure at describing a group. The note must say which functional was requested.""",
        project="""**Deliverable.** MLE fits for Bernoulli, Gaussian, and Poisson on synthetic data from the matching family and from one misspecified family, with holdout log likelihood.

**Constraints.** State the target functional in the first line.

**Rubric.** You refuse at least one MLE as an answer to a decision it does not target, even if the optimization succeeded.""",
        beginner="Derive and code the Bernoulli MLE.",
        university="Include the mixture misspecification case.",
        research="Write the Gaussian score equation and the observed information $-\\partial^2 \\ell / \\partial \\theta^2$ for a scalar parameter. Say what the inverse information estimates when the model is correct, and what sandwich estimation changes when it is not. You may state the sandwich formula without proving it.",
        failures="""- Maximizing likelihood under an independence model for a time series and calling the result efficient.
- Using a boundary MLE of 0 or 1 as a probability in a product that then becomes zero forever.
- Treating a likelihood comparison on the training sample as an unbiased model selection rule.""",
        see_also="Volume 04 for the families. Volume 06 for the losses that are negative log likelihoods.",
    ),
)

chapter(
    v,
    "03_intervals",
    "Tests and intervals",
    entry(
        question="What claim does a confidence interval make, and what claim does it refuse?",
        card=[
            ("Interval", "A random set $C(Z_{1:n})$ computed from the sample."),
            ("Coverage", "$P(T(P) \\in C(Z_{1:n}))$, the probability over samples that the set covers the true functional."),
            ("Not", "A probability that the true value lies in the numerical interval you already computed, unless you did a Bayesian calculation on purpose."),
            ("Test", "A rule that rejects a hypothesis $H_0$ when a statistic falls in a critical region. Size is the rejection probability when $H_0$ is true."),
            ("p-value", "The probability, under $H_0$, of a statistic at least as extreme as the one observed. It is not $P(H_0 \\mid \\mathrm{data})$."),
        ],
        theory="""A 95 percent confidence interval is a procedure that covers the true value on 95 percent of repeated samples, under the assumptions used to build it. After you compute $[1.2, 1.8]$, the true value either is in it or is not. The 95 percent has been spent on the procedure. This frequentist reading is the one the formulas in this chapter use. Bayesian credible intervals answer a different question, in the next chapter, and they should not share a label.

A test of $H_0: \\mu = 0$ that fails to reject is not evidence that $\\mu$ is zero. It is an absence of a particular kind of evidence against zero, at the sample size you have. Report the interval. The interval shows the effects you have not ruled out. A significant result on a huge sample can be an effect too small to matter. Put the scientific threshold in the note before you look at the p-value.

Assumptions enter coverage. A $t$ interval for a mean assumes iid draws with a finite variance, and the $t$ critical value is exact when the draws are Gaussian. Under mild skew and large $n$ it is often a decent approximation for the mean, by the central limit theorem. It is not a check on the median, and it is not valid under the dependence you already know how to construct.""",
        derivation="""If $\\bar{Y} \\approx \\mathcal{N}(\\mu, \\sigma^2/n)$ and $s$ estimates $\\sigma$, then $(\\bar{Y} - \\mu)/(s/\\sqrt{n})$ is approximately standard normal, or Student-$t$ with $n-1$ degrees of freedom under an exact Gaussian model. A level $1-\\alpha$ interval is $\\bar{Y} \\pm t_{n-1, 1-\\alpha/2}\\, s/\\sqrt{n}$. Coverage is the probability this random interval contains $\\mu$.

For a test of $\\mu = \\mu_0$, the statistic is the same ratio at $\\mu_0$. Rejecting when the absolute value exceeds the critical value is the same event as $\\mu_0$ falling outside the interval. The duality is useful: if you would not trust the interval, do not trust the test.""",
        code="""import numpy as np

def mean_interval(y, alpha=0.05):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.size
    mean = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n))
    # Normal critical value is a large-n stand-in. Small n should use a t table.
    z = 1.959963984540054
    if alpha != 0.05:
        raise ValueError("this helper is written for alpha=0.05")
    return mean - z * se, mean + z * se
""",
        experiment="""**Hypothesis.** The normal-approximation interval for a Gaussian mean covers at about 95 percent. The same numerical procedure under block dependence undercover.

**Protocol.** Independent case: $n = 40$ Gaussian draws, 1000 replicates, record the fraction of intervals containing the true mean. Dependent case: 8 independent Gaussian draws repeated 5 times each, and apply `mean_interval` as if $n = 40$.

**Metric.** Empirical coverage.

**Decision rule.** Independent coverage should be near 95 percent, a bit low or high is plausible at 1000 replicates (the standard error of a coverage estimate is about $\\sqrt{0.95 \\cdot 0.05 / 1000} \\approx 0.7$ percentage points, so several points of error can happen; look for a gross miss). Dependent coverage should fall. Using a $z$ critical value instead of $t$ at $n = 40$ is a small effect; dependence is the large one.""",
        project="""**Deliverable.** Intervals for a mean and for a difference of means on synthetic data, plus the dependence counterexample, plus a sentence that translates a non-rejection into "effects larger than this are unlikely," using the interval endpoints rather than the p-value.

**Constraints.** Do not write "the probability that the true mean is in $[a, b]$ is 95 percent."

**Rubric.** The wording matches the coverage definition. The dependent case is not swept into a footnote without a number.""",
        beginner="Compute one interval for a sample of size 30 and state the coverage claim correctly.",
        university="Run both coverage experiments.",
        research="Construct a bootstrap percentile interval for the median and compare its coverage with a mean interval on skewed data (exponential draws). Say which functional each interval is for.",
        failures="""- Peeking at several outcomes and reporting the smallest p-value as if it were the only test.
- Interpreting a confidence interval from a misspecified model as if the assumptions were decorative.
- Stopping data collection when the interval first excludes zero and then quoting a 95 percent coverage that the stopping rule destroyed.""",
        see_also="The Bayesian chapter for a different interval. Volume 07 for paired model comparisons.",
    ),
)

chapter(
    v,
    "04_bayes",
    "Bayesian updating",
    entry(
        question="What extra ingredient does a Bayesian calculation use, and what does the posterior mean?",
        card=[
            ("Prior", "A law $p(\\theta)$ on parameters, chosen before the sample."),
            ("Likelihood", "$p(z \\mid \\theta)$ from the sampling model."),
            ("Posterior", "$p(\\theta \\mid z) \\propto p(z \\mid \\theta) p(\\theta)$."),
            ("Predictive", "$p(z_{\\mathrm{new}} \\mid z) = \\int p(z_{\\mathrm{new}} \\mid \\theta) p(\\theta \\mid z) d\\theta$."),
            ("Credible interval", "A set with posterior probability $1-\\alpha$. It is not a confidence interval."),
        ],
        theory="""Bayesian updating takes the prior times the likelihood and renormalizes. The prior is part of the estimator. Two priors are two procedures. A flat prior is still a prior: it says, in a chosen parameterization, that wide regions start equally weighted, and it is not invariant to reparameterization unless you did the extra work to make it so.

The posterior is a distribution on $\\theta$ given the sample, inside the joint model. A 95 percent credible interval contains 95 percent of that posterior. It answers "where does the updated belief sit?" It does not, by itself, answer "what fraction of repeated samples produce intervals that cover the truth?" Sometimes a Bayesian procedure also has good coverage. That is a separate theorem. Label the interval by the question it answers.

The posterior predictive distribution is what you want for a new observation. A plug-in prediction at the posterior mode ignores leftover uncertainty about $\\theta$. For a Bernoulli with a Beta prior, the predictive probability of the next one is a simple ratio you should be able to derive in your sleep. It shrinks the raw frequency toward the prior mean, which is useful exactly when $n$ is small and dangerous when the prior was a wish.""",
        derivation="""Take $Y \\mid p \\sim \\mathrm{Binomial}(n, p)$ and $p \\sim \\mathrm{Beta}(\\alpha, \\beta)$, with density proportional to $p^{\\alpha-1}(1-p)^{\\beta-1}$. The posterior is $\\mathrm{Beta}(\\alpha + s, \\beta + n - s)$, because the $p$ powers add. The prior mean is $\\alpha/(\\alpha+\\beta)$. The posterior mean is $(\\alpha+s)/(\\alpha+\\beta+n)$, a weighted average of the prior mean and $s/n$.

The probability the next draw is 1, given the past, is the posterior mean of $p$. That is the predictive probability. As $n \\to \\infty$ with $s/n \\to p_\\star$, the posterior mean goes to $p_\\star$ for fixed $\\alpha, \\beta$. The prior washes out when the likelihood concentrates and the model is correct. It does not wash out of a misspecified likelihood.""",
        code="""import numpy as np

def beta_binomial_update(successes, n, alpha=1.0, beta=1.0):
    a = alpha + successes
    b = beta + (n - successes)
    mean = a / (a + b)
    return {"alpha": a, "beta": b, "mean": float(mean), "predictive_next": float(mean)}
""",
        experiment="""**Hypothesis.** With a Beta$(1, 1)$ prior and $s = 0$ successes in $n = 3$ trials, the posterior mean is $1/5$, not 0. With $n = 300$ and $s = 0$, the posterior mean is near 0 but not exactly 0.

**Protocol.** Compare `beta_binomial_update` with the MLE on both samples. Draw posterior samples by `rng.beta` if available and check that their mean matches the formula.

**Metric.** Posterior mean versus MLE.

**Decision rule.** The small-sample gap must be visible. If the $n = 3$ posterior mean is 0, the prior was not added.""",
        project="""**Deliverable.** A note with one Beta-Binomial update, one sentence labeling a credible summary as Bayesian, and one sentence saying what coverage experiment you would need before calling it a confidence interval. Include a prior you consider foolish (Beta$(50, 1)$ on a coin you think is fair) and show that $n = 10$ does not wash it out.

**Constraints.** Do not grid-search the prior on the same data you report as confirmation.

**Rubric.** The foolish prior is visible in the posterior mean, and you say what additional data would be required to move it.""",
        beginner="Compute the Beta$(1,1)$ update for 7 successes in 10 trials by hand and in code.",
        university="Include the foolish-prior calculation.",
        research="State the posterior for a Gaussian mean with known variance and a Gaussian prior. Identify the posterior mean as a precision-weighted average. Compare its shrinkage with ridge regression in one sentence, carefully: related linear algebra, different problem.",
        failures="""- Reporting a credible interval and a p-value as if they were the same object.
- Tuning the prior until the posterior matches a hoped-for story, on the analysis sample.
- Forgetting that a wrong likelihood is not repaired by a careful prior.""",
        see_also="Volume 02 for ridge as penalized least squares. Volume 14 if a posterior over a policy appears later; do not import this chapter's coin math without a new model.",
    ),
)

chapter(
    v,
    "05_power",
    "Experimental design and power",
    entry(
        question="How do you choose a sample size for an effect you would be unwilling to miss?",
        card=[
            ("Effect", "The smallest difference that would change a decision, in the units of the target."),
            ("Power", "The probability that the test rejects $H_0$ when a stated alternative is true."),
            ("Inputs", "Effect, noise scale, test size $\\alpha$, desired power, and the design (paired or not, split unit)."),
            ("Output", "A sample size, plus the assumption list."),
            ("Not", "A sample size copied from a similar paper with a different noise scale."),
        ],
        theory="""Power is the sensitivity of a procedure. A test with size 5 percent and power 10 percent at the effect you care about is a procedure that almost never finds the thing you are looking for, and occasionally raises a false alarm. Collecting that sample and "seeing what happens" is a way to spend the sample without answering the question.

The effect size is a decision input, not a statistical output. "A 1 percent drop in squared error" might matter in a huge system and not matter in a class project. Write it down first. Then estimate the noise from a pilot or from the constant-baseline residual variance. Then solve for $n$. If the required $n$ is impossible, change the question, the design, or the measurement. Do not pretend the affordable $n$ answers the original question.

Paired designs reduce variance when the two measurements share a shock. Entity-level randomization increases the effective noise when you must randomize at the entity even though you measure rows. The sample-size formula has to use the variance of the unit you randomize, not the variance of the noisiest subplot.""",
        derivation="""For a one-sample $z$ test of a mean, $H_0: \\mu = 0$ against $\\mu = \\delta > 0$, the rejection threshold at level $\\alpha$ is about $z_{1-\\alpha} \\sigma/\\sqrt{n}$. Power is the probability that $\\bar{Y}$ exceeds that threshold when the true mean is $\\delta$. Under a normal approximation,

$$1-\\beta \\approx 1-\\Phi\\!\\left(z_{1-\\alpha} - \\frac{\\delta \\sqrt{n}}{\\sigma}\\right).$$

Set this equal to a desired power and solve

$$n \\approx \\left(\\frac{(z_{1-\\alpha} + z_{1-\\beta})\\sigma}{\\delta}\\right)^2.$$

For $\\alpha = 0.05$ and power $0.8$, the $z$ values are about $1.64$ (one-sided) and $0.84$, so $n \\approx (2.48 \\sigma/\\delta)^2$. A two-sided test uses $1.96$ instead of $1.64$. Halving $\\delta$ quadruples $n$.""",
        code="""import numpy as np

def n_one_sample(delta, sigma, z_alpha=1.64485, z_power=0.84162):
    if delta <= 0 or sigma <= 0:
        raise ValueError("delta and sigma must be positive")
    return int(np.ceil(((z_alpha + z_power) * sigma / delta) ** 2))
""",
        experiment="""**Hypothesis.** At the computed $n$, a one-sided $z$ test of a normal mean has power near 80 percent against the $\\delta$ you designed for, and substantially lower power at $n/4$.

**Protocol.** $\\sigma = 1$, $\\delta = 0.2$, compute $n$. Simulate 500 experiments at that $n$ and at $n/4$, testing $\\bar{Y} > 1.64485 \\sigma/\\sqrt{n}$.

**Metric.** Rejection frequency.

**Decision rule.** The designed $n$ should land near 80 percent, with Monte Carlo error of a couple of points. $n/4$ should be obviously worse. A designed-sample power near 50 percent means the $z$ quantiles or the threshold used the wrong $\\sigma$.""",
        project="""**Deliverable.** A design note for an A/B comparison of two prediction models on a metric you specify. Include the smallest effect you care about, a noise estimate, $\\alpha$, desired power, the resulting $n$, and what you will do if that $n$ is unaffordable.

**Constraints.** The noise estimate must say whether it is per row or per entity.

**Rubric.** A reader can recompute $n$ from the note. The effect is in metric units, not in "Cohen's $d$" alone, unless you also translate it.""",
        beginner="Compute $n$ for $\\sigma = 1$, $\\delta = 0.5$, and explain what happens to $n$ if $\\delta$ is halved.",
        university="Run the power simulation.",
        research="Modify the formula for a two-sample test with variance $\\sigma^2$ in each arm. Then say what changes if the true variances differ by a factor of four and you used a pooled estimate.",
        failures="""- Powering a study for a huge effect and then claiming a precise estimate of a small one.
- Using a row-level variance when randomization is by entity.
- Computing power after seeing a p-value, with an effect size taken from the same data, and calling it the design.""",
        see_also="Volume 07 for the metric the design must name. Volume 03 for entity splits.",
    ),
)

chapter(
    v,
    "06_ab_project",
    "Project: analyze a comparison without fooling yourself",
    entry(
        question="How do you report a two-policy comparison so the interval, the unit, and the decision stay attached?",
        card=[
            ("Policies", "Two procedures frozen before the test sample is scored."),
            ("Unit", "The independent randomization or sampling unit."),
            ("Metric", "One primary metric, chosen before looking."),
            ("Estimate", "Difference in means on that unit, with an interval."),
            ("Decision", "Ship, keep, or redesign, compared with the effect the study was powered for."),
        ],
        theory="""This project is the inference volume in one folder. You will generate two policies with a known true difference so you can see whether your analysis recovers it. The discipline is identical when the truth is unknown: the analysis does not get to know it.

A comparison of already-fitted models on a test set estimates the difference in risk for those two frozen predictors. It does not include training-seed variability. Say so. If you refit both models inside each replicate, the interval is about the procedures, which is usually what a paper wants and not what a one-shot deployment check wants.

Peeking at five metrics and highlighting the significant one destroys the size of the test. Predeclare the primary metric in `config.json` before the script prints it. Secondary metrics can be exploratory and must be labeled as such.""",
        derivation="""Let the unit-level outcomes be $A_i$ and $B_i$ under the two policies, and let $D_i = B_i - A_i$ when the comparison is paired on the same unit. The target is $\\mathbb{E}[D]$. The paired interval is the one-sample interval for the $D_i$. Pairing is valid when each unit yields both measurements without interference. If policy B's presence changes policy A's outcome, you do not have a paired difference of stable potential outcomes. For synthetic data you can enforce no interference by simulation. In a letter to your future self, note that production traffic often interferes.""",
        code="""import numpy as np

def paired_difference(a, b):
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    z = 1.959963984540054
    return {"diff": mean, "low": mean - z * se, "high": mean + z * se, "n": int(d.size)}
""",
        experiment="""**Hypothesis.** If $B = A + 0.3 + \\text{noise}$ on paired units, the interval for the mean difference covers 0.3 at roughly 95 percent and excludes 0 in a study powered for 0.3.

**Protocol.** Gaussian noise scale 1, $n$ from the power formula for $\\delta = 0.3$, two-sided, so use $z_\\alpha \\approx 1.96$ in a modified calculator. Run 400 replicates. Also run one underpowered replicate at $n = 20$ and write how you would talk about it.

**Metric.** Coverage of 0.3, and the fraction of intervals that exclude 0.

**Decision rule.** Coverage near 95 percent. Power should be high by construction. The $n = 20$ write-up must not convert "interval contains 0" into "the policies are equal." """,
        project="""**Deliverable.** A folder, in the Volume 03 layout, with the predeclared metric, the power calculation, the paired analysis, the coverage simulation, and a decision paragraph a teammate could act on.

**Constraints.** One primary metric. Any other number is labeled exploratory.

**Rubric.** The decision uses the interval and the predeclared effect, not a p-value alone. The unit of analysis is named in the first five lines of the note.""",
        beginner="Analyze one paired sample of size 40 with a known difference and interpret the interval in the approved wording.",
        university="Complete the coverage simulation and the underpowered write-up.",
        research="Add a second, entity-clustered version: 10 independent pairs repeated 20 times, analyzed once correctly at the entity level and once incorrectly at the repeated-row level. Compare interval widths.",
        failures="""- Scoring the test set, tuning, and scoring it again as if the first look did not happen.
- Reporting a relative improvement with no base rate.
- Letting the randomization unit and the analysis unit differ without a reason written down.""",
        see_also="Volume 07, which uses this comparison style for model selection. Volume 16 for when the metric itself is the wrong decision.",
    ),
)

v = volume(
    6,
    "Supervised learning",
    "University",
    "A reader who can fit least squares, audit a table, and state a sampling unit.",
    "Which predictor should you fit, under which loss, and what baseline must it beat?",
)

chapter(
    v,
    "01_supervised_problem",
    "The supervised problem",
    entry(
        question="What is the object a supervised method is trying to estimate?",
        card=[
            ("Task", "Learn $f$ from samples of $(X, Y)$ so that $\\ell(f(X), Y)$ is small on a new draw."),
            ("Bayes predictor", "A minimizer $f_\\star$ of the risk $R(f) = \\mathbb{E}[\\ell(f(X), Y)]$ over a rich class of functions."),
            ("Model class", "A restricted family $\\mathcal{F}$. Approximation error is the gap between $f_\\star$ and the best element of $\\mathcal{F}$."),
            ("Algorithm", "A map from a sample to $\\hat{f} \\in \\mathcal{F}$. Estimation error is how far $\\hat{f}$ is from that best element."),
            ("Generalization gap", "$R(\\hat{f}) - \\hat{R}_n(\\hat{f})$, a random quantity, not a moral failing."),
        ],
        theory="""Supervised learning splits error into two debts. Approximation error is the price of a small family: linear functions cannot represent a quadratic conditional mean. Estimation error is the price of a finite sample: even the best linear function is not exactly the one you fit. A larger family can cut approximation error and raise estimation error. Regularization, more data, and a better representation are different ways of paying these debts. They are not interchangeable slogans.

The Bayes predictor depends on the loss. Under squared error it is the conditional mean. Under 0-1 loss, for two classes, it is the thresholded conditional probability. A model that estimates $P(Y=1 \\mid X)$ well can still be a poor 0-1 classifier if you choose a bad threshold, and a hard classifier can have good 0-1 loss and be useless when you needed a probability. Say which Bayes predictor you are chasing.

The empirical risk minimizer picks $\\hat{f}$ by looking at $\\hat{R}_n$ on the training sample. Its training loss is optimistic for $R(\\hat{f})$ because the same points were used to choose $\\hat{f}$. The next volume is entirely about not lying with that optimism. This volume builds families that are worth evaluating.""",
        derivation="""For squared error, expand $\\mathbb{E}[(Y - f(X))^2 \\mid X = x] = \\mathrm{Var}(Y \\mid X = x) + (\\mathbb{E}[Y \\mid X = x] - f(x))^2$. The variance term does not depend on $f$. The second term is zero when $f(x) = \\mathbb{E}[Y \\mid X = x]$. So the conditional mean is Bayes for squared error. Restricting to linear $f$ leaves a nonnegative approximation term $\\mathbb{E}[(\\mathbb{E}[Y \\mid X] - X^\\top \\theta)^2]$.

For 0-1 loss, $\\mathbb{E}[\\mathbf{1}\\{f(X) \\ne Y\\} \\mid X = x] = 1 - P(Y = f(x) \\mid X = x)$ in classification. Maximizing the conditional probability of the chosen label minimizes the loss. The decision is $\\arg\\max_y P(Y = y \\mid X = x)$.""",
        code="""import numpy as np

def bayes_squared(conditional_mean, x):
    return conditional_mean(x)

def hard_threshold(prob_one, threshold=0.5):
    return (np.asarray(prob_one) >= threshold).astype(np.float64)
""",
        experiment="""**Hypothesis.** When $Y = \\sin(2\\pi X) + \\varepsilon$ on $[0, 1]$, a linear family has a visible approximation floor, and a piecewise-constant family with enough bins lowers that floor until estimation error in the bins rises.

**Protocol.** $n_{\\mathrm{train}} = 80$, $n_{\\mathrm{test}} = 1000$, noise standard deviation 0.2, 30 seeds. Linear regression on $(1, x)$. Piecewise means on $b \\in \\{2, 8, 20\\}$ equal-width bins, with empty bins falling back to the global training mean.

**Metric.** Median test MSE by method, compared with an oracle that predicts $\\sin(2\\pi x)$.

**Decision rule.** Linear should lose to a moderate binning. Very fine binning should worsen relative to moderate binning at this $n$. If fine binning wins cleanly, $n$ is too large to see estimation error; reduce $n$.""",
        project="""**Deliverable.** A diagram or table that splits test MSE into an oracle Bayes term (noise plus irreducible error), an approximation gap, and the remaining estimation gap, for the linear model and one binning.

**Constraints.** The oracle uses the true $\\sin$ function, which you may use only for evaluation in this synthetic study.

**Rubric.** You can point to which knob cuts which gap. "Use a flexible model" is not an adequate sentence.""",
        beginner="Fit the linear model and the oracle and report the two test MSEs.",
        university="Complete the binning sweep.",
        research="Write the bias-variance decomposition for squared error at a fixed $x$ for the piecewise-constant estimator, assuming bins were fixed before seeing $Y$. Note where data-dependent bins would change the argument.",
        failures="""- Calling training error the approximation error.
- Changing the loss between training and the decision without saying the Bayes predictor changed.
- Using the test set to pick the number of bins and then quoting that same test number as final risk.""",
        see_also="Volume 07 for honest selection of the bin count. Volume 09 for families that are flexible and still structured.",
    ),
)

chapter(
    v,
    "02_linear_logistic",
    "Linear and logistic regression",
    entry(
        question="What do linear regression and logistic regression estimate, and how do you fit the second one?",
        card=[
            ("Linear", "$f(x) = \\theta^\\top x$. Squared loss estimates a linear approximation to the conditional mean."),
            ("Logistic", "$P(Y=1 \\mid X=x) = \\sigma(\\theta^\\top x)$ with $\\sigma(t) = 1/(1+e^{-t})$."),
            ("Log loss", "$-\\big(y \\log p + (1-y)\\log(1-p)\\big)$, the negative Bernoulli log likelihood."),
            ("Gradient", "For one row, $(\\sigma(\\theta^\\top x) - y) x$."),
            ("Fit", "No closed form in general. Use gradient descent or a Newton method on the convex loss."),
        ],
        theory="""Linear regression is the Volume 01 estimator. Logistic regression keeps the linear score and changes the loss so the fitted number is a probability. The mean of a Bernoulli is its probability, and the logistic function maps the real line to $(0, 1)$. The model class is therefore a family of conditional Bernoulli laws, not a hard classifier. You can threshold the probability afterward. The threshold is a decision rule and can be chosen for asymmetric costs. It is not part of the likelihood.

The log loss is convex in $\\theta$ because it is a convex decreasing function of the margin $y' \\theta^\\top x$ in the $\\pm 1$ coding, or by direct Hessian calculation: the Hessian is $X^\\top W X$ with $W$ a diagonal of $\\sigma(1-\\sigma) > 0$ except at degenerate probabilities. Full column rank makes it strongly convex on compact sets where probabilities stay away from 0 and 1. Separable data send $\\|\\theta\\|$ to infinity and probabilities to 0 and 1. A penalty or early stopping is then part of the estimator, not a numerical detail.

Coefficients in a logistic model are changes in log odds per unit of the feature, holding other features fixed, inside the model. They are not causal effects. Collinearity makes them unidentified in the same way as in least squares.""",
        derivation="""Let $p_i = \\sigma(\\theta^\\top x_i)$ and $\\ell(\\theta) = -\\sum_i \\big(y_i \\log p_i + (1-y_i)\\log(1-p_i)\\big)$. The derivative of $\\log \\sigma(t)$ and $\\log(1-\\sigma(t))$ yields $\\partial \\ell / \\partial \\theta = \\sum_i (p_i - y_i) x_i$. Set the step

$$\\theta \\leftarrow \\theta - \\eta X^\\top (p - y) / n$$

if you use the mean loss. The Hessian contribution of one row is $p_i(1-p_i) x_i x_i^\\top$, hence $H = X^\\top W X \\succeq 0$. Newton's method applies when $H$ is invertible. A standard $\\mathrm{L2}$ penalty $\\tfrac{\\lambda}{2}\\lVert\\theta\\rVert_2^2$ adds $\\lambda I$ to $H$ and $\\lambda \\theta$ to the gradient, and keeps separable data from diverging.""",
        code="""import numpy as np

def sigmoid(t):
    t = np.clip(t, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-t))

def fit_logistic(X, y, lam=1.0, steps=200, eta=0.5):
    theta = np.zeros(X.shape[1])
    n = X.shape[0]
    for _ in range(steps):
        p = sigmoid(X @ theta)
        grad = X.T @ (p - y) / n + lam * theta
        theta = theta - eta * grad
    return theta
""",
        experiment="""**Hypothesis.** On linearly separable 2D data, unpenalized gradient descent grows $\\|\\theta\\|$, while $\\lambda = 1$ keeps it bounded, and both can reach low training error if the step is stable.

**Protocol.** Two Gaussian clouds centered at $(\\pm 1.5, 0)$, 40 points each. Fit $\\lambda = 0$ and $\\lambda = 1$ for several step counts. Track $\\|\\theta\\|$ and training log loss.

**Metric.** Norm and mean log loss at the final step.

**Decision rule.** The unpenalized norm should be larger and still growing if you double the step count. If $\\lambda = 0$ converges in norm, the clouds overlap and are not separable; move the centers.""",
        project="""**Deliverable.** A comparison, on one synthetic classification set and one regression set, of least squares on a 0/1 target versus logistic regression. Report log loss and 0-1 error. Threshold logistic probabilities at 0.5 and, if you move the threshold, justify it by a cost.

**Constraints.** Standardize features using training statistics only. Include an intercept.

**Rubric.** You say which loss each method targets. You do not rank them by a metric neither one optimized without saying so.""",
        beginner="Fit logistic regression on the two clouds and print probabilities, not only labels.",
        university="Show the Hessian is PSD and run the penalty comparison.",
        research="Implement one Newton step with an L2 penalty and compare iterations against gradient descent at equal training log loss. Discuss the cost of forming $H$ when $d$ is large.",
        failures="""- Interpreting odds ratios from unstandardized, collinear features as importance.
- Using 0-1 loss as a training objective and expecting a useful gradient.
- Predicting hard labels inside a pipeline that the next stage treats as probabilities.""",
        see_also="Volume 07 for calibration. Volume 10 for penalties as regularization.",
    ),
)

chapter(
    v,
    "03_trees_ensembles",
    "Trees and ensembles",
    entry(
        question="What function class does a tree represent, and why does averaging trees change the error?",
        card=[
            ("Tree", "A partition of feature space into boxes, with a constant prediction in each box."),
            ("Split", "Chosen to reduce a loss: squared error, or a classification impurity such as Gini or entropy."),
            ("Depth", "A capacity knob. A deep tree can memorize. A stump is a single split."),
            ("Bagging", "Average trees fit on bootstrap samples. Variance reduction when the trees are unstable and not perfectly dependent."),
            ("Boosting", "Add trees sequentially to fit the residual or the negative gradient of the loss."),
        ],
        theory="""A regression tree is a piecewise-constant model. It can approximate jumps and interactions that a linear model misses, which is an approximation-error win. The partition is chosen using the training labels, so the family is data-dependent and the "number of leaves" understates the search. Deep trees have low bias and high variance: a small change in the sample can move a split.

Bagging attacks variance. A bootstrap sample draws $n$ rows with replacement. Trees fit on different bootstraps make different errors. Their average has variance $\\rho \\sigma^2 + (1-\\rho)\\sigma^2/M$ in the equicorrelated caricature, where $M$ is the number of trees and $\\rho$ is the correlation of their predictions. If the trees are identical, averaging does nothing. Random feature subsets at each split (a random forest) try to lower $\\rho$.

Boosting attacks bias. Each new tree fits the gradient of the current loss, a functional gradient step in function space. Shrinkage, subsampling, and early stopping are the regularizers. A boosted ensemble can overfit. The staged addition sometimes overfits slowly, which is convenient and not a guarantee.

Trees do not require monotone scaling of a single feature, because splits are threshold rules. They are awkward at linear trends unless depth is spent to staircase the line. A linear baseline remains mandatory.""",
        derivation="""For squared error, the best constant on a fixed region $R$ is the mean of $Y$ on that region, because a constant predictor's optimal value is the mean. A split of $R$ into $R_L$ and $R_R$ is scored by the drop in total squared error

$$\\sum_{i \\in R} (y_i - \\bar{y}_R)^2 - \\sum_{i \\in R_L} (y_i - \\bar{y}_L)^2 - \\sum_{i \\in R_R} (y_i - \\bar{y}_R)^2.$$

Searching over feature indices and thresholds picks the split with the largest drop. This search is part of the estimator. The Bayes predictor is still the conditional mean; the tree is a restricted, data-dependent approximation.""",
        code="""import numpy as np

def best_stump(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    best = None
    for i in range(1, len(y)):
        if x[i] == x[i - 1]:
            continue
        left, right = y[:i], y[i:]
        sse = np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2)
        if best is None or sse < best[0]:
            best = (float(sse), float(0.5 * (x[i - 1] + x[i])), float(left.mean()), float(right.mean()))
    return {"sse": best[0], "threshold": best[1], "left_mean": best[2], "right_mean": best[3]}
""",
        experiment="""**Hypothesis.** A stump recovers a step function in one dimension better than a linear fit, and loses to a linear fit on a straight line with modest noise.

**Protocol.** Case step: $X$ uniform on $[-1, 1]$, $Y = 1\\{X > 0\\} + \\varepsilon$. Case line: $Y = 2X + \\varepsilon$. Noise scale 0.2, $n = 200$, test size 1000, 20 seeds. Compare stump and least squares.

**Metric.** Median test MSE.

**Decision rule.** The ranking should flip between cases. If the stump wins on the line, the line is too noisy or the stump implementation peeks at the test set.""",
        project="""**Deliverable.** The stump, a depth-2 tree you build by splitting each side once, and a linear baseline, on both synthetic cases. If you use a library forest or booster, add it as a fourth row and record its depth and tree count in the config.

**Constraints.** The split search may use training data only.

**Rubric.** You explain the ranking using approximation versus estimation, not using a leaderboard adjective.""",
        beginner="Implement the stump and the step-function case.",
        university="Add the line case and the depth-2 tree.",
        research="Estimate the correlation $\\rho$ of predictions from two bootstrap stumps on the step data, and say whether averaging two of them should help. Check by computing the average's test MSE.",
        failures="""- Letting a tree split on an identifier column.
- Comparing a tuned forest with an untuned linear model.
- Reading a split as a causal threshold.""",
        see_also="Volume 07 for choosing depth on a validation set. Volume 10 for gradient boosting as optimization.",
    ),
)

chapter(
    v,
    "04_kernels",
    "Kernel methods",
    entry(
        question="How can a linear method operate in a large feature space without writing the features down?",
        card=[
            ("Feature map", "$\\phi(x)$, possibly high or infinite dimensional."),
            ("Kernel", "$k(x, x') = \\langle \\phi(x), \\phi(x') \\rangle$."),
            ("Gram", "The $n \\times n$ matrix $K_{ij} = k(x_i, x_j)$."),
            ("Representer", "A penalized linear fit in $\\phi$-space has a solution $\\sum_{i=1}^{n} \\alpha_i \\phi(x_i)$."),
            ("Prediction", "$f(x) = \\sum_{i=1}^{n} \\alpha_i k(x_i, x)$, after the penalty identifies $\\alpha$."),
        ],
        theory="""Kernels are inner products of features you may never form. A polynomial kernel corresponds to polynomial features. A Gaussian (RBF) kernel corresponds to a smooth infinite feature map. The computational object is the Gram matrix, which is $n \\times n$. That is attractive when $n$ is moderate and the feature dimension would have been huge, and unattractive when $n$ is millions.

The representer theorem is the reason the prediction depends on the training points only through $n$ coefficients. It applies to objectives that depend on the data through inner products plus an increasing penalty on the norm in the feature space. It does not say every learning method is a kernel method.

The kernel chooses a notion of similarity. If that notion is wrong, the method is a linear method in the wrong space. Scaling of $x$ changes distances and therefore changes an RBF kernel. Standardize with a purpose, and put the scale in the config. Kernel ridge regression is ridge regression after the feature map, written with $K$ so you solve an $n \\times n$ system instead of a $d \\times d$ system.""",
        derivation="""Ridge regression in feature space minimizes $\\tfrac{1}{2}\\lVert \\Phi \\theta - y \\rVert_2^2 + \\tfrac{\\lambda}{2}\\lVert \\theta \\rVert_2^2$, where $\\Phi$ has rows $\\phi(x_i)^\\top$. The gradient condition is $\\Phi^\\top(\\Phi \\theta - y) + \\lambda \\theta = 0$, so $\\theta = \\Phi^\\top \\alpha$ with $\\alpha = (\\Phi \\theta - y)/(-\\lambda)$ waiting to be identified. Then $\\Phi \\theta = K \\alpha$, and substitution yields $(K + \\lambda I)\\alpha = y$ when the loss is unnormalized squared error with the factor arranged so the $\\lambda$ term matches; if your loss has a $1/n$ or a $1/2$, the $\\lambda$ placement changes. Be consistent. Predictions at new $x$ are $k(x)^\\top \\alpha$ with $k(x)_i = k(x_i, x)$.

For the linear kernel $k(x, x') = x^\\top x'$, this solution matches ordinary ridge when algebra is aligned. That match is the unit test.""",
        code="""import numpy as np

def rbf(a, b, length_scale):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return np.exp(-0.5 * d2 / length_scale**2)

def kernel_ridge(K, y, lam):
    alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), y)
    return alpha
""",
        experiment="""**Hypothesis.** Kernel ridge with a linear kernel and a matching penalty agrees with ridge regression from Volume 02's SVD formula on a small full-rank problem. An RBF kernel beats the linear kernel on the sine wave from the previous volume's approximation example, at a length scale you declare in advance.

**Protocol.** $n = 60$ for the agreement test. For the sine wave, fix $\\lambda$ and the length scale before looking at test error, using a validation slice of the training set if you must choose them.

**Metric.** Parameter or prediction gap on the linear-kernel test. Test MSE on the sine wave.

**Decision rule.** Linear-kernel predictions should match to about $10^{-6}$ relative error once penalties match. If they do not, do not interpret the RBF result; fix the algebra.""",
        project="""**Deliverable.** Linear kernel versus RBF versus the stump, on the step data and the sine data, with $\\lambda$ and length scale in the config hash.

**Constraints.** The Gram matrix is built from training points only. Test points enter only $k(x_{\\mathrm{test}}, x_{\\mathrm{train}})$.

**Rubric.** You state the $n \\times n$ cost and a case where you would refuse a kernel method because $n$ is too large.""",
        beginner="Implement the linear kernel and check it against a dot product.",
        university="Complete the agreement test with ridge and the sine comparison.",
        research="Show that the RBF Gram matrix is PSD by construction in exact arithmetic, and watch its condition number as the length scale shrinks. Connect a tiny length scale to near-interpolation and a large $\\kappa$.",
        failures="""- Building the Gram matrix on train and test together and then slicing.
- Choosing the length scale by the final test number.
- Forgetting that a new categorical level has no training row and no kernel coordinate until you define one.""",
        see_also="Volume 01 for the ridge SVD shrink factor, which is the feature-space version of this penalty. Volume 10 for penalties on neural nets, a different family.",
    ),
)

chapter(
    v,
    "05_calibration",
    "Calibration and proper scoring",
    entry(
        question="When is a predicted probability a probability, and which scores punish a false one?",
        card=[
            ("Calibration", "Among examples with predicted probability $p$, the fraction of positive labels is about $p$."),
            ("Refinement", "How much the predictions vary. A constant $\\bar{y}$ can be calibrated and useless."),
            ("Proper score", "A score minimized in expectation by reporting the true conditional probability. Log loss and squared error on the probability (Brier) are proper."),
            ("Improper", "Accuracy at a threshold can prefer a miscalibrated score that ranks well."),
            ("Fix", "Fit the probability model, then recalibrate on data that did not choose the model class."),
        ],
        theory="""Classification pipelines often emit scores that are not probabilities: margins from a linear model, leaf fractions from a tree grown for Gini, or logits from a network trained long past the point where the probabilities sharpened into 0 and 1. Decisions that need expected cost need probabilities. Ranking the positive class above the negative class needs only an order. The metric has to match the need.

A proper scoring rule makes the truthful conditional probability the optimal report. Log loss is harsh in the tails: predicting $10^{-6}$ when the event happens costs about $13.8$ nats, while predicting $0.5$ costs $\\log 2$. That harshness is correct when a confident mistake is expensive, and it is unstable when your model is wrongly confident. The Brier score $(p - y)^2$ is proper and milder.

Calibration diagrams bin the predictions. Inside a bin, the average prediction should match the average label. A model can be calibrated and have no discrimination, or discriminate well and be miscalibrated. Report both: a calibration plot and a proper score, plus a ranking metric if ranking is the decision.""",
        derivation="""For binary $Y$ and a reported probability $q$, expected log loss given $X=x$ is

$$-p \\log q - (1-p)\\log(1-q), \\quad p = P(Y=1 \\mid X=x).$$

The derivative in $q$ is $-p/q + (1-p)/(1-q)$, zero at $q = p$, and the second derivative is positive. So log loss is proper. The Brier score $p(1-q)^2 + (1-p)q^2$ has derivative zero at $q = p$ as well.

If you pass a score $s$ through a fitted map $q = \\sigma(a + b s)$ using a validation set, you are fitting a one-dimensional logistic regression. That map can repair monotone miscalibration. It cannot create information that was not in $s$.""",
        code="""import numpy as np

def brier(p, y):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((p - y) ** 2))

def reliability_bins(p, y, n_bins=5):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if np.any(mask):
            rows.append((float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())))
    return rows
""",
        experiment="""**Hypothesis.** If the true probability is $p(x) = \\sigma(2x)$ and you report $q = \\sigma(8x)$, the reports are overconfident: reliability bins pull toward the corners relative to the diagonal, and log loss is worse than for $q = p$.

**Protocol.** Draw $x \\sim \\mathcal{N}(0, 1)$, $y \\sim \\mathrm{Bernoulli}(p(x))$, $n = 4000$. Compare $q = p$ and $q = \\sigma(8x)$ by log loss, Brier score, and a 5-bin reliability table. Also compare a constant $q = \\bar{y}$.

**Metric.** The two scores and the bin gaps $\\bar{q} - \\bar{y}$.

**Decision rule.** The true $p$ must win on log loss. The overconfident $q$ should show bin average predictions more extreme than bin frequencies. The constant should be better calibrated than the overconfident model and worse as a ranking of risk. If you only report accuracy at 0.5, you may miss this; that is part of the lesson.""",
        project="""**Deliverable.** Take the logistic regression from earlier in the volume, emit probabilities, and build a reliability table on a validation set. Fit a temperature or an affine logistic map on that validation set only if the table is bent, and quote a final test score once.

**Constraints.** The test set does not choose the map.

**Rubric.** You state whether the downstream decision needs probabilities or only a ranking, and you include the matching metric.""",
        beginner="Compute Brier score for a constant 0.5 and for a constant equal to the base rate, on a sample with base rate 0.1.",
        university="Run the overconfidence experiment.",
        research="Show on paper that accuracy is not a proper scoring rule: give two reported probabilities where the wrong one wins on expected accuracy at threshold 0.5. A simple discrete $p$ is enough.",
        failures="""- Temperature-scaling on the test set.
- Treating a high accuracy on an imbalanced problem as calibration.
- Clipping probabilities to exactly 0 or 1 before log loss, which makes the score infinite at the first mistake.""",
        see_also="Volume 07 for threshold choice under costs. Volume 12 if a token probability is later treated as a calibrated chance; it usually is not.",
    ),
)

chapter(
    v,
    "06_bakeoff_project",
    "Project: a baseline bake-off",
    entry(
        question="How do you compare a small set of supervised methods without a leaderboard fantasy?",
        card=[
            ("Frozen split", "Train, validation, and test, or a cross-validation plan, written before fitting."),
            ("Baselines", "Constant predictor, linear or logistic model, and one nonlinear method."),
            ("Knobs", "Each knob is in the config. The validation set chooses knobs. The test set is used once."),
            ("Table", "Primary metric with uncertainty across seeds or folds, plus the sample size."),
            ("Claim", "One sentence: which method won by enough to matter, or that none did."),
        ],
        theory="""A bake-off is an estimation procedure for the identity of the best method in a finite list. The list is part of the procedure. Searching a thousand hidden variants and publishing the winner is a different, more optimistic procedure. This project keeps the list short and written down: a constant, a linear model, a stump or tree, and optionally kernel ridge or logistic regression.

The validation set is for knobs such as tree depth, $\\lambda$, and length scale. The test set answers one question about the chosen procedures. If you need a second test look, you no longer have that test set. You may say so and call the number exploratory.

Practical significance is the effect size from Volume 05, not the smallest difference a large test set can detect. If the linear model is within that margin of the flexible model, prefer the linear model for interpretation and speed unless you have another reason. Prefer is a decision. The table does not make it for you.""",
        derivation="""Let methods $m = 1, \\ldots, M$ be chosen from a fixed list by validation risk, and let $m^\\star$ be the one you then evaluate on the test set. The test risk of $m^\\star$ is an unbiased estimate of the risk of that frozen method, not of the selection rule's future performance on a new draw of the whole pipeline, unless the test set was never used in any choice. If the validation choice is unstable across seeds, the test number is about whichever method won this time. Report the selection frequency. A method that wins validation on 51 percent of seeds is not a stable champion.""",
        code="""import numpy as np

def pick_by_validation(val_scores):
    names = list(val_scores)
    winner = min(names, key=lambda name: val_scores[name])
    return winner, {name: float(val_scores[name]) for name in names}
""",
        experiment="""**Hypothesis.** On the sine-wave data, a flexible method wins validation more often than linear regression. On the straight-line data, linear regression wins or ties within a margin you set to the noise level.

**Protocol.** Two datasets from earlier chapters, three methods, 20 seeds, a validation slice of 30 percent of the training data, a fixed test set per seed drawn fresh.

**Metric.** Selection frequency and median test MSE of the selected method versus always-linear.

**Decision rule.** If selection frequency is split and test gaps are smaller than your practical margin, the claim is "no meaningful winner," not the name of the method that won eleven times.""",
        project="""**Deliverable.** One experiment folder with the method list, the practical margin, the validation rule, the test table, and a decision. Include the constant baseline in every table.

**Constraints.** No method enters the list after you see the test table.

**Rubric.** The decision would still look reasonable if the seed-level winner flipped. You name what you would need (more data, a different feature, a different loss) if it does not.""",
        beginner="Compare constant versus linear on the line data only, with the test set used once.",
        university="Run the full three-method, two-dataset protocol.",
        research="Add a paired comparison of test losses between the two best methods, using the Volume 05 interval on per-point loss differences. State whether the mean gap's interval excludes your practical margin.",
        failures="""- A bake-off whose only nonlinear method was tuned on the test set.
- Dropping the constant baseline because it "is not a real model."
- Declaring a winner on a gap smaller than the seed-to-seed scatter.""",
        see_also="Volume 07 is this protocol made systematic. Volume 10 adds neural nets to the list only after the baselines exist.",
    ),
)
