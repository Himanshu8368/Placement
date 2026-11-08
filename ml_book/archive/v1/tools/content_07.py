"""Volumes 07–10: evaluation, unsupervised learning, networks, training."""

from booklib import chapter, entry, volume

v = volume(
    7,
    "Evaluation and model selection",
    "University",
    "A reader who can run the bake-off protocol and compute an interval.",
    "Which number is allowed to choose a model, and which number is allowed to be published?",
)

chapter(
    v,
    "01_splits",
    "Train, validation, test, and leakage",
    entry(
        question="What is each split for, and which operations leak across it?",
        card=[
            ("Train", "Fits parameters of a frozen procedure."),
            ("Validation", "Chooses knobs that are not fit inside the training objective: depth, $\\lambda$, step count, thresholds."),
            ("Test", "Estimates risk of the procedure the validation set already chose. One look."),
            ("Leakage", "Any use of a split's labels, or of its inputs when the statistic will be applied outside that split, that the deployment distribution will not provide in the same way."),
            ("Time", "If the world moves forward, the test set is the future, not a random subset of the past."),
        ],
        theory="""A split is a firewall between fitting and bragging. Parameters see the training labels. Knobs see the validation labels. The test labels are a measurement instrument. Using an instrument to build the object changes what the instrument measures.

Leakage is not only a column named `target`. It is a scaler fit on all rows, a feature-selection pass on all rows, an entity that appears on both sides, a moving-average feature that used tomorrow, or a test-set threshold chosen to maximize test accuracy. The deployment system will not have tomorrow, and it will not have the test labels.

When data are a time series, a random split trains on the future to predict the past. Use a cutoff. When rows share an entity, split on the entity. These rules came from Volume 03. Here they are preconditions for every metric in the volume.""",
        derivation="""Let $\\hat{f}_{S}$ be the predictor obtained when the procedure is allowed to see the set of examples $S$. If the test indices $T$ intersect the information used to build $\\hat{f}$, the quantity $m^{-1}\\sum_{i \\in T} \\ell(\\hat{f}(x_i), y_i)$ is not an unbiased estimate of the risk of a predictor built without those examples. In particular, choosing $k = \\arg\\min_k \\widehat{\\mathrm{err}}_T(k)$ and then reporting $\\widehat{\\mathrm{err}}_T(k)$ estimates the minimum of several dependent test errors, which is biased low for the risk of the selected $k$. The bias shrinks only if you pay for selection with a fresh sample or a nested procedure.""",
        code="""import numpy as np

def random_split(n, fractions, rng):
    if abs(sum(fractions) - 1) > 1e-8:
        raise ValueError("fractions must sum to 1")
    idx = rng.permutation(n)
    cuts = (np.cumsum(fractions) * n).astype(int)
    a, b = cuts[0], cuts[1]
    return idx[:a], idx[a:b], idx[b:]
""",
        experiment="""**Hypothesis.** Choosing the polynomial degree on the test set and reporting that same test error is optimistic relative to choosing the degree on a validation set and reporting a fresh test error.

**Protocol.** $Y = X + 0.1 X^3 + \\varepsilon$, $X$ uniform on $[-2, 2]$, noise scale 0.5, $n = 80$. Degrees $1, 3, 5, 9, 15$. Protocol A: pick the degree by test MSE. Protocol B: a 50/25/25 train-validation-test split, pick by validation, report test. 50 seeds.

**Metric.** Median reported test MSE of the selected degree, and the median degree chosen.

**Decision rule.** Protocol A's reported error should be lower than Protocol B's fresh test error. If it is not, the test sets are too small to show selection bias; increase the degree list or the number of seeds before doubting the principle.""",
        project="""**Deliverable.** A split helper that supports random, entity, and time-cutoff splits, plus the polynomial selection experiment.

**Constraints.** The helper returns indices only. Fitting happens outside it, so the firewall is visible.

**Rubric.** You can explain which protocol's number you would put in a report, and you refuse Protocol A's number as a final claim.""",
        beginner="Implement the three-way split and print sizes.",
        university="Run the selection-bias experiment.",
        research="Describe nested cross-validation in a paragraph and say what computational cost it pays to estimate the risk of the selection rule rather than the risk of one chosen model.",
        failures="""- Fitting a vectorizer or a scaler on the full corpus before splitting text data.
- Using a time series' shuffled rows.
- "Just one peek" at the test set to confirm a knob.""",
        see_also="Volume 03 for entity contracts. Volume 15 for the same firewall in a pipeline.",
    ),
)

chapter(
    v,
    "02_metrics",
    "Metrics by decision",
    entry(
        question="Which metric matches the decision, the base rate, and the loss you trained?",
        card=[
            ("Regression", "MSE for the mean, MAE for the median, and a custom weighted loss if costs are asymmetric."),
            ("Classification", "Log loss or Brier for probabilities. Accuracy only when classes are balanced and errors are symmetric."),
            ("Imbalance", "Precision, recall, and false-positive rate at a stated threshold. A curve if the threshold is not yet chosen."),
            ("Ranking", "A pairwise ranking metric if the decision is an order, not a label."),
            ("Cost", "If missing a positive costs $c$ times a false alarm, the threshold solves that cost, and accuracy does not."),
        ],
        theory="""A metric is a functional of predictions and labels. It is the right functional when it ranks procedures the way the decision would rank them. Accuracy on a 99 percent negative problem ranks the constant predictor above a detector that spends a few false alarms to catch rare positives. If the decision needs those positives, accuracy is the wrong functional even though the arithmetic is correct.

Training loss and reporting metric can differ, but the gap must be intentional. Training with log loss and reporting a thresholded cost is coherent if you keep probabilities and then choose the threshold for the cost. Training with accuracy's non-differentiable step and hoping is not a method. Reporting five metrics and letting the best one be the story is a multiple-comparison problem from Volume 05.

Curves are for unfixed thresholds. Once a product has a capacity constraint, such as "at most 10 false alarms per hour," the operating point is fixed and a single point on the curve is the metric. Quote the constraint.""",
        derivation="""Let a positive prediction cost $c_{FP}$ if the label is negative and a missed positive cost $c_{FN}$. The expected cost of predicting positive when $P(Y=1\\mid x) = p$ is $(1-p) c_{FP}$. The expected cost of predicting negative is $p\\, c_{FN}$. Predict positive when $p / (1-p) > c_{FP}/c_{FN}$, i.e. when $p > c_{FP}/(c_{FP}+c_{FN})$. The Bayes threshold is not 0.5 unless the costs are equal. Accuracy assumes they are equal and that you care about the fraction of mistakes, not their weighted cost.""",
        code="""import numpy as np

def binary_counts(pred, y):
    pred = np.asarray(pred).astype(bool)
    y = np.asarray(y).astype(bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

def precision_recall(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"precision": prec, "recall": rec}
""",
        experiment="""**Hypothesis.** On data with 5 percent positives, a constant negative predictor has accuracy 0.95 and recall 0. A predictor that flags the top 5 percent of a slightly informative score trades accuracy for recall.

**Protocol.** $n = 4000$, score equal to the latent log-odds plus noise, threshold the score so that 5 percent of rows are flagged. Compare with predicting all negative.

**Metric.** Accuracy, precision, recall.

**Decision rule.** The constant must win on accuracy and lose on recall. A write-up that ranks by accuracy alone fails the project even if the code is right.""",
        project="""**Deliverable.** A metric card for a problem you define: the decision, the cost or constraint, the primary metric, two metrics you will not use as primary, and the synthetic imbalance experiment as a worked warning.

**Constraints.** The primary metric is one number, or one number at a predeclared operating point.

**Rubric.** A teammate can tell how the constant predictor scores on your primary metric.""",
        beginner="Compute accuracy and recall for the all-negative predictor on a 5 percent base rate.",
        university="Derive the cost threshold and apply it to a logistic model's probabilities.",
        research="Define a precision-recall curve as a set of operating points and say why the area under it is not the quantity you ship when the operating point is constrained by a review budget.",
        failures="""- Macro-averaging away a rare class you actually care about.
- Comparing models at different recall levels and calling the higher precision a better model.
- Using MSE on raw prices when the decision cares about relative error, without a transformation you can justify.""",
        see_also="Volume 06 for proper scores. Volume 16 for metrics that encode a harm you did not intend.",
    ),
)

chapter(
    v,
    "03_error_analysis",
    "Error analysis",
    entry(
        question="How do you turn a residual into a next action?",
        card=[
            ("Slice", "A subset defined by a feature available at decision time, or by a tag you can apply without the prediction."),
            ("Residual", "For regression, $y - \\hat{y}$. For classification, a false positive or false negative."),
            ("Mass", "The slice's share of total loss, not only its average loss."),
            ("Action", "More data, a new feature, a new family, a bug fix, or a decision to stop."),
            ("Not an action", "\"The model struggles on hard examples\" with no slice and no mass."),
        ],
        theory="""Average error hides the slice that dominates the loss. A slice with terrible conditional error and tiny mass may not be the best place to spend the next week. A slice with moderate error and half the mass is. Compute both the conditional metric and the contribution $n_{\\mathrm{slice}}/n$ times that metric, in the same units as the global metric.

Slices must be things you could act on. Slicing by the residual itself ("the points we got wrong") does not tell you a feature to add. Slicing by a region of $X$, by a data source, or by a time period does. Looking at individual losses is still useful as a bug detector: a single row with a corrupted label can dominate MSE. That is a data-contract failure, and the action is to fix the row or clip the loss, then say which you did.

Error analysis is how approximation error becomes a plan. If residuals against a linear fit are shaped like a parabola, the next family should be able to represent a parabola. If residuals are a few huge spikes, the next change is robustness or data cleaning, not a bigger network.""",
        derivation="""Let the test loss be the average of pointwise losses $\\ell_i$. For a partition of the test set into slices $S$,

$$\\frac{1}{n}\\sum_i \\ell_i = \\sum_S \\frac{n_S}{n} \\left(\\frac{1}{n_S}\\sum_{i \\in S} \\ell_i\\right).$$

The term in parentheses is the slice metric. The weight $n_S/n$ turns it into mass. Ranking slices by the product is ranking them by contribution to the number you report. Ranking by the slice metric alone ranks by conditional difficulty and can send you to an empty corner of the space.""",
        code="""import numpy as np

def slice_mass(losses, groups):
    losses = np.asarray(losses, dtype=np.float64)
    groups = np.asarray(groups)
    rows = []
    for g in np.unique(groups):
        mask = groups == g
        rows.append(
            {
                "slice": g.item() if hasattr(g, "item") else g,
                "n": int(mask.sum()),
                "mean_loss": float(losses[mask].mean()),
                "mass": float(losses[mask].sum() / losses.size),
            }
        )
    return sorted(rows, key=lambda r: -r["mass"])
""",
        experiment="""**Hypothesis.** On data with two groups, a model that is slightly worse on a large group contributes more total MSE than a model that is terrible on a tiny group.

**Protocol.** Group A, 90 percent of rows, residual variance 1. Group B, 10 percent, residual variance 9. Compute slice mass for a predictor that predicts the mean everywhere, so the loss is the variance in each group.

**Metric.** Mass of A versus mass of B.

**Decision rule.** Group A should carry more total mass despite the lower variance. An analysis that sends you to "fix B only" because its mean loss is higher needs the mass column before it is allowed to choose the work.""",
        project="""**Deliverable.** An error table for the linear fit on the sine-wave data, sliced by the sign of $x$ and by $|x|$ bins, plus one concrete next action tied to the largest mass.

**Constraints.** At least one slice must be checked for a single corrupted point: multiply one training label by 50, refit, and show the change in MSE.

**Rubric.** The action is something you can implement in a later chapter, not a mood.""",
        beginner="Compute the two-group mass table.",
        university="Add the corruption check and the sine-wave slices.",
        research="Discuss Simpson's paradox in one paragraph: a model can be worse on every slice of a training attribute and better overall if the slice mix differs. Say how a shifted test mix would change which mass matters.",
        failures="""- Slicing only by the prediction and calling it a population subgroup.
- Chasing the noisiest slice regardless of mass.
- Fixing a label in the test set and reporting the improved test number as a model improvement.""",
        see_also="Volume 16 for slices that correspond to people rather than to regions of $x$. Volume 06 for choosing the next family.",
    ),
)

chapter(
    v,
    "04_cross_validation",
    "Cross-validation",
    entry(
        question="How do you reuse a small sample for selection without pretending the test set grew?",
        card=[
            ("K-fold", "Partition the training data into $K$ blocks. Fit on $K-1$, score on the held block, rotate."),
            ("Mean", "The average of the $K$ scores estimates the risk of the training procedure on a sample of size about $n(1-1/K)$."),
            ("Selection", "Choose the knob with the best average validation score. Refit on all training data if you will deploy."),
            ("Nested", "An outer loop estimates the risk of the whole selection procedure. The inner loop selects."),
            ("Dependence", "Folds that share rows' entities or times are not $K$ independent tests."),
        ],
        theory="""A single train-validation split spends a chunk of data on knobs and never trains on it. K-fold cross-validation lets every training row serve as a validation row once. The price is $K$ fits, and the score is not the risk of the model fit on all $n$ rows. It is closer to the risk of a fit on $n(K-1)/K$ rows. When $K$ is 5 or 10 that gap is often small relative to other errors. When $K = n$, leave-one-out trains on almost the full sample and can be expensive and high-variance as an estimate of the risk.

Cross-validation inside the training set does not create a test set. If you select on cross-validated scores and also report those same scores as the final number, you have used the data twice. The optimistic bias is usually smaller than test-set shopping, and it is still bias. A small locked test set, or a nested loop, is the honest final number.

Folds must respect the split contract. A grouped K-fold keeps an entity inside one fold. A time series uses a forward chain: train on the past, validate on the next block, and never the reverse.""",
        derivation="""Let $\\widehat{\\mathrm{CV}}(\\lambda) = K^{-1}\\sum_{k=1}^{K} \\widehat{R}^{(k)}(\\hat{f}^{(-k)}_\\lambda)$, where $\\hat{f}^{(-k)}_\\lambda$ is trained without fold $k$. For a fixed $\\lambda$, each $\\widehat{R}^{(k)}$ is an average of losses on data not used to fit that function, so it is unbiased for the risk of the procedure "train on a sample of that size," up to the dependence among folds that share most of their training rows. The minimizer $\\hat{\\lambda} = \\arg\\min_\\lambda \\widehat{\\mathrm{CV}}(\\lambda)$ is a random knob. $\\widehat{\\mathrm{CV}}(\\hat{\\lambda})$ is a minimum and is biased low as an estimate of $R(\\hat{f}_{\\hat{\\lambda}})$. Nested CV estimates that selected risk by putting the whole minimization inside an outer fold.""",
        code="""import numpy as np

def kfold_indices(n, k, rng):
    if k < 2 or k > n:
        raise ValueError("k must be between 2 and n")
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    splits = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train, val))
    return splits
""",
        experiment="""**Hypothesis.** For ridge $\\lambda$ on a small linear problem, 5-fold CV selects a $\\lambda$ whose test MSE is close to the best $\\lambda$ on a locked test set, and the CV score of the winner is slightly optimistic relative to that test MSE.

**Protocol.** $n = 100$, $d = 10$, sparse linear signal, noise 1, $\\lambda$ on a log grid. Locked test size 500. 30 seeds.

**Metric.** Test MSE of the CV-chosen $\\lambda$ minus test MSE of the oracle $\\lambda$ chosen on the test set (the oracle is an evaluation-only ceiling). Also CV score minus true test MSE of the chosen model.

**Decision rule.** The oracle gap should be small but nonnegative on average. A large negative gap means information leaked into the folds.""",
        project="""**Deliverable.** Grouped or ordinary K-fold selection of one knob, a refit on all training data using the chosen knob, and a locked test evaluation. Document whether you report the CV score, the locked test score, or both, and which one is final.

**Constraints.** The locked test set is not a fold.

**Rubric.** You can say what sample size the CV score refers to, and you do not call the CV score of the winner an unbiased test error.""",
        beginner="Print the five pairs of index arrays and check they cover each index once as validation.",
        university="Run the ridge selection experiment.",
        research="Implement one outer fold of nested CV and compare its estimate with the locked test estimate. Discuss variance: one outer fold is honest and noisy.",
        failures="""- Tuning on CV and shipping the CV score of the best fold rather than the average.
- Random folds on a time series.
- Refitting on train+test after selection because "the test set should not be wasted." """,
        see_also="Volume 05 for the bias of a minimum. Volume 10 for early stopping, which is this chapter applied to a step count.",
    ),
)

chapter(
    v,
    "05_comparisons",
    "Comparing models under uncertainty",
    entry(
        question="When is a measured gap between two models large enough to believe?",
        card=[
            ("Paired", "Compare losses on the same test points. The unit of the interval is the per-point loss difference."),
            ("Seeds", "If both models are refit, the relevant scatter includes seeds, not only the test sample."),
            ("Margin", "A practical margin, declared first, is the smallest gap that would change the decision."),
            ("Interval", "If the interval for the mean gap contains the margin and zero, you have not settled the question."),
            ("Ranking noise", "A leaderboard sort of single runs is not an estimator with a known error bar."),
        ],
        theory="""Two test errors differ by some amount on every run. The question is whether the procedure difference is distinguishable from noise and from your practical margin. Pairing on test points removes a shared component: easy points are easy for both models. The variance of the difference is what matters, and it can be much smaller than the variance of either error.

A significant difference can be smaller than any effect you would ship. A non-significant difference can hide an effect you are not powered to see. Both sentences are Volume 05. Here they apply to model gaps. Report the interval for the mean paired difference and compare it with the margin.

Multiple models inflate the chance that the luckiest gap looks real. If you compared a fixed pair, say so. If you picked the pair after seeing five models, the interval for that post-selected gap does not have its nominal coverage. Predeclare the comparison or correct for the search. The bake-off project already asked you to keep the list short for this reason.""",
        derivation="""Let $D_i = \\ell(f(x_i), y_i) - \\ell(g(x_i), y_i)$ on a test set that neither model was chosen with. The mean $\\bar{D}$ estimates $R(f) - R(g)$ for those two frozen predictors. A normal interval is $\\bar{D} \\pm z s_D/\\sqrt{m}$. If $f$ and $g$ themselves depend on the test set, this derivation does not apply. If $f$ and $g$ are random through the training seed and you only draw one seed, the interval omits that variance. A conservative engineering report repeats the paired test across seeds and shows the seed-wise intervals, or treats the seed-level mean gap as the observation.""",
        code="""import numpy as np

def paired_gap(loss_a, loss_b):
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    z = 1.959963984540054
    return {"gap": mean, "low": mean - z * se, "high": mean + z * se}
""",
        experiment="""**Hypothesis.** Two identical predictors produce a paired interval that covers zero. A predictor with an added bias of 0.3 in its residual produces an interval that eventually excludes zero as the test size grows, and may not at a small test size.

**Protocol.** Shared noise, $m \\in \\{30, 300\\}$, 200 replicates of coverage or exclusion. Practical margin 0.1 in MSE units if you compare squared residuals of a constant shift; set the simulation so you can compute the true gap.

**Metric.** Coverage of the true gap, and fraction of intervals excluding 0.

**Decision rule.** Identical predictors must cover 0 near 95 percent. The shifted predictor at $m = 30$ should often include 0 if the per-point variance is order 1. Do not call that a proof of equality.""",
        project="""**Deliverable.** A paired comparison between the linear model and one flexible model on a locked test set from Volume 06, with a practical margin and a written decision: switch, stay, or collect more test data.

**Constraints.** The models are frozen before the test losses are computed.

**Rubric.** The decision quotes the interval. "The flexible model won" is not sufficient if the interval contains the margin.""",
        beginner="Run `paired_gap` on two identical loss vectors and on a shifted pair.",
        university="Complete the project comparison from a real bake-off folder.",
        research="Explain how a Wilcoxon signed-rank test on the $D_i$ changes the assumptions relative to the normal interval. Say when the mean difference remains the quantity you care about even if you use a rank test as a robustness check.",
        failures="""- Comparing training losses as if they were paired test losses.
- Declaring a winner from one seed when seed-to-seed gaps change sign.
- Using an unpaired test when the models were scored on the same points and a paired test was available.""",
        see_also="Volume 05 for power. Volume 18 for how a paper should display this interval.",
    ),
)

chapter(
    v,
    "06_model_card_project",
    "Project: a model card and an error taxonomy",
    entry(
        question="What does a stranger need to know before reusing your fitted procedure?",
        card=[
            ("Procedure", "Data contract, split, family, knobs, seed policy, and code version."),
            ("Metric", "Primary metric, baseline, interval or seed spread, and test size."),
            ("Slices", "Where the error mass sits."),
            ("Limits", "What the procedure is not for, including populations you did not sample."),
            ("Taxonomy", "A short list of error types with counts, not a gallery of anecdotes alone."),
        ],
        theory="""A model card is the reference page for a procedure you might reuse. It is not a press release. The card should let someone decide not to use the model. That requires the limits and the baseline, not only the best number.

An error taxonomy groups failures by cause you can distinguish: wrong label, missing feature, out-of-range input, threshold cost, or a family that cannot represent the slice. Counts turn the taxonomy into data. A card with twenty cherry-picked mistakes and no denominators is a demo.

This project closes the evaluation volume. Later volumes that train larger models still owe a card. The card gets longer. It does not become optional because the model became harder to train.""",
        derivation="""Treat the card as a specification of the estimator $\\hat{f}$ and of the measurement of $R(\\hat{f})$. If a field is missing, a later user will fill it with an assumption. The dangerous assumptions are a random split when you used entities, a balanced accuracy when the base rate is 1 percent, and an in-domain test set when they will deploy out of domain. Writing "not evaluated under shift" is a result. It blocks a false claim. Volume 16 is where shift is measured rather than only confessed.""",
        code="""def model_card(fields):
    required = [
        "name",
        "target",
        "split",
        "baseline",
        "primary_metric",
        "test_n",
        "limits",
    ]
    missing = [key for key in required if key not in fields or fields[key] in ("", None)]
    if missing:
        raise ValueError(f"model card missing {missing}")
    lines = [f"# {fields['name']}", ""]
    for key in required:
        lines.append(f"- {key}: {fields[key]}")
    return "\\n".join(lines)
""",
        experiment="""**Hypothesis.** A card that omits `limits` is rejected by the helper, and a completed card for the constant predictor on imbalanced data states a limit that accuracy would have hidden.

**Protocol.** Pass a dictionary without `limits` and confirm the exception. Pass a complete card whose primary metric is recall at a fixed false-positive budget, with the limit "not evaluated on a new entity."

**Metric.** Whether the incomplete card raises, and whether the complete card contains the baseline accuracy or the reason you refused it.

**Decision rule.** Both checks pass before you write a card for a flexible model. The constant predictor is the control card.""",
        project="""**Deliverable.** A one-page card for the winner of your bake-off, an error taxonomy with at least three labeled types and counts on the test set, and the slice-mass table from earlier in this volume.

**Constraints.** The card names a population you did not test.

**Rubric.** A reader can reimplement the measurement and can also refuse the model for a stated reason.""",
        beginner="Write the constant-predictor card only.",
        university="Write the bake-off card and the taxonomy.",
        research="Add a known-failure-under-shift line with a concrete shift you can simulate now, such as a change in the noise scale or the base rate, and report the metric under that shift without retuning.",
        failures="""- A card that reports the training loss as performance.
- A taxonomy whose categories overlap so thoroughly that counts cannot be added.
- Limits written as "use responsibly" with no population and no metric.""",
        see_also="Volume 16 for harm and shift sections that this card should grow. Volume 15 for the owner of the deployed procedure.",
    ),
)

v = volume(
    8,
    "Unsupervised learning",
    "University",
    "A reader who can fit a supervised baseline and state a data contract.",
    "What can you estimate when there is no target, and how do you know you estimated it?",
)

chapter(
    v,
    "01_clustering",
    "Clustering",
    entry(
        question="What does a clustering method minimize, and what does a cluster not mean?",
        card=[
            ("K-means", "Partition points into $K$ sets and place a center at each set's mean, minimizing total squared distance to the assigned center."),
            ("Alternation", "Assign each point to the nearest center, then recompute means. Each step drops the objective. The result can be a poor local minimum."),
            ("Shape", "Squared Euclidean K-means prefers blobs of similar variance. It splits long ellipses and chains."),
            ("K", "A knob. Likelihood-style scores on the same points used to fit are optimistic. Stability across seeds is a partial check."),
            ("Not", "A cluster is not a type of person, a market segment, or a cause, unless a later measurement says so."),
        ],
        theory="""Clustering assigns labels the algorithm invented. Those labels can be useful features, compression codes, or hypotheses. They are not supervised targets. Evaluating them with a supervised accuracy requires a human labeling you claim to match, and then you are in a supervised problem with a fuzzy label.

K-means is a vector-quantization method with squared error. Standardize features or the coordinate with the largest variance owns the distance. A random initialization can land in a bad partition; several restarts and the best objective are part of the estimator. The objective falling is not evidence that $K$ is correct. A larger $K$ always can achieve a lower or equal training objective.

Stability is a sanity check: if two runs with different seeds disagree on almost every point, the partition is not a feature of the data under this method. Agreement does not make the clusters real in a scientific sense. It makes the algorithm repeatable.""",
        derivation="""Given centers, the optimal assignment of a point is the nearest center, because the objective is a sum of independent squared distances. Given assignments, the optimal center of a cluster is the mean, because squared error around a constant is minimized by the mean. The alternation therefore cannot increase the objective. It reaches a fixed point in finitely many assignments when ties are broken deterministically, but not necessarily a global minimizer. A standard bad case is two centers that fall into one blob and leave another blob empty or split.""",
        code="""import numpy as np

def kmeans(X, k, rng, n_init=5, steps=30):
    X = np.asarray(X, dtype=np.float64)
    best = None
    for _ in range(n_init):
        centers = X[rng.choice(X.shape[0], size=k, replace=False)].copy()
        labels = np.zeros(X.shape[0], dtype=int)
        for _step in range(steps):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
            labels = np.argmin(d2, axis=1)
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    centers[j] = X[mask].mean(axis=0)
        inertia = float(((X - centers[labels]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, centers, labels)
    return {"inertia": best[0], "centers": best[1], "labels": best[2]}
""",
        experiment="""**Hypothesis.** On two round Gaussian blobs, K-means with $K=2$ and several restarts recovers the generating groups up to label permutation. On one elongated Gaussian, $K=2$ cuts the ellipse rather than refusing to split it.

**Protocol.** Blobs centered at $(\\pm 3, 0)$ with identity covariance, 200 points each. Ellipse: covariance $\\mathrm{diag}(8, 0.2)$, 400 points. Standardize only within a run you declare. Compare a few seeds.

**Metric.** For the blobs, accuracy after the best label flip. For the ellipse, the fraction of points on each side of the cut, and a sentence.

**Decision rule.** Blob recovery should be near perfect. If the ellipse run returns two groups of very different radius in the long direction, you have illustrated the squared-error bias. Do not "fix" it by switching $K$ after seeing a desired story.""",
        project="""**Deliverable.** The two simulations, a note on scaling, and a clustering of a dataset only if you also state what decision the cluster label would feed. If you have no decision, stop at the simulations.

**Constraints.** Report inertia and a stability check across three seeds.

**Rubric.** You explicitly say "these clusters are a quantization of squared distance," not "we discovered two kinds of users," unless you brought an external measurement.""",
        beginner="Run K-means on the two blobs.",
        university="Add the ellipse counterexample and the scaling note.",
        research="Compare K-means with a Gaussian mixture at $K=2$ on the ellipse. Say which likelihood each method is closer to, and why equal spherical covariance is the K-means-like mixture.",
        failures="""- Choosing $K$ by the inertia on the fitting data.
- Clustering on the target column.
- Interpreting a cluster of missingness codes as a behavior.""",
        see_also="The mixture chapter later in this volume. Volume 07 if a human label arrives and the problem becomes classification.",
    ),
)

chapter(
    v,
    "02_dimensionality_reduction",
    "Dimensionality reduction",
    entry(
        question="What information does a low-dimensional projection keep, and what does it destroy?",
        card=[
            ("PCA", "Orthogonal directions of maximal variance. The first $r$ principal components minimize squared reconstruction error among rank-$r$ linear maps of that form."),
            ("SVD", "After centering, the principal directions are the right singular vectors of the data matrix."),
            ("Variance explained", "The sum of the kept eigenvalues divided by the sum of all eigenvalues. A reconstruction metric, not a prediction metric."),
            ("Not invariant", "Scaling a feature changes the components. Centering is part of the method."),
            ("Supervision", "PCA ignores $Y$. A direction of small variance can be the only predictive one."),
        ],
        theory="""PCA compresses $x$ into scores $z = V_r^\\top (x - \\bar{x})$ and reconstructs $\\tilde{x} = \\bar{x} + V_r z$. The discarded coordinates are the ones with least variance, not the ones with least predictive value. Use PCA to denoise a measurement you believe is approximately low rank, to visualize, or to build a baseline representation. Check a supervised model on the original features before you trust the compressed ones.

The number $r$ is a knob. Reconstruction error on the fitting set falls as $r$ grows. A validation reconstruction error, or a downstream validation risk, is the selection criterion. A scree plot is a picture, not a test.

Nonlinear embeddings can preserve local neighborhoods that PCA tears. They also hide the map. If you cannot transform a new point, you have a picture of the training set, not a feature function. A reference implementation should define the map on new $x$.""",
        derivation="""Let the centered data matrix be $X$. Among rank-$r$ matrices, the minimizer of $\\|X - M\\|_F$ is the truncated SVD $U_r \\Sigma_r V_r^\\top$. The columns of $V_r$ are principal axes. The Frobenius identity $\\|X\\|_F^2 = \\sum_j \\sigma_j^2$ says the discarded energy is the sum of the squared singular values you drop. Because centering was done first, this energy is about variance, not about the mean. A new point uses the same $\\bar{x}$ and $V_r$ learned on the training matrix. Recomputing PCA on train plus test leaks.""",
        code="""import numpy as np

def pca_fit(X, r):
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    xc = X - mean
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    r = min(r, vt.shape[0])
    return {"mean": mean, "components": vt[:r], "singular_values": s}

def pca_transform(model, X):
    return (np.asarray(X, dtype=np.float64) - model["mean"]) @ model["components"].T
""",
        experiment="""**Hypothesis.** On data whose third coordinate is small noise, $r = 2$ reconstruction catches the signal plane. A supervised label that depends only on the third coordinate is lost by that PCA, and a linear model on the scores fails while a linear model on the raw third coordinate works.

**Protocol.** $X_{1}, X_{2}$ standard normal, $X_3 \\sim \\mathcal{N}(0, 0.01^2)$, $Y = 5 X_3 + \\varepsilon$. Fit PCA on training $X$ only.

**Metric.** Reconstruction MSE of $X$, and test MSE of $Y$ from PCA scores versus from $X_3$ alone.

**Decision rule.** Reconstruction can look excellent while prediction from scores is near the constant baseline. If prediction from scores works, $X_3$ was not actually low variance.""",
        project="""**Deliverable.** The simulation, plus PCA on a dataset with a declared scaling policy, a plot or table of cumulative energy, and a downstream linear model on the scores compared with the same model on the original features.

**Constraints.** The mean and components come from training rows only.

**Rubric.** You state whether you care about reconstruction or about predicting $Y$, and the conclusion matches that purpose.""",
        beginner="Fit PCA with $r = 1$ on a 2D line of points and reconstruct.",
        university="Run the low-variance predictive coordinate experiment.",
        research="Relate the PCA truncation to the ridge shrink factor from Volume 01. Both shrink directions associated with small singular values, under different objectives. Write the two formulas side by side.",
        failures="""- Fitting PCA on all rows, then splitting.
- Interpreting a component loading as a causal factor.
- Choosing $r$ because it explains 95 percent of variance when the predictive signal is in the last 5 percent.""",
        see_also="Volume 01 for the SVD. Volume 09 for learned nonlinear representations that are trained with a target or with a self-supervised loss.",
    ),
)

chapter(
    v,
    "03_density_anomaly",
    "Density estimation and anomalies",
    entry(
        question="How do you score whether a new point looks like the training distribution?",
        card=[
            ("Density", "An estimate $\\hat{p}(x)$ of the law of inputs, trained without labels, or of residuals, trained on normal operation."),
            ("Anomaly score", "Often $-\\log \\hat{p}(x)$ or a distance to a fitted center. A threshold turns the score into a decision and needs a false-alarm budget."),
            ("Fit", "Use normal data only, or you will teach the detector that failures are typical."),
            ("Dimension", "A kernel density estimate needs a bandwidth and suffers as $d$ grows. A simple Gaussian model is biased and usable as a baseline."),
            ("Evaluation", "If you have labeled anomalies, this becomes a supervised detection metric at a fixed false-alarm rate. If you do not, you only have a score."),
        ],
        theory="""Anomaly detection is a one-class or low-label problem. The training contract is: these rows are the normal class, or they are unlabeled and you are willing to assume anomalies are rare enough not to dominate the fit. A Gaussian density on standardized features is the baseline. Points with large Mahalanobis distance are the candidates. The threshold is a Volume 07 operating point. Without labeled failures you can still set it to flag a chosen fraction of a validation stream, and you cannot quote a recall.

High dimension makes fixed-bandwidth kernel density estimates concentrate in misleading ways: distances become similar and the likelihood values become extreme. A low-dimensional projection or a simple model with a checked residual is often more honest than a flexible density you cannot evaluate.

If the process drifts, yesterday's normal is today's false alarm. A density is a model of a time window. Put the window in the card.""",
        derivation="""For $X \\sim \\mathcal{N}(\\mu, \\Sigma)$, $-2\\log p(x)$ equals the Mahalanobis distance plus $\\log\\det \\Sigma$ plus a constant. Ranking points by Mahalanobis distance is ranking them by Gaussian likelihood. The squared distance is approximately $\\chi^2_d$ when the Gaussian assumption holds, which suggests a threshold, and the suggestion is wrong when the tails are heavier. A validation histogram of the score on held-out normal data is the empirical threshold source. Labeled anomalies, if they exist, are a test set and do not choose the threshold.""",
        code="""import numpy as np

def gaussian_anomaly_score(X_train, X_new, ridge=1e-6):
    mu = X_train.mean(axis=0)
    xc = X_train - mu
    cov = (xc.T @ xc) / X_train.shape[0] + ridge * np.eye(X_train.shape[1])
    left = np.linalg.cholesky(cov)
    delta = np.asarray(X_new, dtype=np.float64) - mu
    # Solve L w = delta^T for whitened coordinates.
    w = np.linalg.solve(left, delta.T)
    return np.sum(w**2, axis=0)
""",
        experiment="""**Hypothesis.** Inliers drawn from the training Gaussian have smaller scores than points shifted by several standard deviations, and a heavier-tailed inlier distribution produces more high scores than the $\\chi^2$ story expects.

**Protocol.** Train on 500 standard normal points in $\\mathbb{R}^4$. Score 500 fresh inliers and 200 points with mean $3 \\mathbf{1}$. Then score Student-$t$ points with 3 degrees of freedom as if they were anomalies under a Gaussian fit.

**Metric.** Fraction of shifted points above the 99th percentile of training scores, and the same fraction for $t$ inliers.

**Decision rule.** The shifted cloud should mostly flag. The $t$ inliers should flag more often than 1 percent. If they do not, the ridge or the percentile is masking tails; look at the score histogram before adding model complexity.""",
        project="""**Deliverable.** A Gaussian anomaly baseline with a predeclared false-alarm rate on a validation slice of normal data, plus one injected shift. If you add a second detector, compare them at the same false-alarm rate.

**Constraints.** The threshold is not chosen on the injected anomalies.

**Rubric.** You report false-alarm rate and detection rate separately. A single accuracy is not accepted.""",
        beginner="Score a 1D normal sample and set a threshold at mean plus three standard deviations.",
        university="Run the shift and heavy-tail experiment in $\\mathbb{R}^4$.",
        research="Explain why a deep density model can assign higher likelihood to a constant input than to training images, as a caution that likelihood is not a perceptual typicality score. You do not need to reproduce a paper; state what evaluation you would require before trusting such a score.",
        failures="""- Fitting the "normal" model on a mixture of failures and normal operation.
- Choosing the threshold to maximize detection on the only labeled anomalies you have, and then quoting that detection rate.
- Ignoring drift and paging a human for a new typical regime.""",
        see_also="Volume 04 for the Gaussian tail. Volume 16 for adversaries who try to sit inside the normal score.",
    ),
)

chapter(
    v,
    "04_mixtures",
    "Mixture models",
    entry(
        question="What does a mixture claim that K-means does not, and how do you fit it?",
        card=[
            ("Model", "$p(x) = \\sum_{k=1}^{K} \\pi_k p(x \\mid k)$, with $\\pi$ on the simplex."),
            ("Latent", "A component indicator $Z$. The responsibilities $p(Z=k \\mid x)$ are soft assignments."),
            ("EM", "Alternate between computing responsibilities under current parameters and maximizing the expected complete log likelihood."),
            ("K-means link", "Hard assignments and equal spherical Gaussians, with variances shrunk toward zero, approach a K-means-like objective."),
            ("Likelihood", "The observed-data likelihood is not convex in the parameters. Restarts matter. A larger $K$ fits the training density better and can overfit."),
        ],
        theory="""A mixture is a density with a discrete latent cause. It gives you a likelihood, a way to score new points, and soft memberships. K-means gives you a partition and a squared-error objective. Use a mixture when you want probabilities or ellipses with covariance. Use K-means when you want a simple quantization and will not pretend you fit a density.

EM is coordinate ascent on a lower bound of the log likelihood. The bound is tight at the current responsibilities, so the observed log likelihood should not decrease. It can converge to a bad local maximum, and a component can collapse onto one point, sending a variance toward zero and the likelihood toward infinity in a Gaussian mixture without a variance floor. A floor or a prior on the covariance is part of a practical estimator.

Selecting $K$ by training likelihood is invalid in the naive way: more components win. Hold out log likelihood, or a penalized criterion you can define, on a validation set. Even then, a good density estimate is not a discovery of $K$ true types.""",
        derivation="""For a Gaussian mixture, the E step sets

$$r_{ik} \\propto \\pi_k \\mathcal{N}(x_i \\mid \\mu_k, \\Sigma_k), \\quad \\sum_k r_{ik} = 1.$$

The M step for the weights is $\\pi_k = n^{-1}\\sum_i r_{ik}$. The mean $\\mu_k$ is the responsibility-weighted average of $x_i$. These updates maximize the expected complete-data log likelihood given the responsibilities. The observed log likelihood after the M step is at least the previous one when the maximization is exact. In code, compute responsibilities in the log domain or the first term underflows to zero and a component dies by rounding.""",
        code="""import numpy as np

def log_gaussian_diag(X, mu, var):
    # var is a vector of per-coordinate variances.
    X = np.asarray(X, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    return -0.5 * np.sum(np.log(2 * np.pi * var) + (X - mu) ** 2 / var, axis=-1)
""",
        experiment="""**Hypothesis.** On two well-separated 1D Gaussians, a two-component EM recovers means near the truth and a higher validation log likelihood than a one-component Gaussian. On a single Gaussian, the two-component model does not win by a meaningful validation margin if you keep a variance floor.

**Protocol.** Implement EM for diagonal or 1D Gaussians with a variance floor of $10^{-3}$, several restarts, $K \\in \\{1, 2\\}$. Validation size at least 1000.

**Metric.** Parameter error when $K$ matches the generator, and validation log likelihood.

**Decision rule.** The separated case should favor $K=2$. The single-Gaussian case should not produce a dramatic likelihood win for $K=2$. A huge win with a near-zero variance means the floor failed and a component collapsed.""",
        project="""**Deliverable.** A 1D two-component EM with a log-likelihood trace that is monotone, a variance floor, and the model-selection comparison above.

**Constraints.** Responsibilities are computed from log densities with a log-sum-exp, not from raw densities that underflow.

**Rubric.** You can show the monotone trace and a restart that found a worse likelihood.""",
        beginner="Fit a single Gaussian by the MLE formulas and score a validation set with `log_gaussian_diag`.",
        university="Implement EM and the two simulations.",
        research="Derive the EM auxiliary function inequality $ \\ell(\\theta) \\ge Q(\\theta \\mid \\theta') $ in words and identify the step that becomes a hard assignment in the small-variance limit.",
        failures="""- Selecting $K$ on training likelihood.
- Letting a component variance go to zero.
- Describing mixture weights as the prior probability of a real latent type without a measurement model.""",
        see_also="Volume 04 for the Gaussian MLE. Volume 13 for mixture-of-experts and latent-variable generative models, which inherit this nonconvexity.",
    ),
)

chapter(
    v,
    "05_self_supervised",
    "Self-supervised representations",
    entry(
        question="What target are you manufacturing when you train without labels?",
        card=[
            ("Pretext", "A supervised task built from the input alone, such as predicting a masked coordinate or identifying a transformed copy."),
            ("Representation", "Intermediate features $h(x)$ that you will reuse for a later task."),
            ("Leakage", "The pretext can be solved by a shortcut that ignores the structure you wanted, such as a border artifact or a timestamp."),
            ("Evaluation", "A downstream task with a frozen or fine-tuned $h$, compared with a baseline that does not use $h$."),
            ("Not free labels", "A pretext loss going down does not mean $h$ helps the decision you care about."),
        ],
        theory="""Self-supervision turns unsupervised data into a supervised problem by hiding part of $x$ and asking the model to recover it, or by declaring two augmentations of the same row to be a match. The hope is that a representation sufficient for the pretext is useful for a later target $Y$. That hope is an empirical claim about the relationship between the pretext and $Y$. It can fail when the pretext is solved by a feature that is independent of $Y$.

Evaluate $h$ by a simple downstream head, often linear, trained on a modest labeled set, with the same split discipline as Volume 07. Compare with the raw features and with PCA scores. If a linear head on $h$ cannot beat a linear head on $x$, you have not yet earned a complicated encoder.

Augmentations are part of the model. If your augmentation destroys the signal, for example by shuffling the only predictive coordinate, the representation will learn what remains, which may be nothing you wanted. Write the augmentation next to the loss.""",
        derivation="""A denoising objective $\\mathbb{E}[\\|x - g(h(\\tilde{x}))\\|^2]$, where $\\tilde{x}$ is a corrupted copy, is minimized by a $g \\circ h$ that approximates the conditional mean of $x$ given $\\tilde{x}$, under squared error. The representation $h$ is whatever bottleneck you force that conditional mean through. If the corruption leaves a coordinate untouched and that coordinate already determines $x$, $h$ can ignore everything else. The conditional-independence chapter is the test: if $Y \\perp h(X) \\mid s(X)$ for a shortcut $s$, the downstream task will not see new information.""",
        code="""import numpy as np

def mask_and_target(X, rng, mask_prob=0.3):
    X = np.asarray(X, dtype=np.float64)
    mask = rng.random(X.shape) < mask_prob
    corrupted = X.copy()
    corrupted[mask] = 0.0
    return corrupted, X, mask
""",
        experiment="""**Hypothesis.** Predicting a masked coordinate of a two-dimensional Gaussian with strong correlation produces a representation that helps impute the missing coordinate, and does not automatically help predict an independent label.

**Protocol.** $X_2 = X_1 + \\text{small noise}$, $Y$ independent of $X$. Train a least-squares map from the unmasked coordinate to the masked one on corrupted rows. Evaluate imputation MSE against predicting the mean. Evaluate predicting $Y$ from the imputed features versus from $X$.

**Metric.** Imputation MSE and test MSE for $Y$.

**Decision rule.** Imputation should beat the mean. Prediction of $Y$ should not beat the constant baseline by a meaningful margin. A win on $Y$ means the generator coupled $Y$ to $X$.""",
        project="""**Deliverable.** The masking experiment, plus a one-paragraph pretext design for a data type you care about, including the shortcut you are afraid of and the downstream check that would catch it.

**Constraints.** The downstream check uses labels that did not enter the pretext fit.

**Rubric.** You do not claim a representation is useful because the pretext loss decreased.""",
        beginner="Implement masking and a mean-imputation baseline.",
        university="Complete the correlated-coordinate versus independent-label experiment.",
        research="Describe a contrastive pretext in one formula: pull together two views of the same row and push apart views of different rows. State a shortcut (a shared border, a camera id) that would satisfy the contrastive loss without helping $Y$.",
        failures="""- Tuning the pretext on the downstream test metric without a validation firewall.
- Augmentations that remove the class-relevant signal and a celebration of invariance to that signal.
- Using labels to build the pretext and still calling the method unsupervised.""",
        see_also="Volume 11 for representation learning with images. Volume 17 for large pretrained models, which are this idea at a scale that does not change the evaluation logic.",
    ),
)

chapter(
    v,
    "06_embedding_project",
    "Project: evaluate a representation",
    entry(
        question="What evaluation would convince you to replace raw features by a learned or reduced representation?",
        card=[
            ("Candidates", "Raw features, PCA scores, and one learned or clustered representation."),
            ("Probe", "A linear model or logistic regression on each representation, with the same split."),
            ("Budget", "The same number of labeled training rows for each probe."),
            ("Map", "A function that applies to new rows. A layout of the training points is not enough."),
            ("Decision", "Adopt, reject, or keep as a visualization only."),
        ],
        theory="""This project stops unsupervised methods from grading their own homework. The representation may be built without $Y$, but the decision to use it is supervised if you have $Y$. If you do not have $Y$, the only honest products are reconstruction, density, or a declared quantization error, and the card should say "no downstream task."

Give every candidate the same labeled budget. A representation that needs 10 times more labels to win has not won at the budget you have. Freeze the representation before the probe is selected, or include its knobs inside the validation loop and pay for that selection.

Visualization is a separate deliverable. A 2D picture can be pedagogically useful and predictively harmful. The project requires you to say which of the two you are claiming.""",
        derivation="""Let $R(h)$ be the risk of the best linear probe on representation $h$ inside a fixed probe family, and let $\\hat{R}(h)$ be a validation estimate. You adopt $h$ over raw $x$ when $\\hat{R}(h)$ beats $\\hat{R}(\\mathrm{id})$ by the practical margin after the probe's own regularization is chosen fairly. The gap $R(\\mathrm{id}) - R(h)$ can be negative even when $h$ is a sufficient statistic for a nonlinear family, because the probe is linear. A loss with a linear probe is a test of linear readability, not of whether any head could use $h$. If you allow a flexible head, say so, and give the raw features the same flexible head or you have confounded the representation with the head.""",
        code="""import numpy as np

def linear_probe_mse(Z_train, y_train, Z_test, y_test, lam=1e-3):
    Z_train = np.asarray(Z_train, dtype=np.float64)
    Z_test = np.asarray(Z_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
    ones_tr = np.concatenate([np.ones((Z_train.shape[0], 1)), Z_train], axis=1)
    ones_te = np.concatenate([np.ones((Z_test.shape[0], 1)), Z_test], axis=1)
    penalty = lam * np.eye(ones_tr.shape[1])
    penalty[0, 0] = 0.0
    theta = np.linalg.solve(ones_tr.T @ ones_tr + penalty, ones_tr.T @ y_train)
    pred = ones_te @ theta
    return float(np.mean((pred - y_test) ** 2))
""",
        experiment="""**Hypothesis.** PCA with $r$ large enough to keep a predictive coordinate matches the raw-feature probe. PCA that drops that coordinate loses. A cluster id represented as a single integer code is a poor linear feature unless you one-hot it, and the one-hot probe still loses to the raw coordinates on a linear truth.

**Protocol.** Use the low-variance predictive coordinate construction from the PCA chapter, and a separate two-blob dataset where the blob id is a one-hot feature. Same labeled budget, $\\lambda$ fixed in advance or chosen on validation.

**Metric.** Probe test MSE or log loss, with a paired comparison if you built per-point losses.

**Decision rule.** The representation that discarded the predictive coordinate must lose. If the one-hot cluster id wins on the blob data, that is allowed: the cluster recovered the Bayes partition. Say so as a special case, not as a general endorsement of clustering.""",
        project="""**Deliverable.** A table of probes, the practical margin, the paired gap if you have per-point losses, and a decision: ship the representation, use it only for a picture, or discard it. Include the map's definition on a new row.

**Constraints.** The same labels and the same split for every row of the table.

**Rubric.** The decision would be unchanged if you replaced adjectives with the table. A picture without a probe is labeled as a picture.""",
        beginner="Probe raw features versus the first principal component on a linear problem.",
        university="Complete the discard-the-signal and the cluster one-hot comparisons.",
        research="Add a self-supervised imputation feature from the previous chapter to the table and interpret a negative result without changing the pretext after seeing the probe.",
        failures="""- A flexible head on the learned features and a linear head on the raw features.
- Fitting the representation on the test rows.
- Declaring success from a 2D scatter plot of the training set.""",
        see_also="Volume 11 and Volume 17, where the same probe protocol applies to features you did not train yourself.",
    ),
)
