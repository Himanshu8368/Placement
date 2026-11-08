"""Volumes 03–06: computation, probability, inference, supervised learning."""

from booklib import chapter, entry, volume

v = volume(
    3,
    "Computation for experiments",
    "Beginner",
    "A reader who can write a Python function and has the least-squares project from Volume 01.",
    "How do you turn a formula into an experiment that still means the same thing next month?",
)

chapter(
    v,
    "01_python_for_reference",
    "Python for reference implementations",
    entry(
        question="What coding habits keep a reference implementation checkable?",
        card=[
            ("Shape", "Every public function documents the shape and dtype of each array."),
            ("Pure core", "The mathematical step does not read files or global random state."),
            ("Generator", "Randomness enters through an explicit `numpy.random.Generator`."),
            ("Check", "A known case with an exact answer runs before any real data."),
            ("Failure", "Non-finite values raise. They are not plotted as a quiet gap."),
        ],
        theory="""A reference implementation is an argument you can execute. If the mathematical step secretly reads a global seed, a previous cell's variable, or a file path, the argument cannot be replayed. Separate three layers: a pure function from arrays to arrays, a driver that builds those arrays and owns the generator, and a reporter that writes the table you will put in the lab note.

Shapes are part of the type. A function that accepts both row and column vectors by guessing will eventually align the wrong axis and still return a finite number. Require $(n, d)$ for a design matrix and $(n,)$ for a target. Convert once at the boundary.

Determinism is a choice about the driver, not about the science. You fix a seed so a bug is reproducible. You vary the seed when the experiment asks about uncertainty. Those are different runs and should have different note names.""",
        derivation="""Let $f$ be the pure step and $D(s)$ the dataset drawn from seed $s$. A known-case test is a pair $(D, y^\\star)$ with $f(D) = y^\\star$ in exact arithmetic, and $\\|f(D) - y^\\star\\| \\le \\tau$ in float64 for a tolerance $\\tau$ you write down. If this fails, do not start the experiment. The experiment's metric is then a function $m(f(D(s)))$ of an explicit seed, so the map $s \\mapsto m$ is a function you can average.""",
        code="""import numpy as np

def as_design(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError(f"expected X (n, d) and y (n,), got {X.shape} and {y.shape}")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("non-finite values")
    return X, y

def rng_from(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
""",
        experiment="""**Hypothesis.** A shape check catches a row-vector target before a silent broadcast inflates the loss.

**Protocol.** Call `as_design` with a target of shape `(1, n)` and with a matching `(n,)` target. Then compute a mean squared error only on the path that passes the check, using a one-column design whose correct residual you know.

**Metric.** Whether the bad shape raises, and whether the good path matches the hand-computed error.

**Decision rule.** Both must hold. A warning instead of an exception is a failed check, because a warning does not stop the experiment.""",
        project="""**Deliverable.** A one-file template `experiment.py` with `pure_step`, `known_case`, `run(seed)`, and `main` writing `notes/result.json`.

**Constraints.** No global `np.random.seed`. The seed is an argument.

**Rubric.** Running the file twice with the same seed writes the same JSON. A wrong shape fails before any metric is printed.""",
        beginner="Wrap least squares from Volume 01 in `as_design` and `known_case`.",
        university="Make the JSON contain the seed, the metric, and the tolerance.",
        research="Add a second backend for the same pure function and require both known cases to agree. This is the pattern you will want when a later chapter has a NumPy path and a library path.",
        failures="""- Sharing one Generator across unrelated experiments without recording the draw order.
- Catching exceptions and returning `NaN` so the table still fills.
- Testing only the happy shape.""",
        see_also="The dataset-audit project at the end of this volume. Volume 07 for splits that must happen inside `run(seed)`, not before it by hand.",
    ),
)

chapter(
    v,
    "02_numpy",
    "NumPy as a mathematical instrument",
    entry(
        question="Which array operations match the linear algebra in Volume 01, and which ones secretly broadcast?",
        card=[
            ("Vector", "Shape `(d,)`. `x @ y` is a dot product."),
            ("Design", "Shape `(n, d)`. Row $i$ is example $i$. `X @ theta` has shape `(n,)`."),
            ("Batch matmul", "`A @ B` on shapes `(b, n, d)` and `(b, d, k)` maps each batch item."),
            ("Reduce", "`mean(axis=0)` averages examples. `axis=1` averages features. The axis you reduce disappears."),
            ("Broadcast", "A `(n, 1)` array and a `(d,)` array become `(n, d)`. That is often an accident."),
        ],
        theory="""NumPy is the computational form of the notation sheet. The dangerous feature is broadcasting: arrays with different shapes are stretched so an operation can continue. Stretching is the right tool for adding a bias of shape `(d,)` to predictions of shape `(n, d)`. It is the wrong tool when a residual of shape `(n, 1)` meets a residual of shape `(n,)` and you think you computed $n$ squared errors rather than an $n \\times n$ outer difference.

Write the shape beside the line in reference code until the operation is boring. Reductions must name the axis. A default `sum` over every entry hides whether you averaged over the batch. In later deep-learning code the same bug is a mean over pixels that was supposed to be a mean over the batch only.""",
        derivation="""Let $X \\in \\mathbb{R}^{n \\times d}$ and $\\theta \\in \\mathbb{R}^{d}$. The $i$th coordinate of $X\\theta$ is $\\sum_{j=1}^{d} X_{ij} \\theta_j$. In NumPy, `X @ theta` performs those $n$ dots. The Gram matrix `X.T @ X` has entries $\\sum_{i=1}^{n} X_{ij} X_{ik}$. If you instead write `X.T * X` you ask for an elementwise product and the shapes `(d, n)` and `(n, d)` do not match. The error is a gift. A silent wrong answer would be worse, and broadcasting is how silent wrong answers happen when one side is a vector.""",
        code="""import numpy as np

def predictions(X, theta, bias):
    X = np.asarray(X, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or theta.shape != (X.shape[1],):
        raise ValueError("X (n, d), theta (d,)")
    return X @ theta + np.asarray(bias, dtype=np.float64)
""",
        experiment="""**Hypothesis.** Subtracting a column vector of shape `(n, 1)` from a prediction of shape `(n,)` builds an `(n, n)` matrix, while matching shapes `(n,)` builds a length-$n$ residual whose mean square agrees with a hand calculation.

**Protocol.** Use $n = 4$, a one-column $X$, and $\\theta = 1$. Compute both residuals inside `try` blocks that print shapes. Compare the flattened mean of squares with the true mean square only for the matching shapes.

**Metric.** Output shape, and absolute error of the mean square on the safe path.

**Decision rule.** The safe path must match to $10^{-12}$. If the unsafe path does not produce shape `(4, 4)`, you have a NumPy version that refused the broadcast; record that, and still keep the shape assert.""",
        project="""**Deliverable.** A shape-cheat sheet of ten operations you will use (`@`, `*`, `mean`, `sum`, `reshape`, `einsum` for a Gram matrix, indexing a minibatch, stacking, concatenating features, adding a bias), each with input shapes and output shape, checked by asserts.

**Constraints.** At least two asserts must fail if an axis is wrong.

**Rubric.** You can implement $X^\\top X$ with `@` and with `einsum` and get the same matrix.""",
        beginner="Do the cheat sheet for `@`, `mean(axis=0)`, and bias add.",
        university="Include the unsafe broadcast experiment in the lab note.",
        research="Write the same three operations as Einstein summations and state the contraction that implements a batched Jacobian-vector product. You will want that language in Volume 09.",
        failures="""- Relying on broadcasting to fix a transpose.
- Reducing over all axes by default.
- Using `matrix` objects or `np.matrix`, which change the meaning of `*`.""",
        see_also="Volume 01 for the identities these operations must match. Volume 09 for batched tensor layouts.",
    ),
)

chapter(
    v,
    "03_data_contracts",
    "Data contracts",
    entry(
        question="What must be true of a table before a model is allowed to see it?",
        card=[
            ("Row", "One observational unit. Say what that unit is."),
            ("Target", "The column you would not have at decision time. It is not a feature."),
            ("Missing", "A missing value is a value of the measurement process, not a quiet zero, unless you justify the fill."),
            ("Split key", "If several rows share an entity, the split is by entity, not by row."),
            ("Schema", "Names, dtypes, units, and an allowed range, checked in code."),
        ],
        theory="""Most false results in applied work are data-contract failures wearing a model's clothes. A contract is a set of predicates the table must satisfy before fitting. Units are part of the predicate: a duration in milliseconds and a duration in days are different features that can share a column name across files.

The split is part of the contract when rows are dependent. Random row splits of a patient table, a time series, or a user's clicks put the same entity on both sides and make the test score a memory test. The contract names the split key. It also names leakage columns: identifiers that determine the label, timestamps from after the decision, and aggregates computed with the test rows included.

Imputation is a model. Filling with the training median is a simple one. Fitting that median on all rows uses the test sample to define the training features. The contract says which statistics are training-only.""",
        derivation="""Let rows be pairs $(e_i, x_i, y_i)$ with entity $e_i$. A row split draws a subset $S$ of row indices. An entity split draws a subset $E$ of entities and sets $S = \\{i: e_i \\in E\\}$. The test risk of an entity split estimates performance on new entities. The test risk of a row split estimates performance on new rows from a mixture that still contains seen entities. Those are different functionals of the data law. No metric can repair a split that answers the wrong question. Write the functional down before choosing the number.""",
        code="""import numpy as np

def check_table(columns, required, split_key):
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if split_key not in columns:
        raise ValueError(f"split key {split_key} not in table")
    return True

def entity_split(entities, test_frac, rng):
    entities = np.asarray(entities)
    uniq = np.unique(entities)
    rng.shuffle(uniq)
    n_test = max(1, int(round(test_frac * len(uniq))))
    test_ents = set(uniq[:n_test])
    test_mask = np.array([e in test_ents for e in entities])
    return ~test_mask, test_mask
""",
        experiment="""**Hypothesis.** A row split of repeated entities yields a lower test error than an entity split when the label is constant inside each entity.

**Protocol.** Create 40 entities, 8 identical rows each, label equal to a standard normal draw per entity, and a single feature equal to the label plus tiny noise. Fit least squares under both splits, 20 seeds.

**Metric.** Median test mean squared error for row split and for entity split.

**Decision rule.** The row split should look optimistically good. If it does not, the feature is not actually a near copy of the label, or the splits coincide by chance and you should increase repeats.""",
        project="""**Deliverable.** A contract checker for a dataset you already have or for the synthetic entity data: required columns, finite features, split key, and a written sentence naming the observational unit.

**Constraints.** The checker runs before fitting and fails the process on a violation.

**Rubric.** You can point to one column you refused to use as a feature, and say why.""",
        beginner="Run the synthetic entity experiment once and write one sentence on the two test errors.",
        university="Add a training-only median imputer and show that fitting it on all rows changes a feature on the test set.",
        research="State the dependence you would need to model if entities are independent but rows inside an entity are a time series. Volume 15 returns to this as leakage in production features.",
        failures="""- Filling missing targets with zero and training anyway.
- Splitting images from the same patient across train and test.
- Using a unique identifier as a numeric feature.""",
        see_also="Volume 07 for metrics after a valid split. Volume 15 for the same contract in a pipeline.",
    ),
)

chapter(
    v,
    "04_reporting",
    "Figures, tables, and claims",
    entry(
        question="What must a figure or table contain before it can support a sentence in the lab note?",
        card=[
            ("Comparison", "A number without a baseline is not a result."),
            ("Uncertainty", "If you report a mean over seeds, report the spread or a interval."),
            ("Axes", "Labeled units, and a title that states the claim, not the chart type."),
            ("n", "The sample size is visible."),
            ("Refuse", "A bar chart of a single run with no baseline does not enter the note."),
        ],
        theory="""A reference book trains judgment about claims. The claim is a sentence with a verb: this method has lower held-out squared error than least squares under this split. The table is the evidence. Decoration is not evidence. A figure that omits the baseline invites the reader to treat a loss of 0.31 as good or bad by mood.

Uncertainty from a finite test set and uncertainty from the training procedure are different. A fixed fitted model still has a test-set standard error if the test points are random. A model refit on new seeds has additional spread. Say which one you computed. Averaging ten seeds and reporting three significant digits beyond the seed-to-seed scatter is a false precision.

Tables beat figures when the reader needs the number. Figures beat tables when the reader needs a shape: a residual plot, a calibration curve, a learning curve against sample size. One of each is enough for a project in this series.""",
        derivation="""Suppose held-out losses $\\ell_1, \\ldots, \\ell_m$ are independent with mean $\\mu$ and variance $\\sigma^2$. The mean $\\bar{\\ell}$ has variance $\\sigma^2/m$. The sample standard deviation $s$ estimates $\\sigma$, and $s/\\sqrt{m}$ estimates the standard error of the mean. This interval says nothing about a new training seed. If you also have means $\\bar{\\ell}^{(1)}, \\ldots, \\bar{\\ell}^{(K)}$ from $K$ training seeds, the sample standard deviation of those $K$ numbers estimates training variability of the reported mean. Quote $K$ and $m$ separately.""",
        code="""import numpy as np

def summarize(losses):
    losses = np.asarray(losses, dtype=np.float64).reshape(-1)
    m = losses.size
    mean = float(np.mean(losses))
    se = float(np.std(losses, ddof=1) / np.sqrt(m)) if m > 1 else np.nan
    return {"n": m, "mean": mean, "se": se}
""",
        experiment="""**Hypothesis.** Two methods with the same mean loss over seeds can have different seed-to-seed spread, and a report of means alone hides it.

**Protocol.** Draw method A as $\\mathcal{N}(1.0, 0.01^2)$ and method B as $\\mathcal{N}(1.0, 0.2^2)$, 30 seeds. Summarize each.

**Metric.** Mean and standard error, plus the standard deviation across seeds.

**Decision rule.** Means match. Spreads do not. A reporting template that prints only the mean fails this experiment and must be changed.""",
        project="""**Deliverable.** A reporting snippet that writes a markdown table with columns `method`, `n_test`, `n_seeds`, `metric`, `mean`, `seed_sd`, and a baseline row.

**Constraints.** The function refuses to emit a row with a missing baseline flag on the table as a whole: the table must contain a row named `baseline`.

**Rubric.** You rebuild the Volume 01 case A versus case B comparison in this table, and the caption states the claim in one sentence.""",
        beginner="Compute a standard error for ten coin flips of a 0-1 loss.",
        university="Produce the required table for the linear versus quadratic simulation.",
        research="Distinguish a confidence interval for a fixed model's test risk from a prediction interval for a future seed. Write one sentence each into the note template.",
        failures="""- Truncating the y-axis of a bar chart to exaggerate a gap smaller than the standard error.
- Reporting the best seed as if it were the procedure.
- Comparing a tuned method against an untuned baseline and calling the gap architectural.""",
        see_also="Volume 07 for paired comparisons. Volume 18 for how a paper's main table is filed.",
    ),
)

chapter(
    v,
    "05_experiment_folders",
    "Reproducible experiment folders",
    entry(
        question="What is the smallest folder layout that lets you rerun a result?",
        card=[
            ("Layout", "`experiments/<id>/{config.json, run.py, metrics.json, notes.md}`."),
            ("Config", "Every knob that changes the metric, including the seed and the data path."),
            ("Command", "The note starts with the exact command."),
            ("Immutable", "Do not edit `metrics.json` by hand."),
            ("Identity", "The id is a slug plus a short hash of the config, not a date alone."),
        ],
        theory="""A result you cannot rerun is a rumor. The folder is the unit you will cite from later chapters. Config is the input. Metrics are the output. The note is the interpretation, and it is allowed to be wrong later; the metrics file is not edited to match a new opinion.

Hashing the config protects you from two runs that share a name and disagree. Dates are for humans. Hashes are for identity. Store the library versions next to the config when you use more than NumPy, because a metric can move when a solver changes.

This layout is intentionally smaller than a platform. You are building the habit a platform would enforce. Volume 15 can wrap the same folder in a pipeline. It should not have to invent the contents.""",
        derivation="""Let $c$ be the config and $h(c)$ a stable hash of its canonical JSON. Let $r(c)$ be the metric map. The folder name `<slug>-<h(c)>` is a key for $r(c)$. If you change a knob and reuse the folder, you have changed the key without changing the address, and later comparisons mix two functions. The rule "new config, new folder" is the statement that $r$ is a function.""",
        code="""import hashlib, json
from pathlib import Path

def canonical_hash(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:12]

def make_folder(root: str, slug: str, config: dict) -> Path:
    path = Path(root) / f"{slug}-{canonical_hash(config)}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return path
""",
        experiment="""**Hypothesis.** Two configs that differ only by seed produce different hashes and different folders. The same config produces the same hash.

**Protocol.** Hash `{n: 32, seed: 0}` twice and `{n: 32, seed: 1}` once.

**Metric.** Equality of the first two hashes, inequality with the third.

**Decision rule.** Both comparisons must hold. If key order in a hand-built dict changes the hash, the JSON was not canonical.""",
        project="""**Deliverable.** Move the Volume 01 simulation into this folder layout. One folder per `(case, n, seed)` is acceptable; or one folder per case with a seed list stored in the config. Choose one and defend it.

**Constraints.** `notes.md` contains the command and the decision. `metrics.json` contains the numbers the decision uses.

**Rubric.** A new terminal, with only the folder, can tell what was run. Deleting `notes.md` does not destroy the numbers.""",
        beginner="Create one folder for the known case of least squares.",
        university="Store the full simulation grid.",
        research="Add `environment.json` with package versions and a check that fails if a rerun's versions differ. Record whether you consider that a failed reproduction or a new experiment.",
        failures="""- Overwriting `metrics.json` after peeking at the test set and changing the model.
- Putting the only copy of the result in a notebook cell you then edit.
- Hashing a config that omits the data path.""",
        see_also="Volume 18 for preregistering the metric before the run. Volume 15 for pipeline runs that should emit this same folder.",
    ),
)

chapter(
    v,
    "06_dataset_audit",
    "Project: audit a dataset",
    entry(
        question="What do you check before any model is allowed to look competent?",
        card=[
            ("Target", "Type, units, missingness, and whether it is a proxy."),
            ("Split", "Who the test score is about: new rows, new entities, or a later time."),
            ("Leakage", "Columns that would be unavailable, or that encode the target."),
            ("Baseline", "A constant predictor and a simple linear or majority predictor, on the same split."),
            ("Stop", "If you cannot name the decision the prediction supports, do not tune a model yet."),
        ],
        theory="""An audit is a reference procedure you run every time the data source changes. It answers whether the prediction problem is well posed. A well-posed problem names the input available at decision time, the target, the population the test set represents, and a baseline a serious person already has. Models come after that.

The constant predictor is not an insult. For squared error it is the training mean, and it is the Bayes predictor when features carry no information. Any later model that cannot beat it has not used the features. For classification, the majority class is the analogous baseline. A rare-event problem can have 99 percent accuracy with a constant predictor. The audit records the base rate so that accuracy cannot launder it.

Proxies matter. Predicting a label that a human assigned from the same features you are using is not the same problem as predicting the event the human was judging. The audit says which problem the columns actually encode.""",
        derivation="""For squared error, $\\mathbb{E}[(Y - c)^2]$ is minimized at $c = \\mathbb{E}[Y]$, with value $\\mathrm{Var}(Y)$. On a training sample the minimizing constant is the training mean $\\bar{y}$, and the training mean squared error is the sample variance with the $1/n$ normalization you chose in the loss, not necessarily the unbiased variance. On a test set you still predict $\\bar{y}_{\\mathrm{train}}$, not the test mean. Using the test mean is leakage and makes the baseline look worse or better for a reason you will not have in deployment.""",
        code="""import numpy as np

def constant_baseline(y_train, y_test):
    pred = float(np.mean(y_train))
    mse = float(np.mean((y_test - pred) ** 2))
    return {"prediction": pred, "test_mse": mse, "test_n": int(len(y_test))}
""",
        experiment="""**Hypothesis.** On data where the feature is independent of the target, least squares does not beat the constant baseline on a reasonably large test set by more than noise.

**Protocol.** Draw $n_{\\mathrm{train}} = n_{\\mathrm{test}} = 500$, feature standard normal, target standard normal, independent. Fit an intercept and a slope. Compare test MSE with `constant_baseline`. Repeat 30 seeds.

**Metric.** Fraction of seeds on which least squares wins, and the median difference in test MSE.

**Decision rule.** The fraction should be near one half and the median difference near zero. A large consistent win means the generator is coupled by accident.""",
        project="""**Deliverable.** An audit note for either the synthetic entity table from the data-contract chapter or a dataset you have the rights to use. Required sections: observational unit, target definition, split, columns refused, missingness, constant baseline, one linear or majority baseline, and a go or no-go sentence.

**Constraints.** No model with more than a few dozen parameters. The point is the audit.

**Rubric.** A reader can tell what a "good" number would mean and whether your baseline already achieved it. The go or no-go sentence names the decision a prediction would support.""",
        beginner="Audit the synthetic independent-feature data only.",
        university="Include the entity-split dataset and both baselines.",
        research="Add a section on what would make the target a proxy, and write a second target definition that would change the decision. You do not need new data if you can argue from the measurement process.",
        failures="""- Tuning a flexible model to rescue a problem you cannot define.
- Reporting accuracy on a 95 percent base rate without the base rate.
- Computing the constant predictor from the test labels.""",
        see_also="Volume 06 will treat the linear baseline as a model class. Volume 07 will refuse metrics that ignore this audit.",
    ),
)

v = volume(
    4,
    "Probability",
    "Beginner to university",
    "A reader who has the notation sheet and can compute an average in NumPy.",
    "What is a probability model claiming about data, and which distributions show up because of a theorem rather than a habit?",
)

chapter(
    v,
    "01_probability_spaces",
    "Probability, events, and random variables",
    entry(
        question="What has to be specified before a probability calculation is about a dataset?",
        card=[
            ("Law", "A probability $P$ on events. A random variable $X$ pushes $P$ forward to a law on values."),
            ("Conditional", "$P(A \\mid B) = P(A \\cap B) / P(B)$ when $P(B) > 0$."),
            ("Independence", "$P(A \\cap B) = P(A) P(B)$. Pairwise independence of many events is weaker than mutual independence."),
            ("iid", "Independent and identically distributed. A strong assumption, not a default about the world."),
            ("Dataset", "One realization of a collection of random variables, not the law itself."),
        ],
        theory="""Probability in this series is a model of ignorance and of variability, written so that expectations become the risk $R(\\theta)$. You choose the sample space by saying what a draw is. If a draw is a patient, two visits are not two independent draws. If a draw is a visit, generalization to a new patient is a different question. The audit in Volume 03 was already this choice.

Conditioning is how features enter. $P(Y \\in \\cdot \\mid X = x)$ is the object a predictor is trying to match, under a loss that says what "match" means. Squared error matches the conditional mean. Absolute error matches a conditional median. A 0-1 loss matches a conditional mode, which for two classes is a threshold on the conditional probability. Using squared error on a problem whose decision needs a probability is a later chapter; the point here is that the loss picks a functional of the conditional law.

Independence is a knife. It lets expectations of sums become sums of expectations of products only under extra conditions, and it lets variances of sums add. It is false for time series, for entities with repeated measures, and for data drawn from a shifting policy. You may still use an independence model. You must then call the risk an approximation and test the approximation, not hide it in the word "sample." """,
        derivation="""If $X$ and $Y$ are independent and the expectations exist, $\\mathbb{E}[XY] = \\mathbb{E}[X]\\mathbb{E}[Y]$. Proof for simple random variables follows by writing both as finite sums of indicators of a partition and factoring the joint probabilities. The same identity fails if you only assume uncorrelatedness when you replace $X$ and $Y$ by nonlinear functions of them. Uncorrelated means $\\mathbb{E}[XY] = \\mathbb{E}[X]\\mathbb{E}[Y]$ for that pair, not for $X^2$ and $Y^2$.

Linearity of expectation does not need independence: $\\mathbb{E}[X + Y] = \\mathbb{E}[X] + \\mathbb{E}[Y]$ whenever the expectations exist. Variance of a sum adds if and only if the covariance is zero, in the two-variable case: $\\mathrm{Var}(X + Y) = \\mathrm{Var}(X) + \\mathrm{Var}(Y) + 2\\mathrm{Cov}(X, Y)$.""",
        code="""import numpy as np

def monte_carlo_mean(sampler, n, rng):
    draws = np.asarray([sampler(rng) for _ in range(n)], dtype=np.float64)
    return float(np.mean(draws)), float(np.std(draws, ddof=1) / np.sqrt(n))
""",
        experiment="""**Hypothesis.** The sample mean of a fair $\\mathrm{Bernoulli}(1/2)$ concentrates, and the standard error $s/\\sqrt{n}$ tracks the known standard deviation $1/2$ divided by $\\sqrt{n}$.

**Protocol.** Draw 2000 independent Bernoulli trials. At $n \\in \\{50, 200, 2000\\}$, compute the mean of the prefix and the estimated standard error. Compare with $0.5$ and with $0.5/\\sqrt{n}$.

**Metric.** Absolute error of the mean, and the ratio of estimated standard error to the true one.

**Decision rule.** The mean error should be on the order of the true standard error, not systematically 0.1. A standard-error ratio far from 1 at $n = 2000$ means the `ddof` or the formula is wrong.""",
        project="""**Deliverable.** A note that defines the sample space for a prediction problem you care about, states whether rows are iid, and computes one expectation by Monte Carlo with a reported standard error.

**Constraints.** If you claim independence, name one way the real data might violate it.

**Rubric.** The note distinguishes the law from the dataset.""",
        beginner="Simulate the Bernoulli mean and write the sample space in one sentence.",
        university="Prove linearity without independence on a two-point example, and give a pair of uncorrelated random variables that are not independent.",
        research="Write the conditional risk $\\mathbb{E}[\\ell(f(X), Y) \\mid X = x]$ for squared error and show the minimizer is the conditional mean. This is the population version of the Volume 01 project.",
        failures="""- Describing a finite dataset as a distribution without saying whether you mean the empirical measure.
- Assuming variance adds for dependent time series.
- Conditioning on an event you selected after seeing the data and still using the original $P$.""",
        see_also="The distribution chapter next. Volume 05 for estimators of these expectations.",
    ),
)

chapter(
    v,
    "02_distributions",
    "Distributions you will actually use",
    entry(
        question="Which families earn a place because of a closure property or a limit theorem?",
        card=[
            ("Bernoulli / binomial", "A bit, or a count of independent bits. Classification labels and error counts."),
            ("Categorical", "One of $K$ classes. Parameterized by a probability vector."),
            ("Poisson", "Counts with variance equal to the mean, as a limit of rare binomials."),
            ("Gaussian", "A limit of normalized sums, and the maximum-entropy law on $\\mathbb{R}$ with fixed mean and variance."),
            ("Exponential / Laplace", "A positive waiting time with a memoryless property, and a symmetric heavy-shouldered location model for absolute error."),
            ("Student-$t$", "A Gaussian scale mixture. Heavier tails. A robustness probe, not a default noise model you skip checking."),
        ],
        theory="""A named distribution is a constraint. $\\mathcal{N}(\\mu, \\sigma^2)$ says the law is determined by a mean and a variance and has light tails. If your residuals have a point mass, a skew, or outliers that dominate the variance, the name is false even if the sample mean and variance match. Matching two moments is not matching a law. A QQ plot or a tail count is the check.

The Gaussian earns its place from the central limit theorem: a standardized sum of independent, not-too-heavy random variables approaches a standard normal. Means of many roughly independent effects look Gaussian. Individual observations often do not. The Laplace distribution earns a place because the median, not the mean, is its location estimator under absolute error, and because it is a simple heavier-tailed alternative. The Bernoulli earns a place because labels are bits. Use a family when a theorem or a measurement process points to it. Do not use it because a library constructor was nearby.""",
        derivation="""If $Y_1, \\ldots, Y_n$ are independent $\\mathrm{Bernoulli}(p)$, then $S = \\sum_i Y_i$ is $\\mathrm{Binomial}(n, p)$, because each sequence with $s$ ones has probability $p^s (1-p)^{n-s}$ and there are $\\binom{n}{s}$ of them. The mean is $np$ and the variance is $np(1-p)$, by independence. As $n \\to \\infty$ and $p = \\lambda/n$, $P(S = k) \\to e^{-\\lambda} \\lambda^k / k!$. That Poisson limit is why rare counts sometimes look Poisson. It fails when the trials are dependent or the rate is not small.

Standardize a sample mean: $\\sqrt{n}(\\bar{Y} - \\mu)/\\sigma \\Rightarrow \\mathcal{N}(0, 1)$ under finite variance and iid sampling. This is a theorem about the mean, and it is the reason standard errors in Volume 03 look Gaussian at large $n$ even when $Y$ itself is Bernoulli.""",
        code="""import numpy as np

def sample_laws(n, rng):
    return {
        "bernoulli": rng.binomial(1, 0.3, size=n),
        "poisson": rng.poisson(2.0, size=n),
        "gaussian": rng.normal(0.0, 1.0, size=n),
        "laplace": rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=n),
        "student_t": rng.standard_t(3, size=n),
    }
""",
        experiment="""**Hypothesis.** Laplace and Student-$t$ draws with matched variance still produce more points beyond 4 standard deviations than a Gaussian does.

**Protocol.** Draw $n = 20000$ from a standard normal, a Laplace scaled to variance 1, and a Student-$t$ with 3 degrees of freedom scaled to variance 1 (the raw $t_3$ has variance 3, so divide by $\\sqrt{3}$).

**Metric.** Fraction of absolute values larger than 4.

**Decision rule.** The Gaussian fraction should be near $6 \\times 10^{-5}$. The others should be visibly larger. If they are not, the scaling is wrong and the comparison is unfair.""",
        project="""**Deliverable.** A one-page card for each of the five families: story, parameters, mean, variance, and one ML loss or model that implies it.

**Constraints.** Include a sixth law you refuse, with the reason (for example a Gaussian for raw counts).

**Rubric.** Each card names a check you would run on residuals.""",
        beginner="Simulate the tail experiment for Gaussian versus Laplace.",
        university="Derive the binomial variance and the Poisson limit probabilities for $k = 0, 1$.",
        research="State the Lindeberg condition in words and give a case with infinite variance where the central limit theorem fails. A stable law is enough as a pointer if you do not want the full proof.",
        failures="""- Using a Gaussian likelihood for nonnegative counts and predicting negative means.
- Matching a histogram in the center and ignoring the tails that dominate squared error.
- Treating the Poisson mean-variance link as optional after you have assumed the Poisson.""",
        see_also="Volume 05 for maximum likelihood under these laws. Volume 06 for the losses they induce.",
    ),
)

chapter(
    v,
    "03_expectation_concentration",
    "Expectation, variance, and concentration",
    entry(
        question="How fast does an average approach its expectation, and what breaks the bound?",
        card=[
            ("Expectation", "The integral of the variable against its law. Linear even without independence."),
            ("Variance", "$\\mathbb{E}[(X - \\mathbb{E} X)^2]$. Scales as $a^2$ when you multiply $X$ by $a$."),
            ("Chebyshev", "$P(|X - \\mu| \\ge t) \\le \\mathrm{Var}(X) / t^2$."),
            ("Hoeffding", "For independent bounded $[0, 1]$ variables, $P(|\\bar{X} - \\mu| \\ge t) \\le 2 e^{-2 n t^2}$."),
            ("Breaks", "Dependence, unbounded heavy tails, or a mean that does not exist."),
        ],
        theory="""Concentration inequalities turn a sample size into a promise about an average. Chebyshev needs only a variance and is loose. Hoeffding needs bounded independent summands and gives a sub-Gaussian tail. Empirical risk is an average of losses. If losses are bounded and rows are independent, Hoeffding applies to $\\hat{R}_n - R$ at a fixed $\\theta$. It does not apply uniformly over an infinite set of $\\theta$ without more work. That gap is learning theory in Volume 06's neighborhood: a uniform law, not a single-average law.

If the loss is unbounded, a few outliers dominate $\\hat{R}_n$ under squared error. The mean may converge, but the constant in the bound is the variance, which those outliers inflate. This is why a reported training loss on unclipped squared error can move when one row is corrected.

Dependence breaks the $1/n$ variance scaling. In a time series with strong correlation, the effective sample size can stay small as $n$ grows. The formula $s/\\sqrt{n}$ then understates uncertainty.""",
        derivation="""Chebyshev: let $Z = (X - \\mu)^2 \\ge 0$. Then $\\mathbb{E}[Z] \\ge t^2 P(Z \\ge t^2)$, which is the claim. For an iid average, $\\mathrm{Var}(\\bar{X}) = \\sigma^2/n$, so $P(|\\bar{X} - \\mu| \\ge t) \\le \\sigma^2/(n t^2)$.

Hoeffding's inequality for independent $X_i \\in [0, 1]$ can be taken as a tool here: the exponent $-2nt^2$ is the fact to remember. Solving $2 e^{-2 n t^2} = \\delta$ for $t$ gives $t = \\sqrt{\\log(2/\\delta)/(2n)}$. A deviation of a few percent at $\\delta = 0.05$ needs $n$ in the hundreds or thousands, not the dozens, when the variables are order-1.""",
        code="""import numpy as np

def chebyshev_bound(variance, t):
    return float(variance / t**2)

def hoeffding_bound(n, t):
    return float(2 * np.exp(-2 * n * t**2))
""",
        experiment="""**Hypothesis.** The fraction of Bernoulli averages that deviate from $p$ by more than $t = 0.1$ is below the Hoeffding bound and well below a loose reading of the data, and it is much larger if you replace independent draws by a constant within each block of length 50.

**Protocol.** $p = 0.5$, $n = 200$, $t = 0.1$, 500 independent experiments. Then a dependent version: draw 4 independent Bernoulli values and repeat each 50 times.

**Metric.** Empirical deviation frequency versus `hoeffding_bound(200, 0.1)`.

**Decision rule.** The independent frequency must sit under the bound. The dependent frequency should be higher. If the independent run violates the bound often, the sampler is wrong; the bound is not the suspect first.""",
        project="""**Deliverable.** A calculator that, given a bounded loss in $[0, 1]$, a tolerance $t$, and a failure probability $\\delta$, returns the Hoeffding sample size, plus a paragraph saying why your actual loss may not be bounded.

**Constraints.** Also compute the Chebyshev sample size using a variance you specify, and compare.

**Rubric.** You apply it to 0-1 loss and explain why squared error on an unbounded target does not get the same promise.""",
        beginner="Compute both bounds for $n = 100$, $t = 0.1$, and simulate the independent Bernoulli case.",
        university="Include the block-dependence comparison.",
        research="Explain, without a full proof, why a uniform bound over a finite class of size $M$ costs an extra $\\log M$ inside the Hoeffding square root. Name the failure mode when $M$ is infinite.",
        failures="""- Quoting $s/\\sqrt{n}$ for a strongly dependent series.
- Applying Hoeffding to unbounded squared error.
- Treating a fixed-$\\theta$ concentration result as a guarantee for the $\\theta$ you picked by looking at the data.""",
        see_also="Volume 05 for central-limit intervals, which are tighter and more assumption-hungry. Volume 07 for test-set intervals.",
    ),
)

chapter(
    v,
    "04_multivariate_gaussian",
    "Multivariate Gaussians",
    entry(
        question="What does a covariance matrix say, and what does it refuse to say?",
        card=[
            ("Law", "$X \\sim \\mathcal{N}(\\mu, \\Sigma)$ in $\\mathbb{R}^d$ has density proportional to $\\exp(-\\tfrac{1}{2}(x-\\mu)^\\top \\Sigma^{-1} (x-\\mu))$ when $\\Sigma$ is SPD."),
            ("Affine", "$AX + b$ is Gaussian with mean $A\\mu + b$ and covariance $A \\Sigma A^\\top$."),
            ("Independence", "For a Gaussian vector, uncorrelated coordinates are independent. This equivalence is special to Gaussians."),
            ("Conditional", "The conditional law of some coordinates given others is Gaussian. The precision matrix encodes conditional independence."),
            ("Mahalanobis", "$(x-\\mu)^\\top \\Sigma^{-1} (x-\\mu)$ is squared distance after whitening."),
        ],
        theory="""The multivariate Gaussian is the linear-algebra distribution. Its geometry is ellipses of constant Mahalanobis distance. PCA, linear discriminant analysis, and Gaussian graphical models are this geometry plus a decision. The covariance $\\Sigma$ controls spread and linear dependence. It does not control nonlinear dependence. A Gaussian cannot represent a ring, a cluster with different tail weights, or a dependence that appears only in extremes.

Whitening by $\\Sigma^{-1/2}$ turns the law into spherical standard normal, when $\\Sigma$ is SPD. In code, a Cholesky factor is the stable square root: $\\Sigma = L L^\\top$ and $L^{-1}(X - \\mu)$ is standard normal. Forming $\\Sigma^{-1}$ explicitly is the same habit Volume 01 discouraged.

If coordinates are jointly Gaussian, a zero in the precision matrix $\\Sigma^{-1}$ means the corresponding pair is conditionally independent given the rest. A zero in $\\Sigma$ itself means ordinary uncorrelatedness, hence independence, of that pair, ignoring the others. Those are different graphs. Do not read a covariance heatmap as a conditional-independence map.""",
        derivation="""Let $Z \\sim \\mathcal{N}(0, I_d)$ and $X = \\mu + L Z$ with $\\Sigma = L L^\\top$. Then $\\mathbb{E}[X] = \\mu$ and

$$\\mathrm{Cov}(X) = \\mathbb{E}[L Z Z^\\top L^\\top] = L L^\\top = \\Sigma,$$

using $\\mathbb{E}[Z Z^\\top] = I$. This is the sampling recipe. The quadratic form in the density is $Z^\\top Z = (x-\\mu)^\\top (L L^\\top)^{-1} (x-\\mu)$.

For a partition of a two-dimensional Gaussian, the conditional mean of $X_1$ given $X_2 = x_2$ is linear in $x_2$: $\\mu_1 + \\Sigma_{12} \\Sigma_{22}^{-1} (x_2 - \\mu_2)$. That linear conditional mean is why Gaussian linear regression and the multivariate Gaussian are the same calculation in different clothes.""",
        code="""import numpy as np

def sample_gaussian(mu, cov, n, rng):
    mu = np.asarray(mu, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    left = np.linalg.cholesky(cov)
    z = rng.normal(size=(n, mu.shape[0]))
    return mu + z @ left.T
""",
        experiment="""**Hypothesis.** Samples from `sample_gaussian` recover the requested covariance as $n$ grows, and a coordinate pair with covariance 0 is empirically uncorrelated.

**Protocol.** $\\mu = 0$, $\\Sigma = \\begin{bmatrix} 2 & 0.5 \\\\ 0.5 & 1 \\end{bmatrix}$, $n \\in \\{200, 5000\\}$, one seed is enough if you also report a second seed as a check.

**Metric.** Frobenius norm of the difference between the sample covariance (with $1/n$) and $\\Sigma$.

**Decision rule.** The error should drop roughly like $1/\\sqrt{n}$. If the Cholesky fails, $\\Sigma$ was not SPD in the code, whatever the formula said.""",
        project="""**Deliverable.** A whitening function using Cholesky, a round-trip test (`whiten` then `unwhiten`), and a short note on a dataset of two coordinates that are uncorrelated but dependent (sample $Z \\sim \\mathcal{N}(0,1)$ and set $X = Z$, $Y = Z^2 - 1$). Show the sample correlation is near 0 while a scatter plot is not a blob.

**Constraints.** Do not call the second pair Gaussian.

**Rubric.** The note says where the Gaussian implication "uncorrelated means independent" would have lied.""",
        beginner="Sample a diagonal Gaussian and check the sample variances.",
        university="Derive the conditional mean formula in two dimensions and check it by binning samples.",
        research="Connect a zero in the precision matrix to a conditional regression coefficient of zero. Implement the precision matrix via `lstsq` or a solver, not an explicit inverse, if you form normal equations.",
        failures="""- Estimating a $d \\times d$ covariance from $n \\le d$ samples and inverting it.
- Reading correlation as causation or as conditional independence.
- Using a Gaussian bottleneck layer and claiming it can represent any dependence.""",
        see_also="Volume 08 for PCA as the eigendecomposition of a covariance. Volume 05 for the Gaussian likelihood.",
    ),
)

chapter(
    v,
    "05_conditional_independence",
    "Conditional independence",
    entry(
        question="When does a feature become irrelevant once you know some other features?",
        card=[
            ("Definition", "$A \\perp B \\mid C$ means $P(A, B \\mid C) = P(A \\mid C) P(B \\mid C)$ whenever the conditioning event has positive probability."),
            ("Chain", "A Markov chain has $X_{t+1} \\perp (X_{s})_{s < t} \\mid X_t$."),
            ("Classifier", "If $Y \\perp X_j \\mid X_{-j}$, feature $j$ adds nothing once the others are known."),
            ("Not closed", "Independence of pairs does not give you independence of functions of those pairs without more structure."),
            ("Graph", "A missing edge is a claim. It is not a default."),
        ],
        theory="""Conditional independence is the grammar of "this feature is redundant given those." It is stronger than a coefficient being small in one particular model. A coefficient can be small because of scaling, collinearity, or a penalty. Conditional independence says the conditional law of the target does not change when the feature moves, once the conditioning set is fixed.

Markov models in sequences and states in reinforcement learning are conditional-independence claims. $S_{t+1}$ depends on the past only through $S_t$ and the action. If the real process remembers a hidden variable, the state you wrote down is not a state. Later, a "Markov decision process" chapter will assume this. The assumption belongs here, where it is just probability.

You cannot test conditional independence by looking at a single correlation. You can test a consequence: after fitting a sufficiently rich model on $X_{-j}$, adding $X_j$ should not improve held-out risk if the independence is true and the first model already captured $P(Y \\mid X_{-j})$. With a weak first model, a gain from $X_j$ may only mean the first model was incomplete.""",
        derivation="""Let $Y, X_1, X_2$ be discrete. $Y \\perp X_2 \\mid X_1$ iff $P(Y = y \\mid X_1 = x_1, X_2 = x_2) = P(Y = y \\mid X_1 = x_1)$ for all values in the support. The conditional risk of any predictor that sees $(X_1, X_2)$ then equals the conditional risk of a predictor that sees only $X_1$, for a loss that is a function of the conditional law, because the conditional law did not change. In particular, the Bayes risk given $(X_1, X_2)$ equals the Bayes risk given $X_1$. Extra features that are conditionally independent of the label given the rest do not improve the best possible risk. They can still make a finite-sample estimate worse.""",
        code="""import numpy as np

def sample_chain(n, rng):
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=0.1, size=n)
    y = x1 + rng.normal(scale=0.5, size=n)
    return x1, x2, y
""",
        experiment="""**Hypothesis.** In `sample_chain`, $Y$ is independent of $X_2$ given $X_1$ by construction. A linear model that already has $X_1$ gains nothing from $X_2$ beyond noise. A model that sees only $X_2$ still predicts $Y$, because $X_2$ carries $X_1$.

**Protocol.** $n_{\\mathrm{train}} = n_{\\mathrm{test}} = 400$, 30 seeds. Compare test MSE of least squares on $(1, X_1)$, on $(1, X_1, X_2)$, and on $(1, X_2)$.

**Metric.** Median test MSE of the three models.

**Decision rule.** The first two medians should be close. The third can be worse but should beat a constant. A large gain from adding $X_2$ after $X_1$ means the code coupled $Y$ to the noise in $X_2$.""",
        project="""**Deliverable.** A note that states one conditional independence you believe in a problem you care about, one you do not, and the chain experiment as a positive control.

**Constraints.** The belief must name the conditioning set.

**Rubric.** You distinguish "the feature is useless alone" from "the feature is useless given the others," using the third model as the illustration.""",
        beginner="Run the chain experiment at one seed and write the three test errors.",
        university="Argue from the sampling code that the conditional law of $Y$ given $(X_1, X_2)$ ignores $X_2$.",
        research="Give a collider example: $X_1$ and $X_2$ independent, $Y$ depends on both, and conditioning on $Y$ induces dependence between $X_1$ and $X_2$. Simulate it. This is the trap in reading conditional correlations as mechanisms.",
        failures="""- Dropping a feature because its marginal correlation with $Y$ is zero, when it matters given other features (a suppressor or a component of an interaction).
- Conditioning on a consequence and calling the induced dependence a direct effect.
- Asserting the Markov property because it makes the code easier.""",
        see_also="Volume 12 for sequences. Volume 14 for state in reinforcement learning. Volume 08 for features that add nothing once a representation is fixed.",
    ),
)

chapter(
    v,
    "06_distribution_project",
    "Project: fit a law and show the misfit",
    entry(
        question="How do you attach a named distribution to data without pretending the name is true?",
        card=[
            ("Fit", "Match the parameters the family uses, on the training sample only."),
            ("Check", "A distance or a plot the family could fail: tails, zeros, skew, or a holdout log likelihood."),
            ("Baseline family", "Always include a simpler or neighboring family, not only the one you like."),
            ("Decision", "Use the family for a stated purpose, or refuse it."),
            ("Purpose", "A Gaussian approximation to a mean can be fine when a Gaussian model of individuals is absurd."),
        ],
        theory="""Fitting a distribution is estimation plus a test of the constraint the family imposes. Maximum likelihood for a Gaussian is the sample mean and the sample variance (with the $1/n$ normalization). The fit can be perfect as a moment match and still a bad model of tails. The project forces a misfit display so the card does not become an identity.

Holdout log likelihood compares families on new draws from the same process. It does not prove the winner is well specified. It says which constraint predicted the new points better. If both lose to a histogram you trust, the parametric families are the wrong summary.

Separate the purpose. Using a Gaussian approximation for the sampling distribution of a mean is the central limit theorem. Using a Gaussian to simulate individual incomes or latencies is a claim about tails and support. Write the purpose in the first line of the note so a later reader does not reuse the fitted law for a purpose you did not check.""",
        derivation="""For $Y_i$ iid $\\mathcal{N}(\\mu, \\sigma^2)$, the log likelihood is

$$-\\tfrac{n}{2}\\log(2\\pi \\sigma^2) - \\tfrac{1}{2\\sigma^2}\\sum_i (Y_i - \\mu)^2.$$

Maximizing in $\\mu$ gives $\\hat{\\mu} = \\bar{Y}$. Plugging in and maximizing in $\\sigma^2$ gives $\\hat{\\sigma}^2 = n^{-1}\\sum_i (Y_i - \\bar{Y})^2$. The unbiased variance with $n-1$ is a different functional. Do not mix them inside a likelihood comparison. For a Laplace location family with known scale, the maximizer of the likelihood in the location is a median, because the loss being minimized is absolute error. The project should show that these two location estimates diverge when the sample is skewed.""",
        code="""import numpy as np

def gaussian_mle(y):
    y = np.asarray(y, dtype=np.float64)
    mu = float(np.mean(y))
    var = float(np.mean((y - mu) ** 2))
    return mu, var

def gaussian_loglik(y, mu, var):
    y = np.asarray(y, dtype=np.float64)
    return float(-0.5 * np.sum(np.log(2 * np.pi * var) + (y - mu) ** 2 / var))
""",
        experiment="""**Hypothesis.** A Gaussian MLE fit on Laplace data assigns lower holdout log likelihood than a Laplace fit, and the reverse is true on Gaussian data.

**Protocol.** Fit on 500 points, evaluate on 2000 fresh points, 20 seeds, both generators, variance matched. For the Laplace fit, use the median and the mean absolute deviation from the median as the scale (a standard Laplace MLE when both parameters are free is acceptable if you implement it).

**Metric.** Mean holdout log likelihood by family and generator.

**Decision rule.** The matching family should win. If the Gaussian wins on Laplace data, the Laplace scale is wrong or the holdout is too small to see the tails.""",
        project="""**Deliverable.** A note with two generators, both fitted families, a tail-fraction table, holdout log likelihoods, and a decision sentence of the form "I would use family A to simulate new individual observations" or "I would use family A only to describe the mean."

**Constraints.** Training points and holdout points are disjoint. Parameters are not refit on the holdout.

**Rubric.** The decision matches the table. A win on likelihood with a failing tail check is called a partial win, not a specification.""",
        beginner="Fit a Gaussian to Gaussian data and report $\\hat{\\mu}$, $\\hat{\\sigma}^2$, and a holdout log likelihood.",
        university="Run the crossed families experiment.",
        research="Add a two-component Gaussian mixture fit by a simple EM or by a library, and show a dataset where both single Gaussians lose. Keep the EM initialization in the config hash from Volume 03.",
        failures="""- Selecting the family on the same sample you use to quote the likelihood, with no holdout and no complexity penalty.
- Using the $n-1$ variance inside a Gaussian likelihood comparison.
- Exporting a fitted individual-level Gaussian for a quantity whose support is nonnegative.""",
        see_also="Volume 05 for a fuller account of maximum likelihood. Volume 08 for mixtures.",
    ),
)
