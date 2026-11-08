"""Volumes 15–18: systems, reliability, foundation models, research practice."""

from booklib import chapter, entry, volume

v = volume(
    15,
    "Systems and production",
    "University to research",
    "A reader who can train a model and can state a data contract. This volume is an original systems guide. The Harvard CS249r volumes are the external deep reference; read them under their license, unmodified.",
    "What has to be true of the data, the job, the serving path, and the monitor before a model is a system?",
)

chapter(
    v,
    "01_lifecycle",
    "The lifecycle as a system",
    entry(
        question="Which artifacts exist between a notebook and a system another person can operate?",
        card=[
            ("Artifacts", "Dataset snapshot, config hash, code version, metrics, model weights, card, and a rollback target."),
            ("Boundary", "Training code and serving code must implement the same feature function, or they are two models."),
            ("Owner", "A named person or rotation for the metric, the data, and the rollback."),
            ("Loop", "Train, evaluate on a locked slice, deploy, monitor, and decide to keep or revert."),
            ("External", "CS249r Volumes I–IV cover single-node systems through physical AI in depth. This chapter does not replace them."),
        ],
        theory="""A model in a notebook is a function on a tensor that happened to be in memory. A system is that function plus the contract that produces the tensor, the job that refreshes the weights, the service that applies them, and the monitor that says the world has changed. Most production incidents are contract mismatches: a feature computed one way in training and another way in serving, a label that arrives late, or a model that cannot be named precisely enough to roll back.

This volume stays at the level of artifacts and checks you can run in a small lab. It is the bridge from the earlier experimental folders to an operable loop. When you need hardware rooflines, distributed training, or serving stacks in full detail, read the CS249r volumes as themselves. Do not paste them into this book. Their textbook text is CC BY-NC-ND.

The config hash from Volume 03 is the identity of a training run. The serving system must record which hash is live. A dashboard that says "the model" without a hash is not a reference.""",
        derivation="""Let $f_{\\mathrm{train}}$ be the feature function in the training job and $f_{\\mathrm{serve}}$ the feature function in the service. The deployed predictor is $g \\circ f_{\\mathrm{serve}}$. The evaluated predictor was $g \\circ f_{\\mathrm{train}}$. These are equal as functions only when $f_{\\mathrm{serve}} = f_{\\mathrm{train}}$ on the serving input domain. A unit test that runs both functions on a fixed batch and compares outputs to a tolerance is the empirical form of that equality. A mismatch larger than the tolerance blocks deployment. It is not a warning.""",
        code="""import hashlib, json

def artifact_id(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]
""",
        experiment="""**Hypothesis.** Two feature functions that differ by a unit conversion produce different predictions from the same weights, and `artifact_id` changes when the conversion flag flips.

**Protocol.** A linear model $y = 2x$ trained on meters. Serving passes centimeters unless a flag converts them. Compare predictions on a fixed batch. Hash both configs.

**Metric.** Max absolute prediction gap, and whether the hashes differ.

**Decision rule.** The gap should be large and the hashes different. If the hashes match, the flag was not in the payload. A serving test that compares predictions is the deployment gate.""",
        project="""**Deliverable.** A one-page system card for a model you have already trained: feature function, label time, artifact id, owner, metric, and rollback id. Include the unit-mismatch test.

**Constraints.** The card names one thing this volume does not measure, such as GPU throughput, and points to CS249r for that measurement rather than inventing a number.

**Rubric.** A teammate can tell which bytes are live and which bytes to restore.""",
        beginner="Hash two configs that differ by one key.",
        university="Complete the unit-mismatch gate and the system card.",
        research="Write the timeline of a label that arrives a day later. Say which training rows would have been illegal to use at decision time, in the sense of Volume 03's leakage definition.",
        failures="""- A notebook path and a serving path with no shared test.
- "Latest model" as an identifier.
- Copying CS249r chapters into this folder.""",
        see_also="Volume 03 for experiment folders. The upstream chapter of this series for how to read CS249r beside these notes.",
    ),
)

chapter(
    v,
    "02_compute_memory",
    "Compute, memory, and batching",
    entry(
        question="Which resource is binding: arithmetic, memory movement, or latency?",
        card=[
            ("Arithmetic", "FLOPs or integer ops. A matmul's count is a function of the shapes."),
            ("Memory", "Bytes moved. A model can be slow because the weights do not fit in cache or RAM, not because the FLOP count is high."),
            ("Batch", "Larger batches improve arithmetic intensity and worsen the latency of the last example in the batch."),
            ("Latency", "The time until one decision. Throughput is decisions per second. They trade off."),
            ("Measure", "A timer around the actual call, with a warmup, beats a FLOP formula you did not check."),
        ],
        theory="""Systems problems in this series start with a budget. A reference model that misses the budget is not "more accurate" in the sense the product means. The earlier volumes produced a function. This chapter asks what it costs to evaluate it.

Arithmetic intensity is operations per byte moved. Small batches and memory-heavy layers are often bandwidth bound. A formula that counts only FLOPs will rank a convolution and an attention block incorrectly if one of them is waiting on memory. Attention's quadratic score matrix, from Volume 12, is a memory object as well as a FLOP object. At long $T$ it may not fit.

Batching raises throughput and delays the first response until the batch is full or a timeout fires. Interactive systems need a timeout. Offline scoring can wait. The serving card says which one you are. A benchmark that uses batch 128 does not describe a batch-1 robot or a batch-1 reviewer tool.""",
        derivation="""A dense layer $y = x W$ with $x \\in \\mathbb{R}^{b \\times d}$ and $W \\in \\mathbb{R}^{d \\times m}$ performs about $2 b d m$ FLOPs. It reads $bd + dm$ numbers and writes $bm$ numbers. Intensity is roughly $2bdm / (bd + dm + bm)$ operations per element moved, up to the constant for the dtype size. When $b$ is small and $d, m$ are large, the $dm$ term can dominate: you reload the weights for a few outputs. Increasing $b$ amortizes the weight read. The latency of an example that arrived just after a batch closed includes the wait for the next batch. That wait is not in the FLOP count.""",
        code="""def dense_flops(batch, d_in, d_out):
    return 2 * batch * d_in * d_out

def attention_score_elements(batch, heads, seq, dim):
    # Elements in the score tensor Q K^T, not a full systems model.
    return batch * heads * seq * seq
""",
        experiment="""**Hypothesis.** Predicted dense FLOPs scale linearly in batch size, while a measured NumPy matmul on a small layer may not, because of overhead. Attention score elements grow quadratically in sequence length.

**Protocol.** Time `X @ W` for batches 1, 16, and 128 on a layer of shape $(256, 256)$, with warmup. Compute `attention_score_elements` at sequence lengths 128 and 1024.

**Metric.** Measured time per example and the quadratic ratio $(1024/128)^2 = 64$ for the score elements.

**Decision rule.** The FLOP prediction must scale exactly in the formula. The timer may not. The note should say "overhead" rather than "the formula is false" when the constant term dominates. The quadratic ratio must be exact in the element count.""",
        project="""**Deliverable.** A cost card for one model you trained: parameter count, a FLOP or element estimate for one forward pass, a measured latency at the batch you will deploy, and a sentence on whether you are latency bound or throughput bound.

**Constraints.** The batch size on the card is the deployment batch, not the largest batch that looked good.

**Rubric.** A reader can see the quadratic attention term if the model has one, or a statement that it does not.""",
        beginner="Compute dense FLOPs at two batch sizes.",
        university="Time the matmul and write the overhead caveat.",
        research="Explain, without a vendor number you cannot measure, the difference between a weight that lives in RAM and a weight that must be read from disk for every request. Name the artifact that would make the second case visible: time that scales with cold start.",
        failures="""- A throughput benchmark used as a latency promise.
- Ignoring dtype: float64 and int8 are not the same bytes.
- An attention model designed at $T = 128$ and deployed at $T = 4096$ without a new measurement.""",
        see_also="CS249r Volume I for hardware-level intensity and Volume II for scale. Volume 12 for the attention definition.",
    ),
)

chapter(
    v,
    "03_data_pipelines",
    "Data pipelines",
    entry(
        question="What does a pipeline guarantee about time, schema, and backfill?",
        card=[
            ("Snapshot", "Training reads a named snapshot, not whatever happens to be in the table now."),
            ("Schema", "A column addition, a unit change, or a null policy is a contract change with a new hash."),
            ("Backfill", "Recomputing a feature with today's code over yesterday's keys. It can create a feature the past decision did not have."),
            ("Late labels", "A label that arrives after the decision may be used for a later training snapshot and must not be used as a feature."),
            ("Test", "A golden batch with expected features, checked in the pipeline, not only in a notebook."),
        ],
        theory="""Pipelines fail by being helpful. A backfill that uses a better definition of a feature improves the training matrix and silently diverges from what serving computed last year. A snapshot freezes the bytes you trained on. A new definition is a new feature and a new model, even if the column name is unchanged.

Late labels are the supervised version of leakage. If the decision at time $t$ could not have seen the label, the label is a target, not an input. Aggregates that include events after $t$ are also inputs the decision did not have. Point-in-time joins are the implementation. A random row from a feature store is not a point-in-time join.

The golden batch is the known case for data. It is small, hand-checked, and run whenever the pipeline code changes. A distributional monitor, later in this volume, catches drift. It does not catch a wrong join that has been wrong since the first day, if the wrongness is stable.""",
        derivation="""Let a feature at decision time $t$ be $\\phi_t = \\Phi(\\mathrm{events}_{\\le t})$. A backfill that computes $\\Phi'(\\mathrm{events}_{\\le T})$ for $T > t$ using a new function $\\Phi'$ is a different random variable. Training on $\\Phi'$ and serving $\\Phi$ is the unit-mismatch bug with a schema. The equality test is the same: compute both on the golden keys and compare. A null policy is part of $\\Phi$. Filling nulls with zero in training and dropping nulls in serving changes the population as well as the number.""",
        code="""def point_in_time(events, decision_time):
    return [e for e in events if e["time"] <= decision_time]
""",
        experiment="""**Hypothesis.** A feature equal to the count of events up to $t$ differs from a count that includes a late event, and a golden batch catches the difference.

**Protocol.** Three events, one after the decision. Compare `point_in_time` counts with a full count.

**Metric.** The two counts.

**Decision rule.** They must differ. A pipeline test that uses the full count as the expected value of the decision-time feature encodes the bug as the specification. The expected value must be the decision-time count.""",
        project="""**Deliverable.** A tiny pipeline: raw events, a point-in-time feature, a snapshot id, and a golden test. Include one backfill that you refuse to mix into the old snapshot.

**Constraints.** The snapshot id changes when the feature code changes.

**Rubric.** The golden test fails if a future event leaks into the feature.""",
        beginner="Run the three-event example by hand.",
        university="Add the golden test and the snapshot id.",
        research="Describe a training-serving skew that comes from a library default, such as a categorical encoder's handling of unseen levels, and the golden row that would catch it.",
        failures="""- Training on a live view that moves during the job.
- A backfill that overwrites the only copy of the old feature.
- Nulls filled differently in two languages.""",
        see_also="Volume 03 for contracts. Volume 07 for leakage. CS249r for production data systems at scale.",
    ),
)

chapter(
    v,
    "04_serving_monitoring",
    "Serving and monitoring",
    entry(
        question="What should a service check before it returns a prediction, and what should a monitor check after?",
        card=[
            ("Schema gate", "Reject or default when the input fails the contract. Do not silently score a NaN."),
            ("Version", "Every response can be tied to a model hash."),
            ("Timeout", "A latency budget with a defined fallback."),
            ("Monitor", "Input distribution, output distribution, and, when labels arrive, a delayed outcome metric."),
            ("Alert", "A threshold on a predeclared statistic, with a human action. An alert with no action is a log."),
        ],
        theory="""Serving is the feature function plus the gates. A gate that fails open and returns a prediction for a malformed input will look available and be wrong. A gate that fails closed must have a fallback the product understands: a previous model, a constant decision, or an error. The fallback is a second policy and needs its own metric.

Monitoring without labels can still see input shift and output shift. It cannot see that the mapping from inputs to outcomes broke while both margins look normal. When labels arrive late, schedule a delayed job that joins predictions to outcomes on the decision id. That job is the production form of the test set. It is biased if unlabeled cases are dropped without a reason.

Dashboards follow Volume 03's reporting rules. A panel without a denominator, a baseline, or a window is not an alert source. The alert threshold is chosen from a quiet period or from a loss budget, not from the first bad day you want to dramatize.""",
        derivation="""Let $p_t$ be the fraction of requests whose features fail the schema in hour $t$, and let $\\hat{y}_t$ be the mean prediction among accepted requests. A change in $p_t$ means the input contract is breaking. A change in $\\hat{y}_t$ with stable inputs means the model or the feature code changed, or a real shift in the accepted population occurred. These are different incidents. An alert on prediction mean alone conflates them. Log $p_t$ beside $\\hat{y}_t$. When outcomes $y$ arrive, the delayed loss is an average over the joinable subset. If joinability depends on $y$, the delayed loss is biased. Check the join rate.""",
        code="""def gate(features, required):
    missing = [k for k in required if k not in features or features[k] is None]
    if missing:
        return {"ok": False, "fallback": "constant", "missing": missing}
    return {"ok": True, "missing": []}
""",
        experiment="""**Hypothesis.** A request missing a required key takes the fallback and is counted in the failure fraction, and it does not produce a model score. A complete request does.

**Protocol.** Two requests and a constant fallback value you record.

**Metric.** Gate decision, whether a score was emitted, and the hourly failure fraction in a toy log of 100 requests with 5 missing keys.

**Decision rule.** The failure fraction should be 0.05. If malformed requests emit scores, the gate is advisory and does not meet this chapter's contract.""",
        project="""**Deliverable.** A serving function with a gate, a version hash in the response, a fallback, and a monitor table: failure fraction, mean output, and a delayed loss on a synthetic label that arrives later for only some rows. State the join rate.

**Constraints.** The fallback is a constant or a previous hash, not an unlogged exception.

**Rubric.** You can reconstruct which hash served a given request id.""",
        beginner="Implement `gate` and a constant fallback.",
        university="Add the monitor table and the delayed join.",
        research="Define a latency budget at the 99th percentile and a timeout policy. Explain why the mean latency can look fine while the budget is missed.",
        failures="""- Scoring NaNs as if they were numbers.
- A monitor that only exists the day after an incident.
- Dropping unjoined labels and reporting a loss as if it were the population loss.""",
        see_also="Volume 07 for the metric. Volume 16 for when the monitor's metric is the wrong harm.",
    ),
)

chapter(
    v,
    "05_reproducibility",
    "Reproducibility in production",
    entry(
        question="What would you need to rebuild today's prediction tomorrow?",
        card=[
            ("Code", "A version identifier. A branch name that moves is not an identifier."),
            ("Data", "A snapshot id the training job recorded."),
            ("Numbers", "Seeds or a statement that the training is nondeterministic and the metric band you accept."),
            ("Environment", "Library versions that affect numerics, when you depend on them."),
            ("Proof", "A rerun that meets a tolerance, or an honest statement that you cannot rerun."),
        ],
        theory="""Reproducibility is the ability to get the same decision from the same inputs, or to explain the band of decisions you accept. Bitwise equality is sometimes available for a small NumPy model and sometimes not available for a distributed job. The card should not claim bitwise equality you did not test.

The production question is narrower than the research question. Can you name the bytes that served this request, and can you train a successor from a named snapshot? If the snapshot was overwritten, you can still have logs of predictions and no ability to explain them. Logging the features for a sample of requests is expensive and is the only way to debug a feature that is gone.

A tolerance belongs in the golden test. Floating-point reductions across different batch sizes can change a prediction in the last bits and should not flip a high-stakes threshold without warning. If your decision is a threshold on a probability, test stability around the threshold, not only the mean squared difference of logits.""",
        derivation="""Let $y = g_h(x)$ be the prediction of hash $h$ at input $x$. A rerun produces $g_{h'}(x)$. The reproducibility claim on a golden set $G$ is $\\max_{x \\in G} |g_h(x) - g_{h'}(x)| \\le \\tau$ when $h$ is supposed to equal $h'$. If the hashes differ, you are comparing different models and the tolerance is the wrong tool. Log the hash first. Then compare. A threshold decision adds the check that $g_h(x) - t$ and $g_{h'}(x) - t$ have the same sign for $x$ near the boundary. A small $\\tau$ can still flip a sign if $|g(x) - t| < \\tau$.""",
        code="""def decisions_match(scores_a, scores_b, threshold, tol):
    for a, b in zip(scores_a, scores_b):
        if abs(a - b) > tol:
            return False
        if (a >= threshold) != (b >= threshold) and min(abs(a - threshold), abs(b - threshold)) <= tol:
            return False
    return True
""",
        experiment="""**Hypothesis.** Two identical score lists match. A perturbation smaller than `tol` that crosses the threshold is flagged by the sign check even when you might have accepted the absolute error.

**Protocol.** Scores `[0.50, 0.80]`, threshold 0.5, tolerance 0.02, and a second list `[0.49, 0.80]`.

**Metric.** The boolean from `decisions_match`.

**Decision rule.** The identical list matches. The perturbed list must not match under this function if you include the sign check as written. If you decide threshold flips inside `tol` are acceptable, change the function and the card together. Do not leave them in conflict.""",
        project="""**Deliverable.** A rerun note for one small training job: code version, snapshot id, seed, metric band, and a golden serving test with a tolerance and a threshold check.

**Constraints.** If you cannot rerun, the note says what was discarded, not a fake hash.

**Rubric.** The serving test fails when the unit conversion from the lifecycle chapter is reintroduced.""",
        beginner="Run the threshold stability example.",
        university="Document a real rerun of a Volume 03 experiment folder.",
        research="List the nondeterministic sources you accept in a GPU training job and the metric band that replaces bitwise equality. Be concrete: data order, atomic reductions, or library versions.",
        failures="""- A floating branch name as a version.
- A tolerance so wide that a unit bug passes.
- Deleting the snapshot after the paper or the launch.""",
        see_also="Volume 18 for research reproducibility, which adds the question of whether someone else can rerun the argument, not only the service.",
    ),
)

chapter(
    v,
    "06_system_project",
    "Project: a batch-scoring service",
    entry(
        question="Can you wrap one trained model in a snapshot, a gate, a versioned score file, and a monitor?",
        card=[
            ("Input", "A snapshot of features with a schema."),
            ("Score", "A file or response that includes the model hash and the fallback flag."),
            ("Gate", "Malformed rows take the fallback and are counted."),
            ("Monitor", "Failure fraction, score distribution, and a delayed metric if you have labels."),
            ("Rollback", "A previous hash you can select without retraining."),
        ],
        theory="""The project is intentionally batch, not a networked service, so you can finish it on a laptop. The artifacts are the same ones a service would log. A batch job that overwrites yesterday's scores without a hash is the anti-pattern. Write a new file per hash.

Use a model you understand from an earlier volume, even a linear one. The systems point will be clearer than if you spend the week training. Include the unit-mismatch gate from the first chapter as a test that must fail closed.

The delayed metric can be synthetic: hide the label until a second script joins it. That script is the monitor. If you skip it, the project is a scorer, not a system.""",
        derivation="""A batch run defines a function from a snapshot id and a model hash to a score file. Rollback is selecting a previous pair $(snapshot, hash)$ and re-emitting scores, or selecting a previous score file if the inputs match. If the feature snapshot has changed, rescoring the old model on new features is a new experiment, not a rollback. The run book must say which of the two you mean. The project note picks one and tests it.""",
        code="""import json
from pathlib import Path

def write_scores(path, model_hash, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model_hash": model_hash, "rows": rows}, indent=2), encoding="utf-8")
""",
        experiment="""**Hypothesis.** Scoring the same snapshot with two hashes produces two files, and a rollback procedure that copies the previous file restores the previous decisions. Scoring a changed snapshot with the old hash does not restore the old decisions when the features changed.

**Protocol.** A two-point linear model, a unit change in the second snapshot, and `write_scores`.

**Metric.** Whether the restored file matches the first file byte for byte, and whether the changed snapshot changes a prediction.

**Decision rule.** Both parts of the hypothesis should hold. A rollback implementation that retrains is a different procedure and must be labeled as retraining.""",
        project="""**Deliverable.** The batch scorer, schema gate, hashed score files, monitor script, rollback test, and a one-page run book.

**Constraints.** The run book states what this system does not do, including online latency and distributed training, and points to CS249r for those topics.

**Rubric.** A classmate can run the rollback test and the golden feature test without asking you which file is live.""",
        beginner="Write and read a score file with a hash.",
        university="Complete the scorer, monitor, and rollback test.",
        research="Add a canary: score 10 percent of rows with a new hash, compare delayed loss, and define the rule that promotes or rejects the canary. Use synthetic labels.",
        failures="""- Overwriting the only score file.
- A fallback that is not logged.
- Retraining and calling it rollback.""",
        see_also="Volume 16 before this system is aimed at people. Volume 18 if the run book is the basis of a paper's reproducibility appendix.",
    ),
)

v = volume(
    16,
    "Reliability, privacy, and harm",
    "University to research",
    "A reader who can evaluate a model and describe a deployment. This volume is about failure of the decision, not about a new architecture.",
    "When the data distribution, the adversary, the privacy constraint, or the harm differs from the average metric, what do you measure instead?",
)

chapter(
    v,
    "01_shift",
    "Distribution shift",
    entry(
        question="What exactly changed between training and deployment, and which metric moves?",
        card=[
            ("Covariate", "The law of $X$ changes. The conditional $P(Y \\mid X)$ stays. A reweighted test can represent the new $X$ if you know the weights."),
            ("Label", "The conditional $P(Y \\mid X)$ changes. Reweighting $X$ is not enough."),
            ("Concept", "Often used for a changing conditional. Say which conditional, of $Y$ given $X$, or of a human label given the event."),
            ("Slice", "A shift can be a new mixture of slices you already measured."),
            ("Action", "Collect data, reweight, restrict the domain, or refuse to deploy. The word retrain is not specific."),
        ],
        theory="""Shift is a change in a named law. Without the name, "it shifted" is not a diagnosis. Covariate shift preserves the conditional you learned and changes where you evaluate it. If the new region of $X$ was absent from training, you are extrapolating, and the conditional estimate there is a hope. Label shift and concept change alter the relationship the model was fit to. A monitor on $X$ alone can miss them, which is why Volume 15 asked for delayed outcomes.

The practical measurement is a test set drawn from the deployment law, or a reweighting of an old test set that you can justify. A synthetic shift, such as brightness in Volume 11, is a unit test of the procedure. It is not a certificate about a new hospital or a new year. Those need their own data.

Report the old metric and the new metric side by side. A model that is best on the old test set and worst on the new one has not "generalized." The selection rule has to see the deployment distribution or you are optimizing the wrong risk.""",
        derivation="""Under covariate shift, $p_{\\mathrm{new}}(x, y) = p_{\\mathrm{new}}(x) p_{\\mathrm{old}}(y \\mid x)$. The new risk is $\\mathbb{E}_{\\mathrm{old}}[w(X) \\ell(f(X), Y)]$ with $w(x) = p_{\\mathrm{new}}(x) / p_{\\mathrm{old}}(x)$ when the support condition holds. An estimated $w$ that is wrong in the tails dominates the weighted average. Clipping $w$ changes the target risk. If the conditional changed, no $w(x)$ recovers the new risk from old labeled data, because the old $y$ values are from the wrong conditional.""",
        code="""import numpy as np

def weighted_mean(losses, weights):
    losses = np.asarray(losses, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("weights must be nonnegative and not all zero")
    return float(np.sum(weights * losses) / np.sum(weights))
""",
        experiment="""**Hypothesis.** On data where $Y = X + \\varepsilon$ and the test $X$ is shifted to a region with the same conditional, a linear model trained on the old region still has small loss if it captured the line, while a nearest-neighbor model degrades. A change in the slope at deployment hurts both, and reweighting old data does not create the new slope.

**Protocol.** Train on $X \\in [-1, 0]$. Covariate test: $X \\in [0, 1]$ with the same slope. Concept test: slope flips sign. Include a constant baseline.

**Metric.** Test MSE under both shifts.

**Decision rule.** The concept shift should be worse for the linear model than a mere covariate shift into a neighboring interval, if the linear fit is accurate. If nearest neighbors beat the line on the original interval and collapse outside it, that is the extrapolation lesson. Record what you actually measure.""",
        project="""**Deliverable.** A shift note for one model: the named change, the old metric, the new metric, the constant baseline under the new law, and a decision to reweight, collect, restrict, or refuse.

**Constraints.** Do not retune on the shifted test set and then call the number a measure of the unshifted procedure.

**Rubric.** The note says whether the conditional was assumed constant, and what data would falsify that assumption.""",
        beginner="Compute a weighted mean and show a large weight dominating it.",
        university="Run the covariate versus slope-change comparison.",
        research="State a support failure: $w(x)$ undefined when the new $x$ has zero density under the old law. Give a feature that makes your model's support fail, such as a new category level.",
        failures="""- Calling every failure "shift" without a law.
- Reweighting with unbounded weights and no effective sample size.
- Selecting the model on the deployment test set.""",
        see_also="Volume 11 for image shifts. Volume 15 for monitors that see only $X$.",
    ),
)

chapter(
    v,
    "02_robustness",
    "Robustness and adversarial examples",
    entry(
        question="Robust to what perturbation, under what threat model?",
        card=[
            ("Threat", "Who can change the input, by how much, and whether they see the model."),
            ("Ball", "A set of allowed perturbations, such as an $\\ell_\\infty$ budget $\\varepsilon$ on pixels or tokens."),
            ("Attack", "A search for a perturbation in the ball that changes the decision. It is a lower bound on badness: failure to find an example is not a proof of robustness."),
            ("Defense", "A training or detection method. It must be tested against an attack that knows the defense, or the result is incomplete."),
            ("Ordinary noise", "A random perturbation is a different experiment from an adversary."),
        ],
        theory="""Robustness is a property of a pair: a model and a set of perturbations. A model can be robust to sensor noise and fragile to a one-pixel change an adversary chose. Reporting "robust" without the set is the same error as reporting accuracy without the base rate.

An attack that fails tells you that this search did not find a bad input. A stronger search might. The honest table is the attack's success rate at a stated budget, plus the clean accuracy. A defense that drops clean accuracy below the useful line has a cost. Put the cost in the table.

Random augmentation during training, from Volume 10, is not an adversarial evaluation. It may or may not improve the adversarial number. Measure it. Defenses that obfuscate gradients can look strong against gradient attacks and fail against a simpler search. Include at least one attack that does not rely on the gradient if you claim a defense.""",
        derivation="""For a linear score $w^\\top x + b$, the perturbation $\\delta$ with $\\|\\delta\\|_\\infty \\le \\varepsilon$ that most increases the score in a chosen direction is $\\delta_j = \\varepsilon \\mathrm{sign}(w_j)$ when you want to increase $w^\\top x$. The change in the score is $\\varepsilon \\lVert w \\rVert_1$. A small $\\varepsilon$ flips the decision when the clean margin is smaller than that change. This is the threat model "bounded $\\ell_\\infty$, full knowledge of $w$." A different norm gives a different worst $\\delta$. The formula is the unit test for any attack code you write on a linear model before you attack a network.""",
        code="""import numpy as np

def linf_attack_linear(w, margin, eps):
    direction = eps * np.sign(w)
    new_margin = margin - eps * np.sum(np.abs(w))
    return direction, float(new_margin)
""",
        experiment="""**Hypothesis.** On a linear classifier, the $\\ell_\\infty$ step reduces the margin by $\\varepsilon \\lVert w \\rVert_1$, and a random perturbation of the same budget reduces it less on average.

**Protocol.** A random $w$ in $\\mathbb{R}^{20}$, several margins, $\\varepsilon = 0.1$. Compare the analytic change with a Monte Carlo average of random $\\delta$ in $[-\\varepsilon, \\varepsilon]^{d}$.

**Metric.** Margin change under the attack and under noise.

**Decision rule.** The attack must match the formula and must be worse than the average noise. If your network attack cannot beat random noise on a linear model, the attack code is wrong.""",
        project="""**Deliverable.** The linear unit test, plus one attack on a small network or a statement that you only claim random-noise robustness, with clean accuracy beside the robust number.

**Constraints.** The threat model is in the first paragraph. The attack budget is not chosen by looking at which budget makes the paper exciting. Pick it from a sensor or a product constraint, or label it as arbitrary.

**Rubric.** You do not claim robustness beyond the attacks you ran.""",
        beginner="Compute the linear $\\ell_\\infty$ margin change by hand for a two-dimensional $w$.",
        university="Compare the attack with random noise.",
        research="Explain gradient masking in a paragraph: a defense that makes $\\nabla_x$ useless while a decision-based or transfer attack still succeeds. You do not need to break a published defense; describe the measurement that would reveal masking.",
        failures="""- Robust accuracy with no clean accuracy.
- An attack budget that includes perturbations a camera cannot produce, presented as a physical threat, or the reverse.
- Testing only the attack the defense was designed on, if you had access to the defense's internals during tuning, without saying so.""",
        see_also="Volume 10 for augmentation. Volume 15 for a monitor that might detect an attack only if the attack leaves the input distribution.",
    ),
)

chapter(
    v,
    "03_privacy",
    "Privacy",
    entry(
        question="What can a released model or statistic reveal about a training row, and what promise would prevent it?",
        card=[
            ("Leak", "A membership inference: an attacker's guess that a particular row was in the training set, better than chance."),
            ("Memorization", "A model that can be prompted or queried to emit a training string."),
            ("Aggregate", "A mean or a model is still a function of the private rows."),
            ("Differential privacy", "A promise about the output distribution of a mechanism: a single row's participation changes probabilities by a bounded factor."),
            ("Cost", "Noise or clipping that enforces the promise usually hurts accuracy. The curve is the result."),
        ],
        theory="""Privacy in this volume is about what the released artifact reveals, not about the virtue of the team. A model that memorizes a rare sentence can reveal it to someone who queries carefully. A statistic of a small group can reveal a member. Removing names is not a privacy analysis if the remaining features identify the person.

Differential privacy is a precise promise with a parameter budget. It is not a synonym for "we added noise." The promise is about neighboring datasets that differ by one row. A mechanism that satisfies it can still be useless if the noise is huge, and a mechanism that feels private can fail it. If you do not implement a mechanism with a known proof, do not claim the guarantee. You may still measure memorization empirically and call it an empirical attack, which is a lower bound on leakage.

The project-scale question is whether you should release the model, a coarser aggregate, or nothing. Accuracy on a held-out set does not answer it.""",
        derivation="""A randomized mechanism $M$ is $(\\varepsilon, \\delta)$-differentially private if for neighboring datasets $D, D'$ and measurable sets $S$,

$$P(M(D) \\in S) \\le e^{\\varepsilon} P(M(D') \\in S) + \\delta.$$

The inequality is a constraint on the mechanism, not on a single run's output. The Laplace mechanism adds noise to a function with a known sensitivity $\\Delta$: scale $\\Delta/\\varepsilon$ for pure $\\varepsilon$-differential privacy in the scalar case, under the usual definition. A function whose sensitivity you do not know, such as an unbounded neural training run, does not become private because you add a little noise at the end. Clipping is how some training mechanisms bound sensitivity. Without the clip, the theory does not apply.""",
        code="""import numpy as np

def laplace_mech(value, sensitivity, epsilon, rng):
    if epsilon <= 0 or sensitivity < 0:
        raise ValueError("epsilon and sensitivity must be valid")
    scale = sensitivity / epsilon
    return float(value + rng.laplace(0.0, scale))
""",
        experiment="""**Hypothesis.** On a counting query, the Laplace mechanism's outputs spread by about the scale $\\Delta/\\varepsilon$, and a tiny noise that ignores sensitivity fails a simple neighboring-dataset distinguishability check you can simulate.

**Protocol.** True count 100, sensitivity 1, $\\varepsilon = 0.5$, many draws. Also a broken mechanism that adds $\\mathcal{N}(0, 10^{-6})$ noise. Compare the distributions on counts 100 and 101 only descriptively: plot or summarize overlap. Do not claim a formal test you did not calibrate.

**Metric.** Empirical standard deviation versus the Laplace scale $\\sqrt{2}\\, \\Delta/\\varepsilon$, and a qualitative note on the broken mechanism.

**Decision rule.** The Laplace spread should match the formula's order. The broken mechanism's outputs on 100 and 101 should be easy to tell apart. If you cannot see that, the simulation is wrong. The point of the broken mechanism is that "some noise" is not the definition.""",
        project="""**Deliverable.** A privacy note for one artifact you might release: what a training row contains, who the attacker is, an empirical extraction or membership check if it is safe and ethical on synthetic data, and either a correctly scoped Laplace mechanism on a simple statistic or an explicit refusal to claim differential privacy.

**Constraints.** Synthetic rows only, unless you have the right to use the real ones for this attack. Do not attack a real person's record as a demonstration.

**Rubric.** The note's claim matches the mechanism. "We care about privacy" is not a result.""",
        beginner="Add Laplace noise to a count and see the scale change with $\\varepsilon$.",
        university="Compare the broken tiny-noise mechanism on neighboring counts.",
        research="Describe membership inference at the level of an attacker's score and a true-positive rate at a low false-positive rate. Say why accuracy of the attacker is the wrong summary when most people are non-members.",
        failures="""- Hashing emails and calling the model private.
- Adding Gaussian noise with no sensitivity and no budget.
- A public model card that includes unique text from the training set.""",
        see_also="Volume 13 for memorization distance. Volume 18 for what not to put in a paper's appendix.",
    ),
)

chapter(
    v,
    "04_fairness",
    "Fairness and measurement",
    entry(
        question="Which disparity did you measure, on which groups, and what can the number not decide?",
        card=[
            ("Group", "A set of people or cases defined by a feature you are willing to use in the audit. The definition is an ethical and a measurement choice."),
            ("Base rate", "Groups can have different outcome rates. A metric that ignores this will be misread."),
            ("Error rates", "False-negative and false-positive rates can differ even when accuracy matches, and the reverse."),
            ("Threshold", "A single threshold can create disparate error rates. Group-specific thresholds are a different policy with their own costs."),
            ("Limit", "A parity number is not a proof of justice. It is one measurement of one definition."),
        ],
        theory="""Fairness metrics conflict. Equalizing accuracy, equalizing false-negative rates, and equalizing predictive values cannot all hold in general when base rates differ and the predictor is not perfect. A reference note picks the disparity that matches the harm, reports the others as context, and does not claim a theorem that all of them improved.

Groups are not discovered by clustering the model's errors and naming the clusters. They are specified by a question: who bears the error? Proxy attributes are dangerous. A postal code can stand in for a protected attribute and for many other things. If you audit only the proxy, say so.

The data that define group membership may be missing, wrong, or illegal to use. An audit that requires collecting a sensitive attribute can itself be the harm. The note should consider that cost. Sometimes the right measurement is a process constraint you can check without new sensitive data, such as "this feature is not an input," which is weaker and clearer.

A model card from Volume 07 grows a disparity table here. The table has denominators. A gap on a group of size 5 is a case review, not a rate.""",
        derivation="""Let group $g$ have base rate $p_g$ and false-positive and false-negative rates $f_g$ and $m_g$. Precision is $p_g(1-m_g) / (p_g(1-m_g) + (1-p_g)f_g)$. Two groups with equal $f$ and $m$ and unequal $p$ have unequal precision. Two groups with equal precision and unequal $p$ generally have unequal error rates. This is algebra, not a moral ranking. It is why a memo that says "we equalized accuracy" has not said what happened to the false-negative rate. Write the table of $p_g, f_g, m_g$ and the policy threshold.""",
        code="""import numpy as np

def group_rates(y, pred, group, g):
    mask = np.asarray(group) == g
    y = np.asarray(y).astype(bool)[mask]
    pred = np.asarray(pred).astype(bool)[mask]
    if y.size == 0:
        raise ValueError("empty group")
    pos = y.sum()
    neg = y.size - pos
    fp = np.sum(pred & ~y)
    fn = np.sum(~pred & y)
    return {
        "n": int(y.size),
        "base_rate": float(y.mean()),
        "fpr": float(fp / neg) if neg else float("nan"),
        "fnr": float(fn / pos) if pos else float("nan"),
    }
""",
        experiment="""**Hypothesis.** Two groups with the same score distribution within the positive class and within the negative class, but different base rates, show equal true and false positive rates at one threshold and unequal precision.

**Protocol.** Simulate scores that depend on the label and not on the group. Vary the base rate. Apply one threshold.

**Metric.** FPR, FNR, and precision by group.

**Decision rule.** Error rates should match up to sampling noise. Precision should not. If error rates differ, the score generator accidentally depends on the group. This simulation is the algebraic point, not a claim that real errors are independent of group.""",
        project="""**Deliverable.** A disparity table for a synthetic or a permitted dataset, with base rates, error rates, sizes, and a paragraph on which disparity you would optimize and which you cannot also satisfy. Include a feature you refuse to use as an input, with the reason.

**Constraints.** Do not invent real demographic claims from a toy generator. Label the data as synthetic when they are.

**Rubric.** The table includes $n$ per group. The prose does not say the model is fair. It says which measured gap remains.""",
        beginner="Compute precision from the formula for two base rates and equal error rates.",
        university="Run the simulation and write the table.",
        research="Discuss a threshold policy that equalizes false-negative rates and the incentive it creates if people can choose which group label is recorded. Keep it concrete and short.",
        failures="""- A fairness claim from accuracy alone.
- Groups of size 10 reported as percentages without $n$.
- Using a protected attribute as a model input in order to "fix" a disparity, without a policy analysis.""",
        see_also="Volume 07 for metrics. Volume 15 for a monitor that should break down by the slices you decided to audit, where that is legitimate.",
    ),
)

chapter(
    v,
    "05_safety",
    "Safety cases",
    entry(
        question="What argument would justify running this system, and what evidence is still missing?",
        card=[
            ("Hazard", "A concrete way the system can cause a loss: a wrong action, a missed case, a leak, a feedback loop."),
            ("Control", "A mechanism that reduces that hazard: a threshold, a human review, a fallback, a limit on actions."),
            ("Evidence", "A measurement that the control works under a stated condition."),
            ("Assumption", "A condition the argument needs and that you have not tested."),
            ("Stop", "A condition under which the system is withdrawn."),
        ],
        theory="""A safety case is a structured argument, not a list of hopes. For each hazard you name a control and the evidence that the control is effective, and you list the assumptions. An assumption that is false breaks the case even if the in-domain accuracy is high. Volume 15's fallback, Volume 14's reward, and this volume's shift and privacy notes are inputs to the case.

Feedback loops are a hazard the earlier metrics miss. A model that denies loans changes who comes back with a label. A recommender changes what is clicked. The logged data then follow the system's own policy. Off-policy evaluation from Volume 14 is relevant, and a static test set is not the world after deployment.

The case should be readable by someone who can stop the launch. If every hazard ends with "the model is state of the art," there is no case. State of the art is not a control.""",
        derivation="""Let $H$ be a hazard and $C$ a control that is claimed to reduce $P(H)$ below a bound $b$ when assumption $A$ holds. The evidence is an experiment whose outcome is informative about $P(H)$ under $A$. If the experiment was run only when $A$ is true by construction, the case must not cite it for worlds where $A$ is false. Write $A$ beside the number. A fallback control has its own hazard: the fallback decision can be harmful. Include it as a row, not as an implementation detail.""",
        code="""def safety_row(hazard, control, evidence, assumption, withdraw_if):
    fields = [hazard, control, evidence, assumption, withdraw_if]
    if any(not str(x).strip() for x in fields):
        raise ValueError("a safety row cannot be empty")
    return {
        "hazard": hazard,
        "control": control,
        "evidence": evidence,
        "assumption": assumption,
        "withdraw_if": withdraw_if,
    }
""",
        experiment="""**Hypothesis.** A row with an empty assumption fails the helper, and a completed row for the batch scorer's unit mismatch names a withdraw condition you can actually test.

**Protocol.** Call `safety_row` with a blank assumption and with a full row whose evidence is the golden feature test.

**Metric.** Whether the blank row raises, and whether the full row's `withdraw_if` is a check rather than a slogan.

**Decision rule.** "Withdraw if users complain" is weaker than "withdraw if the golden feature test fails or the delayed loss exceeds a set budget." Prefer the check. If you keep a human signal, say who reviews it and how fast.""",
        project="""**Deliverable.** A safety case with at least three hazards for a system you described in Volume 15 or for a purely advisory predictor. Each row has evidence you have or an explicit "not yet measured."

**Constraints.** One hazard must be a feedback loop or a fallback harm. One assumption must be marked untested.

**Rubric.** A reader can see what would force withdrawal. The case does not conclude "safe." It concludes "acceptable under these assumptions" or "not acceptable yet." """,
        beginner="Fill one row for a schema-gate failure.",
        university="Complete three rows tied to measurements from earlier volumes.",
        research="Describe a closed-loop data collection problem in which the logging policy is the deployed policy, and the static accuracy number no longer estimates the next period's risk.",
        failures="""- Hazards written as "model error" with no mechanism.
- Evidence cited from a different threat model.
- A withdraw condition nobody is assigned to watch.""",
        see_also="Volume 14 for reward hacking as a hazard. Volume 18 for the difference between a safety case and a paper.",
    ),
)

chapter(
    v,
    "06_audit_project",
    "Project: a shift and harm audit",
    entry(
        question="Can you audit one model for shift, a robustness or privacy limit, and a disparity or safety row without hiding behind the average metric?",
        card=[
            ("Average", "The in-domain primary metric, with $n$ and a baseline."),
            ("Shift", "One named shift and the metric under it."),
            ("Limit", "One robustness budget or one privacy statement that does not over-claim."),
            ("Harm", "One disparity table or one safety row, chosen because it matches the decision."),
            ("Decision", "Ship, restrict, collect data, or stop."),
        ],
        theory="""The audit is the last applied project in the series before the research volumes. It uses a model you already understand so the new work is the measurement of failure. An audit that only restates the best test number is a brochure. The decision paragraph must be able to say stop.

Use synthetic data if real data would involve people you should not instrument for a class project. Label the synthetic world as synthetic. The skill is the structure of the audit, and it transfers. The numbers do not transfer to a real population you did not measure.

If two checks conflict, leave the conflict visible. A robust model that worsens a disparity, or a private mechanism that misses the accuracy budget, is a result. The decision names which constraint wins and who is exposed if you are wrong.""",
        derivation="""The audit estimates several functionals: in-domain risk, shifted risk, an attack success rate or a privacy parameter, and a group-specific error. These estimators have different sample sizes. A decision rule that treats a gap of 0.01 in every column as equal is ignoring the standard errors and the harms. Write the rule in advance: for example, refuse deployment if the shifted risk exceeds a bound or if a group of size at least $n_0$ has a false-negative gap above a bound. Applying the rule after seeing which bound is convenient is the peeking problem from Volume 05.""",
        code="""def audit_decision(shifted_risk, shift_budget, disparity_gap, disparity_budget, group_n, min_n):
    if group_n < min_n:
        disparity_flag = "insufficient_n"
    else:
        disparity_flag = "exceeds" if disparity_gap > disparity_budget else "within"
    shift_flag = "exceeds" if shifted_risk > shift_budget else "within"
    return {"shift": shift_flag, "disparity": disparity_flag}
""",
        experiment="""**Hypothesis.** The decision helper distinguishes a small group, which should not produce a rate-based exceedance, from a large group with a large gap, and it flags a shifted risk above the budget independently.

**Protocol.** Three calls: small $n$, large gap with large $n$, and a shifted risk above budget with disparity within budget.

**Metric.** The returned flags.

**Decision rule.** The flags must match the written rule. If your prose recommends "ship" while a flag says exceeds, the prose loses.""",
        project="""**Deliverable.** An audit memo for one model: average metric, shift, robustness or privacy limit, disparity or safety row, the prewritten decision rule, and the resulting decision.

**Constraints.** Synthetic or permitted data. Claims match the evidence. No borrowed demographic story.

**Rubric.** The memo can say stop. A memo that cannot say stop is not an audit.""",
        beginner="Run the decision helper on the three cases.",
        university="Write the full memo around a model from an earlier project.",
        research="Add a feedback-loop assumption to the memo and a measurement you would need after deployment to test it. Do not pretend the static audit already did.",
        failures="""- An audit written after choosing the decision.
- Private or sensitive real records used as toy examples.
- A conclusion of "fair and robust" from an average accuracy.""",
        see_also="Volume 17 for audits of models you did not train and cannot fully inspect. Volume 18 for turning an audit into a reproducible note.",
    ),
)
