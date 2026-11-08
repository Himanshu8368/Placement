"""Volumes 17–18: foundation models and research practice."""

from booklib import chapter, entry, volume

v = volume(
    17,
    "Foundation models",
    "Research",
    "A reader who has trained small models, evaluated generators, and written an audit. The claims in this volume are narrower than the marketing around large models.",
    "What can you responsibly conclude from a model trained by someone else on data you did not see?",
)

chapter(
    v,
    "01_scale",
    "Scale, data, and claimed emergence",
    entry(
        question="What does a scaling curve show, and what does a jump in a metric not show?",
        card=[
            ("Scale", "Parameters, tokens, and compute. A curve that varies only one of them confounds the others."),
            ("Smooth", "Many error curves fall smoothly on a log compute axis. A plot can look like a jump because the metric is a thresholded accuracy."),
            ("Emergence", "A claim that a capability appears suddenly. Check the metric, the resolution of the x-axis, and the task selection."),
            ("Data", "The mixture, the cutoff date, and the deduplication policy are part of the model."),
            ("Your run", "A small model you train is the place to learn the measurement. A downloaded model is the place to apply it, not to skip it."),
        ],
        theory="""Large pretrained models are empirical objects. Scaling laws, when they hold, are fits of loss against compute on a family of training runs. They predict loss, not a product requirement, and they do not transfer to a new data mixture without a new fit. A figure with three points is not a law.

Threshold metrics create apparent emergence. If the underlying per-token loss improves smoothly, an exact-match accuracy can stay at zero until the loss crosses a cliff and then jump. Report the underlying loss or a partial-credit metric before you announce a phase change. Also beware task selection: a paper that shows the tasks that jumped, and not the tasks that did not, is a selected sample.

You usually cannot reproduce the pretraining. You can still demand the model id, the date, the evaluation protocol, and a baseline a smaller or simpler system can run. Volume 12's bag-of-tokens baseline remains a legitimate comparison on classification. Being embarrassed by it is informative.""",
        derivation="""Suppose a continuous score $s$ improves as $s = a - b \\log c$ with compute $c$, and a reported accuracy is $1\\{s > \\tau\\}$. The indicator jumps at one compute threshold even though $s$ is smooth. Changing $\\tau$ moves the jump. A metric with partial credit, such as mean token accuracy, tracks $s$ more closely. When you see a step-function plot, ask which threshold created it and whether the x-axis is coarse. This does not prove every sudden capability is a metric artifact. It is the first alternative to rule out.""",
        code="""import numpy as np

def threshold_curve(log_compute, tau):
    score = 1.0 - 0.15 * np.asarray(log_compute, dtype=np.float64)
    return (score > tau).astype(np.float64), score
""",
        experiment="""**Hypothesis.** A smooth score and a thresholded version of the same score tell different stories about suddenness.

**Protocol.** Evaluate `threshold_curve` on a fine grid and on three coarse x positions, at two thresholds.

**Metric.** Where the indicator flips, and the smoothness of the raw score.

**Decision rule.** The raw score must not jump. The indicator must. A write-up of this toy that calls the indicator an emergent capability has made the mistake the chapter is about. If your real model's curve remains a jump on a graded metric at high x-resolution, the toy does not explain it away. Say that.""",
        project="""**Deliverable.** A one-page reading of a public scaling or capability plot you can access: what was held constant, what metric was thresholded, what baseline is missing, and what you would need to reproduce even one point.

**Constraints.** Do not invent numbers you did not read. Quote the axis labels.

**Rubric.** The page distinguishes a loss curve from a capability claim.""",
        beginner="Plot or tabulate the toy threshold curve.",
        university="Write the reading note on a real public figure.",
        research="Separate in-distribution loss scaling from out-of-distribution task performance. Give one reason a model with a better pretraining loss can be worse on your deployment shift.",
        failures="""- Three noisy points fit with a law.
- A benchmark suite reported only where the new model wins.
- Compute estimated so loosely that two runs are not comparable.""",
        see_also="Volume 05 for fitting a curve you are not willing to validate. Volume 13 for likelihood versus sample quality, which also diverge.",
    ),
)

chapter(
    v,
    "02_adaptation",
    "Adaptation",
    entry(
        question="Which adaptation method matches the amount of labeled data and the access you have?",
        card=[
            ("Prompt", "Inputs only. Fragile to wording. No gradient. Version the model and the prompt."),
            ("Probe", "A small head on frozen features. The Volume 08 protocol."),
            ("Low-rank update", "A constrained $\\Delta$, as in the language volume. Needs gradients and weights or an adapter API."),
            ("Full fine-tune", "Highest capacity to forget the original model and to memorize a small set."),
            ("Selection", "The adaptation method is chosen on validation. The test set scores the choice once."),
        ],
        theory="""Adaptation is supervised learning on top of a representation you did not choose feature by feature. The same split, baseline, and paired-gap rules apply. A prompt that was edited twenty times on the test examples is a test-set search. Log the attempts or hold out a validation slice of prompts.

Frozen probes tell you whether the representation is linearly readable for your label. If the probe ties a bag-of-tokens model, the foundation model is not earning its cost on this task. That is a successful negative result.

Full fine-tuning on a tiny set needs the regularization chapter: small step, early stopping, and a check that training loss is not the only number that moved. Catastrophic forgetting of a capability you still need is a shift in the model's behavior on a previous task. If you care about that task, measure it. Do not assume the original checkpoint's card still describes the fine-tuned weights.""",
        derivation="""Let $R_{\\mathrm{val}}(m)$ be the validation risk of adaptation method $m$ after its internal knobs are chosen. You pick $m^\\star = \\arg\\min_m R_{\\mathrm{val}}(m)$. The test risk of $m^\\star$ is the number you may publish as the risk of the selected adaptation, if the test set did not enter the prompts, the early stopping, or the method choice. The gap $R_{\\mathrm{val}}(\\mathrm{probe}) - R_{\\mathrm{val}}(\\mathrm{full})$ estimates whether the extra capacity helped on new draws from the same split. It does not estimate performance on a new domain. Run the shift audit if the domain is the question.""",
        code="""def adaptation_record(method, val_metric, n_train, model_id):
    if not model_id or n_train < 0:
        raise ValueError("model id and n_train are required")
    return {
        "method": method,
        "val_metric": float(val_metric),
        "n_train": int(n_train),
        "model_id": model_id,
    }
""",
        experiment="""**Hypothesis.** On a synthetic labeling task that is a keyword, a bag-of-tokens probe matches or beats a method that only sees a paraphrased prompt you deliberately make vague. On a task that needs a long-range phrase, the keyword method fails and a sequence method can win.

**Protocol.** Reuse the Volume 12 generators if you still have them. Treat "prompt" as a fixed lookup table you wrote before seeing the test split: a simulated prompted classifier with a known error rate is acceptable if you do not have local weights. The point is the record and the comparison, not a particular vendor.

**Metric.** Validation metric for each method in `adaptation_record`.

**Decision rule.** The keyword task should not crown an opaque model over the bag of tokens. If it does, the bag baseline is underfit or the test leaked.""",
        project="""**Deliverable.** A comparison card: majority baseline, bag of tokens or linear probe, and one heavier adaptation if you have legal access to it. Model id, date, validation metric, and a contamination caveat.

**Constraints.** No test-set prompt editing. If you only have an API, store the date and do not claim you fine-tuned.

**Rubric.** A negative result that keeps the simpler method is an acceptable outcome.""",
        beginner="Fill `adaptation_record` for a majority baseline and a keyword rule.",
        university="Complete the comparison on the two synthetic tasks.",
        research="Define a forgetting check: a small set of previous-task examples scored before and after fine-tuning. State the budget at which you would stop fine-tuning.",
        failures="""- Prompt iteration on the test set.
- A fine-tune with no simple baseline.
- Reporting the original model's benchmark numbers as if they were your fine-tuned model's numbers.""",
        see_also="Volume 12 for fine-tuning and prompts. Volume 08 for probes.",
    ),
)

chapter(
    v,
    "03_alignment",
    "Preferences and alignment",
    entry(
        question="Whose preference does a post-training procedure encode, and what did it trade away?",
        card=[
            ("Preference data", "Pairs or ratings from a process you should be able to name: who labeled, under what instructions."),
            ("Objective", "A loss that increases the probability of preferred responses and decreases others. It is not a truth certificate."),
            ("Reward model", "A learned proxy for the preferences. It can be gamed by the policy."),
            ("Tradeoff", "Helpfulness, harmlessness, and task accuracy move differently. Report the ones you care about."),
            ("Not included", "A moral certificate. A refusal behavior is a policy, measured on a test suite you specify."),
        ],
        theory="""Post-training that uses human or model-written preferences changes the conditional distribution of answers. The change is as good as the preference process. If the labelers disagree, a single reward model averages them and hides the disagreement. If the labelers share a blind spot, the model inherits it. The audit of a preference dataset is a data contract: who is absent, which instructions were ambiguous, and which behaviors were never rated.

A policy trained against a reward model can find a high reward that a person would not endorse, the reward-hacking pattern from the reinforcement-learning volume. The evaluation has to include checks the reward model does not grade: factuality on a keyed set, refusal on a hazard set, and usefulness on the original task. Optimizing the reward and reporting the reward is circular.

This chapter does not teach you to bypass safety policies. It teaches you to measure a stated policy. A test suite of disallowed requests is scored by whether the system refuses, not by whether you can trick it for sport. If you are building the system, the safety case from Volume 16 names the hazards the preference data were supposed to cover and the ones they do not.""",
        derivation="""A simple preference loss on a pair $(y^+, y^-)$ given a prompt $x$ increases a score $s_\\theta(x, y^+)$ relative to $s_\\theta(x, y^-)$. The implied target is "rank preferred outputs higher under $s_\\theta$," not "match the world." If $s_\\theta$ is a reward model fit to the same pairs, a policy that maximizes $s_\\theta$ can drift off the distribution of $y$ the reward model saw. The KL penalty toward a reference policy is a constraint that limits that drift. Removing it changes the objective. Report whether it was present and the coefficient.""",
        code="""import numpy as np

def pairwise_accuracy(score_pos, score_neg):
    score_pos = np.asarray(score_pos, dtype=np.float64)
    score_neg = np.asarray(score_neg, dtype=np.float64)
    return float(np.mean(score_pos > score_neg))
""",
        experiment="""**Hypothesis.** A score that copies a preference label achieves high pairwise accuracy on those pairs and can still rank a rewarded but empty answer above a correct answer on a keyed factual item you did not put in the preference set.

**Protocol.** Build ten synthetic pairs where the preferred answer is longer, and five factual items where the correct answer is short and a long empty answer scores higher on "length." Compute pairwise accuracy on the preference pairs and accuracy on the factual items under the length score.

**Metric.** The two numbers.

**Decision rule.** High preference accuracy with low factual accuracy is the illustration. A note that reports only pairwise accuracy has not evaluated the reward hack. Do not use this toy as evidence about any real company's model. It is a measurement pattern.""",
        project="""**Deliverable.** A measurement note: the preference process you are willing to describe, the pairwise metric, a task metric that the preference model does not define, and a hazard the preference data do not cover.

**Constraints.** Do not collect real people's private judgments for a toy. Synthetic preferences are enough. Do not write exploits.

**Rubric.** The note says whose preference was modeled and what was not asked of the labelers.""",
        beginner="Compute pairwise accuracy for a perfect ranking and for a reversed one.",
        university="Complete the length-versus-factuality toy.",
        research="State the KL-to-reference term and the qualitative effect of setting its coefficient to zero. Connect it to the offline RL support problem: the policy can leave the region where the reward model is meaningful.",
        failures="""- Calling a preference-tuned model truthful because labelers preferred confident prose.
- Reporting the reward on the training pairs as evaluation.
- A refusal test you iterated on until the demo looked strict, with no held-out hazards.""",
        see_also="Volume 14 for reward hacking. Volume 16 for the safety case that should list uncovered hazards.",
    ),
)

chapter(
    v,
    "04_eval_foundation",
    "Evaluating foundation models",
    entry(
        question="How do you evaluate a model when you do not control training and the test may already have been seen?",
        card=[
            ("Contamination", "Overlap between the evaluation items and the pretraining or post-training data. Exact match is a lower bound on leakage."),
            ("Freshness", "Items written after the model's known data cutoff, when you know the cutoff."),
            ("Protocol", "Temperature, system prompt, number of samples, and parser. These change the score."),
            ("Baseline", "A simpler system on the same items, with the same parser."),
            ("Slice", "The average hides the item type where the model systematically fails."),
        ],
        theory="""Leaderboards saturate and leak. A model can score well because the items were in its training mixture. Exact-string overlap catches the obvious cases and misses paraphrases. A cutoff date helps only if you trust it and the items are actually new. When you cannot rule contamination out, the card says "contamination unknown," which lowers the claim from "the model learned this skill" to "the model scored this number under this protocol."

The protocol is part of the estimator. A chain-of-thought prompt, a tool, and a retry loop are different systems from a single sample. Compare them as different systems. Parsing a free-form answer into a label introduces errors. Check the parser on a sample of outputs. A wrong parse can invent a gain or a loss.

Slices matter more than a new average. If every failure is an arithmetic item or a negation, the next experiment is about that slice, not about a larger model. Volume 07's mass calculation applies to item types.""",
        derivation="""Let an item be contaminated if a normalized copy appears in the training mixture. The observed accuracy $a$ is a mixture $a = \\rho a_{\\mathrm{seen}} + (1-\\rho) a_{\\mathrm{unseen}}$ when you could partition items, with $\\rho$ the contaminated fraction. If you only have a lower bound $\\rho_{\\min}$ from exact match, you cannot recover $a_{\\mathrm{unseen}}$ without more assumptions. You can report $a$ and $\\rho_{\\min}$. Dropping contaminated items and reporting the rest estimates accuracy on the residual set, which is a different, possibly harder or easier set. Say which one you did.""",
        code="""def exact_overlap(eval_strings, train_strings):
    train = {s.strip().lower() for s in train_strings}
    flags = [s.strip().lower() in train for s in eval_strings]
    rho = sum(flags) / len(flags) if flags else float("nan")
    return {"n": len(flags), "rho_min": rho, "flags": flags}
""",
        experiment="""**Hypothesis.** Exact overlap detects copied items and misses a paraphrase. A reported accuracy that includes copies is higher than accuracy on the residual set when copies are answered correctly and new items are not.

**Protocol.** Five eval strings, two copied into the train set, one paraphrase, two new. Assign correct on copies and incorrect on the rest, including the paraphrase.

**Metric.** `rho_min` and accuracy with and without the flagged copies.

**Decision rule.** The paraphrase must not be flagged by exact overlap. The two accuracies must differ. A contamination statement that says "we deduplicated" because $\\rho_{\\min} = 0$ on a paraphrased set overclaims the check.""",
        project="""**Deliverable.** An evaluation protocol for a model you can query or for a small local model: prompt text, decode settings, parser, baseline, exact-overlap check against the data you have, and a slice table.

**Constraints.** If you cannot check overlap with pretraining data, write "unknown" in that cell.

**Rubric.** The protocol is specific enough that a rerun would not quietly change temperature or prompt.""",
        beginner="Run `exact_overlap` on the toy strings.",
        university="Write the full protocol around a real or synthetic item set.",
        research="Describe a cutoff test: items dated after the model freeze, and the selection bias if you only keep items the authors thought were interesting after seeing model behavior.",
        failures="""- A leaderboard number with no protocol.
- Deleting hard items after the run.
- A parser that strips the correct answer into the wrong bucket on a slice you never checked.""",
        see_also="Volume 12 for generation metrics. Volume 16 for harm slices that should appear in the same table when the items concern people.",
    ),
)

chapter(
    v,
    "05_agents",
    "Agents as systems",
    entry(
        question="What changes when a model can call tools and take a sequence of actions?",
        card=[
            ("Loop", "Observe, propose an action, execute under a policy, observe again."),
            ("Tool", "A function with a contract, a permission, and a failure mode. The model does not get every tool."),
            ("State", "The transcript is not automatically a Markov state of the world."),
            ("Stop", "A budget on steps, money, and side effects. A loop without a stop is not a design."),
            ("Eval", "Task success on held-out goals, not the model's confidence, and not a demo transcript you edited."),
        ],
        theory="""An agent is a system from Volume 15 wrapped around a policy from Volume 14, with a foundation model as one possible policy class. The new hazards are tool side effects and long horizons. A wrong email, a deleted file, or a purchased item is not a classification error you can average away. The safety case needs a permission boundary: which tools exist in the evaluation environment, which exist in production, and what the fallback does when the model proposes a forbidden tool.

The transcript can hide state. If a tool's effect is not written back into the observation, the next step acts on a stale belief. That is the partial-observability bug. Log the tool arguments and the tool results. A transcript that stores only the model's prose cannot be debugged.

Evaluate on a fixed suite of tasks with a checker. A sandbox that does not match production will miss the failures that depend on real permissions. Say which environment you measured. Held-out tasks are the test set. Editing a task until the agent succeeds is training on the test set.""",
        derivation="""Let a task have a checker $c(h) \\in \\{0, 1\\}$ on a terminal transcript $h$, and let the policy interact for at most $T$ steps. The success rate is the expectation of $c$ under the policy and the environment. A step limit $T$ is part of the policy-environment pair. Raising $T$ after seeing failures changes the system. Tool permissions enter the transition kernel: a forbidden tool call should transition to an error observation, not to the side effect. If the evaluation harness allows the side effect and production does not, or the reverse, the measured success is for the harness.""",
        code="""def tool_dispatch(name, args, allow):
    if name not in allow:
        return {"ok": False, "error": "forbidden", "name": name}
    if name == "add":
        return {"ok": True, "value": float(args["a"]) + float(args["b"])}
    return {"ok": False, "error": "unknown", "name": name}
""",
        experiment="""**Hypothesis.** A dispatch allow-list blocks a tool that is not permitted and executes a permitted pure function. A checker that looks only at the model's final sentence can mark success when the tool was never called.

**Protocol.** Call `add` when it is allowed and when it is not. Define a toy task "compute 2+2" with a checker on the tool result and a weaker checker on a sentence that says "4" without a tool call.

**Metric.** Whether the forbidden call is refused, and whether the two checkers disagree on a transcript that never called the tool.

**Decision rule.** The allow-list must refuse. The two checkers must be able to disagree. Use the tool-result checker for the project if the task is a tool task. The sentence checker is the leaky metric.""",
        project="""**Deliverable.** A sandboxed agent loop with one or two pure tools, a step budget, a task suite with a checker, a baseline that does not use tools, and a safety row for a forbidden tool.

**Constraints.** No tool with an irreversible external side effect. No attempt to defeat a real system's safety policy.

**Rubric.** The success rate is computed by the checker on held-out tasks. A transcript in the note is an illustration, not the result.""",
        beginner="Implement the allow-list and the two checkers.",
        university="Complete the loop on a few arithmetic or lookup tasks with a step budget.",
        research="Describe a feedback loop in which the agent's earlier action changes the next observation's distribution, and why the offline log of a different policy does not score the new one. This is Volume 14 inside the tool environment.",
        failures="""- An edited demo as the evaluation.
- A tool with production credentials in a notebook.
- A checker that grades the model's self-assessment.""",
        see_also="Volume 14 for the policy and the offline gap. Volume 16 for the safety case. Volume 15 for logs and rollback of the tool-using system.",
    ),
)

chapter(
    v,
    "06_foundation_project",
    "Project: an evaluation harness",
    entry(
        question="Can you evaluate a large or simulated foundation model with a harness that would catch contamination, a weak baseline, and a protocol change?",
        card=[
            ("Items", "A fixed set with ids, slices, and gold answers or checkers."),
            ("Systems", "At least two: a simple baseline and the model under test."),
            ("Protocol file", "Prompt, decode settings, model id, date."),
            ("Contamination", "Exact overlap against the corpora you have, or the explicit status unknown."),
            ("Report", "Overall metric, slice table, and the decision the number supports."),
        ],
        theory="""The harness is the research instrument. It should be boring. Given an item file and a system, it writes a score file with a hash of the protocol. Changing the prompt without changing the hash is the reproducibility failure from Volume 15.

You may not have the compute to train a foundation model. You do not need to. A local small model or a deterministic simulated system still exercises the harness. If you query a real model, the harness stores the raw outputs so a parser change can be rerun without a new query, and you respect the provider's terms.

The decision is a claim about this item set under this protocol. It is not a claim that the model understands a domain. Volume 18 will ask you to write that limitation in a form a reviewer can see.""",
        derivation="""The score file is a function of the item ids, the protocol hash, and the system id. A comparison between two systems is valid when the item ids and the checker match and the protocols are the ones you intended. If one system was allowed three retries and the other was not, the protocol hashes should differ and the comparison is confounded. The harness can still run it, and the report must not call it a matched comparison. This is the ablation rule from Volume 10 applied to evaluation.""",
        code="""import hashlib, json

def protocol_hash(protocol: dict) -> str:
    raw = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]
""",
        experiment="""**Hypothesis.** Changing the temperature in the protocol changes the hash. Reordering JSON keys does not. A simulated system that always returns the gold answer scores perfectly and is rejected as a serious system by a note that also runs a baseline with a known error rate.

**Protocol.** Hash two equivalent dicts and one temperature change. Score a perfect stub and a baseline that misses one slice.

**Metric.** Hash equality, and slice metrics.

**Decision rule.** Key order must not affect the hash. The perfect stub must not be the only system in the report. If the baseline is absent, the harness run is incomplete.""",
        project="""**Deliverable.** The harness, an item file with at least two slices, two systems, a contamination cell, a protocol hash, and a short decision.

**Constraints.** Raw outputs are saved. The checker is versioned. No test-set tuning presented as a final number.

**Rubric.** A second person can rerun the scorer from the saved outputs without calling the model again.""",
        beginner="Hash two protocols and explain a mismatch.",
        university="Complete the harness on synthetic items.",
        research="Add a shift slice whose items use a new format. Show that overall accuracy can hide a collapse on that slice, using the mass or the slice rate from Volume 07.",
        failures="""- A harness that only stores the final accuracy.
- A protocol that lives in an untitled chat rather than in the hashed file.
- A claim about "the model" when only one prompt was tried.""",
        see_also="Volume 18 for writing this harness up as a research note. Volume 16 if the items can harm people when mishandled.",
    ),
)

v = volume(
    18,
    "Research practice",
    "Research",
    "A reader who has finished the projects they cite. This volume is about making a claim that someone else can check.",
    "How do you read, reproduce, and write a result without laundering a guess into a finding?",
)

chapter(
    v,
    "01_reading",
    "Reading a paper into this reference",
    entry(
        question="What do you extract from a paper so it becomes a usable, skeptical reference note?",
        card=[
            ("Claim", "One sentence with a quantity, a population, and a comparison."),
            ("Assumptions", "What must be true for the claim to transfer, in the language of earlier volumes."),
            ("Protocol", "Data split, metric, baselines, compute, and selection of the reported number."),
            ("Threats", "The two or three ways the claim would fail. Leakage, shift, and a weak baseline are common."),
            ("Place", "Which chapter of this series the method extends, and which project of yours would change if the claim is true."),
        ],
        theory="""Reading for a reference is not summarizing the abstract. The abstract is the authors' framing. Your note restates the estimand. A paper that says a method "works" has not named a risk. Translate it into a metric, a split, and a baseline. If you cannot, the paper is not yet usable, and the note says that.

Place the method in the map of this series. A new optimizer belongs next to Volume 10's ablation rules. A new robustness paper belongs next to a threat model. This prevents a technique from floating free of the assumptions that make it meaningful.

Write the threats before you write the praise. A threat is specific: the test set shares users with the training set, the baseline was not tuned, the metric is accuracy at a 99 percent base rate. "More research is needed" is not a threat.""",
        derivation="""A paper reports $\\hat{\\Delta}$, an estimated gap. Your note should say which expectation $\\Delta$ is supposed to be, the unit of randomization, and whether $\\hat{\\Delta}$ was chosen among many gaps. If the paper's table has ten metrics and the abstract quotes one, the selection is part of the estimator. The note records the selection. Recomputing a confidence interval as if the metric were predeclared would be a new, false precision. Volume 05's peeking warning is the formal version.""",
        code="""def paper_note(claim, estimand, baseline, threat):
    for part in (claim, estimand, baseline, threat):
        if not str(part).strip():
            raise ValueError("paper notes need a claim, estimand, baseline, and threat")
    return {"claim": claim, "estimand": estimand, "baseline": baseline, "threat": threat}
""",
        experiment="""**Hypothesis.** A note that omits the baseline fails the helper, and a filled note for a hypothetical accuracy improvement of 2 points on an unbalanced test names a threat a Volume 07 reader would recognize.

**Protocol.** Reject an empty baseline. Accept a note whose threat is the base rate or a shared-entity split.

**Metric.** Whether the helper enforces the fields, and whether the threat is specific.

**Decision rule.** A threat that only says "bias" is incomplete. Require a mechanism. This is a writing standard you can apply to the next paper you read.""",
        project="""**Deliverable.** Two paper notes in this format, on papers or technical reports you can access, each placed in a volume of this series. One threat per note must be something you could check if you had the code, even if you do not run it yet.

**Constraints.** Do not invent experimental details. If the paper does not state the split, the note says "split not stated."

**Rubric.** A classmate can tell what would change their project if the claim is true, and what would make the claim inapplicable.""",
        beginner="Fill one note for the least-squares project in Volume 01, as if it were a paper, and list a real threat.",
        university="Write the two external paper notes.",
        research="Add a what-I-would-rerun-first line: the smallest experiment that could falsify the claim in your setting.",
        failures="""- A summary that copies the abstract's adjectives.
- A threat list that could be attached to every paper unchanged.
- Placing a method in the series without naming the assumption it inherits.""",
        see_also="The reproducibility chapter in this volume. Volume 07 if the paper's number chose the metric after the fact.",
    ),
)

chapter(
    v,
    "02_reproducibility",
    "Reproducing and revising a result",
    entry(
        question="What counts as a reproduction, and what do you owe a negative result?",
        card=[
            ("Reproduction", "The same claim, the same protocol, within a tolerance, on data you are allowed to use."),
            ("Replication", "The same claim under a justified change: new sample, new seed, new implementation."),
            ("Negative", "A failed reproduction is a result if the protocol was followed. It is not an embarrassment to hide."),
            ("Tolerance", "Predeclare how close is close enough. Do not shrink the tolerance after you see the gap."),
            ("Credit", "The original claim keeps its citation. Your note adds what happened when you ran it."),
        ],
        theory="""Reproduction is how a reference stays honest. If you cannot rerun a result that your later chapter depends on, the later chapter depends on a rumor. A small rerun with a stated tolerance is enough for a teaching reference. A failed rerun should stay in the note, with the difference in environment you could identify.

Change one thing when you revise. A new dataset and a new metric and a new architecture is a new project, not a reproduction that "didn't work." Volume 10's ablation discipline is the research discipline.

Negative results need the same artifacts as positive ones: config, metric, and the known-case test that shows your harness works. A buggy rerun is not a negative result about the claim. The debug checklist has to pass on a control problem before the negative result counts.""",
        derivation="""Let $\\hat{\\theta}$ be the published number and $\\tilde{\\theta}$ yours. You predeclare a tolerance $\\tau$ in the units of $\\hat{\\theta}$. The reproduction succeeds if $|\\tilde{\\theta} - \\hat{\\theta}| \\le \\tau$ and the protocol differences are empty or listed as irrelevant. If you list a protocol difference that could move the metric by more than $\\tau$, you do not have a reproduction. You have a confounded comparison. Record it as such. The decision to adopt the method then rests on your protocol, not on the paper's number.""",
        code="""def reproduction_status(published, ours, tol, protocol_differences):
    gap = abs(float(published) - float(ours))
    if protocol_differences:
        return {"status": "not_a_clean_reproduction", "gap": gap, "differences": list(protocol_differences)}
    status = "reproduced" if gap <= tol else "not_reproduced"
    return {"status": status, "gap": gap, "tol": tol}
""",
        experiment="""**Hypothesis.** A gap inside the tolerance with no protocol differences is marked reproduced. The same gap with a different split is marked not clean. A gap above the tolerance is not reproduced, and lowering the tolerance after the fact is a protocol change you can represent by calling the function twice.

**Protocol.** Published value 0.80, yours 0.83, tolerance 0.05 and then 0.01.

**Metric.** The status field.

**Decision rule.** The statuses must match the rule. A lab note that says "roughly reproduced" when the function says not reproduced has to quote the gap and the tolerance instead of the adverb.""",
        project="""**Deliverable.** A reproduction or a negative-result note for one small project from this series: rerun your own least-squares or training experiment from the folder alone, with a tolerance, after a week or on a second seed. If you change a knob, list it.

**Constraints.** The known-case test for the harness passes. A failed known case means you do not interpret the scientific gap.

**Rubric.** The status word matches `reproduction_status`. The folder is enough for someone else to rerun.""",
        beginner="Call the helper on a successful and a failed toy pair.",
        university="Rerun one of your own experiment folders.",
        research="Write what you would publish if a third party failed to reproduce your result: the log you would ask for, and the claim you would withdraw if the known-case test passed on their side.",
        failures="""- Changing seeds until the number matches.
- A negative result from a harness that cannot fit the known case.
- Citing a paper as reproduced because you reimplemented something inspired by it.""",
        see_also="Volume 15 for production reruns. Volume 03 for the folder layout that makes this chapter possible.",
    ),
)

chapter(
    v,
    "03_writing",
    "Writing a research note",
    entry(
        question="What is the smallest document that carries a claim, the evidence, and the limits?",
        card=[
            ("Question", "The decision or estimand, before the method."),
            ("Method", "Enough to rerun, including the baseline and the selection rule."),
            ("Result", "The number, the uncertainty, the $n$, and the comparison."),
            ("Limit", "The population and the threat model you did not cover."),
            ("Claim strength", "A verb that matches the evidence: estimates, suggests, or does not show."),
        ],
        theory="""A research note in this series is a long-form reference entry with a single claim. It is not a short textbook chapter and not a pitch. The reader should be able to stop at the result table and know the comparison, then read the method to see whether they believe it.

Order the document by the logic of the argument, not by the order you suffered in. The question and the predeclared metric come first. The method follows. Exploratory plots are labeled exploratory. A surprise you found after looking is allowed in a section called exploratory. It is not allowed to replace the predeclared result in the first paragraph.

Verbs are part of the science. "Causes" requires a design that identifies a causal effect. "Predicts" requires a test set that matches the prediction claim. "Correlates" is available when that is all you measured. Volume 16's audit language is the right tone for limits. Do not end with a limitation paragraph that withdraws the claim you just overstated. Match them.""",
        derivation="""Let the predeclared comparison be $\\Delta$ and the exploratory comparisons be $\\Delta_j$ for $j = 1, \\ldots, m$. The note's lead result is $\\hat{\\Delta}$ with its interval. The exploratory section can show $\\hat{\\Delta}_j$ without a family-wise guarantee. If you promote one $\\hat{\\Delta}_j$ into the lead because it was significant, the lead is false. A revision of the note after a failed $\\Delta$ is honest when it says the predeclared comparison was null and the new question is future work, or when it labels the new lead as selected from $m$ looks.""",
        code="""def lead_sentence(estimand, comparison, n, limit):
    text = (
        f"On n={n}, we estimate {estimand} for {comparison}. "
        f"This does not establish the result beyond {limit}."
    )
    if n <= 0 or not limit.strip():
        raise ValueError("a lead sentence needs n and a limit")
    return text
""",
        experiment="""**Hypothesis.** A lead sentence that includes a limit is accepted, and a sentence you draft with the word "proves" can be rewritten into the helper's pattern without losing the number.

**Protocol.** Write one overclaimed sentence about a small experiment you actually ran, then rewrite it with `lead_sentence` or the same structure.

**Metric.** Whether the rewritten sentence names $n$, the comparison, and a limit.

**Decision rule.** If the rewritten sentence is still broader than the experiment, the limit clause is decorative. Tighten it. This is a writing experiment with a visible artifact.""",
        project="""**Deliverable.** A research note of at most a few pages on one project from this series: question, predeclared metric, method, result, exploratory section if needed, and limits. Use the lead-sentence discipline.

**Constraints.** The note links the experiment folder. It does not introduce a metric that was not computed.

**Rubric.** A reader who disagrees with you can point to the comparison and the limit, not to a mood.""",
        beginner="Rewrite one overclaimed sentence about the constant baseline.",
        university="Write the full note on a project you completed.",
        research="Add an explicit not-claimed list: causal effect, fairness, privacy, or robustness, unless you actually measured them. Refer to the volume that would be required.",
        failures="""- A lead result chosen after seeing several metrics.
- A method section that omits the baseline.
- Limits that contradict the first paragraph.""",
        see_also="Volume 03 for figures that support a sentence. Volume 16 for limits that are about people.",
    ),
)

chapter(
    v,
    "04_open_problems",
    "A map of open problems",
    entry(
        question="Which questions in this series are still open in a way that a project could bite on?",
        card=[
            ("Open", "A question with a measurable gap, not a topic label."),
            ("Bit", "The smallest experiment that would reduce the gap."),
            ("Dependence", "Which earlier result the question needs, and which failure of that result would dissolve it."),
            ("Non-goal", "A question that is too broad to finish, recorded as a program rather than a project."),
            ("Living", "This map should be edited when you close a question. A static list of buzzwords is not a map."),
        ],
        theory="""An open problem in a reference is a hole you can point to. "Understand intelligence" is not a hole you can point to. "Measure whether this representation helps at a fixed label budget on this shift" is a hole. The map in this chapter is a template you fill with your own holes after the projects you have run. The examples below are illustrations of the grain, not an assignment of the field's priorities.

Each entry names the volume that already constrains it. A robustness question that does not name a threat model is sent back to Volume 16. A systems question that does not name a binding resource is sent back to Volume 15. This is how the series stays one reference rather than a pile of courses.

When you close a problem, leave the entry and add the result. Deleting the question erases the path. A reference reader wants to see what was answered.""",
        derivation="""Treat the map as a set of estimands that lack an accepted measurement in your notes. For an entry $q$, a bite is an experiment whose outcome would change the status of $q$ from open to answered or to ill-posed. If no outcome would change the status, $q$ is a topic, not a problem. Write the outcome that would change your mind before you spend the data. This is the same predeclaration rule as everywhere else in the series, applied to the research agenda.""",
        code="""def problem(question, bite, volume, dissolve_if):
    if not all(str(x).strip() for x in (question, bite, volume, dissolve_if)):
        raise ValueError("a problem needs a bite and a dissolution condition")
    return {
        "question": question,
        "bite": bite,
        "volume": volume,
        "dissolve_if": dissolve_if,
    }
""",
        experiment="""**Hypothesis.** An entry without a bite is rejected, and a concrete entry about evaluation under shift can be dissolved if the conditional $P(Y \\mid X)$ changed and you only reweighted $X$.

**Protocol.** Try to construct `problem` with an empty bite. Then write one full problem whose `dissolve_if` refers to a violated assumption from Volume 16.

**Metric.** Whether the helper enforces the bite, and whether the dissolution condition is an assumption rather than a mood.

**Decision rule.** Keep the problem only if a negative bite result would teach you something named in `dissolve_if` or in the question. Otherwise the experiment cannot fail informatively.""",
        project="""**Deliverable.** Five open problems in your own words, each with a bite you could run with the skills in this series, and one non-goal you explicitly refuse to list as a project.

**Constraints.** At least one problem must be a negative-result-friendly measurement, such as a baseline that might win. At least one must name an ethical or safety constraint that would stop the bite.

**Rubric.** Another reader can tell what data the bite needs and what result would close the question.""",
        beginner="Turn one failed experiment of yours into a problem entry with a bite.",
        university="Write the five entries across different volumes.",
        research="Add a dependence between two entries: a representation-learning question that is ill-posed until an evaluation question is answered. Order them.",
        failures="""- A list of model names.
- A bite that is "train a bigger model" with no metric.
- A dissolution condition that cannot actually occur.""",
        see_also="Every earlier volume. The upstream chapter for problems that are already treated in a book you should read rather than rederive.",
    ),
)

chapter(
    v,
    "05_upstream",
    "Upstream books and how to use them",
    entry(
        question="Which external books should you read beside this series, and what are you not allowed to do with their text?",
        card=[
            ("This series", "Original notes. They do not contain those books' chapters."),
            ("CS249r", "Machine Learning Systems, four volumes. Textbook text is CC BY-NC-ND 4.0. Read it, share it unmodified, do not fold edited text into this book."),
            ("D2L", "Dive into Deep Learning. Text is CC BY-SA 4.0. You may adapt it if you keep attribution and the share-alike duty. This series does not copy it."),
            ("Think books", "Think Python, Think Stats, Think Bayes are CC BY-NC-SA. Non-commercial adaptations require share-alike. We point at them for foundations."),
            ("Hugging Face course", "Apache 2.0. You may reuse it with attribution and a notice of changes. The language volume here is original and smaller."),
            ("Not open", "A free PDF is not permission to republish. Famous textbooks you can read online are not therefore upstream you can integrate."),
        ],
        theory="""The series was built so you can learn the path without pasting someone else's book into your repository. The external books are better than these notes on their own ground: D2L for a broad implemented deep-learning course, the Think books for computational foundations, the Hugging Face course for a library-centered language pipeline, and CS249r for systems from the chip to physical AI. Read them at the point in the path where the volume guide sends you.

Integration has a legal meaning. Unmodified CS249r volumes can sit beside your notes as a separate work with their license intact. Editing them into your chapters is a derivative the ND license does not grant. D2L and the Think books can be adapted only if you accept their share-alike terms for the adapted material. The clean design, and the one this repository uses, is citation and a reading note in your own words, plus your own experiments.

A reading note uses the paper-note format from this volume. It does not reproduce long passages. If a proof is short and you need it, rederive it. If you cannot rederive it, say you are taking the result as a citation and point to the source. That is more honest than a close paraphrase.""",
        derivation="""Suppose you adapt a CC BY-SA chapter and mix it into a note you want to keep under a different license. The adapted portion remains under BY-SA, and if the mixture is one work rather than a separable collection, the share-alike duty can extend further than you intended. The practical rule used here is separation: external text stays in the external repository; your notes stay here; the link is a citation. No derivation in this chapter weakens an upstream license. If you need a combined work, read the license text before you merge repositories.""",
        code="""UPSTREAM = [
    {"id": "cs249r", "use": "read unmodified", "license": "CC BY-NC-ND 4.0 textbook text"},
    {"id": "d2l", "use": "read; adapt only with BY-SA obligations", "license": "CC BY-SA 4.0"},
    {"id": "think", "use": "read; non-commercial adaptation is share-alike", "license": "CC BY-NC-SA 4.0"},
    {"id": "hf-course", "use": "reuse with attribution and change notices", "license": "Apache-2.0"},
]
""",
        experiment="""**Hypothesis.** You can state, without opening an upstream PDF and copying, which book you would read for a systems question and which you would read for a first deep-learning implementation, and you can name the license constraint that stops you from pasting either into this repository.

**Protocol.** Answer in a note of ten lines, then check your answer against the upstream page of this series.

**Metric.** Whether the systems pointer is CS249r, the implementation pointer can be D2L, and both license constraints are present.

**Decision rule.** If the note says "I'll merge the PDFs and fix the wording later," it fails. Separation is the requirement.""",
        project="""**Deliverable.** A personal reading order: which upstream book accompanies which volume you have already studied, plus one paper note in your own words on a chapter you actually read. No pasted paragraphs.

**Constraints.** The note includes the license and the access date or edition.

**Rubric.** A reader can follow your path without receiving a copied chapter they would not have the right to redistribute.""",
        beginner="List the four upstream works and their licenses from memory, then check.",
        university="Write the reading order tied to volumes you have done.",
        research="Pick one result you are relying on and rederive it. If you cannot, mark the dependence as a citation with an unreviewed proof, which is a real status, not a failure of character.",
        failures="""- Pasting ND text because the use is non-commercial.
- A paraphrase that tracks the source sentence by sentence.
- Treating a free download as a license.""",
        see_also="The upstream page at the start of this book, which repeats the pointers without the legal reasoning, and Volume 01 for how a citation should enter your notes.",
    ),
)

chapter(
    v,
    "06_capstone",
    "Capstone: a checkable study",
    entry(
        question="Can you produce one study that uses the series as a reference and could be checked by someone who does not have your chat history?",
        card=[
            ("Question", "One estimand, predeclared."),
            ("Path", "The volumes you actually used, not every volume in the table of contents."),
            ("Baseline", "The simplest method that could answer the question."),
            ("Artifacts", "An experiment folder, a model or audit card if a model is deployed in the study, and a research note."),
            ("Stop", "A result that includes the possibility that the baseline won or that the audit says not to ship."),
        ],
        theory="""The capstone is the series in one folder. It is a reference use, not a tour. You pick a question that needs theory, code, an experiment, and a written limit. You do not demonstrate every algorithm. A study that tries to show every volume usually shows none of them carefully.

The checker's standard is unkind and fair. They have the folder and the note. They can rerun the known case. They can see the split. They can find the baseline. They can tell what you are not claiming. If the answer is no, the missing piece is the next edit, not a longer introduction.

Choose a question from your open-problem map so the capstone closes a bite you already defined. Changing the question after the result is the last peeking temptation in the book. Resist it in the note's first paragraph, where the predeclared question is quoted from the config.""",
        derivation="""The study estimates a gap $\\Delta$ between a chosen method and a baseline under a named distribution, with a tolerance for reproduction and a limit on interpretation. Everything else is appendix. A successful capstone is one in which $\\hat{\\Delta}$ is computed as specified, the known case passes, and the prose matches the table. Whether $\\hat{\\Delta}$ has the sign you hoped is not the success criterion. A reference that only contains hoped-for signs is advertising.""",
        code="""def capstone_manifest(question, baseline, folder, known_case_passed):
    if not known_case_passed:
        raise ValueError("do not write the claim until the known case passes")
    if not folder or not baseline or not question:
        raise ValueError("question, baseline, and folder are required")
    return {"question": question, "baseline": baseline, "folder": folder}
""",
        experiment="""**Hypothesis.** The manifest helper blocks a claim without a known case, and a completed manifest points at a real folder you produced earlier in the series.

**Protocol.** Call it once with `known_case_passed=False` and once with a real folder path after you rerun that folder's check.

**Metric.** The exception and the written manifest.

**Decision rule.** No manifest, no capstone claim. A folder that only contains a draft note without metrics is not the folder this helper is for.""",
        project="""**Deliverable.** The manifest, the research note, the experiment folder, and a short "path through the series" that cites the specific chapters you used for the split, the baseline, the uncertainty, and the limit.

**Constraints.** One question. One primary metric. Upstream books are cited, not copied.

**Rubric.** A checker who starts from the manifest can rerun the known case and can state your claim in one sentence that you would accept.""",
        beginner="Assemble the manifest for the Volume 01 least-squares project, including a limit.",
        university="Write the capstone on the best project you have, with a fresh known-case rerun.",
        research="Add a reproduction status and an audit or safety row if the study recommends a deployment. If it does not recommend one, say so in the lead sentence.",
        failures="""- A capstone that is a syllabus rather than a measurement.
- A claim whose folder is missing the baseline.
- A pasted chapter from an upstream book filling the method section.""",
        see_also="Volume 01 for the standard of a known case. The rest of the series for the pieces the path paragraph should cite by name.",
    ),
)
