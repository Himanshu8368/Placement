"""Volumes 11–14: vision, language, generative models, reinforcement learning."""

from booklib import chapter, entry, volume

v = volume(
    11,
    "Vision",
    "University",
    "A reader who can train a small network and can explain a convolution's shift behavior.",
    "How should images be represented, transformed, and evaluated so a vision model is a measurement rather than a demo?",
)

chapter(
    v,
    "01_images",
    "Images as tensors",
    entry(
        question="What is an image in this reference, and which transforms change the label?",
        card=[
            ("Tensor", "A batch of images as an array of shape $(n, c, h, w)$ in this series. Channels are the color or sensor axis."),
            ("Values", "A declared numeric range. Bytes in $[0, 255]$ and floats in $[0, 1]$ are different inputs."),
            ("Geometry", "Resize, crop, and flip are functions on the tensor and, sometimes, on the label."),
            ("Split", "Augment after the split. Fit normalization statistics on the training images only."),
            ("Baseline", "A linear model on pooled color, or a tiny convolution, before a large architecture."),
        ],
        theory="""An image is a measurement on a grid, not a bag of pixels and not a photograph's caption. The grid spacing is part of the contract. A model trained at $224 \\times 224$ is a different function from the same code at $32 \\times 32$. The label contract says whether a horizontal flip preserves $Y$. For species classification it might. For left-versus-right lesion location it does not.

Channel order and layout bugs are silent. A model can learn to read a permuted channel as if it were color, then fail on a camera that uses the declared order. Assert the shape and the range at the boundary. A mean of 120 and a mean of 0.47 should not both be called "normalized" without the denominator.

The constant and linear baselines still apply. Average brightness predicts some labels and is the thing a careless split will leak through JPEG artifacts or borders. Error analysis by slice, from Volume 07, should include a slice for image source or site, because cameras shift the input law.""",
        derivation="""A per-channel standardization $(x_c - \\mu_c)/\\sigma_c$ uses $\\mu_c, \\sigma_c$ from training pixels. Applying it with test-set moments uses the test distribution to define the features. The same formula with two different moment vectors is two different maps. Put the moments in the config or the checkpoint. At evaluation, reuse them. A resize from $(h, w)$ to $(h', w')$ is a linear interpolation or a pooling rule you must name. It is not invertible, so two different original images can become one tensor. That aliasing is a property of the measurement, and fine labels can become impossible after it.""",
        code="""import numpy as np

def standardize_images(batch, mean, std):
    x = np.asarray(batch, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError("expected (n, c, h, w)")
    mean = np.asarray(mean, dtype=np.float64).reshape(1, -1, 1, 1)
    std = np.asarray(std, dtype=np.float64).reshape(1, -1, 1, 1)
    if np.any(std <= 0):
        raise ValueError("std must be positive")
    return (x - mean) / std
""",
        experiment="""**Hypothesis.** Standardizing with training moments and with test moments produces different tensors when the test images are brighter, and a linear probe on the wrong moments is a different procedure.

**Protocol.** Synthetic $8 \\times 8$ single-channel images, training brightness $\\mathcal{N}(0, 1)$, test brightness $\\mathcal{N}(2, 1)$. Compare the two standardized test batches' means.

**Metric.** Difference of spatial means after standardization.

**Decision rule.** The gap should be near 2 if someone wrongly recenters the test set by its own mean and you compare to a training-moment standardization that leaves the shift in place... Design the comparison explicitly: training-moment standardization of the test set should leave a mean near $(2 - \\mu_{\\mathrm{train}})/\\sigma$, while cheating recentering removes it. If those match, the test brightness was not shifted.""",
        project="""**Deliverable.** A loader contract: layout, range, resize rule, which flips preserve your label, and where moments come from. Demonstrate on synthetic images if you do not have rights to a real set.

**Constraints.** No silent channel swaps. An assert must fail if a batch is $(n, h, w, c)$ under this series' layout, or you document a single layout and convert at the edge.

**Rubric.** A new image can be turned into the tensor the model expects without reading the training loop.""",
        beginner="Standardize a batch and state the shape after the call.",
        university="Run the brightness shift experiment.",
        research="Describe one label that a random crop can destroy, such as a count of objects that the crop cuts in half, and how the target would have to change to a crop-consistent label.",
        failures="""- Dividing uint8 values by 255 in one script and forgetting in another.
- Flipping images whose labels encode orientation.
- Computing dataset mean on a folder that mixes train and test.""",
        see_also="Volume 03 for contracts. Volume 09 for convolutions as the layer that uses the grid.",
    ),
)

chapter(
    v,
    "02_convolution",
    "Convolution and equivariance",
    entry(
        question="What parameter sharing does a convolutional layer implement, and when does padding break the story?",
        card=[
            ("Kernel", "A local weight tensor applied at every spatial position."),
            ("Fan-in", "For the initialization rule, fan-in includes input channels times kernel height times kernel width."),
            ("Stride and padding", "They change the output grid and the exact equivariance. They are part of the architecture, not cosmetics."),
            ("Pooling", "A local reduction. Global pooling yields a vector whose size does not depend on the input size, and it discards position."),
            ("Check", "The shift test from Volume 09, extended to two dimensions, away from the border."),
        ],
        theory="""A convolutional layer is a linear map with tied weights and a local support. The tying is the inductive bias: a detector learned in one corner is available in another. That is appropriate when the same visual pattern can appear at many positions. It is inappropriate when position itself is the label, unless position is reintroduced by a coordinate channel or by avoiding global pooling.

Padding changes the border. "Same" padding tries to preserve spatial size and uses values that were not in the image, often zeros. Those zeros are a cue. A model can detect the border by the padding pattern. The shift test will fail near the border even when the interior is equivariant. Report the region where you claim the property.

Detection and segmentation need positions. Classification of a centered object sometimes does not. Choosing global average pooling is a decision that translation of the object should not change the logit. If the dataset always centers the object, the decision is invisible until a shifted test set.""",
        derivation="""In two dimensions, $(K * X)_{c', r, col} = \\sum_{c, u, v} K_{c', c, u, v} X_{c, r+u, col+v}$, with a stride skipping positions and a padding rule defining $X$ outside the grid. Shifting $X$ by one pixel shifts the unpadded valid output by one pixel. With zero padding, the output near the new border also depends on the zeros that entered the window, so the equality fails for windows that overhang. A gradient check still applies: the convolution is linear in $K$ and in $X$, and a small random kernel should match a numerical gradient of a scalar loss on the output.""",
        code="""import numpy as np

def conv2d_single_channel(image, kernel):
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    out_h = image.shape[0] - kh + 1
    out_w = image.shape[1] - kw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("kernel larger than image")
    out = np.zeros((out_h, out_w))
    for r in range(out_h):
        for c in range(out_w):
            out[r, c] = np.sum(kernel * image[r : r + kh, c : c + kw])
    return out
""",
        experiment="""**Hypothesis.** Valid convolution commutes with a one-pixel shift in the interior: shifting the image and then convolving matches convolving and then shifting, on the overlap. A kernel of all ones computes a local sum you can check by hand on a constant image.

**Protocol.** A $12 \\times 12$ random image and a $3 \\times 3$ kernel. Constant image 7, kernel of ones, expected output $63$ everywhere.

**Metric.** Max absolute error of the shift test and of the constant test.

**Decision rule.** Both errors near $10^{-12}$. A shift error that is large only on the border means you included an invalid region; restrict the comparison.""",
        project="""**Deliverable.** The 2D convolution, the two checks, and a one-layer convolutional featurizer for a synthetic shape task: a bright $3 \\times 3$ block on a dark background, label 1 if present. Compare a linear model on raw pixels with a linear model on the convolution response.

**Constraints.** The block's position varies. The split is by image, not by pixel.

**Rubric.** You predict that the convolutional feature needs fewer labeled images to beat chance, and you measure that learning curve at two sample sizes.""",
        beginner="Hand-check the constant-image case.",
        university="Add the shift test and the synthetic block experiment.",
        research="State how the parameter count of a $3 \\times 3$ convolution depends on channels and not on $h \\times w$, and how the compute does depend on $h \\times w$. This is the systems fact Volume 15 will price.",
        failures="""- Claiming translation invariance for a feature map that is only equivariant.
- Using padding as a location cue and failing on a differently padded deployment image.
- Flattening $224 \\times 224 \\times 3$ into a vector as the first trained model, with 200 images.""",
        see_also="Volume 09 for the 1D shift test. Volume 15 for the cost of the spatial reduction.",
    ),
)

chapter(
    v,
    "03_detection",
    "Detection and segmentation",
    entry(
        question="What is the target when an image can contain many objects, or a label per pixel?",
        card=[
            ("Classification", "One label for the image."),
            ("Detection", "A set of boxes and class labels. The set can be empty. Order does not matter."),
            ("Segmentation", "A label per pixel, or a mask per object."),
            ("Matching", "A predicted box must be matched to a true box before it counts as a true positive. The matching threshold is part of the metric."),
            ("Empty", "Predicting nothing on a negative image is a true negative region, not a failure to find something."),
        ],
        theory="""Detection is not classification of a resized crop you already know how to find. The model must propose locations. The evaluation must not reward a duplicate box that overlaps the same object twice, and it must not punish an empty prediction on an image with no objects. Those rules are the matching protocol. Changing the overlap threshold changes the metric's meaning. Report the threshold.

Class imbalance is structural: most windows are background. A loss that treats every window equally will be happy predicting background. The foreground weight or the sampling of windows is part of the loss, with the calibration consequences from Volume 10.

Segmentation metrics that score pixel accuracy are dominated by the majority class, often background. Mean overlap on the rare class is closer to the decision if the rare class is the point of the system. State the decision.""",
        derivation="""Intersection over union of two boxes is the area of their intersection divided by the area of their union. A true positive at threshold $\\tau$ is a predicted box whose best unmatched true box of the same class has overlap at least $\\tau$. Unmatched predictions are false positives. Unmatched truths are false negatives. Precision and recall then follow Volume 07's definitions. Average precision integrates the precision-recall behavior of a ranked list of detections. It is not the accuracy of an image classifier, and it is not comparable across datasets with different size distributions without a new reading of the protocol.""",
        code="""import numpy as np

def iou(box_a, box_b):
    ay0, ax0, ay1, ax1 = box_a
    by0, bx0, by1, bx1 = box_b
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter = ih * iw
    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
""",
        experiment="""**Hypothesis.** A box that matches a truth with a one-pixel shrink has high IoU. A box shifted completely off the truth has IoU 0. A second identical prediction matched greedily should not count as a second true positive.

**Protocol.** Hand-built boxes. Implement a greedy one-to-one match at $\\tau = 0.5$ for a single class and apply it to a duplicated prediction.

**Metric.** IoU values and the matched counts.

**Decision rule.** The duplicate must become a false positive. If both copies count as true positives, the matcher is wrong even if `iou` is right.""",
        project="""**Deliverable.** The IoU function, the matcher, and a synthetic detection problem: one bright block per some images, none on others. A baseline that always predicts a centered box should fail on the empty images and on the off-center blocks in a way your metric shows.

**Constraints.** Report precision and recall at one predeclared $\\tau$, plus the false-positive count on empty images.

**Rubric.** You can explain a failure that image-level accuracy would have missed.""",
        beginner="Compute IoU for identical boxes and for disjoint boxes.",
        university="Implement the matcher and the synthetic baseline.",
        research="Define mean intersection-over-union for a three-class segmentation and show a case where pixel accuracy is above 95 percent while the rare class has overlap near 0.",
        failures="""- Counting two overlapping predictions of one object as two detections.
- Reporting classification accuracy on a detection task.
- Tuning the overlap threshold on the test set.""",
        see_also="Volume 07 for precision, recall, and operating points. Volume 16 for detectors that fail on a subpopulation the aggregate AP hides.",
    ),
)

chapter(
    v,
    "04_modern_vision",
    "Modern vision architectures",
    entry(
        question="What changed between a small convnet and a current vision backbone, in terms you can test?",
        card=[
            ("Residual path", "A skip connection $x + F(x)$ so a layer can learn a perturbation of the identity."),
            ("Stem and stage", "Early downsampling, then stages that trade spatial size for channels."),
            ("Attention block", "A content-based mix of spatial tokens. It does not by itself know the grid unless position is encoded."),
            ("Pretraining", "Weights fit on a previous dataset. They are a prior, and they can carry that dataset's labels and biases."),
            ("Test", "A shifted-object set, a different camera, and a linear probe on frozen features versus full fine-tuning."),
        ],
        theory="""Modern backbones are still functions from tensors to tensors. Residual branches make depth easier to optimize because the identity path is present at initialization if the branch starts small. Downsampling stages are a computational bet that high resolution is needed only early. Attention mixes information globally within a window or an image and needs a position signal to break permutation symmetry of the tokens.

These are testable designs, not brand names. A residual network that cannot fit the synthetic block task from the convolution chapter has a bug, regardless of its depth. A pretrained backbone that fails when the object moves has learned the centering of its training set. Freeze the backbone and train a linear head before you fine-tune everything. The probe tells you whether the features are readable. Fine-tuning confounds the features with a new head and a new optimization.

Pretraining data are part of the model card. If you cannot name the data, you cannot name the population the features saw. License and consent of that data are part of whether you may use the weights, not a footnote.""",
        derivation="""If $F(x) = 0$ at initialization, $x + F(x) = x$, so a stack of residual blocks starts as an identity in those units. Gradients along the skip do not pass through $F$. A product of many small derivatives is no longer the only path. This does not prevent vanishing gradients inside $F$, and it does not make the function class "more universal" in the sense of the approximation chapter. It changes optimization. A shift test still applies to a convolutional residual block in the interior of the grid, because both the skip and a translation-equivariant $F$ commute with the shift.""",
        code="""import numpy as np

def residual_add(x, branch):
    x = np.asarray(x, dtype=np.float64)
    branch = np.asarray(branch, dtype=np.float64)
    if x.shape != branch.shape:
        raise ValueError(f"residual shape mismatch: {x.shape} vs {branch.shape}")
    return x + branch
""",
        experiment="""**Hypothesis.** A residual block whose branch is zero returns the input unchanged, and a shifted input remains a shifted output. A position-free average of tokens is invariant to permutation and therefore cannot encode "the block is in the left half" unless a coordinate feature is added.

**Protocol.** Implement the zero-branch check. For the position check, compare a global mean of pixel values on a left-bright image and a right-bright image. They match, so the mean cannot classify side. Adding the column index as a feature makes a linear rule possible; demonstrate that with a hand-built score.

**Metric.** Max error of the identity check, and accuracy of the side rule with and without the coordinate.

**Decision rule.** The identity check is exact in float64 up to $10^{-12}$. The mean-only side classifier must fail. If it succeeds, the images differ in total brightness, not only in position.""",
        project="""**Deliverable.** An architecture card for a small convnet you train and, if you use a public pretrained model, a second card for that model naming its training data as well as you can, the probe result, and a shifted-object test.

**Constraints.** The shifted test is not used to pick the architecture and then quoted as a fresh result. If you tune on it, label the number exploratory.

**Rubric.** You can say which inductive bias you relied on, in a sentence a Volume 09 reader would recognize.""",
        beginner="Check residual addition and the global-mean position failure.",
        university="Train the small convnet on the synthetic block task and run a shifted test.",
        research="Compare a frozen pretrained representation's linear probe with full fine-tuning on a tiny labeled set you are allowed to use. Discuss what a win for fine-tuning confounds.",
        failures="""- Treating a leaderboard architecture as a prior that replaces baselines.
- Fine-tuning on a test set because pretrained models "only need a few epochs."
- Ignoring the license of pretrained weights.""",
        see_also="Volume 17 for foundation models that include vision. Volume 08 for probes.",
    ),
)

chapter(
    v,
    "05_vision_eval",
    "Evaluation for vision",
    entry(
        question="Which vision numbers survive a change of camera, crop, or class frequency?",
        card=[
            ("In-domain test", "New images from the same collection protocol as the training images."),
            ("Shift test", "A change you can name: brightness, shift, crop, site, or year."),
            ("Class mix", "Accuracy is not comparable across test sets with different base rates."),
            ("Localization", "If the decision needs a location, an image-level score is not sufficient."),
            ("Leakage", "Near-duplicate frames of the same scene in train and test, or the same patient, inflate the number."),
        ],
        theory="""Vision test sets are often easier than the world the model will see, because the collection protocol is stable. A high in-domain accuracy is a real result about that protocol and a weak result about a new hospital, a new phone, or a new year. The model card should separate these.

Near-duplicates are the entity problem. Frames from one video are not independent images. Splitting them at random lets the model memorize the scene. Group by video, patient, or location. A shift in brightness is the standardization experiment from the first chapter, promoted to a model test: do not recompute moments on the shifted test set if deployment will not do that either. Be consistent about which moments the deployed system uses, and test that procedure.

Calibration of a vision classifier still matters when a human will use the probability. Accuracy does not certify it. A reliability table on an in-domain validation set is the minimum. A reliability table under shift is the honest one, and it usually looks worse.""",
        derivation="""Suppose accuracy on a balanced test set is $a$, and you deploy at base rate $p \\ll 1/2$ with the same class-conditional detection behavior. Accuracy becomes a different function of true and false positive rates and $p$. Two models with the same balanced accuracy can rank differently at the deployment base rate. Report the class-conditional rates, or the metric at the deployment mix, using the threshold rule from Volume 07. Do not transport a balanced accuracy number into an imbalanced decision.""",
        code="""import numpy as np

def accuracy_at_base_rate(tpr, fpr, p):
    # Predict positive with the given class-conditional rates.
    return float(tpr * p + (1 - fpr) * (1 - p))
""",
        experiment="""**Hypothesis.** Two operating points with the same balanced accuracy, $0.5(\\mathrm{TPR}+1-\\mathrm{FPR})$, can have different accuracy at $p = 0.01$.

**Protocol.** Point A: TPR 0.9, FPR 0.1. Point B: TPR 0.99, FPR 0.19. Check balanced accuracy matches. Evaluate `accuracy_at_base_rate` at $p = 0.5$ and $p = 0.01$.

**Metric.** The two accuracies.

**Decision rule.** They should agree at $p = 0.5$ and disagree at $p = 0.01$. If they do not, the formula was applied to the wrong rate. This numerical fact is the reason a vision paper's headline accuracy is not your operating metric.""",
        project="""**Deliverable.** An evaluation note for the synthetic block detector: in-domain metric, a shifted-block metric, a brightness-shifted metric, and a grouped split if you created near-duplicate images. Include the base-rate calculation if you threshold a score.

**Constraints.** Moments and thresholds follow the deployment rule you wrote down.

**Rubric.** You refuse to quote a single accuracy as the performance of the system.""",
        beginner="Compute the two base-rate accuracies by hand and in code.",
        university="Write the full synthetic evaluation note.",
        research="Design a grouped split for a video dataset and explain a leakage path through background identity rather than through the object.",
        failures="""- A random frame split.
- Re-tuning on each shifted test set and calling it robustness.
- Comparing two papers' top-1 accuracy across different class mixes.""",
        see_also="Volume 16 for a fuller shift and harm audit. Volume 07 for the threshold.",
    ),
)

chapter(
    v,
    "06_vision_project",
    "Project: a small vision study",
    entry(
        question="Can you carry a vision claim from a contract through a baseline, a convnet, and a shift test?",
        card=[
            ("Task", "Synthetic or a dataset you have the right to use. The label definition fits in five lines."),
            ("Baselines", "Constant, linear on pooled features, and one convolutional model."),
            ("Ablation", "One factor: depth, augmentation, or input size."),
            ("Shift", "One predeclared shift."),
            ("Card", "Limits include the collection protocol you did not vary."),
        ],
        theory="""This project is the volume's integration test. A reader of Volumes 07, 09, and 10 should recognize every piece: a split, a baseline bake-off, a checked or library model with a stated reduction, one ablation, and a card. The new piece is the grid. If the convolutional model cannot beat pooled linear features on a task you designed to be local and shift-friendly, believe that measurement. Either the task is easier than you thought or the convnet is buggy. Both outcomes are publishable in your lab note.

Keep the images small enough that a CPU run finishes. The point is the protocol. A large backbone you cannot train carefully is a worse reference than a small kernel you can ablate.""",
        derivation="""The claim structure is the same as the paired gap in Volume 07. Let $D_i$ be the per-image loss difference between the convnet and the pooled linear model. The interval for the mean difference is the in-domain result. Repeat the losses under the shifted distribution without retuning, and report a second interval. A model that wins in-domain and ties under the shift you care about has not won the deployment claim. The card's primary metric must say which distribution it refers to.""",
        code="""import numpy as np

def pooled_brightness(images):
    x = np.asarray(images, dtype=np.float64)
    return x.mean(axis=(2, 3))
""",
        experiment="""**Hypothesis.** On the moving bright-block task, a convolutional feature beats pooled brightness, and pooled brightness fails once positive and negative images are matched for total brightness.

**Protocol.** Construct negatives with the same total intensity and no compact block, and positives with a compact block. Fit the pooled baseline and a one-filter threshold or a small trained convnet.

**Metric.** Accuracy or average precision at a predeclared score threshold, in-domain and with the block shifted.

**Decision rule.** If pooled brightness matches the convnet, the negatives are darker and the task is not spatial. Rebuild the negatives. Do not celebrate a model for measuring brightness.""",
        project="""**Deliverable.** The dataset generator, the three baselines or models, one ablation, the shift evaluation, and a model card.

**Constraints.** Training moments and the decision threshold are frozen before the shift test.

**Rubric.** The note's first paragraph states the label, the split unit, and the shift. A screenshot of a detection is not a substitute.""",
        beginner="Generate the blocks and show the pooled baseline failing the equal-brightness control.",
        university="Complete the project as specified.",
        research="Add a grouped near-duplicate split: several jitters of the same block location. Show the optimistic gap of a random split versus a group split.",
        failures="""- Tuning on the shifted images.
- A task that pooled color already solves.
- Weights from a dataset whose license you have not read.""",
        see_also="Volume 15 if this model is served. Volume 16 if the images are of people.",
    ),
)

v = volume(
    12,
    "Sequences and language",
    "University to research",
    "A reader who can train a classifier and can state a split unit.",
    "How do you model ordered symbols without leaking the future or trusting a probability you did not calibrate?",
)

chapter(
    v,
    "01_tokens",
    "Tokens and sequences",
    entry(
        question="What is the unit of a language or sequence model, and what does a split have to protect?",
        card=[
            ("Token", "An element of a finite vocabulary, or a bin of a continuous stream. The tokenizer is part of the model."),
            ("Document", "The leakage boundary. Sentences from one document are not independent draws."),
            ("Vocabulary", "Built on the training text only. A test token that is absent has a defined unknown id."),
            ("Length", "Padding is not content. The loss mask must ignore pad positions."),
            ("Order", "A bag-of-tokens baseline is mandatory. If it wins, the model is not yet using order."),
        ],
        theory="""A tokenizer is a function from raw text or signals to ids. Changing it changes the input space. Two runs with different vocabularies are not the same architecture on the same data. Fit the vocabulary, including frequency cutoffs, on the training split. A vocabulary that saw the test documents has leaked rare names and labels.

The split unit is the document, the user, or the time period. Splitting sentences independently lets a model memorize a document from its other sentences. Time-ordered text should be tested on the future. A random document split answers a different question: new documents from the same period.

Padding exists so a batch can be rectangular. A loss that predicts the pad id will look strangely accurate and will waste capacity. Mask it. A metric that does not mask it is a different metric.""",
        derivation="""Let a document be a token sequence $w_{1:T}$. A next-token model factorizes $p(w_{1:T}) = \\prod_t p(w_t \\mid w_{<t})$ under the model. The average negative log likelihood per non-pad token is the usual training loss. Dividing by the number of documents instead of tokens gives a different weight to long documents. The mask $m_t \\in \\{0, 1\\}$ enters as $\\sum_t m_t \\ell_t / \\sum_t m_t$. If $m_t$ is zero on pads and also on the token you are not allowed to predict, the denominator is the count of real predictions. A bug that sets $m_t = 1$ on the position being predicted from itself is label leakage inside the sequence.""",
        code="""import numpy as np

def masked_mean(values, mask):
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    denom = mask.sum()
    if denom <= 0:
        raise ValueError("empty mask")
    return float((values * mask).sum() / denom)
""",
        experiment="""**Hypothesis.** Including pad positions in the loss, when pads are easy to predict, lowers the reported mean loss without improving token predictions on real positions.

**Protocol.** A batch with half the positions masked as pad, real-token loss 2.0, pad loss 0.0. Compare `masked_mean` with a plain mean.

**Metric.** The two means.

**Decision rule.** The plain mean should be optimistically smaller. If your training log uses the plain mean, the number is not the masked language-model loss. Fix the log before comparing runs.""",
        project="""**Deliverable.** A tokenizer or a simple whitespace vocabulary fit on training documents only, an unknown-token rule, a pad mask, and a bag-of-tokens logistic or least-squares baseline on a synthetic labeling task.

**Constraints.** The task must have a document id so the split can be grouped. Synthetic documents are acceptable: a keyword determines the label, and an order-sensitive variant uses a two-token phrase.

**Rubric.** You show one case where the bag of tokens suffices and one where order matters, analogous to the stump-versus-linear flip in Volume 06.""",
        beginner="Implement `masked_mean` and the pad experiment.",
        university="Build the vocabulary and the two synthetic tasks.",
        research="Describe byte-pair encoding in a paragraph and the leakage path if merge rules are learned on the full corpus, including the test set.",
        failures="""- A shared vocabulary built before the split.
- Padding tokens in the accuracy.
- Treating two sentences from one article as independent test points.""",
        see_also="Volume 03 for grouped splits. Volume 10 for the reduction.",
    ),
)

chapter(
    v,
    "02_recurrent",
    "Recurrent models",
    entry(
        question="What state does a recurrent net carry, and what does truncation do to the dependency it can learn?",
        card=[
            ("Recurrence", "$h_t = f(h_{t-1}, x_t)$. The same $f$ at every time, which is weight sharing along the sequence."),
            ("State", "$h_t$ is a function of $x_{1:t}$ and $h_0$. It is a bottleneck."),
            ("BPTT", "Backpropagation through the unrolled graph. Gradients accumulate along the path."),
            ("Truncation", "Stopping the backward path after $K$ steps makes the update blind to dependencies longer than $K$, even if the forward state could carry them."),
            ("Vanish", "Products of Jacobians along time vanish or explode. Gates are a mitigation, not a proof."),
        ],
        theory="""A recurrent net is a deep net in time, with tied weights. The inductive bias is that the same update is useful at every position. That fits a stationary process and misfires when the beginning of a document should be treated by a different rule than the end, unless the inputs mark position.

The hidden state is a finite summary. If the label depends on a detail the state was too small to store, the model cannot represent the function, whatever the optimizer does. This is the approximation question again, with a bottleneck you chose.

Truncated backpropagation is an optimization approximation. You can still, in the forward pass, carry state across the whole sequence, and only backpropagate through the last $K$ steps. The early inputs then affect the forward state but not that update. Long-range learning can fail even though a long-range path exists. If the experiment needs a dependency of length 20, truncation at 5 is a reason.""",
        derivation="""Unroll $h_t = \\tanh(W_h h_{t-1} + W_x x_t + b)$ for three steps and differentiate a loss on $h_3$ with respect to $h_0$. The chain contains the product $J_3 J_2 J_1$ of Jacobians of the tanh map. Each Jacobian includes $\\mathrm{diag}(1 - \\tanh^2)$ times $W_h$. If the singular values of that product are much less than 1, the gradient with respect to early inputs is tiny. Clipping the gradient norm bounds the step but does not restore information that the product destroyed. It only stops a divergent step.""",
        code="""import numpy as np

def rnn_step(h, x, Wh, Wx, b):
    z = h @ Wh + x @ Wx + b
    return np.tanh(z)
""",
        experiment="""**Hypothesis.** A hand-built state that copies a bit can remember an input from $K$ steps ago, and a truncated analytic story that ignores the path cannot update the copy weight from a loss that depends only on that memory. In code, demonstrate memory with an identity recurrence, and demonstrate a vanished product by powering a scalar $0.5$ twenty times.

**Protocol.** Identity: $h_t = x_t$ if you overwrite, or $h_t = h_{t-1}$ if $x_t$ is a blank. Report $h_{20}$ given a bit at step 0. Separately compute $0.5^{20}$.

**Metric.** Whether the bit survived, and the value of the product.

**Decision rule.** The bit should survive the identity recurrence. The product should be numerically tiny. If you claim a gated RNN fixes this, the experiment must show a gate that is open on the remembered path, not a citation.""",
        project="""**Deliverable.** A tiny recurrent classifier on the order-sensitive synthetic phrase task, compared with the bag-of-tokens baseline, plus a truncation or depth note: what is the longest dependency in the synthetic language, and can the unroll see it?

**Constraints.** Pad masks apply. Documents are the split unit.

**Rubric.** You can say whether a failure is the bag-of-tokens baseline winning, a state that is too small, or a dependency longer than the truncation.""",
        beginner="Step a one-dimensional recurrence by hand for three steps.",
        university="Complete the memory demo and the synthetic comparison.",
        research="Derive the gradient clip's effect: it rescales $g$ when $\\|g\\| > c$ and leaves it otherwise. Explain why clipping can stabilize training without lengthening the dependency the optimizer can credit.",
        failures="""- Feeding the target token as an input at the same time step.
- Evaluating a language model with teacher forcing and calling that a free-running generation error.
- Splitting a sequence in the middle so the state at the boundary has no past, then scoring the prefix.""",
        see_also="Volume 09 for products of Jacobians. The attention chapter for a path that does not multiply through every intermediate time.",
    ),
)

chapter(
    v,
    "03_attention",
    "Attention and transformers",
    entry(
        question="What does an attention layer compute, and what cost and bias come with it?",
        card=[
            ("Scores", "A compatibility $q_i^\\top k_j / \\sqrt{d}$ between a query at position $i$ and a key at position $j$."),
            ("Weights", "A softmax over allowed positions. The mask sets disallowed scores to a large negative number before the softmax."),
            ("Output", "A weighted sum of values. The layer is a routed average, not a lookup of a single token, unless the softmax is peaked."),
            ("Causal mask", "Position $i$ may not read $j > i$ in a next-token model. A bug here is label leakage."),
            ("Cost", "Full attention is quadratic in the sequence length, in memory and in the matmul."),
        ],
        theory="""Self-attention lets a position read any other allowed position in one step. The path length between distant tokens is constant, which is the contrast with the recurrent product of Jacobians. The price is quadratic compute and a weak inductive bias about order. Without a position encoding, the layer is permutation-equivariant in a precise sense: it does not know which token came first. Sequences are ordered, so you add positions, and those features are part of the model.

The causal mask is the difference between a language model and a model that peeks at the answer. Bidirectional encoders are appropriate when the whole sequence is an input, as in classification of a complete document. They are inappropriate when you claim to predict the future. The mask test is the unit test. It is not optional because a library "usually" masks.

Multi-head attention is several of these maps concatenated. It raises capacity. It does not by itself create linguistic understanding. Evaluate it as a function.""",
        derivation="""Let $Q, K, V$ be $T \\times d$ matrices for one head and one example. Scores $S = Q K^\\top / \\sqrt{d}$. For a causal mask, set $S_{ij} = -\\infty$ when $j > i$, then $A = \\mathrm{softmax}(S)$ row-wise, and the output is $AV$. Row $i$ of $A$ sums to 1 and is zero after $i$. The gradient of a loss on the output with respect to $V$ is $A^\\top \\nabla_{\\mathrm{out}}$. The gradient with respect to $S$ uses the Jacobian of the softmax. A numerical check on a short $T$ catches a mask that was applied after the softmax, which cannot zero the future once the weights are positive.""",
        code="""import numpy as np

def causal_attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    scale = np.sqrt(Q.shape[-1])
    scores = (Q @ K.T) / scale
    t = scores.shape[0]
    future = np.triu(np.ones((t, t), dtype=bool), k=1)
    scores = np.where(future, -1e9, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V, weights
""",
        experiment="""**Hypothesis.** With a causal mask, attention weights on the future are numerically zero and the output at position $i$ does not change when a future value vector changes. Without the mask, it does.

**Protocol.** $T = 4$, random $Q, K, V$. Perturb $V$ at the last position and compare outputs at position 0.

**Metric.** Max future weight, and the change in the position-0 output.

**Decision rule.** Future weights below $10^{-8}$ and a zero change at position 0 for the masked layer. If the unmasked layer also shows a zero change, the query-key pair assigned that position no weight by chance; resample.""",
        project="""**Deliverable.** The masked attention function, the peek test, and a tiny trained or closed-form sequence task where a bag of tokens fails and a one-layer causal or bidirectional model matches the task's information constraint. Classification of a finished string may be bidirectional. Next-token prediction must be causal.

**Constraints.** The peek test is in the debug checklist.

**Rubric.** You can explain the quadratic cost at $T = 1024$ versus $T = 128$ in arithmetic operations for the score matrix.""",
        beginner="Implement causal attention and print the weight matrix.",
        university="Run the peek test and the bag-of-tokens comparison.",
        research="Describe rotary or absolute position features in one paragraph and the failure mode if two sequences are concatenated without a segment boundary the model can see.",
        failures="""- Applying the mask after the softmax.
- A bidirectional encoder advertised as a next-token model.
- Ignoring the pad mask so padding positions become keys the model attends to.""",
        see_also="Volume 15 for the quadratic cost as a systems constraint. Volume 17 for models that use this block at large scale.",
    ),
)

chapter(
    v,
    "04_finetune",
    "Training and adapting language models",
    entry(
        question="What is the difference between training a small model from scratch and adapting a model you did not train?",
        card=[
            ("From scratch", "You own the data, the tokenizer, and the loss. The baselines of this volume apply."),
            ("Fine-tune", "Start from someone else's parameters and continue training. You inherit their data and their failures."),
            ("Freeze", "Train a head on frozen features. A probe, in the sense of Volume 08."),
            ("Prompt", "No gradient. The model is a fixed function and the prompt is an input. It is not a guarantee of the same behavior next month if the model is hosted."),
            ("Split", "The downstream split is still yours. Their pretraining data may already contain your test documents."),
        ],
        theory="""Adapting a pretrained model can beat a small model on a tiny labeled set because the pretraining was a large self-supervised task. That is a hypothesis about feature reuse, tested by a probe and by a fine-tune, not a law. The pretrained model may have seen text that overlaps your test set. Deduplication and a truly new domain are the checks. If you cannot check, the card says the overlap is unknown.

Fine-tuning all weights on a small set can erase useful features and can memorize the small set. A frozen head, a small learning rate, and early stopping are regularizers from Volume 10. The bag-of-tokens baseline remains mandatory. On some classification tasks it is close enough that a transformer is an expensive tie.

A hosted model you call through an API is not a checkpoint you can hash. The config must store the model name and the date. A result that cannot name the model version is not reproducible. This series prefers a local checkpoint when the claim is about your training.""",
        derivation="""Let $\\theta_0$ be the pretrained parameter and $\\theta_0 + \\Delta$ the fine-tuned one. A penalty $\\lVert \\Delta \\rVert$ or a low-rank constraint on $\\Delta$ keeps the new function near the old one. Full fine-tuning is the unconstrained case. The validation loss decides the constraint. The test set does not. If pretraining already optimized a loss that includes your test documents, then even $\\Delta = 0$ is not an out-of-sample model. No penalty on $\\Delta$ repairs contamination that happened before you started.""",
        code="""import numpy as np

def lora_delta(A, B, x):
    # x (n, d), A (d, r), B (r, d_out) or match your layout and document it.
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return (x @ A) @ B
""",
        experiment="""**Hypothesis.** A low-rank update can represent a small correction to a linear map and cannot, at rank 1, represent an arbitrary full-rank correction of a $d \\times d$ map when $d > 1$.

**Protocol.** $d = 4$. Fit a rank-1 $\\Delta = ab^\\top$ by least squares to a random full-rank target matrix on many $x$, and to a rank-1 target.

**Metric.** Frobenius residual of the two fits.

**Decision rule.** The rank-1 target should be fit closely. The full-rank target should leave a visible residual. If both residuals vanish, the parameterization is not rank 1.""",
        project="""**Deliverable.** A downstream classification note: bag-of-tokens baseline, a small from-scratch model or a frozen probe, and a fine-tune only if you have a checkpoint you may use. Include a contamination caveat.

**Constraints.** The tokenizer and the split follow this volume's rules.

**Rubric.** You do not claim a general language ability. You claim a metric on a named split.""",
        beginner="Show the rank-1 residual gap with `lora_delta` or an outer product.",
        university="Complete the baseline and one adapted or from-scratch model on the synthetic document task.",
        research="Write a deduplication check: exact overlap between train and test strings, and a limit of that check for paraphrases it will not catch.",
        failures="""- Fine-tuning on the test split because the set is "small."
- An API result with no date and no model id.
- Dropping the bag-of-tokens baseline because a transformer "should" win.""",
        see_also="Volume 08 for probes. Volume 17 for scale and contamination as a research topic.",
    ),
)

chapter(
    v,
    "05_generation_eval",
    "Evaluation of generation",
    entry(
        question="What may you conclude from a generation metric, and what requires a human or a task-specific check?",
        card=[
            ("Likelihood", "A held-out negative log likelihood or perplexity, under a stated tokenization, measures the predictive model. It is not a judgment of truth."),
            ("Reference overlap", "A score against one reference string punishes valid paraphrases and rewards overlap."),
            ("Task metric", "If generation is a means to an extracted answer, score the answer, not the prose."),
            ("Degeneration", "Repetition, empty answers, and copied inputs need their own counters."),
            ("Calibration", "A high token probability is not a calibrated probability that a claim is true."),
        ],
        theory="""Generation has several different questions. "Does this model predict the next token of this corpus?" is a likelihood question. "Does the string match a reference translation?" is a string-overlap question with known failure modes. "Is the answer correct?" is a task question and needs an answer key or a checker. "Should a person trust the prose?" is not settled by any of those.

Perplexity is $\\exp$ of the mean token negative log likelihood. It depends on the tokenizer. You cannot compare perplexities across different tokenizations without a conversion you justify. A model that memorizes the test set has excellent perplexity and has not generalized. The contamination check sits next to the number.

Automatic overlap metrics are baselines. They are useful when they correlate with a decision you have validated on a sample of human or programmatic checks. They are not the decision. For a synthetic task, prefer a programmatic checker: did the output contain the required phrase, did the arithmetic hold, did the format parse.""",
        derivation="""If the model factorizes token probabilities, the log probability of a completed string under teacher forcing is the sum of conditional log probabilities. That sum uses the true prefix at every step. Free-running generation feeds the model's own samples back in, so errors compound. A high teacher-forced likelihood does not imply a high probability for the string the model actually emits when it is allowed to condition on its own mistakes. Report which protocol you used. A metric computed on teacher-forced completions of the reference is not a metric of free-running text.""",
        code="""import numpy as np

def perplexity_from_nll(mean_nll):
    if mean_nll < 0:
        raise ValueError("mean negative log likelihood must be nonnegative")
    return float(np.exp(mean_nll))
""",
        experiment="""**Hypothesis.** Two tokenizations of the same characters can yield different per-token negative log likelihoods for a model that is uniform over its vocabulary, and the character-level numbers can be reconciled only if you account for tokens per character.

**Protocol.** Vocabulary sizes 30 and 100, uniform model, sequences of 10 tokens. Compute per-token NLL and perplexity. Note that the uniform NLL is $\\log |V|$.

**Metric.** The two perplexities.

**Decision rule.** They must differ. A comparison of two models that hides a tokenizer change is the same kind of mismatch. Record both vocabulary sizes whenever you quote perplexity.""",
        project="""**Deliverable.** An evaluation note for a small generator or a prompted fixed model on a synthetic task with a programmatic checker, plus likelihood on held-out text if you trained a model. Include repetition rate or exact-copy rate.

**Constraints.** The checker is written before you generate the test outputs, as far as that is operationally possible. If you adjust the checker after seeing outputs, say so.

**Rubric.** You separate "predicts the corpus," "matches a reference," and "satisfies the checker." You do not use one of those phrases to mean the others.""",
        beginner="Compute perplexity of a uniform model over a vocabulary of size 10.",
        university="Complete the synthetic checker evaluation.",
        research="Discuss why a model can assign high probability to a false statement that is common in the training text. Tie this to proper scoring: the likelihood scores the string distribution, not the truth of a proposition under an external world.",
        failures="""- Comparing perplexity across tokenizers.
- Selecting checkpoints by a free-running demo you reran until it looked good.
- Treating a fluent paragraph as a cited fact.""",
        see_also="Volume 16 for claims that affect people. Volume 17 for evaluation of large models.",
    ),
)

chapter(
    v,
    "06_language_project",
    "Project: a small language pipeline",
    entry(
        question="Can you ship a language classification or tagging pipeline whose baseline, mask, and errors are visible?",
        card=[
            ("Task", "Document label or a synthetic tag. Not an open-ended claim about understanding."),
            ("Pipeline", "Raw text, tokenizer, features or model, masked metric, error slices."),
            ("Baselines", "Majority class and bag of tokens."),
            ("Model", "A linear model, a small recurrent net, or a small attention layer."),
            ("Errors", "A taxonomy: unknown tokens, order failures, leaked duplicates, pad-mask bugs."),
        ],
        theory="""The pipeline is the product. A good attention block inside a leaky split is a bad system. This project scores the pipeline. The write-up should let someone rerun it from the experiment folder and disagree with your decision using your table.

Prefer a task with a known mechanism so error analysis can name the mechanism. A synthetic phrase task, a copied bag-of-tokens task, and a duplicate-document leak task are enough. If you use a public dataset, the card still needs the split unit and the license.

The decision is whether the order-sensitive model earns its cost over the bag of tokens on the primary metric, by the paired-gap rule. If it does not, ship the linear model in the note's recommendation and keep the neural result as a negative. Negative results are part of a reference.""",
        derivation="""The pipeline risk is $R(\\mathrm{decode} \\circ f \\circ \\mathrm{tokenize})$. A comparison that swaps the tokenizer between the baseline and the neural model confounds the representation with the family. Use one tokenizer for both, or treat the tokenizer as a factor in an ablation with its own row. The primary metric uses the pad mask and the document-level split. The optimistic sentence-level split can appear only as a labeled anti-pattern, the way Protocol A did in Volume 07.""",
        code="""import numpy as np

def majority_label(y_train, n_test):
    y_train = np.asarray(y_train)
    values, counts = np.unique(y_train, return_counts=True)
    label = values[np.argmax(counts)]
    return np.full(n_test, label)
""",
        experiment="""**Hypothesis.** On documents whose label is the majority keyword, the bag-of-tokens model matches a sequence model. On documents whose label is whether token A appears before token B, the bag model falls to chance if the counts are balanced, and a causal or bidirectional sequence model can exceed it.

**Protocol.** Generate both datasets. Same vocabulary and document-level split. Majority baseline on both.

**Metric.** Accuracy or log loss, with a paired comparison on the order-sensitive set.

**Decision rule.** A sequence model that does not win on the order-sensitive set is not finished. A bag model that wins on the order-sensitive set means the generator leaks order into counts; fix the generator.""",
        project="""**Deliverable.** The generator, the majority and bag baselines, one sequence model, the mask, the document split, an error taxonomy, and a model card. Include the leaky sentence split only as a warning row.

**Constraints.** Unknown tokens and pads are defined. The primary metric ignores pads.

**Rubric.** The recommendation names the simpler model if the gap does not clear the practical margin.""",
        beginner="Implement the majority baseline and the keyword dataset.",
        university="Complete both datasets and the card.",
        research="Add a contamination row: copy 10 percent of test documents into training and measure the optimistic gap. Leave the row in the note as a warning, not as a system to ship.",
        failures="""- A fluent demo in place of the table.
- Different tokenizers for the baseline and the model.
- Pads counted as correct predictions.""",
        see_also="Volume 17 if you replace the small model with a pretrained one. The same baselines stay.",
    ),
)
