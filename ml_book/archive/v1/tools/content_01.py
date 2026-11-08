"""Volumes 01–03: language, calculus, and computation."""

from booklib import chapter, entry, volume

v = volume(
    1,
    "Mathematical language",
    "Beginner, first weeks",
    "A reader who can do high-school algebra and is willing to learn notation as a tool.",
    "What do the symbols in the rest of this reference mean, and which linear-algebra facts are load-bearing?",
)

chapter(
    v,
    "01_how_to_use",
    "How to use this reference",
    entry(
        question="How should a beginner-to-research reader move through a long reference without treating it as a short textbook?",
        card=[
            ("Object", "A reference entry: card, theory, derivation, code, experiment, project."),
            ("Order", "Volume order the first time. Chapter cards on every later visit."),
            ("Unit of work", "One experiment folder with a hypothesis, a command, a metric, and a decision."),
            ("Done", "You can restate the result, rerun the code, and say what would falsify it."),
            ("Not done", "You highlighted a paragraph and did not change a number."),
        ],
        theory="""This series is a guide you return to, not a narrative you finish once. Each volume widens the object under study: a vector, a probability, a predictor, a training run, a deployed system, a research claim.

A result in this book is a triple: a statement, the assumptions that make it true, and a check you can run. If you drop the assumptions, you do not have the result. If you cannot check it, you do not yet own it. The four layers exist so that a lookup (the card), an argument (theory and derivation), an implementation (code), and a decision (experiment and project) stay attached to each other.

The progression has three speeds. Beginner work builds the objects. University work proves the standard lemmas and compares methods on a fixed split. Research work changes an assumption and writes the consequence. You may be a beginner in Volume 14 and at research depth in Volume 04 at the same time. The track labels are about the chapter, not about your identity.""",
        derivation="""Fix a study session of 90 minutes as the unit. Spend about 20 minutes on the card and theory, 25 on the derivation or a line-by-line reading of the proof sketch, 25 on the code until a known case passes, and 20 on writing the experiment note. A project spans several sessions and must leave a rerunnable artifact.

Let $C$ be a chapter and $L(C)$ the set of layers you completed. The chapter is usable as a reference when $\\{\\text{card}, \\text{code}\\} \\subseteq L(C)$. It is university-complete when all six layers are in $L(C)$. It is research-complete when you have added one written failure of an assumption, stored next to the lab note.""",
        code="""from pathlib import Path

def new_note(slug: str, hypothesis: str, metric: str) -> Path:
    path = Path("notes") / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\\n".join(
            [
                f"# {slug}",
                "",
                f"Hypothesis. {hypothesis}",
                "",
                f"Metric. {metric}",
                "",
                "Protocol.",
                "",
                "Result.",
                "",
                "Decision.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path
""",
        experiment="""**Hypothesis.** A fixed note template increases the chance that a later session can reconstruct the decision.

**Protocol.** For the next four chapters, write the note before running code. On two of them, skip the template and write free prose. A week later, try to rerun both kinds from the note alone.

**Metric.** Fraction of notes from which you can recover the command, the metric, and the decision without opening old chat logs.

**Decision rule.** If templated notes are not easier to rerun, change the template. Do not keep a ritual that does not restore the result.""",
        project="""**Deliverable.** A `notes/` directory, this function, and one completed note for the linear-algebra chapter.

**Constraints.** The note names the assumption you used and one place it can fail.

**Rubric.** Someone else can tell what you measured and what you would do differently. The code creates the file you claim it creates.""",
        beginner="Read this chapter once, create the notes folder, and move to notation.",
        university="Write the session rule down and follow it for all of Volume 01.",
        research="After Volume 06, reread this chapter and add a fifth field to the note: the assumption you are worried is false in your data.",
        failures="""- Treating the card as a substitute for the experiment.
- Collecting chapters without a question you are trying to answer.
- Skipping prerequisites and then blaming the later formula.""",
        see_also="Volume 03 for the experiment folder layout. Volume 18 for how a paper is filed into this reference. The upstream reading list names books you read beside this series. Their text is not copied here.",
    ),
)

chapter(
    v,
    "02_notation",
    "Sets, functions, and notation",
    entry(
        question="Which symbols are fixed for the rest of the series, and what does a function claim?",
        card=[
            ("Data", "A pair $(x, y)$ with $x \\in \\mathcal{X}$ and $y \\in \\mathcal{Y}$. A sample is $\\{(x_i, y_i)\\}_{i=1}^{n}$."),
            ("Parameters", "$\\theta \\in \\Theta$. A predictor is $f_\\theta: \\mathcal{X} \\to \\mathcal{A}$, where $\\mathcal{A}$ is the action or prediction space."),
            ("Loss", "$\\ell(f_\\theta(x), y) \\ge 0$. Empirical risk is $\\hat{R}_n(\\theta) = n^{-1}\\sum_{i=1}^{n} \\ell(f_\\theta(x_i), y_i)$."),
            ("Expectation", "$R(\\theta) = \\mathbb{E}[\\ell(f_\\theta(X), Y)]$ is the risk under the unknown data law $P$."),
            ("Convention", "Vectors are columns. $x^\\top x$ is a scalar. $\\|x\\|_2 = \\sqrt{x^\\top x}$. Indices start at 1 in sums and at 0 in code."),
        ],
        theory="""A set is a collection we are willing to test membership in. A function $f: \\mathcal{X} \\to \\mathcal{Y}$ assigns exactly one output to each input in its domain. In this series the domain is part of the claim. A neural network that expects a tensor of shape $(n, d)$ is a different function from one that expects $(n, c, h, w)$.

Random variables are functions on a sample space. Writing $X \\sim P$ means we are about to use the law of $X$, not a single draw. A single dataset is one draw of the collection $(X_1, Y_1), \\ldots, (X_n, Y_n)$. The risk $R(\\theta)$ depends on $P$. The empirical risk $\\hat{R}_n(\\theta)$ depends on the draw. Almost every later confusion between training loss and future error is a confusion between these two functions.

We write $\\arg\\min_{\\theta \\in \\Theta} \\hat{R}_n(\\theta)$ for a minimizer, not for a unique one. If the set of minimizers has more than one point, say so. Uniqueness is a theorem, not a notation.""",
        derivation="""Let $\\ell$ be squared error, $\\ell(\\hat{y}, y) = (\\hat{y} - y)^2$, and let $f_\\theta(x) = \\theta^\\top x$ with $x \\in \\mathbb{R}^d$ already containing a constant coordinate if you want an intercept. Then

$$\\hat{R}_n(\\theta) = n^{-1}\\lVert X\\theta - y \\rVert_2^2,$$

where $X \\in \\mathbb{R}^{n \\times d}$ has rows $x_i^\\top$ and $y \\in \\mathbb{R}^n$. Expanding,

$$\\lVert X\\theta - y \\rVert_2^2 = \\theta^\\top X^\\top X \\theta - 2 y^\\top X \\theta + y^\\top y.$$

Differentiate in $\\theta$ and set the gradient to zero:

$$X^\\top X \\theta = X^\\top y.$$

This is the normal equation. It is a claim about critical points of squared error, under no claim yet that $X^\\top X$ is invertible.""",
        code="""import numpy as np

def empirical_risk(theta, X, y):
    residual = X @ theta - y
    return float(np.mean(residual**2))

def normal_equation(X, y):
    gram = X.T @ X
    rhs = X.T @ y
    # lstsq remains defined when gram is singular.
    theta, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
    return theta
""",
        experiment="""**Hypothesis.** Adding a duplicate column to $X$ leaves the minimal squared error unchanged and makes the parameter vector non-unique.

**Protocol.** Draw $X_0 \\in \\mathbb{R}^{40 \\times 3}$ with independent standard normal entries and $y = X_0 \\theta_\\star + \\varepsilon$. Build $X$ by copying the first column. Compare `lstsq` on $X_0$ and on $X$, and compare $\\hat{R}_n$.

**Metric.** Absolute difference of the two minimal risks, and the dimension of the numerical null space of $X^\\top X$ (count singular values below $10^{-8}$).

**Decision rule.** The risks should match to about $10^{-8}$ relative error. If they do not, the implementation, not the theorem, is wrong.""",
        project="""**Deliverable.** A one-page notation sheet in your own words, plus the script above, plus the duplicate-column note.

**Constraints.** The sheet must define $R$, $\\hat{R}_n$, $\\theta$, $X$, and $y$ with shapes.

**Rubric.** A reader can translate one formula from a later chapter back onto your sheet without asking you.""",
        beginner="Copy the notation table by hand. Run the code on a 5 by 2 matrix you type yourself.",
        university="Prove the expansion of the squared norm and the gradient calculation without looking.",
        research="Rewrite the normal equation for a weighted risk $\\sum_i w_i (x_i^\\top \\theta - y_i)^2$ with $w_i > 0$. State when the weighted Gram matrix is singular.",
        failures="""- Using the same letter for a random variable and a single sample.
- Assuming $\\arg\\min$ returns one vector.
- Mixing row and column layouts between the formula and the array.""",
        see_also="The next chapter is the linear-algebra reference those shapes rely on. Volume 04 rereads $R$ and $\\hat{R}_n$ as expectations.",
    ),
)

chapter(
    v,
    "03_linear_algebra",
    "Linear algebra for models",
    entry(
        question="Which facts about matrices, ranks, and norms do models actually use?",
        card=[
            ("Space", "$\\mathbb{R}^d$ with the dot product $\\langle u, v \\rangle = u^\\top v$ and the Euclidean norm."),
            ("Matrix", "A linear map $A: \\mathbb{R}^d \\to \\mathbb{R}^n$. Columns of $A$ are the images of the standard basis."),
            ("Rank", "$\\mathrm{rank}(A) = \\dim \\mathrm{col}(A)$. $\\mathrm{rank}(A) \\le \\min(n, d)$."),
            ("SVD", "$A = U \\Sigma V^\\top$ with orthonormal $U, V$ and nonnegative singular values on the diagonal of $\\Sigma$."),
            ("PSD", "$G = G^\\top$ is positive semidefinite if $z^\\top G z \\ge 0$ for every $z$. Gram matrices $X^\\top X$ are PSD."),
            ("Norms", "$\\|A\\|_2$ is the largest singular value. $\\|A\\|_F^2 = \\sum_{ij} A_{ij}^2 = \\sum_k \\sigma_k(A)^2$."),
        ],
        theory="""A model that predicts with $X\\theta$ is using the column space of $X$. If $y$ is not in that column space, squared error cannot be zero. The residual $y - X\\hat{\\theta}$ from least squares is orthogonal to the column space: $X^\\top (y - X\\hat{\\theta}) = 0$. That geometric sentence is the normal equation.

Rank tells you how many independent directions the features supply. Two copies of the same feature do not give a new direction. A one-hot encoding of a categorical variable with a redundant level, together with an intercept, is the same phenomenon: the columns are linearly dependent, and the parameter is not identified even when predictions are.

Norms measure size. Parameter size and prediction size are different. $\\|\\theta\\|_2$ can be large while $X\\theta$ is moderate, which is what happens when columns of $X$ are small or nearly collinear. Later regularization chapters penalize $\\|\\theta\\|_2$ or $\\|\\theta\\|_1$. They are changing the parameter, not directly the residual, unless the design is well scaled.""",
        derivation="""Suppose $X^\\top X$ is invertible and $\\hat{\\theta} = (X^\\top X)^{-1} X^\\top y$. The fitted vector is $\\hat{y} = P y$ with $P = X (X^\\top X)^{-1} X^\\top$. Then $P^2 = P$ and $P^\\top = P$, so $P$ is the orthogonal projector onto $\\mathrm{col}(X)$. For any $v = X a$,

$$(y - \\hat{y})^\\top v = y^\\top (I - P) X a = 0,$$

because $(I - P) X = 0$. So the residual is orthogonal to every column. If $X = U \\Sigma V^\\top$ is a thin SVD with $\\mathrm{rank}(X) = r$, the same projector is $U_r U_r^\\top$, and the least-squares solution of minimal Euclidean norm is $\\hat{\\theta} = V_r \\Sigma_r^{-1} U_r^\\top y$.""",
        code="""import numpy as np

def svd_least_squares(X, y, tol=1e-10):
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    keep = s > tol
    coef = np.zeros(X.shape[1])
    if np.any(keep):
        coef = vt[keep].T @ ((u[:, keep].T @ y) / s[keep])
    fitted = X @ coef
    residual_norm = np.linalg.norm(y - fitted)
    return coef, fitted, float(residual_norm), int(np.sum(keep))
""",
        experiment="""**Hypothesis.** The residual from minimal-norm least squares is numerically orthogonal to the column space, including when two columns are copies.

**Protocol.** Build a rank-2 matrix in $\\mathbb{R}^{30 \\times 4}$ by drawing two independent columns and repeating each once. Draw $y$ at random. Compute `svd_least_squares`. Measure $\\|X^\\top (y - \\hat{y})\\|_\\infty$ and the reported rank.

**Metric.** The infinity norm of $X^\\top$ times the residual, and the gap between the kept and discarded singular values.

**Decision rule.** The orthogonality residual should be near $10^{-8}$ or smaller for this scale. A reported rank other than 2 means the tolerance does not match the singular-value gap.""",
        project="""**Deliverable.** A script that, for a feature matrix you care about, prints rank, condition number of the kept singular values, residual norm, and whether an intercept column is redundant.

**Constraints.** Do not drop columns silently. Report them.

**Rubric.** The script distinguishes "predictions are fine" from "parameters are not unique." """,
        beginner="Compute rank and a least-squares fit for a 4 by 2 example by hand and by code.",
        university="Prove that $P$ is the orthogonal projector and that the SVD solution has minimal norm.",
        research="State the condition number $\\sigma_{\\max}/\\sigma_{\\min}$ and how a small singular value amplifies noise in $y$ into $\\hat{\\theta}$. Relate this to ridge regression in Volume 02.",
        failures="""- Inverting $X^\\top X$ when it is singular or nearly singular.
- Comparing $\\|\\theta\\|_2$ across features that have not been scaled.
- Interpreting a large coefficient on a duplicated column as a real effect.""",
        see_also="Volume 02 uses the same SVD to describe ridge shrinkage. Volume 08 uses projectors again for principal components.",
    ),
)

chapter(
    v,
    "04_matrix_calculus",
    "Matrix calculus",
    entry(
        question="How do you differentiate a scalar loss with respect to a vector or a matrix without dropping a transpose?",
        card=[
            ("Layout", "The gradient $\\nabla_\\theta L$ is a column vector, the same shape as $\\theta$."),
            ("Differential", "$dL = \\langle \\nabla_\\theta L, d\\theta \\rangle$. This identity defines the gradient."),
            ("Linear", "$d(A\\theta) = A\\, d\\theta$. $d(\\theta^\\top A \\theta) = \\theta^\\top (A + A^\\top) d\\theta$ if you do not assume symmetry yet."),
            ("Quadratic", "If $A = A^\\top$, then $\\nabla_\\theta (\\theta^\\top A \\theta) = 2 A \\theta$."),
            ("Least squares", "$\\nabla_\\theta \\lVert X\\theta - y \\rVert_2^2 = 2 X^\\top (X\\theta - y)$."),
            ("Chain rule", "If $z = W x$ and $L$ depends on $z$, then $\\nabla_W L = (\\nabla_z L) x^\\top$."),
        ],
        theory="""Code autodiff and a hand derivative answer the same question: which direction of a small change in the parameter increases the scalar loss. The differential $dL$ is the linear part of that change. Once you write $dL$ as a dot product against $d\\theta$, the vector on the other side is the gradient. This is safer than memorizing a table, because the shape checks itself.

Matrix parameters follow the same rule with the Frobenius inner product $\\langle A, B \\rangle = \\sum_{ij} A_{ij} B_{ij} = \\mathrm{tr}(A^\\top B)$. If $dL = \\langle G, dW \\rangle_F$, then $\\nabla_W L = G$. The outer product $(\\nabla_z L) x^\\top$ is the gradient of a loss that sees $z = Wx$, and it is the rank-one update used throughout backpropagation.""",
        derivation="""Let $r = X\\theta - y$ and $L = r^\\top r$. Then $dr = X\\, d\\theta$ and

$$dL = 2 r^\\top dr = 2 r^\\top X\\, d\\theta = \\langle 2 X^\\top r, d\\theta \\rangle.$$

Hence $\\nabla_\\theta L = 2 X^\\top (X\\theta - y)$. Setting it to zero recovers the normal equation.

For $z = Wx$ with $L = \\ell(z)$, the differential is $dz = dW\\, x$, so

$$dL = \\langle \\nabla_z L, dW\\, x \\rangle = \\langle (\\nabla_z L) x^\\top, dW \\rangle_F.$$

Therefore $\\nabla_W L = (\\nabla_z L) x^\\top$. If a bias $b$ is added, $dz$ also contains $db$, and $\\nabla_b L = \\nabla_z L$.""",
        code="""import numpy as np

def mse_grad(theta, X, y):
    residual = X @ theta - y
    return 2.0 * X.T @ residual

def numerical_grad(loss, theta, eps=1e-6):
    grad = np.zeros_like(theta)
    for j in range(theta.size):
        step = np.zeros_like(theta)
        step[j] = eps
        grad[j] = (loss(theta + step) - loss(theta - step)) / (2 * eps)
    return grad
""",
        experiment="""**Hypothesis.** The analytic least-squares gradient matches the central-difference gradient on a random instance, to an accuracy limited by the step size.

**Protocol.** Draw $X \\in \\mathbb{R}^{25 \\times 4}$, $\\theta$, and $y$ from a standard normal. Compare `mse_grad` with `numerical_grad` of $L(\\theta) = \\lVert X\\theta - y \\rVert_2^2$ at `eps` in $\\{10^{-4}, 10^{-6}, 10^{-8}\\}$.

**Metric.** $\\|g_{\\mathrm{ana}} - g_{\\mathrm{num}}\\|_\\infty / (1 + \\|g_{\\mathrm{ana}}\\|_\\infty)$.

**Decision rule.** The relative gap should be smallest near $10^{-6}$ and worse at $10^{-8}$ because subtraction cancels digits. If $10^{-6}$ is already wrong in the third digit, the analytic gradient is wrong.""",
        project="""**Deliverable.** A gradient-check helper that takes any scalar loss and a parameter vector, plus a short note on a case you intentionally broke (for example, forgetting the factor 2).

**Constraints.** The helper must report the relative gap, not only a boolean.

**Rubric.** You can explain why a too-small `eps` fails even when the mathematics is right.""",
        beginner="Check the gradient of $L(\\theta) = \\theta^\\top \\theta$ by hand and with the numerical helper.",
        university="Derive $\\nabla_W L$ for $z = Wx + b$ and a squared loss on $z$.",
        research="Write the differential of $\\log\\det G(\\theta)$ for a symmetric positive-definite $G$, and name a model in Volume 04 that needs it.",
        failures="""- Treating the gradient of a vector-valued map as if it were a column. That object is a Jacobian.
- Dropping a transpose so that a $(d,)$ array is added to a $(1, d)$ array and broadcasts.
- Trusting autodiff on a loss that has detached a tensor or stopped the graph.""",
        see_also="Volume 09 uses the same outer-product gradient inside backpropagation. Volume 02 uses gradients for descent.",
    ),
)

chapter(
    v,
    "05_numerical_linear_algebra",
    "Numerical linear algebra",
    entry(
        question="When is a formula that is true in exact arithmetic the wrong thing to compute?",
        card=[
            ("Condition", "For $A\\theta = b$, relative errors can grow by about $\\kappa_2(A) = \\sigma_{\\max}(A)/\\sigma_{\\min}(A)$."),
            ("Stability", "An algorithm is stable when it returns the exact answer to a nearby problem."),
            ("Prefer", "`solve` on a linear system, `lstsq` or a thin SVD on least squares, a Cholesky factor when the matrix is SPD."),
            ("Avoid", "Forming an explicit inverse, or forming $X^\\top X$ when $X$ itself is ill-conditioned."),
            ("Float", "float64 has about 16 decimal digits. Cancellation removes leading digits before you round."),
        ],
        theory="""Exact identities are not execution plans. The normal equation squares the condition number: $\\kappa_2(X^\\top X) = \\kappa_2(X)^2$. A design that is merely somewhat collinear becomes a Gram matrix that is unusable in float64. QR or SVD on $X$ avoids that squaring.

An ill-conditioned problem is hard even for a perfect algorithm, because the true answer moves a lot when $y$ moves a little. An unstable algorithm is worse: it adds error that the problem did not require. Forming $A^{-1}$ and multiplying is a common unstable habit. Solving $A\\theta = b$ with a factorization answers the same mathematics and perturbs the data less.

Scaling features before forming a Gram matrix is not a cosmetic preprocessing step. It changes $\\kappa_2$. Report the condition number you actually computed.""",
        derivation="""Let $X = U \\Sigma V^\\top$ with singular values $\\sigma_j$. The eigenvalues of $X^\\top X$ are $\\sigma_j^2$. If $\\sigma_{\\min}/\\sigma_{\\max} = 10^{-8}$, then the Gram matrix has eigenvalue ratio $10^{-16}$, which is at the edge of float64. A perturbation of size $10^{-16}$ in the Gram matrix is then as large as the smallest eigenvalue, and the computed inverse is noise.

The minimal-norm least-squares map is $\\theta = \\sum_{\\sigma_j > 0} \\sigma_j^{-1} (u_j^\\top y) v_j$. Truncating singular values below a tolerance $\\tau$ replaces an explosive $1/\\sigma_j$ with zero. That is a modeling choice: you refuse to fit directions the data do not identify. Ridge regression makes a related choice by replacing $\\sigma_j^{-1}$ with $\\sigma_j / (\\sigma_j^2 + \\lambda)$.""",
        code="""import numpy as np

def condition_report(X):
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 0]
    kappa = float(s[0] / s[-1]) if s.size else np.inf
    return {"singular_values": s, "kappa": kappa, "gram_kappa_if_formed": kappa**2}

def ridge_from_svd(X, y, lam):
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    shrink = s / (s**2 + lam)
    return vt.T @ (shrink * (u.T @ y))
""",
        experiment="""**Hypothesis.** Solving the normal equation in float64 loses accuracy before an SVD solver does, once columns are nearly collinear.

**Protocol.** Let the columns of $X$ be $v$, $v + 10^{-8} z$, and an independent third column, with $v, z$ standard normal in $\\mathbb{R}^{200}$. Let $y = X \\theta_\\star$ with no noise. Compare $\\hat{\\theta}$ from `lstsq` on $X^\\top X$ against `ridge_from_svd` with $\\lambda = 0$ implemented as a tiny $\\lambda = 10^{-12}$, and against `np.linalg.lstsq(X, y)`.

**Metric.** $\\|\\hat{\\theta} - \\theta_\\star\\|_2 / \\|\\theta_\\star\\|_2$ for each solver, and $\\kappa_2(X)$.

**Decision rule.** The solver that forms the Gram matrix should be worse. If it is not, the columns are not collinear enough; shrink the $10^{-8}$.""",
        project="""**Deliverable.** A preprocessor that standardizes columns, refuses to drop them, prints $\\kappa_2$ before and after, and fits least squares with `lstsq` on $X$ rather than on the Gram matrix.

**Constraints.** Keep a column of ones unscaled, or handle the intercept separately, and say which you did.

**Rubric.** On the collinear construction above, your relative parameter error is in the same ballpark as `lstsq` on $X$, not worse by orders of magnitude.""",
        beginner="Print $\\kappa_2$ for a well-scaled matrix and for the collinear construction.",
        university="Show that the eigenvalues of $X^\\top X$ are the squared singular values.",
        research="Compare truncation of small singular values with the ridge shrink factor at matched effective degrees of freedom.",
        failures="""- Using `np.linalg.inv` as a default.
- Comparing algorithms on well-conditioned toy data and concluding they are equivalent on real data.
- Standardizing using statistics computed on the test split.""",
        see_also="Volume 02 for ridge as a numerical and a statistical method. Volume 06 for feature scaling inside model pipelines.",
    ),
)

chapter(
    v,
    "06_least_squares_project",
    "Project: least squares from the axioms",
    entry(
        question="Can you build an identified, numerically stable least-squares baseline and know what it does not estimate?",
        card=[
            ("Estimator", "$\\hat{\\theta} \\in \\arg\\min_\\theta \\lVert X\\theta - y \\rVert_2^2$, choosing the minimal-norm solution if needed."),
            ("Identification", "Unique $\\theta$ if and only if $\\mathrm{rank}(X) = d$."),
            ("Prediction", "Unique fitted vector $\\hat{y} = P_{\\mathrm{col}(X)} y$ even when $\\theta$ is not unique."),
            ("Noise model", "If $y = X\\theta_\\star + \\varepsilon$ and $\\mathbb{E}[\\varepsilon \\mid X] = 0$, then $\\mathbb{E}[\\hat{\\theta} \\mid X] = \\theta_\\star$ when $X$ has full column rank."),
            ("Out of scope", "This chapter does not claim that squared error is the right loss for counts, ranks, or heavy tails."),
        ],
        theory="""Least squares is the first complete object in the series: a loss, a geometric estimator, a numerical method, and a statistical claim with an assumption. The statistical claim is about the conditional mean. If $\\mathbb{E}[Y \\mid X = x] = x^\\top \\theta_\\star$, squared error is a coherent way to estimate that function, because the conditional mean minimizes expected squared error over all predictors. The linear restriction can still be wrong. Unbiasedness of $\\hat{\\theta}$ given $X$ does not imply that $x \\mapsto x^\\top \\theta_\\star$ is the conditional mean. It implies unbiasedness inside a model that may be false.

Use this chapter as the pattern for later projects. The artifact is not a plot alone. It is a statement of the target, the assumption, the algorithm, a check on a known case, a check on a broken case, and a decision about whether the estimator is usable for prediction, for parameter interpretation, or for neither.""",
        derivation="""Among all functions $g$, $\\mathbb{E}[(Y - g(X))^2]$ is minimized by $g_\\star(x) = \\mathbb{E}[Y \\mid X = x]$, when the expectation exists. Restricting $g$ to linear functions $x \\mapsto x^\\top \\theta$ yields a different problem, whose population solution satisfies $\\mathbb{E}[X X^\\top] \\theta = \\mathbb{E}[X Y]$ when the Gram matrix is invertible. The sample normal equation replaces those expectations by averages. Nothing in that derivation forces $g_\\star$ itself to be linear. The gap $\\mathbb{E}[(g_\\star(X) - X^\\top \\theta)^2]$ is approximation error. Later volumes separate it from estimation error.""",
        code="""import numpy as np

def fit_predict(X, y, X_new):
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta, X @ theta, X_new @ theta

def known_case():
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    theta_star = np.array([1.0, 2.0])
    y = X @ theta_star
    theta, fitted, _ = fit_predict(X, y, X)
    return np.allclose(theta, theta_star), np.allclose(fitted, y)
""",
        experiment="""**Hypothesis.** With a correct linear conditional mean and full rank, the parameter error shrinks as $n$ grows. With a quadratic conditional mean, prediction error on a fresh sample stays bounded away from zero.

**Protocol.** Case A: $x \\sim \\mathrm{Uniform}[-1, 1]$, $Y = 1 + 2x + \\varepsilon$, $\\varepsilon \\sim \\mathcal{N}(0, 0.1^2)$. Case B: $Y = x^2 + \\varepsilon$. Fit an intercept and a slope. Repeat 50 times at $n \\in \\{32, 128, 512\\}$. Hold out 200 fresh points each time.

**Metric.** Median $\\|\\hat{\\theta} - \\theta_\\star\\|_2$ in case A. Median held-out mean squared error in both cases.

**Decision rule.** Case A parameter error should drop roughly like $1/\\sqrt{n}$. Case B test error should stop improving once the linear approximation error dominates. If case B test error goes to zero, the data generator is wrong.""",
        project="""**Deliverable.** A short report with the known-case check, the collinear non-uniqueness check from the linear-algebra chapter, and the two simulation cases. Include one table and a statement of what you would refuse to interpret.

**Constraints.** No library linear-model object. `lstsq` or SVD only, so the numerics stay visible.

**Rubric.** The report says whether you are estimating a parameter, a conditional mean, or a linear approximation to a conditional mean, and it matches the evidence.""",
        beginner="Make `known_case` pass. Fit one line to ten points you choose.",
        university="Run both simulation cases and write the decision paragraph.",
        research="Add a heavy-tailed noise law (Student-$t$ with 3 degrees of freedom) and compare squared error with a Huber fit you implement by reweighted least squares. Say which functional of the conditional distribution each one estimates.",
        failures="""- Reporting coefficient signs when columns are collinear.
- Quoting in-sample $R^2$ as if it were risk.
- Centering or scaling using the full sample before a train-test split.""",
        see_also="Volume 06 returns to this estimator as a baseline. Volume 07 shows how to compare it with another model without fooling yourself.",
    ),
)

v = volume(
    2,
    "Calculus and optimization",
    "Beginner moving into university",
    "A reader who can differentiate a polynomial and has finished Volume 01.",
    "How does a training algorithm move in parameter space, and which guarantees require convexity?",
)

chapter(
    v,
    "01_gradients",
    "Derivatives, gradients, and Jacobians",
    entry(
        question="What does a derivative say, and which object do you need for a vector parameter?",
        card=[
            ("Derivative", "For $f: \\mathbb{R} \\to \\mathbb{R}$, $f'(t)$ is the slope of the best linear approximation at $t$."),
            ("Gradient", "For $L: \\mathbb{R}^d \\to \\mathbb{R}$, $\\nabla L(\\theta)$ satisfies $L(\\theta + h) = L(\\theta) + \\nabla L(\\theta)^\\top h + o(\\|h\\|)$."),
            ("Jacobian", "For $F: \\mathbb{R}^d \\to \\mathbb{R}^k$, $J_F(\\theta) \\in \\mathbb{R}^{k \\times d}$ is the matrix of partial derivatives."),
            ("Chain", "$J_{F \\circ G}(\\theta) = J_F(G(\\theta)) J_G(\\theta)$."),
            ("Descent", "For small $\\eta > 0$, $\\theta^+ = \\theta - \\eta \\nabla L(\\theta)$ decreases $L$ if the gradient is nonzero and $L$ is differentiable."),
        ],
        theory="""A derivative is a local linear model, not a slogan about change. The gradient is the direction in parameter space in which that linear model increases fastest per unit Euclidean length. Steepest descent walks the opposite way. The step length $\\eta$ is part of the method. The gradient does not choose it.

When the loss depends on parameters through a vector of predictions, the Jacobian is the object that carries derivatives through the chain rule. Backpropagation computes one row of interest, the gradient of a scalar, without forming the full Jacobian. You still need to know the Jacobian exists as the thing being multiplied.

Differentiability can fail at kinks, as with a ReLU at 0 or an absolute value. Subgradients are a later, convex repair. For now, check kinks explicitly or use a smooth surrogate when you need a classical gradient.""",
        derivation="""Let $L(\\theta + h) = L(\\theta) + g^\\top h + r(h)$ with $r(h)/\\|h\\| \\to 0$. Then

$$L(\\theta - \\eta g) = L(\\theta) - \\eta \\lVert g \\rVert_2^2 + r(-\\eta g).$$

Choose $\\eta$ small enough that $|r(-\\eta g)| \\le (\\eta/2) \\lVert g \\rVert_2^2$. If $g \\ne 0$, the right-hand side is at most $L(\\theta) - (\\eta/2)\\lVert g \\rVert_2^2$. So some positive step decreases the loss. This argument does not say that a constant step works for every $\\eta$, and it does not say the iteration converges to a global minimizer.""",
        code="""import numpy as np

def jacobian_rows(f, theta, eps=1e-6):
    base = f(theta)
    rows = []
    for j in range(theta.size):
        step = np.zeros_like(theta, dtype=float)
        step[j] = eps
        rows.append((f(theta + step) - base) / eps)
    return np.vstack(rows)
""",
        experiment="""**Hypothesis.** Forward differences recover the Jacobian of $F(\\theta) = (\\theta_1^2, \\theta_1\\theta_2)$ at $(2, -1)$, whose exact Jacobian is $\\begin{bmatrix} 2\\theta_1 & 0 \\\\ \\theta_2 & \\theta_1 \\end{bmatrix}$.

**Protocol.** Compare `jacobian_rows` at three step sizes.

**Metric.** Max absolute entrywise error.

**Decision rule.** Error should fall as `eps` shrinks, until float noise dominates. A systematic factor-of-two error means an index bug, not a step-size issue.""",
        project="""**Deliverable.** A function `descend(L, grad, theta0, eta, steps)` that records the loss every step, plus a plot or table for $L(\\theta) = \\theta^\\top \\theta$ and for $L(\\theta) = \\theta^4$ in one dimension, where a large constant step oscillates.

**Constraints.** Refuse a non-finite loss and stop.

**Rubric.** You can point to a step size that decreases $L$ and one that does not, on the second function.""",
        beginner="Differentiate three scalar functions by hand and match them with central differences.",
        university="Write the descent inequality with the little-o remainder in your own words.",
        research="Give an example of a differentiable function, bounded below, whose gradient descent with any fixed step fails to approach the infimum. Start from a function with increasingly narrow valleys if you need a picture, and then look up a standard counterexample before claiming one.",
        failures="""- Calling a Jacobian a gradient.
- Using a step size copied from another model.
- Ignoring non-finite values after a too-large step.""",
        see_also="The stochastic-gradient chapter later in this volume. Volume 10 for practical step-size schedules.",
    ),
)

chapter(
    v,
    "02_taylor",
    "Taylor expansions and local models",
    entry(
        question="What local polynomial is a training step actually using?",
        card=[
            ("First order", "$L(\\theta + h) \\approx L(\\theta) + \\nabla L(\\theta)^\\top h$."),
            ("Second order", "Add $\\tfrac{1}{2} h^\\top H(\\theta) h$, where $H_{ij} = \\partial_i \\partial_j L(\\theta)$."),
            ("PSD Hessian", "If $H(\\theta) \\succeq 0$, the quadratic model is convex in $h$."),
            ("Newton", "If $H$ is SPD, the quadratic model is minimized at $h = -H^{-1} \\nabla L$."),
            ("Use", "First order is cheap and global only by repetition. Second order is local and expensive."),
        ],
        theory="""Gradient descent trusts a linear model of the loss and then takes a short step so that the trust is not absurd. Newton's method trusts a quadratic model and jumps to that model's minimizer. When the Hessian is positive definite and the quadratic approximation is accurate, Newton can be dramatically faster. When the Hessian is indefinite, the quadratic model has no minimum, and the Newton step can climb.

In high dimension the Hessian is $d \\times d$. For a network with millions of parameters you do not form it. You still use the Taylor picture to understand curvature: a learning rate that is safe in a flat direction can explode in a sharp one. Adaptive methods and normalization are attempts to rescale those directions without forming $H$.""",
        derivation="""The second-order expansion with remainder says

$$L(\\theta + h) = L(\\theta) + g^\\top h + \\tfrac{1}{2} h^\\top H(\\theta) h + r_3(h),$$

where $r_3(h)/\\|h\\|^2 \\to 0$ as $h \\to 0$ if third derivatives exist and are continuous. Drop $r_3$ and differentiate in $h$: $g + H h = 0$. If $H$ is invertible, $h = -H^{-1} g$.

If $H$ has an eigenvalue $\\lambda < 0$ with eigenvector $u$, then along $h = t u$ the quadratic term is $(\\lambda/2) t^2 \\to -\\infty$ as $|t|$ grows. The local model is then not a bowl.""",
        code="""import numpy as np

def newton_step(grad, hess):
    return -np.linalg.solve(hess, grad)

def quadratic_bowl():
    # L(theta) = 0.5 * theta^T diag(1, 100) theta
    g = np.array([0.2, -4.0])
    H = np.diag([1.0, 100.0])
    return newton_step(g, H)
""",
        experiment="""**Hypothesis.** On a quadratic bowl, one Newton step from any start reaches the minimizer at the origin, while gradient descent with a constant step that is safe for the sharp axis is slow on the flat axis.

**Protocol.** Take $L(\\theta) = \\tfrac{1}{2}(\\theta_1^2 + 100 \\theta_2^2)$. Start at $(1, 1)$. Run 50 gradient steps at $\\eta = 1/100$ and one Newton step.

**Metric.** Distance to the origin after one Newton step, and after 50 gradient steps.

**Decision rule.** Newton should land within $10^{-8}$. Gradient descent should still have a visible $\\theta_1$ component. If Newton does not land, `solve` was applied to the wrong Hessian.""",
        project="""**Deliverable.** A contour or a table of the bowl, the Newton step, and a divergent gradient step at $\\eta = 0.1$.

**Constraints.** Annotate the sharp and flat axes with their curvatures 100 and 1.

**Rubric.** A reader can see why one learning rate is not a property of the loss alone.""",
        beginner="Expand $e^t$ and $\\log(1 + t)$ through second order and name the remainder.",
        university="Derive the Newton step and run the bowl experiment.",
        research="Replace the Hessian by a diagonal approximation and measure the extra iterations on this bowl. This is a toy version of the compromise adaptive optimizers make.",
        failures="""- Using Newton when $H$ is indefinite and treating the step as progress.
- Reading a local quadratic decrease as a global convergence proof.
- Forming a dense Hessian for a model whose $d$ makes that matrix larger than memory.""",
        see_also="Volume 10 discusses curvature without forming $H$. Convex functions in the next chapter are where the local bowl is global.",
    ),
)

chapter(
    v,
    "03_convexity",
    "Convex sets and convex functions",
    entry(
        question="When is every local minimizer a global one?",
        card=[
            ("Set", "$C$ is convex if $x, y \\in C$ and $t \\in [0, 1]$ imply $tx + (1 - t)y \\in C$."),
            ("Function", "$f$ is convex when $\\mathrm{epi} f = \\{(x, r): r \\ge f(x)\\}$ is convex."),
            ("Inequality", "$f(tx + (1-t)y) \\le t f(x) + (1-t) f(y)$."),
            ("Smooth", "Differentiable $f$ is convex iff $f(y) \\ge f(x) + \\nabla f(x)^\\top (y - x)$."),
            ("Strong", "$f(y) \\ge f(x) + \\nabla f(x)^\\top (y - x) + (\\mu/2)\\lVert y - x \\rVert_2^2$ for some $\\mu > 0$."),
            ("Payoff", "A local minimum of a convex function on a convex set is global. Strict convexity gives uniqueness."),
        ],
        theory="""Convexity is the assumption that makes optimization theory look like a guarantee rather than a heuristic. Linear least squares with a convex constraint, logistic regression, and ridge regression are convex in the parameters when the predictor is linear in those parameters. Deep networks are not convex in their weights. Carrying a convex guarantee into a nonconvex training run is a category error.

The first-order inequality says the graph sits above its tangents. That is why a point with $\\nabla f(x) = 0$ is a global minimizer when $f$ is convex and differentiable: every other point is at least $f(x)$. Without convexity, a zero gradient might be a saddle or a poor local minimum.

Strong convexity adds a quadratic bowl under the function and gives a unique minimizer and a linear rate for gradient descent with a suitable step. Squared error with a full-rank design and an L2 penalty is the prototype.""",
        derivation="""Let $f$ be convex and differentiable, and suppose $\\nabla f(x_\\star) = 0$. The tangent inequality at $x_\\star$ says $f(y) \\ge f(x_\\star)$ for every $y$. So $x_\\star$ is a global minimizer.

Conversely, if $x_\\star$ is a local minimizer and $f$ is convex, take any $y$ and any small $t > 0$ so that $x_t = (1 - t) x_\\star + t y$ lies in the local neighborhood. Then $f(x_\\star) \\le f(x_t) \\le (1 - t) f(x_\\star) + t f(y)$, which rearranges to $f(x_\\star) \\le f(y)$. A local minimizer is global.

For $f(\\theta) = \\tfrac{1}{2}\\lVert X\\theta - y \\rVert_2^2 + (\\lambda/2)\\lVert \\theta \\rVert_2^2$ with $\\lambda > 0$, the Hessian is $X^\\top X + \\lambda I \\succeq \\lambda I$, so $f$ is strongly convex with modulus $\\lambda$.""",
        code="""import numpy as np

def ridge_objective(theta, X, y, lam):
    resid = X @ theta - y
    return 0.5 * np.dot(resid, resid) + 0.5 * lam * np.dot(theta, theta)

def is_above_tangent(f, grad, x, y):
    return f(y) + 1e-8 >= f(x) + np.dot(grad(x), y - x)
""",
        experiment="""**Hypothesis.** The ridge objective lies above its tangents. The nonconvex function $f(t) = t^4 - t^2$ does not, at $t = 0$, where the derivative is zero and nearby values are negative.

**Protocol.** Check the tangent inequality for ridge at 20 random pairs $(x, y)$ in $\\mathbb{R}^5$. Check $f(0.1)$ against the tangent of $f$ at 0.

**Metric.** The smallest slack $f(y) - f(x) - \\nabla f(x)^\\top (y - x)$ for ridge, and the slack for the scalar nonconvex example.

**Decision rule.** Ridge slacks should be nonnegative up to $10^{-8}$. The scalar slack at $0.1$ should be negative. If ridge fails, the gradient of the objective is inconsistent with the function.""",
        project="""**Deliverable.** A note that classifies least squares, ridge, logistic regression with linear logits, and a one-hidden-layer network as convex or not convex in their parameters, with a one-line reason each.

**Constraints.** "Neural networks can have local minima" is not enough. Point at the composition of a nonlinear activation with a linear map.

**Rubric.** Each classification names the variable of convexity. A problem can be convex in $\\theta$ and nonconvex in a feature transformation you also learn.""",
        beginner="Draw a convex function and a function with two basins. Mark a point where the tangent inequality fails.",
        university="Prove the local-to-global argument in your own words.",
        research="Show that strong convexity plus smoothness $L$ implies that gradient descent with step $1/L$ contracts the distance to the minimizer by the factor $1 - \\mu/L$. State where a deep network violates the hypothesis.",
        failures="""- Calling a loss convex because the log loss is convex, after composing it with a nonlinear network.
- Using uniqueness of the prediction to claim uniqueness of the parameter.
- Importing linear convergence rates into nonconvex training logs.""",
        see_also="Volume 06 for logistic regression, which is convex in the linear weights. Volume 09 for the point where convexity is given up on purpose.",
    ),
)

chapter(
    v,
    "04_constraints",
    "Constraints and optimality",
    entry(
        question="What changes when the parameter must lie in a set, such as a simplex or a box?",
        card=[
            ("Problem", "Minimize $f(\\theta)$ subject to $\\theta \\in C$, with $C$ convex."),
            ("Projection", "$\\Pi_C(z) = \\arg\\min_{\\theta \\in C} \\lVert \\theta - z \\rVert_2^2$ when the minimizer exists and is unique."),
            ("Projected step", "$\\theta^+ = \\Pi_C(\\theta - \\eta \\nabla f(\\theta))$."),
            ("Simplex", "Nonnegative vectors summing to 1. Projection is a standard algorithm, not a softmax."),
            ("Stationarity", "For differentiable convex $f$, $\\theta_\\star$ is optimal iff $\\nabla f(\\theta_\\star)^\\top (\\theta - \\theta_\\star) \\ge 0$ for every $\\theta \\in C$."),
        ],
        theory="""Unconstrained stationarity is $\\nabla f = 0$. On a constraint set, the negative gradient may point outside $C$. The right first-order test is that no feasible direction is a descent direction. Projected gradient descent follows the gradient and then returns to $C$ by Euclidean projection.

Softmax outputs lie in the simplex, but softmax is a parameterization of an unconstrained vector, not a projection. Projection onto the simplex is what you want if the parameter itself must be a probability vector, as in some mixture weights or discrete optimal-transport problems. Box constraints $\\theta_j \\in [0, 1]$ project by clipping. Clipping is not a general substitute for projection onto other sets.

KKT conditions add multipliers for equality and inequality constraints. You need them when you want a certificate, dual variables, or a Lagrangian method. For many ML constraints, projection is the algorithm and the variational inequality above is the check.""",
        derivation="""Let $f$ be convex and differentiable. Suppose $\\nabla f(\\theta_\\star)^\\top (\\theta - \\theta_\\star) \\ge 0$ for all $\\theta \\in C$. Convexity gives $f(\\theta) \\ge f(\\theta_\\star) + \\nabla f(\\theta_\\star)^\\top (\\theta - \\theta_\\star) \\ge f(\\theta_\\star)$. So $\\theta_\\star$ is optimal.

If $\\theta_\\star$ is interior to $C$, every small direction is feasible, which forces $\\nabla f(\\theta_\\star) = 0$. The unconstrained test is the special case $C = \\mathbb{R}^d$.

Euclidean projection onto the unit box $[0, 1]^d$ is coordinatewise clipping, because the squared distance separates across coordinates. Projection onto the probability simplex does not separate in that naive way once the sum constraint binds.""",
        code="""import numpy as np

def project_box(theta, low=0.0, high=1.0):
    return np.clip(theta, low, high)

def projected_step(theta, grad, eta):
    return project_box(theta - eta * grad)
""",
        experiment="""**Hypothesis.** Minimizing $(\\theta - 2)^2$ on $[0, 1]$ by projected gradient descent ends at 1, and the variational inequality holds there even though the derivative is not zero.

**Protocol.** Start at 0.2, use $\\eta = 0.1$, run 50 steps of `projected_step` with gradient $2(\\theta - 2)$. Evaluate $\\nabla f(1) \\cdot (\\theta - 1)$ for 20 random $\\theta \\in [0, 1]$.

**Metric.** Final $\\theta$, and the most negative directional derivative you find inside the interval.

**Decision rule.** The final value should be 1, and every feasible directional derivative should be nonnegative. A final value of 2 means you forgot to project.""",
        project="""**Deliverable.** Projected gradient descent for nonnegative least squares: $\\min_{\\theta \\ge 0} \\lVert X\\theta - y \\rVert_2^2$, with clipping as the projection.

**Constraints.** Compare with unconstrained least squares on a problem whose true $\\theta_\\star$ has a negative coordinate.

**Rubric.** You report both the objective value and the constraint violation, which must be zero for the projected run.""",
        beginner="Project ten random numbers onto $[0, 1]$ and onto the nonnegative reals.",
        university="Prove the box-constrained optimality argument for $(\\theta - 2)^2$ on $[0, 1]$.",
        research="Implement projection onto the probability simplex and test it on a 5-dimensional vector. Use the variational inequality as the unit test.",
        failures="""- Using softmax and calling it a Euclidean projection.
- Checking only $\\nabla f = 0$ on a problem whose solution is on the boundary.
- Projecting after every epoch of an algorithm whose theory assumed an unconstrained step, and still quoting the unconstrained rate.""",
        see_also="Volume 08 for mixture weights. Volume 14 if you later parameterize a policy on a simplex.",
    ),
)

chapter(
    v,
    "05_stochastic_gradients",
    "Stochastic gradients",
    entry(
        question="What remains true when each step uses a minibatch instead of the full gradient?",
        card=[
            ("Full gradient", "$g(\\theta) = \\nabla \\hat{R}_n(\\theta) = n^{-1} \\sum_{i=1}^{n} \\nabla \\ell_i(\\theta)$."),
            ("Unbiased minibatch", "If $I$ is a random index, uniform on $1, \\ldots, n$, then $\\mathbb{E}[\\nabla \\ell_I(\\theta) \\mid \\theta] = g(\\theta)$."),
            ("Noise", "The step is $g(\\theta) + \\xi$ with $\\mathbb{E}[\\xi \\mid \\theta] = 0$. Variance need not be zero."),
            ("Step sizes", "A constant step can hover. Steps with $\\sum \\eta_t = \\infty$ and $\\sum \\eta_t^2 < \\infty$ are the classical convergent schedule for noisy gradients."),
            ("Batch", "Averaging $B$ independent indices cuts the variance by about $B$, at $B$ times the cost per step."),
        ],
        theory="""Empirical risk is an average. Its gradient is an average of per-example gradients. A minibatch gradient is a Monte Carlo estimate of that average. Unbiasedness says the estimate points the right way on average. It does not say a given step descends.

Variance is why training curves jitter and why a learning rate that worked on the full gradient can diverge on batch size 1. Increasing the batch reduces variance and raises the cost of each step. The useful comparison is loss decrease per unit of compute, not per iteration.

A constant step size on a strongly convex smooth objective converges to a neighborhood whose radius shrinks if you later decrease the step. People anneal learning rates for this reason, not only for folklore. In nonconvex deep learning the same update is used without the guarantee. You still diagnose it as noisy descent: if the batch gradient is systematically biased, by a bug in shuffling or by dropout left in the wrong mode, the theory you think you are using is not the algorithm you are running.""",
        derivation="""Let $\\ell_i$ be the loss on example $i$ and $g_B = B^{-1} \\sum_{b=1}^{B} \\nabla \\ell_{I_b}(\\theta)$ with $I_b$ drawn uniformly with replacement. Linearity of expectation gives $\\mathbb{E}[g_B \\mid \\theta] = g(\\theta)$. If the per-example gradients have covariance bounded by $\\sigma^2 I$ and the draws are independent,

$$\\mathbb{E}[\\lVert g_B - g(\\theta) \\rVert_2^2 \\mid \\theta] \\le \\frac{C \\sigma^2}{B}$$

for a constant $C$ that depends only on dimension in the crude bound. The $1/B$ scaling is the part to remember. It is a variance calculation, not a claim about generalization to new data. The batch is still drawn from the training set.""",
        code="""import numpy as np

def full_grad(theta, X, y):
    return 2.0 * X.T @ (X @ theta - y) / X.shape[0]

def minibatch_grad(theta, X, y, batch, rng):
    idx = rng.integers(0, X.shape[0], size=batch)
    return full_grad(theta, X[idx], y[idx])
""",
        experiment="""**Hypothesis.** On a fixed $\\theta$, the mean of minibatch gradients approaches the full gradient, and the empirical variance of the gradient norm scales like $1/B$.

**Protocol.** Draw a least-squares problem with $n = 2000$, $d = 5$. At one $\\theta$, draw 200 minibatches for $B \\in \\{1, 8, 32\\}$.

**Metric.** Distance between the average minibatch gradient and the full gradient. Sample variance of $\\|g_B\\|_2$.

**Decision rule.** The mean gap should shrink as you average more batches. Variance should drop by roughly 4 when $B$ grows by 4. A bias that does not shrink means the index sampler is not uniform.""",
        project="""**Deliverable.** Stochastic gradient descent for least squares with batch sizes 1, 16, and $n$, same number of data touches, and a plot of training objective against data touches.

**Constraints.** Use a decreasing step $\\eta_t = \\eta_0 / \\sqrt{t}$ and also a constant step. Report both.

**Rubric.** You can say which schedule reached a lower objective per data touch on this convex problem, and you do not claim the same ranking for a neural network.""",
        beginner="At one parameter, average 100 single-example gradients and compare with the full gradient.",
        university="Run the variance experiment and the data-touch project.",
        research="Derive a neighborhood bound for constant-step SGD on a $\\mu$-strongly convex, $L$-smooth objective in terms of $\\eta$ and $\\sigma^2$. Identify which hypothesis a deep network drops.",
        failures="""- Confusing minibatch noise with test-set uncertainty.
- Forgetting to turn off dropout or data augmentation when you need the training gradient of a fixed loss.
- Comparing optimizers at equal iteration counts and unequal amounts of data read.""",
        see_also="Volume 10 for Adam and schedules. Volume 07 for the difference between training loss and risk.",
    ),
)

chapter(
    v,
    "06_gradient_check_project",
    "Project: a gradient-checking harness",
    entry(
        question="How do you certify a derivative before you trust a training run?",
        card=[
            ("Check", "Compare an analytic gradient with a central difference on a tiny problem."),
            ("Central", "$\\partial_j L(\\theta) \\approx (L(\\theta + \\varepsilon e_j) - L(\\theta - \\varepsilon e_j)) / (2\\varepsilon)$."),
            ("Relative gap", "$\\|g_a - g_n\\| / (1 + \\|g_a\\|)$, not a raw difference."),
            ("Where", "On a one-example or tiny-batch problem in float64, with a deterministic loss."),
            ("Not a proof", "Agreement at one point does not prove the code is right at every point."),
        ],
        theory="""A training run can decrease a loss for the wrong reason, or fail because the gradient is transposed. The cheap insurance is a finite-difference check on a problem small enough that $d$ extra loss evaluations are irrelevant. Central differences cancel the even part of the Taylor error and are worth the extra evaluation.

The harness belongs in the repository next to the model, and it should fail loudly. Relative error protects you when the gradient is large. You also test a point where the gradient is supposed to be zero, and a point where a constraint or a kink makes the classical gradient the wrong object. ReLU at exactly zero is a bad place to demand agreement unless you define the subgradient you implemented.

This project is the template for every later model you write by hand. Library autodiff still deserves a check against a hand derivative on one layer, because the bug is often in the loss reduction, the mask, or the parameter you forgot to register.""",
        derivation="""Taylor expansion gives

$$L(\\theta + \\varepsilon e_j) = L + \\varepsilon g_j + \\tfrac{1}{2}\\varepsilon^2 H_{jj} + O(\\varepsilon^3),$$

$$L(\\theta - \\varepsilon e_j) = L - \\varepsilon g_j + \\tfrac{1}{2}\\varepsilon^2 H_{jj} + O(\\varepsilon^3).$$

Subtract and divide by $2\\varepsilon$: the constant and the second-order term cancel, and the leading error is $O(\\varepsilon^2)$. Forward differences leave an $O(\\varepsilon)$ error. That is why the experiment should show central differences winning as $\\varepsilon$ decreases, until roundoff, which is about $\\mathrm{eps}_{\\mathrm{mach}} / \\varepsilon$ in the subtraction, starts to grow.""",
        code="""import numpy as np

def central_grad(loss, theta, eps=1e-6):
    grad = np.zeros_like(theta, dtype=float)
    for j in range(theta.size):
        step = np.zeros_like(theta, dtype=float)
        step[j] = eps
        grad[j] = (loss(theta + step) - loss(theta - step)) / (2.0 * eps)
    return grad

def relative_gap(a, b):
    return float(np.linalg.norm(a - b) / (1.0 + np.linalg.norm(a)))
""",
        experiment="""**Hypothesis.** For $L(\\theta) = \\lVert X\\theta - y \\rVert_2^2$, the relative gap between $2 X^\\top (X\\theta - y)$ and the central difference is smallest at a moderate $\\varepsilon$ and worse for both very large and very small $\\varepsilon$.

**Protocol.** Use $n = 8$, $d = 3$, float64, and $\\varepsilon \\in \\{10^{-2}, 10^{-4}, 10^{-6}, 10^{-8}\\}$.

**Metric.** Relative gap at each $\\varepsilon$.

**Decision rule.** $10^{-5}$ to $10^{-6}$ should be clearly better than $10^{-2}$. If every $\\varepsilon$ is bad, the analytic formula is wrong. If only the smallest $\\varepsilon$ is bad, you are seeing roundoff, and the formula may still be right.""",
        project="""**Deliverable.** A harness that registers several losses: a quadratic, least squares, and logistic loss on a linear logit (you may implement the logistic gradient from $\\sigma(x^\\top \\theta) - y$ times $x$). The harness prints a table of relative gaps and exits with a nonzero status if any gap at $\\varepsilon = 10^{-6}$ exceeds $10^{-6}$.

**Constraints.** Include one deliberately wrong gradient and show that the harness catches it.

**Rubric.** The command is one line, the table is saved, and the failure case is documented in the lab note.""",
        beginner="Check $L(\\theta) = \\theta_1^2 + 3\\theta_2$ only.",
        university="Complete the harness including the intentional bug.",
        research="Add a complex-step or a two-sided check on a matrix-shaped parameter using the Frobenius pairing, and document the kink you refuse to test with a classical gradient.",
        failures="""- Testing in float32 and treating noise as a bug.
- Checking only shapes, not values.
- Disabling the harness in the training configuration and never turning it back on.""",
        see_also="Volume 09, where this harness is the acceptance test for a hand-written backward pass.",
    ),
)
