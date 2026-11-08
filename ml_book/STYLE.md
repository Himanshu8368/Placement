# Chapter style guide (v2)

Every chapter is a hand-written Quarto file, `books/<part-folder>/NN_<slug>.qmd` (folders are named by topic, see PLAN.md §3). The model is UDL/MML for the math narrative and D2L for running code. On top of that, each chapter has worked examples, exercises with solutions, and interview Q&A.

## Depth standard (this is a comprehensive guide, not a short textbook)

A chapter is complete only when **all** of the following hold:

| Requirement | Minimum |
|---|---|
| Coverage | Everything the matching chapters of **at least 2 primary sources** cover (e.g. MML + PML1, ESL + ISLP, UDL + D2L). List the coverage checklist in the Sources box, and do not drop a subtopic because it is "advanced" |
| Length | About 25–60 printed pages. If a chapter would go past 60 pages, **split it into two chapters**. Never compress to save space |
| Derivations | Every main result is derived step by step. Results whose proofs are long are stated with a proof sketch and the exact source location |
| Variants and extensions | A table of the method's variants (what changes, when to use it, cost) |
| History and landmark papers | A short "Where this came from" note, plus a *Further reading* list of key papers with one line on each |
| Complexity | Time and memory cost for training and inference, wherever relevant |
| Practice | Hyperparameters and how to tune them, defaults that work, failure symptoms and fixes |
| Worked examples | ≥ 3 per chapter and ≥ 2 per major idea (plus a *By hand* example in architecture chapters) |
| Exercises | ≥ 15, mixing [E]/[M]/[H] and including coding exercises, each with a full solution |
| Interview questions | ≥ 12, covering conceptual, derivation, design, and "what goes wrong" questions |

## Skeleton

The number of sections is flexible. Use as many core sections as the topic needs.

```markdown
# Linear Algebra for Machine Learning {#sec-linalg}

::: {.callout-note title="Sources"}
MML ch. 2–4 · UDL App. B · CS229 linear algebra review · ISLP §…
:::

::: {.callout-tip title="What you will be able to do"}
- 3–6 concrete, checkable outcomes
:::

## Why this matters            <- short, motivating, ML-flavoured
## <Core idea 1> {#sec-linalg-xxx}
### Definition / intuition
### Derivation                  <- every step, nothing "left to the reader"
### Implementation              <- executable {python} cell, from scratch
### Check                       <- assert np.allclose(ours, library)
### Worked example 1 / 2        <- ≥2 per major idea, numbers shown
### By hand                     <- architecture chapters: one tiny forward (and backward) pass
                                   with integer-friendly matrices, every intermediate shown
                                   (AI-by-Hand style), then the same numbers reproduced in code
## <Core idea 2> …
## Variants and extensions       <- table
## Complexity                    <- train/inference time and memory
## Practical guide               <- hyperparameters, defaults, failure symptoms → fixes
## Pitfalls
## Where this came from          <- history + landmark papers
## Exercises                     <- ≥15, tagged [E]/[M]/[H]; each solution in a collapsed callout
## Interview questions           <- ≥12, each answer in a collapsed callout
## Summary                       <- "Key equations" callout + 6–10 bullet takeaways
## Further reading               <- papers and book sections, one line each
```

## Rules

- **No YAML `title:`.** The chapter title is the single H1 with its `{#sec-…}` label. Having both makes Quarto number the chapter twice.

- **Math.** Use `$…$` and `$$…$$` with labels `{#eq-name}` when an equation is referenced. Use one notation across the book (Appendix A): vectors are bold lowercase $\mathbf{x}$, matrices are bold uppercase $\mathbf{A}$, scalars are italic, and $\top$ marks a transpose. Data matrix $\mathbf{X}\in\mathbb{R}^{n\times d}$ (rows are examples), parameters $\boldsymbol\theta$ / $\mathbf{w}$.
- **Code.** Use executable cells (` ```{python} `), and give important cells a label `#| label: …`. Use NumPy for from-scratch code, and sklearn/scipy/PyTorch for checks. Each chapter starts with a hidden setup cell (`#| include: false`) that imports packages and sets a seed. Every from-scratch implementation ends with an `assert` that compares it with a library or a closed form. Code must run on CPU in under about 60 s per chapter.
- **Figures.** Use matplotlib, with `#| label: fig-…` and `#| fig-cap: …`. Plot only where it teaches something.
- **Cross-references.** Use `@sec-…`, `@eq-…`, `@fig-…`. Never write "Chapter 7" by hand. Mention a future chapter by name only if its section label doesn't exist yet.
- **Collapsed solutions.**
  ```markdown
  ::: {.callout-note collapse="true" title="Solution"}
  …
  :::
  ```
- **Callouts.** Use `callout-important` for key equations, `callout-warning` for pitfalls, and `callout-tip` for interview tips.
- **Sources.** Adapting explanations, exercises and figures from the free books is allowed, because this book is private. Name them in the Sources box.
- **What not to include:** meta-advice about lab notes or study rituals, licence discussion, text that is unfinished or "TODO", or claims you haven't checked.

## Verifying a chapter

```bash
.venv/Scripts/python scripts/check_chapter.py books/math/01_linear_algebra.qmd
```

This runs every Python cell in order and reports the first failure. After that, run a full `scripts/build.ps1 -To html`.
