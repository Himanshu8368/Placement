# Machine Learning: From Foundations to LLMs

A personal ML book. It is written as Quarto chapters with executable Python: math, from-scratch code checked against libraries, worked examples, exercises, and interview questions.

- `PLAN.md`: the full outline (5 volumes, 16 parts, ~132 chapters) and the list of free source books.
- `STYLE.md`: the chapter format and rules.
- `books/`: the Quarto book (`_quarto.yml`, plus one topic folder per part, e.g. `math/`, `supervised/`, `agents/`).
- `sources/`: free reference PDFs (`sources/pdf/`) and code repos (gitignored, see `sources/README.md`).
- `archive/v1/`: the old 18-volume version, kept for reference only.

## Setup (once)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install numpy scipy matplotlib scikit-learn pandas jupyter pytest
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install the [Quarto CLI](https://quarto.org/docs/download/). PDF output also needs `quarto install tinytex`.

## Check a chapter, then build

```powershell
.venv\Scripts\python scripts\check_chapter.py books\part01_math\01_linear_algebra.qmd
.\scripts\build.ps1 -To html      # or pdf / epub / all
```

The site is written to `books/_build/index.html`. Executed results are cached in `books/_freeze/`, so a chapter re-runs only when it changes.
