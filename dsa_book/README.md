# DSA Patterns

A Quarto book on data structures and algorithms for interviews and competitive programming, organised by pattern: 13 volumes, 70 parts, 369 chapters (`outline.yml`).

## Everyday workflow

```bash
python scripts/new_chapter.py 230 --prefix dp1d --owner me              # chapter file from outline.yml + template; claims it (status, owner)
python scripts/new_problem.py LC 70 "Climbing Stairs" --difficulty Easy  # problem in the registry
.venv/Scripts/python scripts/check_chapter.py books/dp/230_one_d.qmd     # run every Python cell (includes expanded)
.venv/Scripts/python scripts/check_changed.py                            # ... only the chapters that changed since they last passed
.venv/Scripts/python scripts/check_cpp.py books/dp/230_one_d.qmd         # compile every C++ listing
.venv/Scripts/python scripts/review_figs.py books/dp/230_one_d.qmd       # light + dark contact sheets of the figures
.venv/Scripts/python scripts/check_links.py --online                     # judge links (CF, LC, CSES, AtCoder) + "solved once" rule
powershell scripts/build.ps1                                             # generate, lint, render (html | pdf | epub | all)
python -m http.server 8766 --directory books/_build                      # preview
```

`build.ps1` runs `gen_book.py`, then `check_style.py` (errors stop the build), then Quarto, then `fix_search.py` (book chapter numbers in the search index). `gen_book.py` regenerates everything derived from the outline and the chapters:

- the sidebar, the home page (status pills), and a part opener page per part (`books/parts/`)
- registry cards, the problem index (with its filter bar) and per-page data for `site.js` (numbers, status, reading time, read-first and next links)
- the revision pages: cheat sheets, question bank (quiz mode), glossary (from `glossary.yml`)
- frozen output is cleared for chapters whose includes or imported modules changed

When you finish a chapter, set `status: written` in `outline.yml` and remove its `owner:`.

## Where things live

| Path | What |
|---|---|
| `outline.yml` | the structure: volumes → parts → chapters, with `status`, `owner`, `needs` (edit this to restructure) |
| `glossary.yml` | glossary terms, each with the section that introduces it |
| `books/<folder>/NNN_<slug>.qmd` | chapters |
| `books/problems/<key>/` | each solved problem, written once (`problem.qmd`, optional `solution.qmd` and `traces.py`, generated `card.qmd`) |
| `books/traces/` | step-board data for `V.story`, per chapter |
| `books/shared/` | `dsa_viz.py` (figures, animations, 3D tables), `registry.py`, `style.css`, themes, `site.js` |
| `books/appendix/` | anatomy and gallery (hand-written); cheat sheets, question bank, glossary, problem index (generated) |
| `PLAN.md` / `STYLE.md` | decisions / how to write a chapter |

Setup: `python -m venv .venv` then `.venv/Scripts/python -m pip install jupyter-core nbformat nbclient ipykernel pyyaml matplotlib numpy pillow`. C++ checks need `g++` (GCC 7 or newer verifies every listing).
