# DSA Book — Plan (v3)

Status (2026-10-29): **16 chapters written** — Part I (1–4), arrays 5–7, trees 163–164, graphs 199–200, DP 229–233 — each with the full chapter flow, animations, and the checks below passing. v3 (2026-10-26): every LaTeX book in `D:\Prep\DSA` becomes a volume that mirrors it chapter for chapter (arrays, binary search, recursion, stacks/queues/heaps, strings, trees, graphs, DP, greedy, bits, math), with overlaps resolved so each topic lives in exactly one chapter.

---

## 1. Principles

| Decision | Why |
|---|---|
| **Chapters are organised by pattern.** Each chapter lists its patterns; each pattern gets **≥ 2 fully solved problems** | Interviews and contests reward recognising a pattern, not memorising a problem |
| As many patterns per chapter as the topic needs | Fixed counts either pad small topics or squeeze large ones |
| Every kind of problem appears (textbook classics, concept builders, recent contest/interview problems, tricky edge-case problems), but **no kind tags are shown** | Tags give away the trick before you try |
| Practice lists are ordered **by concept, not by rating**, with **≥ 1 practice problem per solved problem**. Codeforces preferred, then LeetCode, CSES, AtCoder | Rating order mixes unrelated ideas; CF has the deepest pool |
| Python cells are executable and end in `assert`s (brute force vs. optimised). C++ is shown as listings | The book stays checkable when it is built |
| **Pictures are welcome wherever they help.** The web edition is not limited by the print notes' page budget. Add a diagram or a film strip (`V.trace`) when it makes a structure, a sample, or a changing walk easier to see. Skip it when the prose is already clear. Rules in STYLE.md | A picture is used for understanding, not to fill a quota |
| **Every LaTeX book in `D:\Prep\DSA` is mirrored chapter for chapter**: `array_book` (36), `bs_book` (24), `recursion_book` (36), `sqh_book` (38), `string_book` (40), `tree_book` (36), `graph_book` (36), `dp_book` (35, with practice from `dp_ladder`), `greedy_book` (28), `bit_book` (31) and the math notes (45). Also the GfG PDFs in `DSA/notes/` and the solved `.cpp` files | Port and extend them rather than rewrite from zero. A written LaTeX chapter is ported with its patterns and solved problems; a planned one uses the pattern list from its stub. When two books cover a topic, one chapter owns it (see §3). `D:\Prep\DSA` itself is never modified |
| **Each problem is solved in exactly one chapter** (it may appear as practice or as a card elsewhere), stored once in `books/problems/<key>/` | `scripts/check_links.py` fails the check when a problem is written up in full (a `Source:` line) in two chapters |

---

## 2. Folder layout

The full layout, and how to add chapters and problems, is in STYLE.md ("Project layout"). In short:

```
dsa_book/
├── outline.yml          the book's structure: volumes → parts → chapters (single source of truth)
├── PLAN.md  STYLE.md  README.md
├── templates/chapter.qmd
├── glossary.yml         glossary terms → books/appendix/glossary.qmd
├── scripts/             gen_book (+ gen_revise) · build.ps1 · fix_search · qmdlib
│                        check_chapter · check_changed · check_style · check_links · check_cpp · review_figs
│                        new_chapter · new_problem · to_registry
└── books/
    ├── <folder>/NNN_<slug>.qmd      chapters
    ├── problems/<key>/              problem registry: problem.qmd, solution.qmd, traces.py, card.qmd (generated)
    ├── traces/                      V.story board data, one module per chapter
    ├── shared/                      dsa_viz.py, registry.py, style.css, themes, site.js
    ├── parts/                       part opener pages (generated from outline.yml)
    ├── appendix/                    anatomy, gallery (hand-written); glossary, cheat sheets,
    │                                question bank, problem index (generated)
    └── 404.qmd
```

---

## 3. Outline: 13 volumes, 70 parts, 369 chapters

The outline lives in **`outline.yml`**: every volume, part and chapter with its number, title, folder and pattern list, plus the list of source-book chapters that are merged into another chapter (`merged`). `gen_book.py` builds the sidebar, the home page and the problem index from it, and `new_chapter.py` scaffolds a chapter from it. To restructure the book, edit `outline.yml` and rebuild.

| Volume | Source | Chapters |
|---|---|---|
| 1 · Foundations | written here | 1–4 |
| 2 · Arrays | `array_book` + 2 sorting chapters | 5–37 |
| 3 · Binary Search | `bs_book` | 38–61 |
| 4 · Recursion and Backtracking | `recursion_book` | 62–89 |
| 5 · Linear Data Structures | linked lists + `sqh_book` | 90–124 |
| 6 · Strings | `string_book` | 125–162 |
| 7 · Trees | `tree_book` | 163–198 |
| 8 · Graphs | `graph_book` | 199–228 |
| 9 · Dynamic Programming | `dp_book` + `dp_ladder` | 229–260 |
| 10 · Greedy | `greedy_book` | 261–287 |
| 11 · Bit Manipulation | `bit_book` | 288–317 |
| 12 · Mathematics | math notes | 318–360 |
| 13 · Interviews and Contests | all volumes | 361–369 |

### Flow inside a volume and a chapter

- **A volume is one climb.** Its parts go from the first idea to contest level; each part's first chapter says which earlier chapters it builds on.
- **A chapter is one argument.** Patterns form a ladder (each adds one idea), every pattern opens with a bridge from the previous one, every solved problem ends with a connector naming the next twist, and a closing paragraph names the thread. See STYLE.md, "Chapter flow".
- **Problems are written once.** Each solved problem lives in the registry and is included by its home chapter; later chapters reuse it through its card or its solution, and say only what is new.
- **Prerequisites are explicit.** `needs:` in `outline.yml` names what to read first; by default it is the previous chapter (a part's first chapter: the previous part's last). A volume's first chapter always lists its `needs` by hand. Chapter pages show "Read first" and "Next chapter"; part opener pages show the ladder.

### Revision and practice features

| Feature | Source | Page |
|---|---|---|
| Cheat sheets | each chapter's Summary template callout + Pattern map | Appendix: Cheat Sheets |
| Question bank, quiz mode with spaced repetition | each chapter's Interview questions | Appendix: Question Bank |
| Glossary | `glossary.yml` | Appendix: Glossary |
| Problem browser, progress export / import | every judge link from Part II on | Appendix: Problem Index |
| Practice tracker, continue reading | the reader's browser (`localStorage`) | chapter pages, home page |
| Run Python in the browser | plain Python listings (Pyodide on first click) | chapter pages |
| Every diagram type, with its code | `dsa_viz.py` | Appendix: Visual Gallery |

### Open decisions

- **Chapter 231 (about 126 pages) should be split** (`check_style.py` warns above 120). Splitting inserts a chapter and renumbers every later one in `outline.yml`.
- **PDF chapter numbers.** The website and search use the outline numbers (229); the PDF still shows Quarto's order (12). Fix on the LaTeX side when the PDF is next built (`build.ps1 -To pdf`; the default build is the website only).
- **C++17 checks.** GCC 6.3 cannot compile structured bindings, so about 32 listings are reported as unverified; GCC 7 or newer verifies all of them.

---

## 4. Writing order

1. Volume 1 (chapters 1–4): done.
2. Volume 9, Dynamic Programming, from chapter 229: port the written `dp_book` chapters 1–7 first (chapters 229–235), with `dp_ladder` practice.
3. The written chapters of the other books: `tree_book` 1–3 (Volume 7) and `graph_book` 1–2 (Volume 8), math notes 1 (Volume 12).
4. Volumes 2–6 (arrays, binary search, recursion, linear structures, strings) from their stubs' pattern lists.
5. The remaining planned chapters of Volumes 7–12, then Volume 13.

Chapters may be written out of order; the sidebar shows each part as soon as one of its chapters exists, and the problem index lists problems from Volume 2 on.

Each chapter is written from `templates/chapter.qmd`, checked with `scripts/check_chapter.py` and `scripts/check_links.py --online`, then built with `scripts/build.ps1` (which adds it to the sidebar, the outline status and the problem index).
