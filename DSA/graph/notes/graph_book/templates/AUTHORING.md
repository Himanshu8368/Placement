# Authoring guide (read before writing any chapter file)

This book follows the system of the companion book `D:\Prep\DSA\dp\notes\dp_book` (same preamble, boxes, badges,
tools and conventions). The DP book's Chapter 1 is the reference for style and density; this guide lists what is
different for graphs.

## 0. Structure lives in `tools/outline.py`
Parts, chapter titles, levels, lessons, patterns and the planned solved problems are defined once in
`tools/outline.py`. Running `python tools/outline.py` regenerates the part openers, the part chapter lists, the
roadmap and a PLANNED stub `chNN.tex` for every chapter not yet written. When a chapter is written, its `chNN.tex`
loses the `% STATUS: PLANNED` marker and is never overwritten again. To change the plan (a pattern, a problem),
edit `outline.py` and rerun it; never edit the generated files by hand. `python tools/verify_outline.py` checks
every planned problem against its judge and against the DP book.

## 1. Files and labels
* Chapter folder `chapters/chNN_<slug>/`: `chNN.tex`, `lessons/`, `p1_<name>/` ... `p5_<name>/`, `summary.tex`,
  `problem_set.tex`, `problem_set_solutions.tex` (see `templates/chapter_template.tex`).
* One pattern folder holds `theory.tex`, `prob1_<name>.tex`, `prob2_<name>.tex` (and `prob3_<name>.tex`, ... when
  needed) and `exercises.tex`.
* **How many patterns and solved problems:** a chapter has as many patterns as its topic needs (5 is a
  typical number, not a limit). Every pattern has AT LEAST 2 fully solved problems; add a 3rd, 4th, ... solved
  problem (`prob3_<name>.tex`, ...) whenever the pattern is not fully explained by two (a different sub-case,
  a harder twist, a second technique). Never pad a pattern with a near-duplicate.
* Labels: `ch:<slug>`, `ssec:chN-pK-theory`, `ssec:chN-<short-problem-name>`, `ssec:chN-pK-ex`,
  `ssec:chN-quiz-answers`.

* A solved problem's label is set with `\problemlabel{N.K.M}{ssec:chN-<name>}` right after its
  `\subsection{Problem N.K.M: ...}` (never a plain `\label`), so `Problem~\ref{...}` prints the problem number.

## 2. Structure of each file
**theory.tex:** `patternbox` "Pattern K at a Glance" (picture + the core algorithm in 5-8 lines) -> from the previous
idea to this one -> when to recognise it (use / do NOT use) -> the algorithm with its invariant and proof sketch
(`invariantbox`) -> variations map -> compilable C++ templates with a `main()` -> common mistakes -> `checklistbox`
"I'm stuck".

**prob*.tex** (`templates/problem_template.tex`):
1. `\problemcard{N.K.M}{Title}{Source}{URL}{book badge + judge badge}{technique}`
2. Statement, constraints, samples, a drawing of the sample graph
3. **Graph Model** (`modelbox`): vertices, edges, weights, the question in graph language, sizes. This replaces the
   DP book's design sheet and is the heart of every solution.
4. Key observations
5. Approaches: `bruteforcebox` -> `algorithmbox` (+ `invariantbox` with the correctness argument) ->
   `alternativebox` (a second correct algorithm, e.g. BFS vs DFS vs DSU) -> `tricksbox` (iterative form, memory,
   recursion depth) when useful
6. `tracebox`: the algorithm run by hand on the sample, with vertex colours and the data-structure contents
7. Test suite, common mistakes, takeaways, `variantsbox` (3-5 variants: change, model, algorithm change; every
   variant checked by code on the official samples or a brute force; problems already in the book use `\varin`)

**exercises.tex:** summary card, `\exlegend`, exercise list, hints and answers (`solutionbox`).

## 3. Difficulty and links
Identical to the DP book (`preamble/difficulty.tex`): book badge `\exeasy|\exmedium|\exhard|\exexpert`, judge badge
`\cf{rating}` / `\lc{label}` from the official APIs. No premium LeetCode problems.

## 4. Diagram vocabulary (`preamble/tikz-styles.tex`)
* Vertices: `gv` (plain), `gvsrc` (source), `gvdst` (target / answer), `gvcur` (being processed), `gvseen`
  (discovered / in the queue or on the stack), `gvdone` (finished), `gvdead` (removed), `gvsel` (selected),
  `gvA` / `gvB` (two colour classes or the two sides of a bipartite graph).
* Edges: `ge` (undirected), `gde` (directed), `gtree` (tree edge), `gback`, `gfwd`, `gcross` (DFS edge types),
  `gpath` (shortest / chosen path), `gfaded`, `gcut` (a cut line).
* Labels: `gw` (edge weight), `gcap` (capacity / flow, e.g. `3/5`), `gdist` (distance next to a vertex).
* Grids: `gcell`, `gwall`; DP tables (for DP on graphs) keep the DP-book styles `cell`, `basecell`, ...
* Aim for at least 6 diagrams per solved problem (sample graph, model, trace steps) and 5 per `theory.tex`.

## 5. LaTeX rules (same as the DP book)
* Box titles containing `[` or `]` must be braced. No `verbatim` / `\verb` in boxes.
* `[style=cpp]` listings must compile alone (`tools/check_code.py`); C++14 only (local g++ 6.3).
* Recursive DFS on 10^5-10^6 vertices can overflow the stack: whenever constraints allow deep recursion, show an
  iterative version or say why recursion is safe.
* Never write LaTeX through a bash heredoc (it mangles `\\`): use the Write/Edit tools.

## 6. Verification (mandatory)
1. Every number in a trace, table or sample comes from running code.
2. `python tools/check_code.py chapters/<chapter>` reports 100%.
3. Each solution reproduces the official samples (Codeforces samples via the Luogu mirror
   `https://www.luogu.com.cn/problem/CF<id>` with header `x-lentille-request: content-only`, field
   `data.problem.samples`; LeetCode examples via GraphQL `exampleTestcases`; CSES / AtCoder pages via curl) and
   matches a brute force on random small graphs (random generator + slow reference).
4. Compile in isolation with a private `_test_chNN.tex` driver in the book root (`\setcounter{chapter}{NN-1}`): no
   errors, no overfull box over 5pt; render pages with `pdftoppm` and look at every diagram; delete the driver.
5. `python tools/check_cross_book.py`: no problem may also be used in the DP book.

## 6b. A good mix of problems in every chapter (practice set and exercises)
Every practice problem carries **two** labels:
* **Level** (where it sits): Tier 1 Warm-up `\exeasy`, Tier 2 Core `\exmedium`, Tier 3 Challenge `\exhard`, and a fourth
  group **Stretch** `\exstretch` for problems ABOVE the chapter's level (CF rating > the chapter's upper bound in
  `chapterfacts`). Stretch problems are optional and never needed to continue; they are not mixed into Tiers 1-3.
* **Kind** (why it is in the set). The marker is INVISIBLE: it prints nothing (the author does not want kind tags in
  the book) and the text never mentions kinds; it only lets `check_problem_mix.py` check the mix. Exactly one of:
  `\pclassic` a well-known problem every graph programmer should have solved;
  `\pconcept` isolates one idea of the chapter in its purest form;
  `\precent` from a recent contest (last ~3 years: Codeforces contest id >= 1900, LeetCode number >= 2900,
  AtCoder ABC >= 330 / ARC >= 170);
  `\ptricky` a trap: an edge case, a tempting wrong idea, or a twist in the model (say which in the sketch).
Format (marker followed by a plain space, never `\precent\ `): `\problembox{P2.07}{\exmedium\ \precent Title --- Codeforces 1985F \cf{1500}}{hint}{url}`.

Rules for a set of about 30: at least 5 problems of each kind; at least 1 Recent and 1 Tricky in every tier;
every problem must teach a DIFFERENT idea --- the topic index lists the one-line idea of each problem, and two problems
with the same model and the same algorithm (only the story differs) are near-duplicates: keep the better one. The
same rule holds across pattern exercises, solved problems and the practice set of a chapter, and across chapters.
Pattern exercises should contain at least one Tricky and, where one exists, one Recent problem.
`python tools/check_problem_mix.py chapters/chNN_*` reports the kind counts per tier and flags Stretch problems that are
not above the chapter level and Tier problems that are.

## 7. Uniqueness
Each problem appears in exactly one place in this book and never in the DP book. The same task on another judge
counts as the same problem (`tools/known_duplicates.txt`). If an assigned problem is unsuitable, replace it with an
own exercise (`\exown`) and update `tools/outline.py`.
