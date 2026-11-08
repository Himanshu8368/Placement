# Authoring guide (read before writing any chapter file)

This book follows the system of the companion books `D:\Prep\DSA\dp\notes\dp_book` and
`D:\Prep\DSA\graph\notes\graph_book` (same preamble, boxes, badges, tools and conventions). The DP book's
Chapter 1 (`dp_book/chapters/ch01_introduction/`) and the graph book's Chapter 1 are the references for style and
density; this guide lists what is specific to arrays.

## 0. Structure lives in `tools/outline.py`
Parts, chapter titles, levels, lessons, patterns and the planned solved problems are defined once in
`tools/outline.py`. Running `python tools/outline.py` regenerates the part openers, the part chapter lists, the
roadmap and a PLANNED stub `chNN.tex` for every chapter not yet written. When a chapter is written, its `chNN.tex`
loses the `% STATUS: PLANNED` marker and is never overwritten again. To change the plan (a pattern, a problem),
edit `outline.py` and rerun it; never edit the generated files by hand. `python tools/verify_outline.py` checks
every planned problem against its judge, against the six companion books, and for duplicates in this book.

## 1. Files and labels
* Chapter folder `chapters/chNN_<slug>/`: `chNN.tex`, `lessons/`, `p1_<name>/` ... `pK_<name>/`, `summary.tex`,
  `problem_set.tex`, `problem_set_solutions.tex` (see `templates/chapter_template.tex`).
* One pattern folder holds `theory.tex`, `prob1_<name>.tex`, `prob2_<name>.tex` (and `prob3_<name>.tex`, ... when
  needed) and `exercises.tex`.
* **How many patterns and solved problems:** a chapter has as many patterns as its topic needs (5 is a
  typical number, not a limit). Every pattern has AT LEAST 2 fully solved problems; add a 3rd, 4th, ... solved
  problem whenever the pattern is not fully explained by two (a different sub-case, a harder twist, a second
  technique). Never pad a pattern with a near-duplicate.
* Labels: `ch:<slug>`, `ssec:chN-lessonK` (or a descriptive lesson label), `ssec:chN-pK-theory`,
  `ssec:chN-<short-problem-name>`, `ssec:chN-pK-ex`, `ssec:chN-quiz-answers`.
* A solved problem's label is set with `\problemlabel{N.K.M}{ssec:chN-<name>}` right after its
  `\subsection{Problem N.K.M: ...}` (never a plain `\label`), so `Problem~\ref{...}` prints the problem number.

## 2. Structure of each file
**theory.tex:** `patternbox` "Pattern K at a Glance" (a picture of the array + the core loop in 5-8 lines) -> from the
previous idea to this one -> when to recognise it (use / do NOT use) -> the technique with its invariant and proof
sketch (`invariantbox`) -> variations map (diagram) -> compilable C++ templates with a `main()` -> common mistakes ->
`checklistbox` "I'm stuck".

**prob*.tex** (`templates/problem_template.tex`):
1. `\problemcard{N.K.M}{Title}{Source}{URL}{book badge + judge badge}{technique}`
2. Statement (in your own words), constraints, samples, a drawing of the sample array
3. **Problem Model** (`modelbox`): input, what is computed (value / count / index / array), over which objects
   (elements, subarrays, pairs, ...), sizes and value ranges, and the time budget they allow. Then the brute
   force follows directly.
4. Key observations (why the obvious idea fails, with a concrete counter-example; the fact the algorithm uses)
5. Approaches: `bruteforcebox` -> `algorithmbox` (+ `invariantbox` with the correctness argument) ->
   `alternativebox` (a second correct method: hashing vs sorting, prefix sums vs window, ...) -> `tricksbox`
   (memory, constant factor, overflow, in-place version) when useful
6. `tracebox`: the algorithm run by hand on the sample: the array after each step, pointers, window, running state
   and a table of the variables
7. Test suite (`testbox`), common mistakes, takeaways, `variantsbox` (3-5 variants: change, what the model becomes,
   what the algorithm changes; every variant checked by code on the official samples or a brute force; problems
   already in this book use `\varin`)

The `\variant` macro's last two fields are labelled "Subproblem" and "Recurrence" (shared with the DP book). In this
book write in them: field 4 = the new model / quantity in <= 2 lines, field 5 = the algorithm change + answer in
<= 2 lines.

**exercises.tex:** summary card (`patternbox`), `\exlegend`, exercise list, hints and answers (`solutionbox`).

## 3. Difficulty and links
Identical to the DP book (`preamble/difficulty.tex`): book badge `\exeasy|\exmedium|\exhard|\exexpert`, judge badge
`\cf{rating}` / `\lc{label}` from the official APIs (never from memory). No premium LeetCode problems.
Links: `\exprob{title}{source}{url}` in exercise lists, `\problembox` in practice sets, `\problemcard` for solved
problems.

## 4. Diagram vocabulary (`preamble/tikz-styles.tex`)
* Draw arrays with `\dparray{a}{y}{label}{v/style, ...}` (cells `a0, a1, ...`, 0.9 cm apart) and indices with
  `\dpindex{i}{y}{label}{first}{last}`.
* Array cell styles: `acell` (plain), `aread` (being read / current), `awrite` (just written), `awin` (inside the
  window), `aout` (outside / discarded), `akeep` (final, kept part), `adone` (finished, bold); the DP styles
  `basecell`, `compcell`, `anscell`, `pathcell`, `blockcell`, `dupcell` are available for prefix arrays and tables.
* Pointers: `\aptr{a3}{l}{successgreen}` (arrow from below), `\aptrup{a5}{r}{accentorange}` (from above). Use green
  for the left / read pointer, orange for the right / write pointer.
* Ranges: `\arange{a1}{a4}{window}{memoorange}` (brace under a range with a label).
* Matrices: `cell` grids via `\dparray` rows, or `gcell` / `gwall`.
* Arrows: `dep` (dependency), `flow`, `back`, `faded`. Flowcharts: `fcstart`, `fcproc`, `fcdec`, `fcout`, `fcarrow`.
* Captions: `\figcap{...}` inside a `diagrambox`. Aim for at least 6 diagrams per solved problem (sample array,
  model, trace steps) and 5 per `theory.tex`.

## 5. LaTeX rules (same as the DP book)
* Box titles containing `[` or `]` must be braced: `\begin{warningbox}[{Mistake --- $a[i+1]$}]`.
* No `verbatim` / `\verb` in boxes. Complete programs in `\begin{lstlisting}[style=cpp]` must compile alone
  (`tools/check_code.py`); fragments `[style=snippet]`, traces `[style=trace]`.
* C++14 only (local g++ 6.3): no structured bindings, no `std::gcd` without `<numeric>` care, no `if constexpr`.
* Overflow: sums of up to 2*10^5 values up to 10^9 need `long long`; say so where it matters.
* Long display equations overflow: split them with `align*`.
* Inside `\foreach` lists, brace any item text that contains a comma.
* Never write LaTeX through a bash heredoc (it mangles `\\`): use the Write/Edit tools.

## 6. Verification (mandatory)
1. Every number in a trace, table or sample comes from running code.
2. `python tools/check_code.py chapters/<chapter>` reports 100%.
3. Each solution reproduces the official samples (Codeforces samples via the Luogu mirror
   `https://www.luogu.com.cn/problem/CF<id>` with header `x-lentille-request: content-only`, field
   `data.problem.samples`; LeetCode examples via GraphQL `exampleTestcases`; CSES / AtCoder pages via curl) and
   matches a brute force on random small arrays (`python tools/stress.py`, generators in `code/gen/`).
   For class-based LeetCode code, write a small driver in your scratch folder.
4. Compile in isolation with a private `_test_chNN_x.tex` driver in the book root:
   ```
   \documentclass[11pt,a4paper,oneside]{book}
   \input{preamble/preamble}
   \begin{document}\setcounter{chapter}{N-1}
   \input{chapters/chNN_.../pK_.../theory} ... (your files)
   \end{document}
   ```
   Run `pdflatex -interaction=nonstopmode -halt-on-error` twice: no errors, no overfull box over 5pt; render pages
   with `pdftoppm -r 60` and look at every diagram; then delete the driver and its outputs.
5. `python tools/check_cross_book.py`: no problem may also be used in a companion book.

## 6b. A good mix of problems in every chapter (practice set and exercises)
Every practice problem carries **two** labels:
* **Level** (where it sits): Tier 1 Warm-up `\exeasy`, Tier 2 Core `\exmedium`, Tier 3 Challenge `\exhard`, and a fourth
  group **Stretch** `\exstretch` for problems ABOVE the chapter's level (CF rating > the chapter's upper bound in
  `chapterfacts`). Stretch problems are optional and never needed to continue; they are not mixed into Tiers 1-3.
* **Kind** (why it is in the set). The marker is INVISIBLE: it prints nothing (the author does not want kind tags in
  the book) and the text never mentions kinds; it only lets `check_problem_mix.py` check the mix. Exactly one of:
  `\pclassic` a well-known problem every array programmer should have solved;
  `\pconcept` isolates one idea of the chapter in its purest form;
  `\precent` from a recent contest (Codeforces contest id >= 1900, LeetCode number >= 2900, AtCoder ABC >= 330 /
  ARC >= 170);
  `\ptricky` a trap: an edge case, a tempting wrong idea, or a twist in the model (say which in the sketch).
Format (marker followed by a plain space, never `\precent\ `): `\problembox{P2.07}{\exmedium\ \precent Title --- Codeforces 1985F \cf{1500}}{hint}{url}`.

Rules for a set of about 30: at least 5 problems of each kind; at least 1 Recent and 1 Tricky in every tier;
every problem must teach a DIFFERENT idea. Two problems with the same model and the same algorithm (only the story
differs) are near-duplicates: keep the better one, or make the other a `\variant`. The same rule holds across
pattern exercises, solved problems and the practice set of a chapter, and across chapters.
Pattern exercises contain at least one Concept and one Tricky and, where one exists, one Recent problem.
`python tools/check_problem_mix.py chapters/chNN_*` reports the kind counts per tier.

## 7. Uniqueness and borders
Each problem appears in exactly one place in this book and never in a companion book (DP, graph, tree, string,
recursion, stack-queue-heap; `tools/check_cross_book.py` also reserves the problems those books have only
planned). The same task on another judge counts as the same problem (`tools/known_duplicates.txt`). If an assigned
problem is unsuitable, replace it with an own exercise (`\exown`) and update `tools/outline.py`.

Topics owned by a companion book are referenced, not taught: Fenwick / segment trees / sparse tables
(`\treebook{..}{..}`), monotonic stack / deque / heaps (`\sqhbook`), string windows and string hashing
(`\stringbook`), merge sort / inversions / quickselect / CDQ / meet in the middle (`\recbook`), Kadane / LIS / any
DP (`\dpbook`), graphs (`\graphbook`).
