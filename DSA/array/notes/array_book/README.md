# Arrays --- From Foundations to World Finals

A systematic array book from beginner to CF 3500, built on the same system as the companion books
`D:\Prep\DSA\dp\notes\dp_book` and `D:\Prep\DSA\graph\notes\graph_book` (same preamble, boxes, difficulty badges,
tools and verification rules). The rough notes `..\array_notes.tex` (60,000 lines) are source material only; they
are never edited.

## Build

```bash
bash tools/build.sh main         # whole book            -> main.pdf
bash tools/build.sh vol1         # Part I only            -> vol1_foundations.pdf
python tools/outline.py          # regenerate parts, roadmap and PLANNED chapter stubs from the outline
python tools/verify_outline.py   # check every planned problem against its judge, the companion books and duplicates
python tools/check_code.py chapters/ch01_introduction   # compile every C++ listing of a chapter
python tools/check_cross_book.py # no problem may be used in this book and a companion book
python tools/check_varin.py      # every \varin cross-reference resolves
python tools/check_problem_mix.py chapters/ch01_introduction   # kind mix of exercises and practice sets
python tools/stress.py sol.cpp brute.cpp gen.py [runs]          # stress test against a brute force
```

## Folder layout

```
array_book/
├── main.tex                 whole book (all six parts)
├── vol1_foundations.tex     Part I only
├── preamble/                shared with the DP and graph books, plus array TikZ styles (acell, awin, aout, aread,
│                            awrite, akeep, adone; \aptr, \aptrup pointers; \arange braces) and the
│                            cross-book macros \dpbook \graphbook \treebook \stringbook \recbook \sqhbook
├── frontmatter/             titlepage, how_to_use (anatomy, workflow, legend), roadmap (generated), constraints
├── parts/                   partK_<name>.tex openers + partK_chapters.tex include lists (generated)
├── chapters/chNN_<slug>/    one folder per chapter; chNN.tex is a PLANNED stub until the chapter is written
├── appendix/                problem index (generated)
├── code/                    gen/ (random array generators, C++ and Python), lib/ and tests/ (to be filled)
├── templates/               AUTHORING.md, chapter / pattern / problem / exercises templates
└── tools/                   outline.py (the plan), companions.py, build.sh, verify_outline.py, gen_problem_index.py,
                             check_code.py, check_varin.py, check_cross_book.py, check_problem_mix.py, stress.py,
                             known_duplicates.txt
```

## The plan: 6 parts, 36 chapters

| Part | Level | Chapters |
|---|---|---|
| I Foundations | Beginner -> CF 1800 | 1 Introduction to Arrays, 2 Prefix Sums & Difference Arrays, 3 Hashing & Counting, 4 Two Pointers, 5 Sliding Window, 6 Sorting as a Tool, 7 Binary Search on Arrays (basics) |
| II Core Techniques | CF 1100 -> 2000 | 8 Intervals & Sweep Line, 9 Partitioning & In-Place Rearrangement, 10 Index Tricks (Cyclic Sort, Marking, Permutation Cycles), 11 Matrices & 2D Arrays, 12 Designing Array-Based Data Structures |
| III Counting, Greedy & Number Arrays | CF 1500 -> 2400 | 13 Counting Subarrays, Pairs & Triples (Contribution), 14 Greedy on Arrays, 15 Bitwise Operations, 16 GCD, Divisibility & Pigeonhole, 17 MEX, Distinctness & Value Domain, 18 Coordinate Compression & Offline Processing |
| IV Square Root & Query Techniques | CF 1600 -> 2600 | 19 Square Root Decomposition, 20 Mo's Algorithm, 21 Multidimensional & Higher-Order Prefix Sums, 22 Hashing Arrays & Multisets, 23 Selection, Streaming & Order Statistics |
| V Constructive, Operations & Ad-hoc | CF 1400 -> 2500 | 24 Constructive Arrays, 25 Operations & Invariants, 26 Interactive Problems, 27 Games on Arrays, 28 Circular Arrays & Rotations, 29 Deletions, Next Pointers & Simulation |
| VI Advanced & Expert | CF 1800 -> 3500 | 30 Scanline over the Right Endpoint, 31 Permutations (Cycles, Inverses, Powers), 32 Bitsets, 33 Heavy/Light Splits & Root Thresholds, 34 Expert Greedy & Exchange Arguments, 35 Expert Mixed Problems, 36 Contest Mastery |

Binary search (2026-10-26): the former Chapters 7 "Binary Search on Arrays" and 8 "Binary Search on the Answer" and
all 22 of their planned problems moved to the Binary Search book (`D:\Prep\DSA\binary_search\notes\bs_book`). At the
author's request a basic Chapter 7 "Binary Search on Arrays" was then added back (bounds and counting, nearest
values, binary search on prefix sums, a first search on the answer) with 8 new problems that are not in the Binary
Search book; everything beyond the basics is referenced with `\bsbook`.

Every chapter: Part A (lessons) -> as many patterns as the topic needs, usually 5 (theory, at least 2 fully solved
problems -- more when two do not explain the pattern fully --, exercises) -> summary + quiz -> practice set (30+)
with design sketches. Every solved problem: Problem Model -> brute force -> efficient algorithm + invariant ->
alternative method -> refinements -> trace -> tests -> mistakes -> takeaways -> Variants box.
The detailed lessons, patterns and planned problems of every chapter are in `tools/outline.py` and printed in each
PLANNED chapter of the PDF.

## Borders with the companion books

Arrays touch every topic, so the borders are explicit (`tools/outline.py` docstring, `templates/AUTHORING.md` 7):

| Topic | Lives in |
|---|---|
| Fenwick trees, segment trees, sparse tables, Mo on trees, treaps | Trees book (`\treebook`) |
| Monotonic stack, histogram, deques, sliding-window extrema, heaps, top-k | Stacks, Queues & Heaps book (`\sqhbook`) |
| Two pointers / windows / hashing on strings | Strings book (`\stringbook`) |
| Merge sort, inversions, quickselect, CDQ, meet in the middle | Recursion book (`\recbook`) |
| Kadane, LIS, knapsack, every DP | DP book (`\dpbook`) |
| Graphs, functional graphs | Graph book (`\graphbook`) |
| Binary search beyond Ch 7: rotated arrays, the answer, ternary, interactive, offline | Binary Search book (`\bsbook`) |

`tools/companions.py` collects every problem the six books use in their chapters *or plan* in their outlines;
`verify_outline.py` and `check_cross_book.py` refuse any overlap. All 73 solved problems planned for Part I and all
candidates of Parts II-VI were checked against Codeforces, LeetCode (no premium problems), CSES, AtCoder and SPOJ,
and none is used or planned in a companion book.

## What changed from the rough notes (`array_notes.tex`)

Kept (reorganised): introduction, searching, sorting, rotation and reversal, frequency counting, rearrangement,
two pointers, sliding window, subarray problems, prefix/suffix arrays, hashing, merge and intersection, matrices,
partitioning / Dutch flag, cyclic sort and missing numbers, bit manipulation, binary search on the answer, greedy,
special patterns, array construction, Mo's algorithm and sqrt decomposition, compression, online vs offline queries.

Moved to a companion book (referenced, not repeated): stack-based array problems (stack-queue-heap book), segment
trees, Fenwick trees, sparse tables, persistent and advanced range structures (tree book), divide and conquer and
D&C optimisation (recursion and DP books), advanced sorting algorithms (recursion book).

Added: binary search on arrays and on the answer as separate chapters, intervals & sweep line, the contribution
technique, GCD/divisibility arrays, MEX and value-domain problems, multidimensional and higher-order prefix sums,
random hashing of arrays and multisets, counting pairs and triples, selection and streaming algorithms,
constructive arrays, operations & invariants, interactive problems, circular arrays, linked lists and next pointers
in arrays, scanline over the right endpoint, permutation algebra, bitsets, heavy/light value splits, expert greedy.

Dropped: "Research-Level Array Problems" (too vague to teach; its contest-relevant parts are in Part VI).

Second review (structure pass, before any chapter is written):
* New chapter 13 "Designing Array-Based Data Structures" (dynamic arrays, swap-with-last deletion, timestamps and
  sparse sets, snapshot arrays, open addressing) -- the old notes' "Array Construction and Design" had no home.
* New chapter 28 "Games on Arrays" (parity, mirror strategies, first decisive element, Nim sums) -- a frequent
  Codeforces problem type that no companion book covers.
* The planned "Counting Pairs & Triples" chapter overlapped the contribution chapter; both are now Chapter 14.
* New patterns: split points (Ch 2), runs of equal / consecutive elements and set operations on sorted arrays
  (Ch 4), binary search on prefix sums (Ch 7), ternary search (Ch 8), pigeonhole arguments (Ch 17); new lessons:
  subarrays vs subsequences vs subsets (Ch 1), leaders / suffix extrema (Ch 2), the gap method (Ch 4), fractional
  cascading (Ch 7), interval and online compression (Ch 19).
* Ch 20 (sqrt decomposition) no longer duplicates Ch 34 (heavy/light splits): it uses blocks over the value domain.
* Stubs and part openers show the real number of patterns and planned solved problems.

## Status
Structure complete (parts, roadmap, legend, constraints guide, templates, tools, generators; PLANNED stubs for
the 35 unwritten chapters).

**Chapter 1 written** (Introduction to Arrays, about 8,000 lines, about 150 pages): 5 lessons, 5 patterns x 2 solved
problems, 37 pattern exercises, summary + 12-question quiz, 32 practice problems with verified sketches.
78/78 listings compile under C++14 (`CODE_STD=c++14 python tools/check_code.py chapters/ch01_introduction`); every
solved problem, exercise, variant and practice problem reproduces its official samples, and the solutions were
stress-tested against brute forces on random arrays; 97 distinct problems, none shared with a companion book
(`tools/check_cross_book.py`); the practice set passes `tools/check_problem_mix.py` (at least 5 of each kind, a recent
and a tricky problem in every tier). The whole book builds with 0 errors and 0 undefined references.
