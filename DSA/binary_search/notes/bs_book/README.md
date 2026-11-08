# Binary Search --- From Foundations to World Finals

A systematic binary search book from beginner to CF 3500, built on the same system as the companion books
(`dp_book`, `graph_book`, `array_book`, `bit_book`: same preamble, boxes, difficulty badges, tools and verification
rules). The rough notes `..\binary_search_notes.tex` (31,000 lines) and the solved programs in `..\..\*.cpp` are
source material only; they are never edited.

## Build

```bash
bash tools/build.sh main         # whole book            -> main.pdf
bash tools/build.sh vol1         # Part I only            -> vol1_foundations.pdf
python tools/outline.py          # regenerate parts, roadmap and PLANNED chapter stubs from the outline
python tools/verify_outline.py   # check every planned problem against its judge, the companion books and duplicates
python tools/check_new.py LC:35 CF:706B     # vet candidate problems before using them
python tools/check_code.py chapters/ch01_introduction   # compile every C++ listing of a chapter
python tools/check_cross_book.py # no problem may be used in this book and a companion book
```

## The plan: 5 parts, 24 chapters

| Part | Level | Chapters |
|---|---|---|
| I Foundations | Beginner -> CF 1900 | 1 Introduction to Binary Search, 2 Lower Bound, Upper Bound and Counting, 3 Binary Search on Partly Sorted Arrays, 4 Binary Search over Functions and Implicit Sequences, 5 Binary Search on 2D Arrays |
| II Binary Search on the Answer | CF 1300 -> 2300 | 6 Binary Search on the Answer, 7 Minimise the Maximum, 8 Maximise the Minimum, 9 Time, Budget and Resource Problems, 10 Median and k-th Element Search |
| III Advanced Techniques | CF 1600 -> 2600 | 11 Designing Checks, 12 Binary Search on Real Numbers, 13 Ternary Search and Unimodal Functions, 14 Parametric Search and Fractional Programming, 15 Binary Search with Graph and DP Checks |
| IV Searching Beyond Arrays | CF 1500 -> 2800 | 16 Sorted Containers and Timelines, 17 Exponential, Interpolation and Bitwise Search, 18 Binary Search in Geometry, 19 Fractional Cascading |
| V Interactive, Offline & Expert | CF 1300 -> 3500 | 20 Interactive Binary Search, 21 Offline and Parallel Binary Search, 22 Expert Binary Search on the Answer, 23 Expert Mixed Problems, 24 Contest Mastery |

Every chapter: Part A (lessons) -> as many patterns as the topic needs (theory, at least 2 fully solved problems,
exercises) -> summary + quiz -> practice set (30+) with design sketches. The detailed lessons, patterns and planned
problems are in `tools/outline.py` and printed in each PLANNED chapter of the PDF.

## Borders with the companion books

| Topic | Lives in |
|---|---|
| Binary search as one recursive call; merge sort, quickselect | Recursion book Ch 13 (`\recbook`) |
| Parallel binary search written as recursion on time (Meteors) | Recursion book Ch 29 |
| LIS in O(n log n); the Aliens trick (WQS / Lagrangian relaxation) | DP book Ch 4, Ch 24 (`\dpbook`) |
| Binary search + 2-SAT / flow / minimum ratio cycles | Graph book (`\graphbook`) |
| Binary lifting search, Fenwick and segment-tree descent, BSTs | Trees book (`\treebook`) |
| Greedy over bits (deciding the answer bit by bit) | Bit book Ch 7 (`\bitbook`) |
| Two pointers, prefix sums, sorting around a search | Arrays book (`\arraybook`) |

Moved here from the array book (2026-10-26): the array book's former Chapters 7 "Binary Search on Arrays" and 8
"Binary Search on the Answer" and their 22 planned problems. The array book keeps a basic Chapter 7 "Binary
Search on Arrays" with 8 different problems (topic overlap with Chapters 1--4 here is intended; no problem is shared).
Two of those were near-duplicates inside the new plan and were dropped: Aggressive Cows (SPOJ, the same task as
Magnetic Force Between Two Balls) and Peak Index in a Mountain Array (a one-line change of Find Peak Element).

`tools/companions.py` collects every problem the eight companion books use in their chapters *or plan* in their
outlines; `verify_outline.py`, `check_new.py` and `check_cross_book.py` refuse any overlap. All 62 problems named in
the outline (26 solved problems of Part I, 36 candidates of Parts II-V) were checked against Codeforces, LeetCode
(no premium problems), CSES and SPOJ. Rejected because a companion book already has them: Search in Rotated Sorted
Array and Binary Search (recursion book), Concert Tickets (tree book), No Palindromes (string book).

## What changed from the rough notes (`binary_search_notes.tex`)

Kept (reorganised): introduction, lower/upper bound, sorted arrays, predicate functions, 2D arrays, introduction to
binary search on the answer, allocation, placement (aggressive cows), speed and time, cutting and division (merged
into time/budget and allocation), median and k-th element, binary search with greedy, floating point, ternary
search, binary search with DP and graphs (as checks), fractional cascading, binary search with data structures,
geometric binary search.

Added (missing from the notes): binary search over implicit sequences and prefix sums, designing checks
(prefix-sum and difference-array checks), parametric search and fractional programming (Dinkelbach), timelines in
sorted containers, exponential / interpolation / bitwise binary search, interactive binary search, expert nested
searches.

Moved to a companion book (referenced, not repeated): parallel binary search as recursion (recursion book; this book
keeps the batched round-by-round form in Ch 21), the Aliens trick (DP book Ch 24).

Dropped: "Research-Level Binary Search Problems" (too vague to teach; contest parts are in Part V).

## Status
Structure complete (parts, roadmap, legend, constraints guide, templates, tools, generators; PLANNED stubs for all
24 chapters). No chapter is written yet.
