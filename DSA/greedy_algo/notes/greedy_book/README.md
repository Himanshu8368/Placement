# Greedy Algorithms --- From Foundations to World Finals

A systematic greedy-algorithms book from beginner to CF 3500, built on the same system as the companion books
(`dp_book`, `graph_book`, `array_book`, `bit_book`, `bs_book`: same preamble, boxes, difficulty badges, tools and
verification rules). The rough notes `..\greedy_notes.tex` and `..\..\1_455_assign_cookies.cpp` are source material
only; they are never edited.

## Build

```bash
bash tools/build.sh main         # whole book            -> main.pdf
bash tools/build.sh vol1         # Part I only            -> vol1_foundations.pdf
python tools/outline.py          # regenerate parts, roadmap and PLANNED chapter stubs from the outline
python tools/verify_outline.py   # check every planned problem against its judge, the companion books and duplicates
python tools/check_new.py LC:455 CF:158B    # vet candidate problems before using them
python tools/check_code.py chapters/ch01_introduction   # compile every C++ listing of a chapter
python tools/check_cross_book.py # no problem may be used in this book and a companion book
```

## The plan: 6 parts, 28 chapters

| Part | Level | Chapters |
|---|---|---|
| I Foundations | Beginner -> CF 1600 | 1 Introduction to Greedy Algorithms, 2 First Proofs: Stays Ahead and Exchange, 3 Sorting-Based Greedy, 4 Interval Greedy, 5 Greedy on Numbers and Digits, 6 Greedy on Strings |
| II Core Techniques | CF 1300 -> 2200 | 7 Pairing and Matching Two Lists, 8 Scheduling and Deadlines, 9 Running Balances and Prefix Constraints, 10 Mathematical Greedy, 11 Greedy versus DP, 12 Lexicographic Greedy with Feasibility Checks |
| III Proofs & Structure | CF 1700 -> 2600 | 13 Exchange Arguments and Comparators, 14 Matroids, 15 Greedy with Regret, 16 Potentials and Amortised Greedy, 17 Hall-Type Conditions |
| IV Greedy with Tools | CF 1200 -> 2600 | 18 Greedy with Sorted Containers, 19 Advanced Intervals and Multi-Machine Scheduling, 20 Greedy in Games, 21 Multi-Criteria and Two-Phase Greedy, 22 Constructive Greedy |
| V Beyond Exact Greedy | CF 1700 -> 2800 | 23 Online Algorithms, 24 Greedy Approximation Algorithms, 25 Submodular and Randomised Greedy |
| VI Expert & Contest Mastery | CF 2400 -> 3500 | 26 Expert Exchange Arguments, 27 Expert Mixed Problems, 28 Contest Mastery |

Every chapter: Part A (lessons) -> as many patterns as the topic needs (theory, at least 2 fully solved problems,
exercises) -> summary + quiz -> practice set (30+) with design sketches. Greedy-specific rule (AUTHORING.md): every
solved problem has a counter-example to a tempting wrong rule, a written proof, and a stress test against an
exhaustive brute force.

## Borders with the companion books

| Topic | Lives in |
|---|---|
| Greedy with a heap, Huffman, heap scheduling, regret with heaps | Stacks, Queues & Heaps book Ch 19, 23 (`\sqhbook`) |
| Remove-k-digits and other monotonic-stack greedies | Stacks, Queues & Heaps book Ch 11 |
| Minimum spanning trees, Dijkstra | Graph book (`\graphbook`) |
| Greedy on trees | Trees book Ch 33 (`\treebook`) |
| Greedy over bits | Bit book Ch 7 (`\bitbook`) |
| Greedy checks inside binary search on the answer | Binary Search book Ch 11 (`\bsbook`) |
| Jump game, gas station, candy, interval merging and sweeps | Arrays book Ch 14 and Ch 8 (`\arraybook`) |
| Weighted intervals, knapsack, coin change: every problem where greedy fails | DP book (`\dpbook`) |

Topic overlap with the array book's Ch 14 "Greedy Algorithms on Arrays" and Ch 34 "Expert Greedy & Exchange
Arguments" is intended (the same rule as the binary search chapter there); no problem is shared.

`tools/companions.py` collects every problem the nine companion books use in their chapters *or plan* in their
outlines; `verify_outline.py`, `check_new.py` and `check_cross_book.py` refuse any overlap. All 46 problems named in
the outline (28 solved problems of Part I, 18 candidates of Parts II-VI) were checked against Codeforces, LeetCode
(no premium problems) and CSES. Rejected because a companion book already has them: Maximum Length of Pair Chain,
Two Best Non-Overlapping Events, Minimum Initial Energy to Finish Tasks, Largest Merge Of Two Strings, Minimum Number
of Taps (DP book); Valid Parenthesis String, Split a String in Balanced Strings (strings book). Maximum Matching of
Players With Trainers was dropped as the same task as Assign Cookies.

## What changed from the rough notes (`greedy_notes.tex`)

Kept (reorganised): introduction, activity selection and intervals, fractional greedy (in Ch 11), array and number
greedy (digits), string greedy, two-pointer greedy (matching), DP vs greedy, mathematical greedy, exchange arguments
and proofs, advanced intervals, matroids, task scheduling, constructive greedy, multi-objective greedy, game theory
greedy, online algorithms, submodular optimisation, approximation algorithms, data-structure greedy (sorted
containers), probabilistic greedy (in Ch 25).

Added (missing from the notes): first proofs as their own chapter (stays ahead, exchange, local repairs), running
balances and prefix constraints, lexicographic greedy with feasibility checks, greedy with regret, potential
functions, Hall-type feasibility conditions, the stress-testing discipline in every chapter.

Moved to a companion book (referenced, not repeated): stack and priority-queue greedy (stack-queue-heap book), graph
greedy and advanced graph greedy (graph book), greedy on trees (trees book), bit manipulation with greedy (bit book),
string algorithms with greedy (strings book).

Dropped (not contest material): computational geometry greedy, parallel and distributed greedy, quantum models,
advanced optimisation theory, "research-level" and "world finals championship" chapters (their contest parts are
in Part VI).

## Status
Structure complete (parts, roadmap, legend, constraints guide, templates, tools, generators; PLANNED stubs for all
28 chapters). No chapter is written yet.
