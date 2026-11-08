# Bit Manipulation --- From Foundations to World Finals

A systematic bit-manipulation book from beginner to CF 3500, built on the same system as the companion books
(`dp_book`, `graph_book`, `array_book`: same preamble, boxes, difficulty badges, tools and verification rules).
The rough notes `..\bit_manipulation.tex` (138,000 lines) and `..\temp\` are source material only; they are never
edited.

## Build

```bash
bash tools/build.sh main         # whole book            -> main.pdf
bash tools/build.sh vol1         # Part I only            -> vol1_foundations.pdf
python tools/outline.py          # regenerate parts, roadmap and PLANNED chapter stubs from the outline
python tools/verify_outline.py   # check every planned problem against its judge, the companion books and duplicates
python tools/check_new.py LC:191 CF:1421A   # vet candidate problems before using them
python tools/check_code.py chapters/ch01_introduction   # compile every C++ listing of a chapter
python tools/check_cross_book.py # no problem may be used in this book and a companion book
```

## The plan: 6 parts, 31 chapters

| Part | Level | Chapters |
|---|---|---|
| I Foundations | Beginner -> CF 1700 | 1 Binary Representation, 2 Bitwise Operators and Their Algebra, 3 Classic Bit Tricks, 4 Counting Bits and Parity, 5 XOR Fundamentals, 6 Sets and Subsets as Bitmasks |
| II Core Techniques | CF 1300 -> 2400 | 7 Greedy over Bits, 8 Per-Bit Independence and Contribution, 9 AND and OR: Monotonicity, Submasks and Closure, 10 Bitwise Equations and Identities, 11 Gray Codes, Hypercubes and Bit Orders, 12 Bits in Number Theory |
| III XOR Algebra | CF 1500 -> 3200 | 13 Prefix XOR and XOR Hashing, 14 Linear Basis over GF(2), 15 Gaussian Elimination and Matrices over GF(2), 16 Walsh--Hadamard and Bitwise Convolutions, 17 Nim and XOR Games |
| IV Bitmask Techniques & Word Parallelism | CF 1600 -> 2800 | 18 Advanced Mask Enumeration, 19 Masks as Sets, 20 Bitsets, 21 Word-Level Parallelism and Low-Level Bit Hacks, 22 Bit Trees: 64-ary Fast Sets |
| V Bits in Problem Solving | CF 1500 -> 2600 | 23 Randomisation and Hashing with Bits, 24 Counting over Binary Prefixes, 25 Arithmetic Meets Bitwise: Carries and Borrows, 26 Interactive Bit Problems, 27 Constructive Bit Problems |
| VI Advanced & Expert | CF 2200 -> 3500 | 28 Expert XOR Algebra, 29 Bit-Level Optimisation and Performance, 30 Expert Mixed Problems, 31 Contest Mastery |

Every chapter: Part A (lessons) -> as many patterns as the topic needs (theory, at least 2 fully solved problems,
exercises) -> summary + quiz -> practice set (30+) with design sketches. Every solved problem: Problem Model ->
brute force -> efficient algorithm + invariant -> alternative method -> refinements -> trace -> tests -> mistakes ->
takeaways -> Variants box. The detailed lessons, patterns and planned problems are in `tools/outline.py` and printed
in each PLANNED chapter of the PDF.

## Borders with the companion books

| Topic | Lives in |
|---|---|
| Bitmask DP (TSP, profiles), digit DP, SOS DP, subset convolution | DP book Ch 13, 14, 26 (`\dpbook`) |
| Binary tries, maximum XOR with a trie, XOR-MST | Strings book Ch 15 and Trees book Ch 25 |
| Bit-parallel string matching (Shift-And, alphabet masks) | Strings book Ch 10 (`\stringbook`) |
| Bitmask search vs DP, exact cover / DLX, meet in the middle | Recursion book Ch 12, 16, 28 (`\recbook`) |
| XOR / AND / OR over subarrays, per-bit sums over subarrays | Arrays book Ch 15 (`\arraybook`) |
| XOR paths and the cycle space, Hamiltonian bitmasks | Graph book (`\graphbook`) |
| Fenwick trees (lowbit as an index), sparse tables | Trees book (`\treebook`) |

This book owns: representation and operators, bit tricks, popcount and parity, XOR algebra, the linear basis,
Gaussian elimination over GF(2), Walsh--Hadamard transforms, Nim as XOR, Gray codes, bitwise equations,
mask enumeration, bitsets as a data structure, word-level parallelism, 64-ary bit trees, bit-mixing hashes,
interactive and constructive bit problems, bit-level optimisation.

`tools/companions.py` collects every problem the seven companion books use in their chapters *or plan* in their
outlines; `verify_outline.py`, `check_new.py` and `check_cross_book.py` refuse any overlap. All 76 problems named in
the outline (50 solved problems of Part I, 26 candidates of Parts II-VI) were checked against Codeforces (English
names), LeetCode (no premium problems), CSES and Library Checker, and none is used or planned in a companion book.
Rejected because a companion book already has them: LC 231 Power of Two, LC 78 Subsets, LC 89 Gray Code, LC 1238
(recursion book), LC 338 Counting Bits, CF 1856E2 (DP book), CSES 1623 Apple Division (recursion book).

## What changed from the rough notes (`bit_manipulation.tex`)

Kept (reorganised): introduction, fundamental techniques, XOR properties, bit counting and parity, subsets as
bitmasks, number theory, advanced XOR techniques, bitwise optimisation, matrices over GF(2), game theory (Nim),
randomised algorithms, parallel bit algorithms (word-level parallelism), advanced subset convolution (as the
Walsh--Hadamard chapter; subset convolution itself is the DP book).

Added (missing from the notes): greedy over bits, per-bit independence, AND/OR monotonicity and submask order,
bitwise equations, Gray codes and hypercubes, the linear basis as its own chapter, mask enumeration done right
(3^n submasks, Gosper), masks as compatibility sets, bitsets as a data structure, 64-ary bit trees, counting over
binary prefixes, carries and borrows, interactive bit problems, constructive bit problems.

Moved to a companion book (referenced, not repeated): bitmask DP and advanced bitmask DP, SOS DP (DP book); trie and
binary representation, advanced trie techniques (strings and trees books); bit manipulation in strings (strings
book); segment trees and bits (trees book); bit manipulation in graphs and advanced graph algorithms (graph book).

Dropped (not contest material): computational geometry with bits (Morton codes kept as a lesson in Ch 21),
cryptographic applications (hash mixers kept in Ch 23), machine learning, complexity theory, quantum computing,
"research-level techniques".

## Status
Structure complete (parts, roadmap, legend, constraints guide, templates, tools, generators; PLANNED stubs for all
31 chapters). No chapter is written yet.
