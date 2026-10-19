# Graph Algorithms --- From Foundations to World Finals

A systematic graph book from beginner to CF 3500, built on the same system as the companion book
`D:\Prep\DSA\dp\notes\dp_book` (same preamble, boxes, difficulty badges, tools and verification rules).
The rough notes `..\graph_notes.tex` are the source material; they are never edited.

## Build

```bash
bash tools/build.sh main      # whole book            -> main.pdf
bash tools/build.sh vol1      # Part I only            -> vol1_foundations.pdf
python tools/outline.py       # regenerate parts, roadmap and PLANNED chapter stubs from the outline
python tools/verify_outline.py   # check every planned problem against its judge and the DP book
python tools/check_code.py chapters/ch01_introduction   # compile every C++ listing of a chapter
python tools/check_cross_book.py # no problem may be used in both books
python tools/check_varin.py      # every \varin cross-reference resolves
```

## Folder layout

```
graph_book/
├── main.tex                 whole book (all six parts)
├── vol1_foundations.tex     Part I only
├── preamble/                shared with the DP book, plus graph boxes (modelbox, bruteforcebox, algorithmbox,
│                            alternativebox, tricksbox, tracebox, invariantbox, plannedbox) and graph TikZ
│                            styles (gv*, ge, gde, gtree, gback, gfwd, gcross, gpath, gw, gcap, gdist, gcut)
├── frontmatter/             titlepage, how_to_use (anatomy, workflow, legend), roadmap (generated)
├── parts/                   partK_<name>.tex openers + partK_chapters.tex include lists (generated)
├── chapters/chNN_<slug>/    one folder per chapter; chNN.tex is a PLANNED stub until the chapter is written
├── appendix/                code_library.tex (Appendix A), testing.tex (Appendix B), problem index (generated)
├── code/                    lib/ (20 tested C++14 headers), gen/ (random generators), tests/, examples/
├── templates/               AUTHORING.md, chapter / pattern / problem / exercises templates
└── tools/                   outline.py (the plan), build.sh, verify_outline.py, gen_problem_index.py,
                             check_code.py, check_varin.py, check_cross_book.py, known_duplicates.txt
```

## The plan: 6 parts, 36 chapters

| Part | Level | Chapters |
|---|---|---|
| I Foundations | Beginner -> CF 2000 | 1 Introduction to Graphs, 2 DFS, 3 BFS, 4 Components & Bipartite Graphs, 5 DAGs & Topological Sorting, 6 DSU |
| II Shortest Paths, Spanning Trees & Modelling | CF 1400 -> 2200 | 7 Dijkstra, 8 Negative Weights / Bellman-Ford / 0-1 BFS, 9 All-Pairs Shortest Paths, 10 MST, 11 Functional Graphs & Binary Lifting, 12 Graph Construction Techniques |
| III Trees | CF 1600 -> 2400 | 13 Tree Fundamentals, 14 LCA, 15 Subtree & Path Queries, 16 Heavy-Light Decomposition, 17 Centroid Decomposition |
| IV Connectivity & Classic Path Problems | CF 1700 -> 2500 | 18 Bridges / Articulation Points / Biconnectivity, 19 SCC, 20 2-SAT, 21 Euler Paths, 22 Cycles, Triangles & Degree Tricks, 23 Hard Problems on Small Graphs |
| V Flows & Matchings | CF 2000 -> 2800 | 24 Max Flow, 25 Min Cut & Flow Modelling, 26 Bipartite Matching & Covers, 27 Min-Cost Flow & Assignment, 28 General Matching |
| VI Advanced & Expert | CF 2400 -> 3500 | 29 Offline & Dynamic Connectivity, 30 Link-Cut Trees, 31 Dominator Trees, 32 Global Cuts & Gomory-Hu, 33 Special Graph Classes, 34 Counting & Algebraic Graph Theory, 35 Constructive & Interactive Graph Problems, 36 Contest Mastery |

Every chapter: Part A (lessons) -> as many patterns as the topic needs, usually 5 (theory, at least 2 fully solved
problems -- more when two do not explain the pattern fully --, exercises) -> summary + quiz ->
practice set (30+) with design sketches. Every solved problem: Graph Model -> brute force -> efficient algorithm +
invariant -> alternative algorithm -> trace -> tests -> mistakes -> takeaways -> Variants box.
The detailed lessons, patterns and planned problems of every chapter are in `tools/outline.py` and printed in
each PLANNED chapter of the PDF.

Status: structure complete (parts, roadmap, legend, templates, tools; PLANNED stubs for unwritten chapters).
**Chapter 1 written** (about 10,000 lines: 5 lessons, 5 patterns x 2 solved problems, 31 exercises, summary, 32 practice
problems with verified sketches): 77/77 listings compile, every solution reproduces the official samples and matches a
brute force on random inputs, 65 distinct problems, none shared with the DP book.
**Chapter 2 written** (Depth-First Search, about 200 pages: 5 lessons, 5 patterns with 12 solved problems, 30 exercises,
summary + quiz, 30 practice problems with verified sketches): 82/82 listings compile, every solution reproduces the
official samples and matches a brute force on random inputs, no problem shared with the DP or tree book.
The 60 solved problems of Part I and all candidate problems of Parts II-VI were checked against Codeforces,
LeetCode (no premium problems), CSES and AtCoder, and none is used in the DP book.

## What changed from the rough notes (`graph_notes.tex`)

Kept (reorganised): DFS, BFS, components, bipartite graphs, topological sorting, Dijkstra, DSU, MST,
Bellman-Ford, Floyd-Warshall, bridges & articulation points, SCC, Euler paths, Hamiltonian paths, 2-SAT, max flow,
min cut, bipartite matching, tree basics, LCA, HLD, centroid decomposition, link-cut trees, min-cost flow, general
matching, dominator trees, cactus graphs, Gomory-Hu trees, randomised algorithms, dynamic graphs, Steiner trees,
graph colouring, planar and chordal graphs.

Added (common in contests, missing from the notes): functional graphs & binary lifting (Ch 11), graph construction
techniques --- virtual vertices, layered graphs, segment-tree edges, complement traversal (Ch 12), 0-1 BFS as a
first-class topic (Ch 8), subtree & path queries with Euler tours, differences on trees, small-to-large and virtual
trees (Ch 15), tree isomorphism and Pruefer codes (Ch 13), offline connectivity: rollback DSU, segment tree over
time, Kruskal reconstruction tree, parallel binary search (Ch 29), walk counting, random walks, matrix-tree and BEST theorems,
XOR hashing (Ch 34), a final mixed contest chapter (Ch 36).

Added after a second review: Ch 22 "Cycles, Triangles & Degree Tricks" (girth, triangle / 4-cycle counting in
O(m sqrt m), heavy and light vertices, the cycle space and XOR paths, DFS-tree cycle constructions); Ch 35
"Constructive & Interactive Graph Problems" (building graphs and trees with given properties, reconstructing from
BFS/DFS orders and distances, interactive search on trees and graphs); games on graphs with cycles (retrograde
BFS, Ch 12); minimum mean and ratio cycles (Ch 8); directed MST / Chu-Liu-Edmonds (Ch 10); long-path
decomposition (Ch 16); stable marriage and bipartite edge colouring (Ch 26).

Merged: Hamiltonian paths + graph colouring + Steiner tree + cliques -> Ch 23 "Hard Problems on Small Graphs";
planar + chordal + cactus -> Ch 33 "Special Graph Classes"; dynamic graphs -> Ch 29/30; randomised algorithms ->
Ch 32 (Karger) and Ch 34 (hashing).

Dropped (research topics that do not appear in contests, or too vague to teach): graph minor theory, graph limits
and graphons, spectral graph theory (beyond walk counting), parameterised algorithms, "research-level graph
problems", general graph isomorphism (tree isomorphism is kept), matroid theory as a chapter (matroid intersection
can be a "going further" note in Ch 27), "graph decomposition" (tree decompositions are mentioned in Ch 33).

Problems: the notes' problem lists were not copied blindly. Several descriptions did not match the linked problem,
several were Kattis-only or premium, and some were the same task on two judges. Every problem in this book is chosen
again, checked against the judge's API, and appears exactly once (`tools/known_duplicates.txt`).

## Relation to the DP book
DP on DAGs (DP book Ch 12), DP on trees (Ch 11), rerooting (Ch 19), bitmask DP / TSP (Ch 13), matrix
exponentiation (Ch 25) and Dilworth / chains (Ch 4) are taught there; this book refers to them with `\dpbook{N}{Title}`
and solves no problem that the DP book solves (`tools/check_cross_book.py`).

## Code library and testing

`code/lib/` holds one tested C++14 header per reusable algorithm (graph storage, DFS/BFS, components, bipartite,
cycles, topological sort, functional graphs, DSU, 0-1 BFS, Dijkstra, Bellman-Ford, Floyd-Warshall, Kruskal, Prim,
LCA). `python tools/test_library.py` checks every header against a brute force on thousands of random graphs
(20/20 pass). `python tools/stress.py <solution.cpp> <brute.cpp> <generator> [runs]` is the generic stress tester.
Appendix A (`appendix/code_library.tex`) documents the library; Appendix B (`appendix/testing.tex`) teaches testing
graph code (eight pitfalls, generators, brute-force references, a real stress-testing run).
