r"""outline.py -- the single source of truth for the book's structure.

Running  python tools/outline.py  (re)generates
    parts/partK_<name>.tex           part opener pages
    parts/partK_chapters.tex         \include list of the part
    frontmatter/roadmap.tex          the roadmap chapter
    chapters/chNN_<slug>/chNN.tex    a PLANNED stub for every chapter that is not written yet
A chapter counts as written as soon as its chNN.tex no longer contains the marker '% STATUS: PLANNED';
written chapters are never overwritten. Edit this file, not the generated ones.

Problem references are (title, judge, id). Judges: CF (contest+index), LC (number), CSES (task id),
AC (AtCoder task id), LIB (Library Checker), SPOJ. Part I problems were checked against the judges'
APIs; later parts are candidates and are re-checked when the chapter is written.
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

PARTS = [
    dict(n=1, key='foundations', title='Foundations', level=r'Beginner $\to$ CF 2000', color='successgreen',
         tagline='Modelling, traversal and the first structures: everything later is built from these six chapters.',
         goals=[r'model any problem as vertices and edges, including grids, numbers, words and game states;',
                r'use DFS and BFS fluently: timestamps, cycles, layers, multi-source and state-space search;',
                r'decompose graphs into components, bipartite classes and topological orders;',
                r'maintain connectivity online with a disjoint-set union, including weighted and offline variants.']),
    dict(n=2, key='paths', title='Shortest Paths, Spanning Trees \\& Modelling', level=r'CF 1400 $\to$ CF 2200',
         color='primaryblue',
         tagline='Weights enter the picture: shortest paths of every kind, minimum spanning trees, and the art of building the right graph.',
         goals=[r'choose between BFS, 0-1 BFS, Dijkstra, Bellman--Ford and Floyd--Warshall from the constraints alone;',
                r'prove and use the cut and cycle properties of minimum spanning trees;',
                r'answer jump queries on functional graphs with binary lifting;',
                r'build helper graphs: virtual vertices, layered states, segment-tree edges, complement traversal.']),
    dict(n=3, key='trees', title='Trees', level=r'CF 1600 $\to$ CF 2400', color='softpurple',
         tagline='The most common graphs in contests: diameters, ancestors, paths and subtrees, answered fast.',
         goals=[r'compute diameters, centres, centroids and canonical forms of trees;',
                r'answer LCA, distance and path-aggregate queries in $\bigO{\log n}$ or $\bigO{1}$;',
                r'turn subtree and path queries into array queries (Euler tour, heavy--light decomposition);',
                r'count paths with centroid decomposition and merge information small-to-large.']),
    dict(n=4, key='connectivity', title='Connectivity \\& Classic Path Problems', level=r'CF 1700 $\to$ CF 2500',
         color='accentorange',
         tagline='The deeper structure of a graph: bridges, cut vertices, strong components, and the paths that use every edge or every vertex.',
         goals=[r'find bridges, articulation points, 2-edge- and 2-vertex-connected components with one DFS;',
                r'condense directed graphs into DAGs of strongly connected components and solve 2-SAT;',
                r'construct Euler paths and circuits, including de Bruijn sequences;',
                r'recognise NP-hard problems and solve them exactly on small graphs.']),
    dict(n=5, key='flows', title='Flows \\& Matchings', level=r'CF 2000 $\to$ CF 2800', color='warningred',
         tagline='Networks, cuts and matchings: the most powerful modelling tool in competitive programming.',
         goals=[r'compute maximum flows with Dinic and recover minimum cuts;',
                r'model selection, assignment and covering problems as flows or cuts;',
                r"use matchings, K\"onig's and Hall's theorems, and path covers;",
                r'solve min-cost flow and assignment problems, and matchings in general graphs.']),
    dict(n=6, key='advanced', title='Advanced \\& Expert Topics', level=r'CF 2400 $\to$ CF 3500', color='deepblue',
         tagline='Dynamic and offline connectivity, dynamic trees, dominators, global cuts, special graph classes and algebra on graphs.',
         goals=[r'answer connectivity queries under insertions and deletions, offline and online;',
                r'maintain dynamic forests with link-cut trees;',
                r'use dominator trees, Gomory--Hu trees and global minimum cuts;',
                r'count spanning trees, Euler circuits and walks with linear algebra.']),
]

# (num, slug, title, part, lo, hi, promise, lessons, patterns, connections)
# patterns: list of (name, [(title, judge, id), ...])
CH = []


def ch(num, slug, title, part, lo, hi, promise, lessons, patterns, connections=''):
    num = len(CH) + 1   # chapters are numbered by their order in this file (the first argument is a reminder only)
    CH.append(dict(num=num, slug=slug, title=title, part=part, lo=lo, hi=hi, promise=promise,
                   lessons=lessons, patterns=patterns, connections=connections))


# ------------------------------------------------------------------ PART I
ch(1, 'introduction', 'Introduction to Graphs', 1, 'Beginner', '1400',
   'Learn to see graphs everywhere: in maps and friendships, but also in grids, numbers, words and game states.',
   ['Vertices and edges: directed, undirected, weighted, simple graphs, multigraphs and loops',
    'Vocabulary: degree and the handshake lemma, walks, paths, cycles, connectivity, trees, DAGs, bipartite and complete graphs',
    'Storing a graph: edge list, adjacency matrix, adjacency list, compressed (CSR) form; reading input safely',
    'A first traversal: reachability with DFS and BFS (previewing Chapters 2 and 3)',
    'Modelling: implicit graphs (grids, numbers, strings, states) and recognising graph shapes from the constraints'],
   [('Degrees and counting', [('Find the Town Judge', 'LC', '997'), ('Maximal Network Rank', 'LC', '1615')]),
    ('Reachability', [('Find if Path Exists in Graph', 'LC', '1971'), ('Keys and Rooms', 'LC', '841')]),
    ('Grids as graphs', [('Counting Rooms', 'CSES', '1192'), ('Max Area of Island', 'LC', '695')]),
    ('Implicit state graphs', [('Two Buttons', 'CF', '520B'), ('Open the Lock', 'LC', '752')]),
    ('Functional graphs: one outgoing edge', [('Badge', 'CF', '1020B'), ('Longest Cycle in a Graph', 'LC', '2360'), ('Find the Duplicate Number', 'LC', '287')])])

ch(2, 'dfs', 'Depth-First Search', 1, '1000', '1900',
   'One recursive idea --- go as deep as possible, then back up --- and everything it reveals: cycles, subtrees, timestamps and the DFS tree.',
   ['Recursion and the call stack; iterative DFS and recursion-depth limits',
    'Vertex states (white, gray, black) and entry/exit times; the parenthesis theorem and the O(1) ancestor test (flattening is Chapter 13, queries are Chapter 15)',
    'The DFS tree and the four edge types (tree, back, forward, cross)',
    'Cycle detection in undirected and directed graphs, and recovering the cycle',
    'DFS as a toolbox: subtree aggregation, path reconstruction, backtracking over paths'],
   [('Cycle detection and recovery', [('Round Trip', 'CSES', '1669'), ('Round Trip II', 'CSES', '1678')]),
    ('Subtree aggregation and push-down', [('White-Black Balanced Subtrees', 'CF', '1676G'), ('Count Nodes With the Highest Score', 'LC', '2049'), ('Ki', 'AC', 'abc138_d')]),
    ('Entry and exit times', [('Military Problem', 'CF', '1006E'), ('Count Descendants', 'AC', 'abc202_e'), ('Maze', 'CF', '377A')]),
    ('Backtracking over paths', [('All Paths From Source to Target', 'LC', '797'), ('Path with Maximum Gold', 'LC', '1219')]),
    ('Reverse search: start from the target', [('All Ancestors of a Node in a Directed Acyclic Graph', 'LC', '2192'), ('Solve The Maze', 'CF', '1365D')])])

ch(3, 'bfs', 'Breadth-First Search', 1, '1000', '1900',
   'Explore in layers and every vertex is reached by a shortest path --- the engine behind most unweighted shortest-path problems.',
   ['The queue and the layer structure; BFS vs DFS',
    'The shortest-path theorem and its proof; parents and path reconstruction',
    'Grid BFS and multi-source BFS',
    'Multi-dimensional states: (position, keys, budget, ...) and bidirectional BFS for huge state spaces (puzzle-state modelling itself is Chapter 1, Pattern 4)',
    'The shortest-path DAG: counting shortest paths, forced vertices, lexicographically smallest paths'],
   [('Shortest paths and reconstruction', [('Message Route', 'CSES', '1667'), ('Labyrinth', 'CSES', '1193')]),
    ('Multi-source BFS', [('Rotting Oranges', 'LC', '994'), ('Monsters', 'CSES', '1194')]),
    ('Multi-dimensional states', [('Shortest Path in a Grid with Obstacles Elimination', 'LC', '1293'), ('Shortest Path to Get All Keys', 'LC', '864')]),
    ('Distances from two ends and second-shortest paths', [('Fight Against Traffic', 'CF', '954D'), ('Second Minimum Time to Reach Destination', 'LC', '2045')]),
    ('The shortest-path DAG', [('Number of Shortest paths', 'AC', 'abc211_d'), ('Navigation System', 'CF', '1320B')])])

ch(4, 'components', 'Components \\& Bipartite Graphs', 1, '1100', '1700',
   'Split a graph into the pieces that matter --- connected components and the two sides of a bipartite graph.',
   ['Connectivity as an equivalence relation; components',
    'Labelling components with DFS, BFS or DSU; storing component data',
    'Bipartite graphs and the odd-cycle theorem',
    'Two-colouring, counting colourings and parity constraints',
    'Components as units: answering questions per component (grids were Chapter 1, Pattern 3)'],
   [('Counting and connecting components', [('Building Roads', 'CSES', '1666'), ('Ice Skating', 'CF', '217A')]),
    ('Properties of a component', [('Cyclic Components', 'CF', '977E'), ('Count the Number of Complete Components', 'LC', '2685')]),
    ('Bipartite check', [('Building Teams', 'CSES', '1668'), ('Mahmoud and Ehab and the bipartiteness', 'CF', '862B')]),
    ('Counting colourings and parity', [('Beautiful Graph', 'CF', '1093D'), ('Even Relation', 'AC', 'abc126_d')]),
    ('Components as units', [('Smallest String With Swaps', 'LC', '1202'), ('Learning Languages', 'CF', '277A')])])

ch(5, 'dag', 'DAGs \\& Topological Sorting', 1, '1200', '2000',
   'Directed graphs without cycles are exactly the graphs you can put in order --- and ordered graphs are where DP lives.',
   ['DAGs and the existence theorem for topological orders',
    "Kahn's algorithm (in-degrees) and the DFS finishing order",
    'Cycle detection and reporting in directed graphs',
    'Lexicographically smallest and unique topological orders',
    r'Processing a DAG in topological order (DP on DAGs: \dpbook{12}{DP on DAGs \& Graphs})'],
   [('Ordering with dependencies', [('Course Schedule', 'CSES', '1679'), ('Build a Matrix With Conditions', 'LC', '2392')]),
    ('Priority and grouped orders', [('Course Schedule II', 'CSES', '1757'), ('Sort Items by Groups Respecting Dependencies', 'LC', '1203')]),
    ('Paths in a DAG', [('Longest Flight Route', 'CSES', '1680'), ('Game Routes', 'CSES', '1681')]),
    ('Cycles versus order', [('Find Eventual Safe States', 'LC', '802'), ('Directing Edges', 'CF', '1385E')]),
    ('Implicit DAGs', [('Fox And Names', 'CF', '510C'), ('Substring', 'CF', '919D')])])

ch(6, 'dsu', 'Disjoint Set Union', 1, '1200', '1900',
   'Merge sets and ask "same set?" in almost constant time --- the data structure behind Kruskal, offline queries and much more.',
   ['The disjoint-set problem; the forest representation',
    'Path compression and union by size or rank',
    'Why it is fast: an amortised analysis sketch',
    'DSU with data: sizes, sums, parity and potentials (weighted DSU)',
    'Offline tricks: reverse deletion, sorting by threshold, skip pointers'],
   [('Incremental connectivity', [('Road Construction', 'CSES', '1676'), ('Count Unreachable Pairs of Nodes in an Undirected Graph', 'LC', '2316')]),
    ('Equivalence classes', [('Accounts Merge', 'LC', '721'), ('Satisfiability of Equality Equations', 'LC', '990')]),
    ('Weighted (potential) DSU', [('Evaluate Division', 'LC', '399'), ('People on a Line', 'AC', 'arc090_b')]),
    ('Offline queries', [('Checking Existence of Edge Length Limited Paths', 'LC', '1697'), ('Destroying Array', 'CF', '722C')]),
    ('DSU as a building block', [('Secret Passwords', 'CF', '1263D'), ('Vessels', 'CF', '371D')])])

# ------------------------------------------------------------------ PART II
ch(7, 'dijkstra', "Dijkstra's Algorithm", 2, '1400', '2200',
   'Shortest paths with non-negative weights: the greedy algorithm, why it is correct, and the many graphs it runs on.',
   ['Weighted shortest paths and relaxation',
    "Dijkstra's algorithm and its proof (the settled-set invariant)",
    'Implementations: binary heap with lazy deletion, $\\bigO{n^2}$ for dense graphs',
    'Shortest-path trees, counting shortest paths, the shortest-path DAG',
    'State-space Dijkstra (extra dimensions in the vertex); one target only: bidirectional Dijkstra and A* search with an admissible heuristic'],
   [('Single-source shortest paths', [('Shortest Routes I', 'CSES', '1671'), ('Dijkstra?', 'CF', '20C')]),
    ('Counting and tie-breaking', [('Number of Ways to Arrive at Destination', 'LC', '1976')]),
    ('State-space Dijkstra', [('Flight Discount', 'CSES', '1195')]),
    ('Several sources and reversed graphs', [('Minimum Weighted Subgraph With the Required Paths', 'LC', '2203'), ('Reducing Delivery Cost', 'CF', '1433G')]),
    ('k shortest paths', [('Flight Routes', 'CSES', '1196')])])

ch(8, 'negative_weights', 'Negative Weights, Bellman--Ford \\& 0-1 BFS', 2, '1500', '2200',
   'When edges can be negative --- or only 0 and 1 --- the right algorithm changes; learn when and why.',
   ['0-1 BFS with a deque; Dial\'s algorithm for small integer weights',
    'Bellman--Ford: relaxation rounds and paths with at most $k$ edges',
    'Negative cycles: detection, reporting, and vertices affected by them',
    'SPFA and why it can be slow',
    'Difference constraints, potentials and Johnson\'s reweighting; minimum mean cycles (Karp) and ratio cycles (binary search + negative cycle)'],
   [('0-1 BFS', [('Labyrinth', 'CF', '1063B'), ('Minimum Cost to Make at Least One Valid Path in a Grid', 'LC', '1368')]),
    ('At most k edges', [('Cheapest Flights Within K Stops', 'LC', '787')]),
    ('Negative cycles', [('Cycle Finding', 'CSES', '1197'), ('High Score', 'CSES', '1673')]),
    ('Difference constraints', []),
    ('Mean and ratio cycles', [])])

ch(9, 'all_pairs', 'All-Pairs Shortest Paths', 2, '1500', '2100',
   'Distances between every pair of vertices: Floyd--Warshall, its DP meaning, and the problems it quietly solves.',
   ['Floyd--Warshall as DP over intermediate vertices',
    'Path reconstruction and negative cycles',
    'Transitive closure with bitsets',
    'Adding vertices or edges one at a time',
    'Minimax and maximin paths; Johnson\'s algorithm for sparse graphs'],
   [('Floyd--Warshall', [('Shortest Routes II', 'CSES', '1672'), ('Find the City With the Smallest Number of Neighbors at a Threshold Distance', 'LC', '1334')]),
    ('Vertex insertion order', [('Greg and Graph', 'CF', '295B'), ('Roads in Berland', 'CF', '25C')]),
    ('Transitive closure', [('Course Schedule IV', 'LC', '1462')]),
    ('Minimax paths', []),
    ('Reconstructing from distances', [('Anna, Svyatoslav and Maps', 'CF', '1204C')])])

ch(10, 'mst', 'Minimum Spanning Trees', 2, '1400', '2200',
   'Connect everything as cheaply as possible --- and understand the cut and cycle properties that make greed correct.',
   ['Spanning trees; the cut property and the cycle property',
    "Kruskal's algorithm (with DSU)",
    "Prim's algorithm (heap and dense $\\bigO{n^2}$ versions); Bor\\r{u}vka's algorithm",
    'MST structure: uniqueness, bottleneck paths, second-best MST',
    'Critical edges; MSTs of implicit and geometric graphs; directed MST (minimum arborescence, Chu--Liu/Edmonds)'],
   [('Kruskal and Prim', [('Road Reparation', 'CSES', '1675'), ('Min Cost to Connect All Points', 'LC', '1584')]),
    ('Virtual vertices', [('Shichikuji and Power Grid', 'CF', '1245D')]),
    ('Which edges can be in an MST?', [('Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree', 'LC', '1489'), ('Edges in MST', 'CF', '160D')]),
    ('Bottleneck and minimax', []),
    ('Special and directed MSTs', [('0-1 MST', 'CF', '1242B'), ('Directed MST', 'LIB', 'directedmst')])])

ch(11, 'binary_lifting', 'Binary Lifting \\& Jump Pointers', 2, '1500', '2200',
   'Answer "where am I after $k$ steps?" in $\\bigO{\\log k}$: double every jump, and carry sums, minima or counts along with it.',
   ['From Chapter 1 (the rho shape of functional graphs) to many queries: why one walk per query is too slow',
    'Binary lifting: the table $\\mathrm{up}[j][v]$ and the $k$-th successor in $\\bigO{\\log k}$',
    'Carrying aggregates along jumps: sums, minima, counts',
    'Distances and meeting points in functional graphs; permutations: powers and cycle structure',
    'Jump pointers on arrays: next-greater links, greedy segment covers, "first time above a threshold" by binary descent'],
   [('k-th successor', [('Planets Queries I', 'CSES', '1750')]),
    ('Distances in functional graphs', [('Planets Queries II', 'CSES', '1160')]),
    ('Permutation powers and cycles', [('Books Exchange (hard version)', 'CF', '1249B2')]),
    ('Doubling with aggregates', [('Analysis of Pathes in Functional Graph', 'CF', '702E')]),
    ('Jump pointers on arrays', [('Minimal Segment Cover', 'CF', '1175E')])],
   r'Cycle finding in functional graphs is Chapter~1 (Pattern~5); binary lifting on trees (LCA) is Chapter~14.')

ch(12, 'construction', 'Graph Construction Techniques', 2, '1700', '2400',
   'The hardest step is often building the graph: virtual vertices, layers, segment-tree edges, complements --- and game graphs.',
   ['Super sources and virtual vertices',
    'Layered (product) graphs: vertex = (state, layer)',
    'Segment-tree-optimised edges: ranges of edges in $\\bigO{\\log n}$',
    'Complement graphs (traversal with a set of unvisited vertices) and vertices for values (bucket and divisor graphs)',
    'Game graphs with cycles: win / lose / draw by retrograde BFS from the terminal states'],
   [('Virtual vertices', []),
    ('Layered graphs', []),
    ('Segment-tree graphs', [('Legacy', 'CF', '786B')]),
    ('Complements and value vertices', [('Connected Components?', 'CF', '920E'), ('Greatest Common Divisor Traversal', 'LC', '2709')]),
    ('Games on graphs (retrograde analysis)', [('Cat and Mouse', 'LC', '913'), ('Berzerk', 'CF', '786A')])],
   r'Games on DAGs and Grundy numbers are \dpbook{10}{Game Theory DP}; this chapter adds games whose graphs have cycles (draws).')

# ------------------------------------------------------------------ PART III
ch(13, 'tree_basics', 'Tree Fundamentals', 3, '1400', '2200',
   'Trees are the graphs you will meet most: root them, measure them, and recognise them up to isomorphism.',
   ['Rooting a tree: parents, depths, subtree sizes',
    'Diameter: two searches, and the DP view',
    'Centres and centroids',
    'The Euler tour: flattening a tree so every subtree is a contiguous range (ancestor tests were Chapter 2; range queries are Chapter 15)',
    'Tree isomorphism by hashing; Pr\\"ufer sequences and counting labelled trees'],
   [('Diameter and eccentricity', [('Tree Diameter', 'CSES', '1131'), ('Tree Distances I', 'CSES', '1132')]),
    ('Centres and centroids', [('Link Cut Centroids', 'CF', '1406C')]),
    ('Euler tour flattening', []),
    ('Isomorphism and hashing', [('Symmetree', 'CF', '1800G')]),
    ('Counting trees', [('Clues', 'CF', '156D')])],
   r'Tree DP is \dpbook{11}{DP on Trees}; rerooting is \dpbook{19}{Rerooting DP}.')

ch(14, 'lca', 'Lowest Common Ancestor', 3, '1500', '2300',
   'The question behind most tree queries: where do two root paths meet?',
   ['Binary lifting: ancestors at every power of two',
    'Euler tour + sparse table: LCA in $\\bigO{1}$',
    "Tarjan's offline LCA with DSU",
    'Level ancestors and the $k$-th vertex on a path',
    'Path aggregates (min, max, sum) with lifting'],
   [('Binary lifting', [('Company Queries I', 'CSES', '1687'), ('Tree', 'CF', '932D')]),
    ('LCA queries', [('Company Queries II', 'CSES', '1688')]),
    ('Distances in trees', [('Distance Queries', 'CSES', '1135')]),
    ('Path aggregates', [('Minimum spanning tree for each edge', 'CF', '609E')]),
    ('LCA reasoning', [('Passable Paths (hard version)', 'CF', '1702G2')])])

ch(15, 'tree_queries', 'Subtree \\& Path Queries', 3, '1600', '2400',
   'Turn trees into arrays and merge information cleverly: Euler tours, differences on trees, small-to-large and virtual trees.',
   ['Euler tour + Fenwick / segment tree for subtree queries',
    'Root-path prefixes for path queries',
    'Differences on trees: counting how many paths cover each vertex or edge',
    'Small-to-large merging (DSU on tree)',
    'Virtual (auxiliary) trees for queries on vertex subsets'],
   [('Subtree queries', [('Subtree Queries', 'CSES', '1137')]),
    ('Path queries', [('Path Queries', 'CSES', '1138')]),
    ('Differences on trees', [('Counting Paths', 'CSES', '1136'), ('Fools and Roads', 'CF', '191C')]),
    ('Small-to-large', [('Lomsat gelral', 'CF', '600E'), ('Distinct Colors', 'CSES', '1139')]),
    ('Virtual trees', [('Kingdom and its Cities', 'CF', '613D')])])

ch(16, 'hld', 'Heavy--Light Decomposition', 3, '1900', '2500',
   'Cut every root path into $\\bigO{\\log n}$ chains and answer path queries with a segment tree.',
   ['Heavy edges and chains; the $\\bigO{\\log n}$ light-edge bound',
    'Numbering vertices so chains (and subtrees) are contiguous',
    'Path queries and updates with a segment tree',
    'Vertex values versus edge values; non-commutative path aggregates',
    'Long-path decomposition: depth-indexed data merged in $\\bigO{n}$'],
   [('Path queries', [('Path Queries II', 'CSES', '2134')]),
    ('Path updates', []),
    ('Edge values', [('Query on a tree', 'SPOJ', 'QTREE')]),
    ('Subtree and path together', [('Water Tree', 'CF', '343D')]),
    ('Long-path decomposition', [('Dominant Indices', 'CF', '1009F')])])

ch(17, 'centroid', 'Centroid Decomposition', 3, '2000', '2600',
   'Divide a tree at its centroid, recurse, and every path is handled at exactly one level.',
   ['Centroids exist and halve the tree',
    'The decomposition and the centroid tree ($\\bigO{\\log n}$ depth)',
    'Counting paths through a centroid',
    'Nearest-marked-vertex queries on the centroid tree',
    'Combining with data structures and small-to-large'],
   [('Paths of fixed length', [('Fixed-Length Paths I', 'CSES', '2080'), ('Fixed-Length Paths II', 'CSES', '2081')]),
    ('Nearest marked vertex', [('Xenia and Tree', 'CF', '342E')]),
    ('Centroid-tree properties', [('Ciel the Commander', 'CF', '321C')]),
    ('Path counting with weights', []),
    ('Centroid + data structures', [])])

# ------------------------------------------------------------------ PART IV
ch(18, 'bridges', 'Bridges, Articulation Points \\& Biconnectivity', 4, '1700', '2500',
   'One DFS and the low-link value reveal every single point of failure in a network.',
   ['The DFS tree and low-link values',
    'Bridges and 2-edge-connected components; the bridge tree',
    'Articulation points and biconnected components; the block-cut tree',
    'Strong orientations (Robbins\' theorem)',
    'Iterative low-link for deep graphs'],
   [('Bridges', [('Critical Connections in a Network', 'LC', '1192'), ('Necessary Roads', 'CSES', '2076')]),
    ('Articulation points', [('Necessary Cities', 'CSES', '2077')]),
    ('The bridge tree', [('We Need More Bosses', 'CF', '1000E')]),
    ('The block-cut tree', [('Tourists', 'CF', '487E')]),
    ('Orientations', [('Bertown roads', 'CF', '118E')])])

ch(19, 'scc', 'Strongly Connected Components', 4, '1700', '2400',
   'Collapse every directed cycle into a single vertex and the graph becomes a DAG.',
   ['Strong connectivity and the condensation DAG',
    "Kosaraju's two-pass algorithm",
    "Tarjan's one-pass algorithm",
    'DP on the condensation',
    'Making a graph strongly connected with the fewest new edges'],
   [('Finding SCCs', [('Flight Routes Check', 'CSES', '1682'), ('Planets and Kingdoms', 'CSES', '1683')]),
    ('DP on the condensation', [('Coin Collector', 'CSES', '1686')]),
    ('Sources and sinks', [('Checkposts', 'CF', '427C')]),
    ('Fewest edges to strong connectivity', [('Scheme', 'CF', '22E')]),
    ('Modelling with SCCs', [('Catowice City', 'CF', '1239D')])])

ch(20, 'two_sat', '2-SAT', 4, '1800', '2500',
   'Boolean constraints on pairs of variables become an implication graph --- and SCCs decide them in linear time.',
   ['Clauses, literals and the implication graph',
    'Satisfiability via strongly connected components',
    'Constructing an assignment',
    'Modelling: equivalence, XOR, at-most-one, implications',
    'Prefix variables for at-most-one constraints in groups'],
   [('Classic 2-SAT', [('Giant Pizza', 'CSES', '1684')]),
    ('Switches and toggles', [('The Door Problem', 'CF', '776D')]),
    ('Binary search + 2-SAT', []),
    ('Interval and time constraints', [('Radio Stations', 'CF', '1215F')]),
    ('At-most-one encodings', [])])

ch(21, 'euler', 'Eulerian Paths \\& Circuits', 4, '1600', '2400',
   'Walk every edge exactly once: when it is possible, and Hierholzer\'s algorithm to do it.',
   ['Existence: degrees and connectivity, undirected and directed',
    "Hierholzer's algorithm",
    'Lexicographically smallest Euler paths',
    'De Bruijn sequences',
    'Adding edges to make a graph Eulerian; pairing tricks'],
   [('Undirected circuits', [('Mail Delivery', 'CSES', '1691')]),
    ('Directed paths', [('Teleporters Path', 'CSES', '1693'), ('Reconstruct Itinerary', 'LC', '332')]),
    ('De Bruijn sequences', [('De Bruijn Sequence', 'CSES', '1692')]),
    ('Pairing via Euler tours', [('One-Way Reform', 'CF', '723E')]),
    ('Euler tours in constructions', [('Neko and Flashback', 'CF', '1152E')])])

ch(22, 'cycles', 'Cycles, Triangles \\& Degree Tricks', 4, '1700', '2500',
   'Short cycles, triangles and the cycle space: counting small subgraphs fast and using every cycle a DFS tree reveals.',
   ['The shortest cycle (girth): BFS from every vertex',
    'Triangles and 4-cycles in $\\bigO{m\\sqrt{m}}$: orienting edges from low to high degree',
    'Heavy and light vertices: square-root decomposition on degrees',
    'The cycle space: XOR basis of cycles and maximum / minimum XOR paths',
    'Cycles from a DFS tree: odd cycles, long cycles, independent sets (constructive proofs)'],
   [('Shortest cycle', [('Graph Girth', 'CSES', '1707')]),
    ('Triangles and small subgraphs', [('Minimum Degree of a Connected Trio in a Graph', 'LC', '1761')]),
    ('Heavy and light vertices', [('Similar Sets', 'CF', '1468M')]),
    ('XOR paths and the cycle space', [('Shortest Path Problem?', 'CF', '845G')]),
    ('Cycles from DFS trees', [("Ehab's Last Corollary", 'CF', '1364D'), ("Ehab's Last Theorem", 'CF', '1325F')])])

ch(22, 'hard_small', 'Hard Problems on Small Graphs', 4, '1800', '2600',
   'Hamiltonian paths, cliques and colourings are NP-hard --- recognise them, then solve them exactly when $n$ is small.',
   ['Recognising NP-hard problems; budgets for $n \\le 20$, $40$, $60$',
    r'Hamiltonian paths and TSP by bitmask DP (\dpbook{13}{Bitmask DP})',
    'Maximum clique and independent set: branching and meet in the middle',
    'Chromatic number by inclusion--exclusion in $\\bigO{2^n n}$',
    'Steiner trees by DP over subsets'],
   [('Hamiltonian paths', []),
    ('Cliques and independent sets', [('Helping Hiasat', 'CF', '1105E')]),
    ('Colouring', []),
    ('Steiner trees', []),
    ('Exponential search with pruning', [])],
   r'Bitmask DP itself is \dpbook{13}{Bitmask DP}; this chapter focuses on the graph side.')

# ------------------------------------------------------------------ PART V
ch(23, 'max_flow', 'Maximum Flow', 5, '1900', '2600',
   'Push as much as possible from a source to a sink: residual graphs, augmenting paths and Dinic\'s algorithm.',
   ['Networks, flows and residual graphs',
    'Ford--Fulkerson and Edmonds--Karp',
    "Dinic's algorithm: level graphs and blocking flows; unit networks",
    'Edge-disjoint and vertex-disjoint paths; vertex splitting',
    'Flow decomposition and capacity scaling'],
   [('Maximum flow', [('Download Speed', 'CSES', '1694')]),
    ('Disjoint paths', [('Distinct Routes', 'CSES', '1711')]),
    ('Vertex capacities', []),
    ('Assignment by flow', [('Soldier and Traveling', 'CF', '546E')]),
    ('Binary search on a flow', [])])

ch(24, 'min_cut', 'Minimum Cut \\& Flow Modelling', 5, '2000', '2700',
   'Every maximum flow certifies a minimum cut --- and cuts model selection, separation and closure problems.',
   ['The max-flow min-cut theorem and recovering the cut',
    'Project selection and maximum-weight closure',
    'Infinite edges as constraints',
    'Flows with lower bounds; circulations',
    'Cuts in grids and images'],
   [('Recovering the cut', [('Police Chase', 'CSES', '1695')]),
    ('Project selection', [('Petya and Graph', 'CF', '1082G')]),
    ('Lower bounds and circulations', []),
    ('Grid cuts', []),
    ('Cut modelling', [])])

ch(25, 'bipartite_matching', 'Bipartite Matching \\& Covers', 5, '1800', '2600',
   'Pair things up optimally --- and the theorems that connect matchings, covers and independent sets.',
   ["Augmenting paths and Kuhn's algorithm",
    'Hopcroft--Karp in $\\bigO{E\\sqrt{V}}$',
    "K\\\"onig's theorem: vertex cover and independent set; Hall's theorem",
    r"Minimum path cover in DAGs and Dilworth's theorem (\dpbook{4}{Subsequence DP})",
    'Edge covers, matchings on grids, bipartite edge colouring; stable marriage (Gale--Shapley)'],
   [('Maximum matching', [('School Dance', 'CSES', '1696')]),
    ("Hall's theorem", []),
    ('Vertex covers', []),
    ('Path covers', []),
    ('Grid matchings', [])])

ch(26, 'min_cost_flow', 'Min-Cost Flow \\& Assignment', 5, '2100', '2800',
   'Flows where every unit has a price: successive shortest paths, potentials and the Hungarian algorithm.',
   ['Min-cost flow; successive shortest paths',
    'Potentials: Dijkstra on reduced costs',
    'The assignment problem and the Hungarian algorithm',
    'Cycle cancelling and negative costs',
    'Modelling with costs: $k$ disjoint cheapest paths, convex costs'],
   [('Successive shortest paths', [('Distinct Routes II', 'CSES', '2130')]),
    ('Assignment', [('Task Assignment', 'CSES', '2129')]),
    ('k disjoint cheapest paths', []),
    ('Convex costs', []),
    ('Cost modelling', [('Binary Tree on Plane', 'CF', '277E')])])

ch(27, 'general_matching', 'Matching in General Graphs', 5, '2400', '3000',
   'Odd cycles break augmenting paths --- blossoms repair them.',
   ['Alternating paths and why odd cycles are a problem',
    "Edmonds' blossom algorithm",
    'The Tutte--Berge formula',
    'Randomised matching with the Tutte matrix',
    'Weighted general matching: an overview'],
   [('Maximum matching', [('General Matching', 'LIB', 'general_matching')]),
    ('Tutte--Berge reasoning', []),
    ('Randomised algebraic matching', []),
    ('Games and matchings', []),
    ('Weighted matching', [])])

# ------------------------------------------------------------------ PART VI
ch(28, 'offline_connectivity', 'Offline \\& Dynamic Connectivity', 6, '2200', '3000',
   'Connectivity when edges come and go: rollback DSU, divide and conquer over time, and Kruskal reconstruction trees.',
   ['DSU with rollback',
    'Segment tree over time (offline dynamic connectivity)',
    'The Kruskal reconstruction tree',
    'Parallel binary search on graphs',
    'Dynamic bipartiteness and offline bridges'],
   [('Rollback DSU', []),
    ('Divide and conquer over time', [('Bipartite Checking', 'CF', '813F')]),
    ('Kruskal reconstruction tree', [('Graph and Queries', 'CF', '1416D')]),
    ('Parallel binary search', [('Stamp Rally', 'AC', 'agc002_d')]),
    ('Fully dynamic connectivity offline', [('Dynamic Graph Vertex Add Component Sum', 'LIB', 'dynamic_graph_vertex_add_component_sum')])])

ch(29, 'link_cut', 'Link-Cut Trees \\& Dynamic Forests', 6, '2400', '3100',
   'Forests that change: link, cut and query paths online in amortised $\\bigO{\\log n}$.',
   ['Splay trees in a nutshell',
    'Preferred paths and the access operation',
    'make\\_root, link and cut; path aggregates',
    'Euler-tour trees for subtree queries in dynamic forests',
    'Top trees and when to use what'],
   [('Dynamic connectivity in forests', []),
    ('Path aggregates', [('Dynamic Tree Vertex Add Path Sum', 'LIB', 'dynamic_tree_vertex_add_path_sum')]),
    ('Rerooting dynamic trees', []),
    ('LCT + offline tricks', []),
    ('Euler-tour trees', [])])

ch(30, 'dominators', 'Dominator Trees', 6, '2300', '3000',
   'Which vertices must every path from the source pass through? The dominator tree answers for all vertices at once.',
   ['Dominance and immediate dominators',
    'The dominator tree of a DAG via LCA',
    'The Lengauer--Tarjan algorithm',
    'Edge versions and articulation-style questions in directed graphs',
    'Applications: critical vertices on shortest paths'],
   [('Dominator tree', [('Dominator Tree', 'LIB', 'dominatortree')]),
    ('DAG dominators', []),
    ('Shortest-path dominators', [('Team Rocket Rises Again', 'CF', '757F')]),
    ('Counting with dominators', []),
    ('Applications', [])])

ch(31, 'global_cuts', 'Global Cuts \\& Gomory--Hu Trees', 6, '2400', '3100',
   'The cheapest way to disconnect a graph, and all pairwise minimum cuts in one tree.',
   ['Global minimum cut: Stoer--Wagner',
    "Karger's random contraction",
    'Gomory--Hu trees: $n-1$ flows give every pairwise min cut',
    'Applications of Gomory--Hu trees',
    'Cut enumeration and probability amplification'],
   [('Global min cut', []),
    ('Randomised contraction', []),
    ('Gomory--Hu construction', []),
    ('All-pairs min cut queries', []),
    ('Cut applications', [])])

ch(32, 'special_classes', 'Special Graph Classes', 6, '2200', '3000',
   'Cacti, planar graphs, chordal and interval graphs, tournaments: structure that turns hard problems easy.',
   ['Cactus graphs and their tree of cycles',
    "Planar graphs: Euler's formula, faces and dual graphs",
    'Chordal graphs: perfect elimination orders and maximum cardinality search',
    'Interval graphs and tournaments',
    'Bounded treewidth: the idea and when it appears'],
   [('Cactus graphs', [('Cactus', 'CF', '231E')]),
    ('Planar graphs', []),
    ('Chordal graphs', []),
    ('Tournaments', []),
    ('Treewidth', [])])

ch(33, 'algebraic', 'Counting \\& Algebraic Graph Theory', 6, '2200', '3200',
   'Linear algebra and counting on graphs: walks, spanning trees, Euler circuits and randomised hashing.',
   ['Counting walks with adjacency-matrix powers; min-plus matrix powers; random walks and expected hitting times (Gaussian elimination)',
    "Kirchhoff's matrix-tree theorem",
    'The BEST theorem: counting Euler circuits',
    'Counting colourings and independent sets by inclusion--exclusion',
    'Random hashing on graphs (XOR hashing of edges and cycles)'],
   [('Counting walks', [('Graph Paths I', 'CSES', '1723'), ('Graph Paths II', 'CSES', '1724')]),
    ('Counting spanning trees', [('Counting Spanning Trees', 'LIB', 'counting_spanning_tree_undirected')]),
    ('Counting Euler circuits', []),
    ('Inclusion--exclusion on graphs', []),
    ('Graph hashing', [])],
   r'Matrix exponentiation itself is \dpbook{25}{Matrix Exponentiation}.')

ch(34, 'constructive', 'Constructive \\& Interactive Graph Problems', 6, '1700', '2700',
   'Build a graph with the required properties, or discover a hidden one with few queries --- two problem types Codeforces loves.',
   ['Constructing trees and graphs with given degrees, distances or diameter',
    'Spanning trees under constraints (fixed degrees, forced edges)',
    'Validating and reconstructing BFS / DFS orders and distance arrays',
    'Interactive problems on trees: binary search, centroids and query budgets',
    'Interactive problems on graphs: finding hidden vertices and edges with distance queries'],
   [('Trees with given parameters', [('Tree Constructing', 'CF', '1003E'), ('Multihedgehog', 'CF', '1067B')]),
    ('Constrained spanning trees', [('Spanning Tree with One Fixed Degree', 'CF', '1133F2')]),
    ('Reconstruction from orders and distances', [('Valid BFS?', 'CF', '1037D'), ('Restore Graph', 'CF', '404C')]),
    ('Interactive on trees', [('Hemose in ICPC ?', 'CF', '1592D')]),
    ('Interactive on graphs', [('The Hidden Pair (Hard Version)', 'CF', '1370F2')])])

ch(35, 'mastery', 'Contest Mastery: Mixed Hard Graph Problems', 6, '2400', '3500',
   'No hints about the technique: recognise the model, pick the tool, and combine chapters under contest pressure.',
   ['A recognition guide: from constraints to the technique',
    'Combining techniques: shortest paths + DSU, flows + binary search, trees + data structures',
    'Proof habits for graph greedy and exchange arguments',
    'Implementation templates and pitfalls (recursion depth, multi-tests, overflow)',
    'Mixed contest sets by rating'],
   [('Mixed set 2400', []), ('Mixed set 2600', []), ('Mixed set 2800', []), ('Mixed set 3000', []), ('Mixed set 3200+', [])])


# ------------------------------------------------------------------ helpers
def judge_name(j, i):
    return {'CF': 'CF ' + i, 'LC': 'LeetCode ' + i, 'CSES': 'CSES ' + i, 'AC': 'AtCoder ' + i.replace('_', '\\_').upper(),
            'LIB': 'Library Checker', 'SPOJ': 'SPOJ ' + i}[j]


def esc(t):
    return t.replace('&', '\\&').replace('#', '\\#')


def folder(c):
    return 'ch%02d_%s' % (c['num'], c['slug'])


def stub(c):
    L = []
    L.append('%' + '=' * 60)
    L.append('%% Chapter %d: %s' % (c['num'], c['title'].replace('\\&', '&')))
    L.append('% STATUS: PLANNED  (generated by tools/outline.py; replace this file when the chapter is written)')
    L.append('%' + '=' * 60)
    L.append('\\chapter{%s}' % c['title'])
    L.append('\\label{ch:%s}' % c['slug'])
    L.append('')
    L.append('\\begin{tcolorbox}[enhanced, colback=white, colframe=secondaryblue, boxrule=1pt, arc=2mm]')
    L.append('\\textbf{\\large\\textcolor{primaryblue}{%s}}' % c['promise'])
    L.append('\\end{tcolorbox}')
    L.append('')
    L.append('\\chapterfacts{%s}{%s}{5}{10}{30}' % (c['lo'], c['hi']))
    L.append('')
    L.append('\\begin{plannedbox}[Planned Contents --- this chapter is not written yet]')
    L.append('\\textbf{Part A --- Lessons}')
    L.append('\\begin{enumerate}[leftmargin=*, label=\\textbf{A\\arabic*.}, itemsep=1pt]')
    for l in c['lessons']:
        L.append('    \\item %s' % l)
    L.append('\\end{enumerate}')
    L.append('\\textbf{Part B --- Patterns} %s' % (
        '(with the two solved problems of each pattern)' if c['part'] == 1 else '(with candidate problems, checked when the chapter is written)'))
    L.append('\\begin{enumerate}[leftmargin=*, label=\\textbf{P\\arabic*.}, itemsep=1pt]')
    for name, probs in c['patterns']:
        s = ', '.join('%s {\\small\\color{darkgray}(%s)}' % (esc(t), judge_name(j, i)) for t, j, i in probs)
        L.append('    \\item \\textbf{%s}%s' % (name, (' --- ' + s) if s else ''))
    L.append('\\end{enumerate}')
    if c['connections']:
        L.append('\\textbf{Connections.} %s' % c['connections'])
    L.append('\\end{plannedbox}')
    return '\n'.join(L) + '\n'


def write(path, text):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    open(full, 'w', encoding='utf8').write(text)


def is_planned(path):
    full = os.path.join(ROOT, path)
    return (not os.path.exists(full)) or ('% STATUS: PLANNED' in open(full, encoding='utf8').read())


def main():
    written = []
    for c in CH:
        p = 'chapters/%s/ch%02d.tex' % (folder(c), c['num'])
        if is_planned(p):
            write(p, stub(c))
        else:
            written.append(c['num'])
    for P in PARTS:
        cs = [c for c in CH if c['part'] == P['n']]
        n_pat = sum(len(c['patterns']) for c in cs)
        o = ['%' + '=' * 60, '%% Part %s opener  (generated by tools/outline.py; design: preamble/partpage.tex)' % P['n'],
             '%' + '=' * 60,
             '\\dppartopener{%s}{%s}{%s}%%' % (P['title'], P['level'], P['color']),
             '  {%s}' % P['tagline'], '',
             '\\dppartstats{%d}{%d}{%d}{%d}' % (len(cs), n_pat, 2 * n_pat, 30 * len(cs)), '',
             '\\begin{dpjourney}']
        for c in cs:
            o.append('\\dpstep{%d}{%s}{%s}{%s}{%s}' % (
                c['num'], c['title'], ' $\\cdot$ '.join(name for name, _ in c['patterns'][:5]),
                ('Beginner' if c['lo'] == 'Beginner' else 'CF ' + c['lo']) + ' -- ' + c['hi'],
                'yes' if c['num'] in written else 'no'))
        o += ['\\end{dpjourney}', '', '\\dppartgoals{'] + ['  \\item ' + g for g in P['goals']] + ['}', '']
        write('parts/part%d_%s.tex' % (P['n'], P['key']), '\n'.join(o))
        inc = ['%' + '=' * 60, '%% Part %d --- chapter list (generated by tools/outline.py)' % P['n'], '%' + '=' * 60]
        inc += ['\\include{chapters/%s/ch%02d}' % (folder(c), c['num']) for c in cs]
        write('parts/part%d_chapters.tex' % P['n'], '\n'.join(inc) + '\n')
    write('frontmatter/roadmap.tex', roadmap(written))
    print('chapters: %d (%d written, %d planned stubs); parts: %d' % (len(CH), len(written), len(CH) - len(written), len(PARTS)))


def roadmap(written):
    R = ['%' + '=' * 60, '% roadmap.tex --- the whole book at a glance (generated by tools/outline.py)', '%' + '=' * 60,
         '\\chapter{Roadmap of the Book}', '', '\\section*{Six Parts, One Staircase}', '',
         '\\begin{diagrambox}',
         '\\begin{tikzpicture}[x=2.45cm, y=1.05cm,',
         '    stair/.style={rectangle, rounded corners=2pt, draw=#1, fill=#1!12, line width=1pt,',
         '                  align=center, font=\\scriptsize, minimum width=2.3cm, minimum height=1.6cm}]']
    short = {1: 'Foundations', 2: 'Paths \\& MSTs', 3: 'Trees', 4: 'Connectivity', 5: 'Flows', 6: 'Advanced'}
    for P in PARTS:
        cs = [c['num'] for c in CH if c['part'] == P['n']]
        lv = P['level'].replace('$\\to$', '--').replace('CF ', '')
        R.append('    \\node[stair=%s] at (%d,%d) {\\textbf{Part %s}\\\\%s\\\\Ch %d--%d\\\\{\\tiny %s}};' % (
            P['color'], P['n'] - 1, P['n'] - 1, ['I', 'II', 'III', 'IV', 'V', 'VI'][P['n'] - 1], short[P['n']],
            cs[0], cs[-1], lv))
    R += ['    \\draw[flow, line width=1.5pt] (-0.45,-1.1) -- (5.45,4.1);', '\\end{tikzpicture}',
          '\\figcap{Each part assumes fluency with everything below it. Do not skip steps.}', '\\end{diagrambox}', '',
          '\\section*{Chapter List}', '',
          'Status: \\yes~written \\quad $\\circ$~planned. For Part~I the two \\emph{solved} problems of every pattern are '
          'listed; every problem appears in exactly \\textbf{one} place in the book, and no problem is solved both here and '
          'in the companion \\textit{Dynamic Programming} book.', '',
          '\\begingroup', '\\small', '\\renewcommand{\\arraystretch}{1.25}']
    roman = ['I', 'II', 'III', 'IV', 'V', 'VI']
    for P in PARTS:
        R.append('')
        R.append('\\subsection*{Part %s --- %s (%s)}' % (roman[P['n'] - 1], P['title'], P['level']))
        R.append('\\begin{tabularx}{\\linewidth}{@{}c c >{\\raggedright\\arraybackslash}p{3.6cm} X@{}}')
        R.append('\\toprule')
        R.append('& \\textbf{Ch} & \\textbf{Title} & \\textbf{%s} \\\\' % (
            'Patterns (solved problems)' if P['n'] == 1 else 'Patterns'))
        R.append('\\midrule')
        for c in [c for c in CH if c['part'] == P['n']]:
            if P['n'] == 1:
                pats = ' $\\cdot$ '.join('%s (%s)' % (name, ', '.join(esc(t) for t, _, _ in probs)) for name, probs in c['patterns'])
            else:
                pats = ' $\\cdot$ '.join(name for name, _ in c['patterns'])
            R.append('%s & %d & %s & %s \\\\' % ('\\yes' if c['num'] in written else '$\\circ$', c['num'], c['title'], pats))
        R.append('\\bottomrule')
        R.append('\\end{tabularx}')
    R += ['\\endgroup', '']
    return '\n'.join(R)


if __name__ == '__main__':
    main()
