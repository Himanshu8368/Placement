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
APIs and against the six companion books (tools/verify_outline.py); later parts are candidates and are
re-checked when the chapter is written.

Topics owned by a companion book are NOT taught here (the text points to them instead):
    Fenwick / segment trees / sparse tables / Mo on trees ......... tree book
    monotonic stack, histogram, deque windows, heaps, top-k ........ stack-queue-heap book
    string two pointers / windows / hashing of strings ............. string book
    merge sort, inversions, quickselect, CDQ, meet in the middle ... recursion book
    binary search beyond the basics (rotated, answer, ternary, ...) . binary search book (Ch 7 here is the basics)
    Kadane, LIS, knapsack and every DP .............................. DP book
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

PARTS = [
    dict(n=1, key='foundations', title='Foundations', level=r'Beginner $\to$ CF 1800', color='successgreen',
         tagline='The array and the seven tools you will use in every later chapter: scans, prefix sums, hashing, two pointers, windows, sorting and binary search.',
         goals=[r'reason about arrays as contiguous memory: indices, half-open ranges, in-place updates and the cost model;',
                r'answer range questions with prefix sums and apply range updates with difference arrays, in 1D and 2D;',
                r'count pairs and subarrays with hash maps, and replace nested loops by two pointers or a sliding window;',
                r'use sorting as a preprocessing step, and binary search sorted data with bounds and prefix sums.']),
    dict(n=2, key='core', title='Core Techniques', level=r'CF 1200 $\to$ CF 2100', color='primaryblue',
         tagline='Intervals, in-place rearrangement, index tricks, matrices and array-based data structures: the techniques interviews and Div.~2 rounds test most.',
         goals=[r'merge, cover and sweep intervals and events;',
                r'partition and rearrange arrays in place, and use values as indices (cyclic sort, marking, permutation cycles);',
                r'traverse, rotate and query matrices, and reduce 2D problems to 1D ones;',
                r'design array-based data structures: swap-with-last deletion, timestamps, snapshots.']),
    dict(n=3, key='counting', title='Counting, Greedy \\& Number Arrays', level=r'CF 1500 $\to$ CF 2400', color='softpurple',
         tagline='Count over all subarrays and pairs without listing them, choose greedily with a proof, and exploit bits, divisors and values.',
         goals=[r'count subarrays and pairs by contribution, by fixing one end, or by complementary counting;',
                r'prove greedy choices on arrays with exchange arguments and invariants;',
                r'split problems by bits, and use the "few distinct values" property of OR, AND and GCD over subarrays;',
                r'handle MEX, distinctness and value-domain questions, and process queries offline after compression.']),
    dict(n=4, key='queries', title='Square Root \\& Query Techniques', level=r'CF 1600 $\to$ CF 2600', color='accentorange',
         tagline="When prefix sums are not enough and a tree is not needed: blocks, Mo's order, multidimensional prefixes, hashing and selection.",
         goals=[r'decompose arrays into $\sqrt{n}$ blocks and balance update and query costs;',
                r"answer offline range queries with Mo's algorithm, including updates;",
                r'use higher-order and multidimensional prefix sums and difference arrays;',
                r'check multisets and permutations with random hashing, and select order statistics without sorting.']),
    dict(n=5, key='adhoc', title='Constructive, Operations \\& Ad-hoc', level=r'CF 1400 $\to$ CF 2500', color='warningred',
         tagline='The problems with no standard algorithm: build an array, reach a target by operations, find invariants, ask few queries.',
         goals=[r'construct arrays with given properties and prove the construction;',
                r'find the invariant of an operation and decide reachability and the minimum number of operations;',
                r'solve interactive problems on hidden arrays within a query budget;',
                r'handle circular arrays and arrays with deletions (linked lists and next pointers in arrays).']),
    dict(n=6, key='advanced', title='Advanced \\& Expert Topics', level=r'CF 1800 $\to$ CF 3500', color='deepblue',
         tagline='Scanlines over the right endpoint, permutation algebra, bitsets, heavy--light value splits and mixed expert problems.',
         goals=[r'sweep the right endpoint and maintain answers for all left endpoints at once;',
                r'use the cycle structure, inverses and powers of permutations;',
                r'speed up array algorithms with bitsets and heavy/light splits by frequency;',
                r'recognise the technique of an unlabeled hard array problem under contest pressure.']),
]

# (num, slug, title, part, lo, hi, promise, lessons, patterns, connections)
# patterns: list of (name, [(title, judge, id), ...])
CH = []


def ch(num, slug, title, part, lo, hi, promise, lessons, patterns, connections=''):
    num = len(CH) + 1   # chapters are numbered by their order in this file (the first argument is a reminder only)
    CH.append(dict(num=num, slug=slug, title=title, part=part, lo=lo, hi=hi, promise=promise,
                   lessons=lessons, patterns=patterns, connections=connections))


# ------------------------------------------------------------------ PART I
ch(1, 'introduction', 'Introduction to Arrays', 1, 'Beginner', '1300',
   'Contiguous memory, an index, and a loop: learn to read, scan and rewrite arrays correctly before learning to do it fast.',
   ['What an array is: contiguous memory, $\\bigO{1}$ indexing, cache lines; static arrays, \\texttt{std::vector} and amortised growth',
    'Indices and ranges: 0- vs 1-based, half-open $[l, r)$, off-by-one errors and loop invariants',
    'The cost model: from the constraints to the allowed complexity; overflow and the range of values',
    'Scanning: one pass with running state, two passes, and when a second array is needed; subarrays, subsequences and subsets and how many of each there are',
    'In place versus extra memory: read/write pointers, reversal, swapping, and encoding two values in one cell'],
   [('Single-pass aggregates', [('Third Maximum Number', 'LC', '414'), ('Advantage', 'CF', '1760C')]),
    ('Read/write pointers: compacting in place', [('Move Zeroes', 'LC', '283'), ('Remove Duplicates from Sorted Array II', 'LC', '80')]),
    ('Reversal and rotation', [('Rotate Array', 'LC', '189'), ('Check if Array Is Sorted and Rotated', 'LC', '1752')]),
    ('Simulation and index arithmetic', [('Plus One', 'LC', '66'), ('Build Array from Permutation', 'LC', '1920')]),
    ('Frequency arrays over small values', [('How Many Numbers Are Smaller Than the Current Number', 'LC', '1365'), ('Unique Number of Occurrences', 'LC', '1207')])])

ch(2, 'prefix_sums', 'Prefix Sums \\& Difference Arrays', 1, '900', '1700',
   'Precompute once, answer every range sum in $\\bigO{1}$; record range updates at their two ends and recover the array with one pass.',
   ['Prefix sums: definition, the off-by-one-free convention $P[0]=0$, range sums as $P[r]-P[l]$',
    'Difference arrays: the inverse of prefix sums; range add in $\\bigO{1}$, rebuild in $\\bigO{n}$',
    'Two dimensions: inclusion--exclusion for rectangle sums and rectangle updates',
    'Prefix sums with a hash map: counting subarrays by their sum, remainder or balance',
    'Prefix and suffix of other operations: XOR, product, minimum, maximum (leaders), and when an operation has no inverse',
    'Split points: every way to cut an array into two or three parts, from prefix and suffix sums'],
   [('Range sum queries', [('Static Range Sum Queries', 'CSES', '1646'), ('Find Pivot Index', 'LC', '724')]),
    ('Prefix sums with a hash map', [('Subarray Sum Equals K', 'LC', '560'), ('Subarray Sums Divisible by K', 'LC', '974'), ('Good Subarrays', 'CF', '1398C')]),
    ('Difference arrays', [('Corporate Flight Bookings', 'LC', '1109'), ('Greg and Array', 'CF', '295A')]),
    ('Two-dimensional prefix sums', [('Range Sum Query 2D - Immutable', 'LC', '304'), ('Maximum Side Length of a Square with Sum Less than or Equal to Threshold', 'LC', '1292')]),
    ('Prefix and suffix of other operations', [('Product of Array Except Self', 'LC', '238'), ('XOR Queries of a Subarray', 'LC', '1310')]),
    ('Split points: cutting the array into parts', [('Number of Ways to Split Array', 'LC', '2270'), ('Three Parts of the Array', 'CF', '1006C')])])

ch(3, 'hashing', 'Hashing \\& Counting', 1, '900', '1700',
   'Trade memory for time: a hash map remembers what you have seen, so the second loop of a pair search disappears.',
   ['Hash sets and hash maps: expected $\\bigO{1}$, collisions, \\texttt{unordered\\_map} versus \\texttt{map} versus a plain array',
    'Anti-hash tests and custom hashes on Codeforces',
    'Complement lookups: seen-so-far maps for pairs, and why the order of insert and query matters',
    'Grouping by a key: canonical forms, signatures and transformed values ($a_i - i$, $a_i \\bmod k$)',
    'Counting without a map: Boyer--Moore voting and other $\\bigO{1}$-memory counters'],
   [('Complement lookups', [('Two Sum', 'LC', '1'), ('Max Number of K-Sum Pairs', 'LC', '1679')]),
    ('Grouping by a transformed key', [('Same Differences', 'CF', '1520D'), ('Equal Row and Column Pairs', 'LC', '2352')]),
    ('Distinct values and nearby duplicates', [('Distinct Numbers', 'CSES', '1621'), ('Contains Duplicate II', 'LC', '219')]),
    ('Majority voting', [('Majority Element', 'LC', '169'), ('Majority Element II', 'LC', '229')]),
    ('Set membership and remainders', [('Longest Consecutive Sequence', 'LC', '128'), ('Make Sum Divisible by P', 'LC', '1590')])])

ch(4, 'two_pointers', 'Two Pointers', 1, '1000', '1700',
   'Two indices that only move forward turn $\\bigO{n^2}$ pair searches into one linear pass --- if you can prove a pointer never needs to come back.',
   ['The two-pointer invariant: what the region between, left of and right of the pointers means',
    'Opposite ends on a sorted array: why discarding one end is safe',
    'Fixing one element and running two pointers on the rest: 3-sum, $k$-sum and their complexity',
    'Merging two sorted arrays: from the front, from the back, in place (the gap method); union, intersection and difference of sorted arrays',
    'Same-direction pointers for counting pairs: monotone predicates and the "for each right, how many left" view',
    'Runs: the group pointer that jumps over a block of equal (or consecutive) elements'],
   [('Opposite ends on a sorted array', [('Two Sum II - Input Array Is Sorted', 'LC', '167'), ('Container With Most Water', 'LC', '11')]),
    ('Fix one, two pointers on the rest', [('3Sum', 'LC', '15'), ('3Sum Closest', 'LC', '16')]),
    ('Merging and set operations on sorted arrays', [('Merge Sorted Array', 'LC', '88'), ('Squares of a Sorted Array', 'LC', '977')]),
    ('Greedy pairing', [('Boats to Save People', 'LC', '881'), ('Number of Pairs', 'CF', '1538C')]),
    ('Counting pairs with two pointers', [('Pair of Topics', 'CF', '1324D'), ('Count the Number of Fair Pairs', 'LC', '2563')]),
    ('Runs of equal or consecutive elements', [('Summary Ranges', 'LC', '228'), ('Alternating Subsequence', 'CF', '1343C')])])

ch(5, 'sliding_window', 'Sliding Window', 1, '1100', '1800',
   'A window that grows on the right and shrinks on the left: longest, shortest and counting questions about subarrays in one pass.',
   ['Fixed-size windows: add the entering element, remove the leaving one',
    'Variable windows: the grow--shrink loop and the invariant "the window is valid"',
    'Longest valid window versus shortest valid window: where the answer is updated',
    'Counting windows: "at most $k$" minus "at most $k-1$", and counting all subarrays ending at $r$',
    'When the window breaks: negative numbers, non-monotone predicates, and the fallback to prefix sums (window extrema are the stack-queue-heap book)'],
   [('Fixed-size windows', [('Maximum Average Subarray I', 'LC', '643'), ('Maximum Points You Can Obtain from Cards', 'LC', '1423')]),
    ('Longest valid window', [('Max Consecutive Ones III', 'LC', '1004'), ('Books', 'CF', '279B')]),
    ('Shortest valid window', [('Minimum Size Subarray Sum', 'LC', '209'), ('Minimum Operations to Reduce X to Zero', 'LC', '1658')]),
    ('Counting windows', [('Subarray Product Less Than K', 'LC', '713'), ('Subarrays with K Different Integers', 'LC', '992')]),
    ('Windows with a frequency table', [('Maximum Sum of Distinct Subarrays With Length K', 'LC', '2461'), ('Count Subarrays Where Max Element Appears at Least K Times', 'LC', '2962')])])

ch(6, 'sorting', 'Sorting as a Tool', 1, '1000', '1800',
   'Sorting costs $\\bigO{n \\log n}$ once and makes neighbours meaningful: sort first, then scan, pair, group or rank.',
   ['\\texttt{std::sort}, stability and \\texttt{stable\\_sort}; what sorting buys: neighbours, monotonicity, binary search',
    'Comparators: strict weak ordering, sorting by several keys, sorting indices instead of values',
    'Non-comparison sorts: counting sort, bucket sort, radix sort and their limits (merge sort and quickselect are the recursion book)',
    'Ranks and coordinate compression in one line',
    'Sorting to minimise a cost: the median minimises absolute deviations, exchange arguments for orders'],
   [('Sort, then scan neighbours', [('Minimum Absolute Difference', 'LC', '1200'), ('Honest Coach', 'CF', '1360B')]),
    ('Custom comparators', [('Largest Number', 'LC', '179'), ('Queue Reconstruction by Height', 'LC', '406')]),
    ('Counting and bucket sort', [('Relative Sort Array', 'LC', '1122'), ('Maximum Gap', 'LC', '164')]),
    ('Ranks and sorting by derived keys', [('Rank Transform of an Array', 'LC', '1331'), ('Sort Array by Increasing Frequency', 'LC', '1636')]),
    ('The median and exchange arguments', [('Stick Lengths', 'CSES', '1074'), ('Minimum Cost to Make Array Equal', 'LC', '2448')])])

ch(7, 'binary_search', 'Binary Search on Arrays', 1, '1000', '1600',
   'Sorted data answers ``where is $x$?\'\' in $\\bigO{\\log n}$: bounds, counting, nearest values and prefix sums --- the basic toolkit every array chapter uses.',
   ['The monotone predicate and the loop invariant: ``everything left of $lo$ is false, everything from $hi$ is true\'\'',
    '\\texttt{lower\\_bound} and \\texttt{upper\\_bound}: counting elements in a value range',
    'Sort, then search: answering many queries against a sorted copy',
    'Binary search on prefix sums of non-negative arrays and on prefix maxima',
    'A first search on the answer over a small range (the full technique is the binary search book)'],
   [('Counting with bounds', [('Maximum Count of Positive Integer and Negative Integer', 'LC', '2529'), ('Find Target Indices After Sorting Array', 'LC', '2089')]),
    ('Nearest values in sorted arrays', [('Find K Closest Elements', 'LC', '658'), ('Maximum Distance Between a Pair of Values', 'LC', '1855')]),
    ('Binary search on prefix sums', [('Eating Queries', 'CF', '1676E'), ('Scuza', 'CF', '1742E')]),
    ('A first search on the answer', [('Special Array With X Elements Greater Than or Equal X', 'LC', '1608'), ('Sum of Mutated Array Closest to Target', 'LC', '1300')])],
   r'Rotated arrays, peaks, binary search on the answer, ternary, real-valued, interactive and offline binary search are the \textit{Binary Search} book.')

# ------------------------------------------------------------------ PART II
ch(9, 'intervals', 'Intervals \\& Sweep Line', 2, '1200', '2000',
   'Sort the endpoints and walk left to right: merging, covering, scheduling and counting overlaps all become one sweep.',
   ['Representing intervals; open, closed and half-open ends',
    'Merging and inserting intervals after sorting by start',
    'Events: turning intervals into $+1$/$-1$ points and sweeping',
    'Greedy interval scheduling and covering: sort by end, sort by start',
    'Intervals on a circle and intervals with weights (weighted scheduling is the DP book)'],
   [('Merging intervals', [('Merge Intervals', 'LC', '56'), ('Insert Interval', 'LC', '57')]),
    ('Event sweep and overlap counts', [('Restaurant Customers', 'CSES', '1619'), ('Car Pooling', 'LC', '1094')]),
    ('Scheduling by end time', [('Movie Festival', 'CSES', '1629'), ('Movie Festival II', 'CSES', '1632')]),
    ('Covering with intervals', [('Minimum Number of Arrows to Burst Balloons', 'LC', '452'), ('Remove Covered Intervals', 'LC', '1288')]),
    ('Intersections and gaps', [('Interval List Intersections', 'LC', '986')])])

ch(10, 'partitioning', 'Partitioning \\& In-Place Rearrangement', 2, '1100', '1900',
   'Rearrange without a second array: partition schemes, the Dutch national flag, next permutation and interleavings.',
   ['Partition invariants: Lomuto and Hoare, three-way partition',
    'Stable versus unstable rearrangement, and what extra memory buys',
    'Rearranging by sign, parity or index; interleaving two halves',
    'Next and previous permutation: the algorithm and its proof (ranking and the $k$-th permutation are the recursion book)',
    'Wiggle and zig-zag orders'],
   [('Three-way partition', [('Sort Colors', 'LC', '75'), ('Partition Array According to Given Pivot', 'LC', '2161')]),
    ('Rearrange by a property', [('Rearrange Array Elements by Sign', 'LC', '2149'), ('Sort Array By Parity II', 'LC', '922')]),
    ('Next permutation', [('Next Permutation', 'LC', '31'), ('Next Greater Element III', 'LC', '556')]),
    ('Wiggle orders', [('Wiggle Sort II', 'LC', '324')]),
    ('Rotations and block swaps', [])])

ch(11, 'index_tricks', 'Index Tricks: Cyclic Sort, Marking \\& Permutation Cycles', 2, '1200', '2000',
   'When the values are $1..n$, a value is also an index: place every value home, mark by sign, and walk permutation cycles.',
   ['Values in $[1, n]$: the array as a function $i \\mapsto a_i$',
    'Cyclic sort: put every value at its home index in $\\bigO{n}$ swaps',
    'Marking in place: negation, adding $n$, and restoring the array',
    'Permutation cycles: decomposition, minimum swaps to sort, applying a permutation in place',
    'Missing and duplicate numbers: sums, XOR and index tricks compared'],
   [('Cyclic sort', [('First Missing Positive', 'LC', '41'), ('Find All Numbers Disappeared in an Array', 'LC', '448')]),
    ('Marking in place', [('Find All Duplicates in an Array', 'LC', '442'), ('Set Mismatch', 'LC', '645')]),
    ('Permutation cycles', []),
    ('Missing numbers by sums and XOR', [('Missing Number', 'LC', '268')]),
    ('Permutations as functions', [])])

ch(12, 'matrices', 'Matrices \\& 2D Arrays', 2, '1100', '2000',
   'Two indices instead of one: traversal orders, rotation, in-place marking and reducing a 2D question to many 1D ones.',
   ['Row-major storage, flattening $(i, j) \\mapsto i\\cdot m + j$, and cache-friendly loops',
    'Traversal orders: spiral, diagonal, anti-diagonal, boundary layers',
    'Transpose, rotate and reflect in place',
    'Using the first row and column as markers',
    'Fix two rows, solve 1D on the columns between them: submatrix sums and counts'],
   [('Traversal orders', [('Spiral Matrix', 'LC', '54'), ('Diagonal Traverse', 'LC', '498')]),
    ('Rotate and transpose in place', [('Rotate Image', 'LC', '48'), ('Determine Whether Matrix Can Be Obtained By Rotation', 'LC', '1886')]),
    ('In-place marking', [('Set Matrix Zeroes', 'LC', '73'), ('Game of Life', 'LC', '289')]),
    ('Fix two rows, solve in 1D', [('Max Sum of Rectangle No Larger Than K', 'LC', '363')]),
    ('Matrix invariants and parity', [])])

ch(13, 'design', 'Designing Array-Based Data Structures', 2, '1200', '2000',
   'Arrays are the engine inside most data structures: dynamic arrays, $\\bigO{1}$ deletion by swapping, lazy resets with timestamps, versioned arrays and array-backed hash sets.',
   ['The dynamic array from scratch: growth, shrinking and the amortised bound',
    'Delete in $\\bigO{1}$ by swapping with the last element; index maps for random access',
    'Lazy initialisation and resets in $\\bigO{1}$: timestamps and the sparse-set trick',
    'Versioned arrays: snapshots by per-cell history and binary search (full persistence is the tree book)',
    'Open addressing: a hash set in a plain array; bit arrays and \\texttt{vector<bool>}'],
   [('Swap-with-last deletion', [('Insert Delete GetRandom O(1)', 'LC', '380'), ('Insert Delete GetRandom O(1) - Duplicates allowed', 'LC', '381')]),
    ('Snapshots and versions', [('Snapshot Array', 'LC', '1146')]),
    ('Counters of counters', [('Frequency Tracker', 'LC', '2671')]),
    ('Array-backed hashing', [('Design HashSet', 'LC', '705')]),
    ('Lazy resets with timestamps', [])],
   r'Circular buffers and queues are the stack-queue-heap book; persistent arrays via segment trees are the tree book.')

# ------------------------------------------------------------------ PART III
ch(13, 'contribution', 'Counting Subarrays, Pairs \\& Triples: the Contribution Technique', 3, '1500', '2400',
   'There are $\\bigO{n^2}$ subarrays and $\\bigO{n^2}$ pairs, but the answer only needs how much each element contributes.',
   ['Counting subarrays: fix the right end, fix the left end, complementary counting',
    'Contribution of an element: in how many subarrays (or pairs) does it count, and with what weight',
    'Sum over all pairs: sorting and prefix sums for $\\sum |a_i - a_j|$ and similar',
    'Fix the middle element: left counts times right counts; triples and the cost of each extra dimension',
    'Pairs with an index condition and a value condition; small values: frequency arrays and convolutions',
    'Contribution with nearest smaller/greater indices (computing them is the stack-queue-heap book; per-bit sums are Chapter~\\ref{ch:bitwise})'],
   [('Sum over all subarrays', []),
    ('Sum over all pairs', []),
    ('Counting subarrays with a property', []),
    ('Complementary counting', []),
    ('Fix the middle: triples', []),
    ('Index and value conditions', []),
    ('Small-value counting', [])])

ch(14, 'greedy', 'Greedy Algorithms on Arrays', 3, '1300', '2300',
   'Make the locally best choice --- and prove it with an exchange argument, a potential, or a stays-ahead invariant.',
   ['What makes a greedy correct: exchange arguments and stays-ahead proofs',
    'Counter-examples: how to break a wrong greedy quickly (stress testing)',
    'Reachability greedies: jump games and gas stations',
    'Greedy from both sides: two passes',
    'Greedy with sorting and with prefix minima / maxima'],
   [('Reach as far as possible', [('Gas Station', 'LC', '134'), ('Partition Labels', 'LC', '763')]),
    ('Two-pass greedy', [('Candy', 'LC', '135')]),
    ('Exchange arguments', []),
    ('Greedy with prefix extrema', []),
    ('Greedy on operations', [])])

ch(15, 'bitwise', 'Bitwise Operations on Arrays', 3, '1400', '2300',
   'Split every number into bits and solve 30 easy problems instead of one hard one; and use the fact that OR and AND change only $\\bigO{\\log V}$ times.',
   ['XOR identities: cancelling pairs, prefix XOR, XOR of a range of integers',
    'Per-bit decomposition: counting, sums and choices bit by bit from the top',
    'OR and AND over subarrays: at most $\\log V$ distinct values ending at each index',
    'Maximum XOR by greedy over bits (binary tries are the string and tree books)',
    'Bit tricks on arrays: masks of present values, subsets of small arrays'],
   [('XOR cancellation', [('Single Number', 'LC', '136'), ('Single Number III', 'LC', '260')]),
    ('Per-bit decomposition', []),
    ('Distinct OR and AND values of subarrays', [('Bitwise ORs of Subarrays', 'LC', '898')]),
    ('Greedy over bits', []),
    ('Prefix XOR', [])])

ch(16, 'number_arrays', 'GCD, Divisibility \\& Number-Theoretic Arrays', 3, '1500', '2300',
   'Arrays of integers carry arithmetic structure: prefix GCDs, remainders, divisor counts and harmonic-sum loops over values.',
   ['Prefix and suffix GCD; the GCD of a subarray changes $\\bigO{\\log V}$ times',
    'Remainders: counting pairs and subarrays modulo $k$',
    'Loops over multiples: $\\sum V/d = \\bigO{V \\log V}$ and counting by divisors',
    'Sieve-based preprocessing of the value domain',
    'Inclusion--exclusion and M\\"obius over the values of an array',
    'The pigeonhole principle on prefix sums and remainders: when a large $n$ forces the answer'],
   [('Prefix and suffix GCD', []),
    ('Distinct GCDs of subarrays', []),
    ('Remainder counting', []),
    ('Harmonic loops over values', []),
    ('Counting coprime pairs', []),
    ('Pigeonhole: large inputs force the answer', [])])

ch(17, 'value_domain', 'MEX, Distinctness \\& Value-Domain Problems', 3, '1500', '2400',
   'Questions about which values appear rather than where: MEX, the number of distinct values, last occurrences and frequency of frequencies.',
   ['The MEX: computing it, maintaining it, and how it changes when elements are added or removed',
    'Distinct values in subarrays: last occurrence arrays',
    'Frequency of frequencies and the "at most $\\sqrt{2n}$ distinct frequencies" fact',
    'Permutation checks: a subarray is a permutation of $1..k$',
    'Arrays over small value domains: counting arrays per value'],
   [('MEX', []), ('Distinct values in subarrays', []), ('Frequency of frequencies', []), ('Permutation subarrays', []), ('Small value domains', [])])

ch(18, 'offline', 'Coordinate Compression \\& Offline Processing', 3, '1600', '2300',
   'Read every query first, sort queries and elements together, and the problem often reduces to a simple sweep.',
   ['Coordinate compression with sorting and \\texttt{unique}; compressing intervals (keep $l$ and $r+1$); online compression',
    'Offline versus online; sorting queries by a threshold',
    'Processing queries in reverse: deletions become insertions',
    'Answering queries at the moment the sweep reaches them',
    'Combining a sweep with a Fenwick tree (the tree itself is the tree book)'],
   [('Coordinate compression', []), ('Sorting queries by threshold', []), ('Reverse processing', []), ('Sweep and answer', []), ('Compression in two dimensions', [])])

# ------------------------------------------------------------------ PART IV
ch(19, 'sqrt', 'Square Root Decomposition', 4, '1700', '2400',
   'Cut the array into $\\sqrt{n}$ blocks: whole blocks are answered from summaries, partial blocks element by element.',
   ['Blocks and block summaries; choosing the block size',
    'Point update and range query; range update with lazy block tags',
    'Rebuilding: sqrt decomposition over the sequence of queries',
    'Blocks over the value domain (heavy/light splits by frequency are Chapter~\\ref{ch:heavy_light_values})',
    'Balancing update and query cost: $\\bigO{1}$ update with $\\bigO{\\sqrt n}$ query and vice versa'],
   [('Block summaries', []), ('Lazy block tags', []), ('Periodic rebuilding', []), ('Blocks over values', []), ('Unbalanced costs', [])])

ch(20, 'mo', "Mo's Algorithm", 4, '1900', '2600',
   'Reorder offline range queries so that moving from one window to the next is cheap: $\\bigO{(n+q)\\sqrt{n}}$ pointer moves.',
   ["The order of Mo's algorithm and its $\\bigO{(n+q)\\sqrt n}$ analysis",
    'Add and remove operations; queries whose answer has no remove (rollback Mo)',
    "Mo's algorithm with updates (time as a third dimension)",
    'Hilbert order and constant-factor tricks',
    'When not to use Mo: offline sweeps that are simpler (Mo on trees is the tree book)'],
   [("Classic Mo's", [('Powerful array', 'CF', '86D')]), ('Rollback Mo', []), ('Mo with updates', []), ('Hilbert order', []), ('Mo plus a value structure', [])])

ch(21, 'multidim_prefix', 'Multidimensional \\& Higher-Order Prefix Sums', 4, '1700', '2400',
   'Prefix sums in more dimensions and of higher order: 2D and 3D difference arrays, prefix sums of prefix sums, polynomial range updates.',
   ['2D difference arrays: rectangle updates in $\\bigO{1}$',
    'Higher dimensions and the $2^d$ inclusion--exclusion terms',
    'Prefix sums of prefix sums: adding arithmetic progressions to ranges',
    'Diagonal and triangular prefix sums',
    'Sum over subsets is the DP book; prefix sums over masks are only referenced'],
   [('2D difference arrays', []), ('Higher dimensions', []), ('Arithmetic progressions on ranges', []), ('Diagonal prefix sums', []), ('Weighted prefix sums', [])])

ch(22, 'array_hashing', 'Hashing Arrays \\& Multisets', 4, '1800', '2500',
   'Give every value a random number: sums and XORs of random values compare multisets, permutations and parities in $\\bigO{1}$.',
   ['Random values per element: sum hashing and XOR hashing',
    'Is this subarray a permutation? Is this multiset equal to that one?',
    'Parity of occurrences with XOR hashing',
    'Polynomial hashing of integer sequences (string hashing is the string book)',
    'Collision probability and anti-hash defences'],
   [('Sum hashing of multisets', []), ('XOR hashing of parities', []), ('Equal multisets of subarrays', []), ('Sequence hashing', []), ('Hashing with prefix sums', [])])

ch(24, 'selection', 'Selection, Streaming \\& Order Statistics', 4, '1600', '2400',
   'The $k$-th element without sorting everything, majorities in one pass, and samples from streams of unknown length.',
   ['Order statistics in sorted structures and \\texttt{nth\\_element}',
    'Heavy hitters: Misra--Gries and the $n/k$ majority',
    'Reservoir sampling and random shuffles (Fisher--Yates)',
    'Medians and order statistics by binary search on the value',
    'Ordered sets with order statistics (\\texttt{pb\\_ds}) as a tool'],
   [('Partial sorting', []), ('Heavy hitters', []), ('Random sampling', []), ('Order statistics by value search', []), ('Ordered sets', [])])

# ------------------------------------------------------------------ PART V
ch(25, 'constructive', 'Constructive Array Problems', 5, '1500', '2400',
   'Build an array with the required sums, orders or differences --- and prove that your construction always works.',
   ['Reading a constructive problem: small cases first, then a pattern',
    'Constructions by parity, by blocks and by alternating patterns',
    'Permutations with given properties',
    'Impossibility proofs: invariants and counting arguments',
    'Checking a construction with a brute-force validator'],
   [('Small cases and patterns', []), ('Parity constructions', []), ('Permutation constructions', []), ('Impossibility', []), ('Constructions with sums', [])])

ch(26, 'operations', 'Operations, Invariants \\& Reachability', 5, '1600', '2500',
   'An array and an operation: which arrays are reachable, and in how few steps? Find what the operation never changes.',
   ['Invariants: sums, parities, sortedness, multisets',
    'Monovariants and termination',
    'Adjacent swaps, reversals and prefix operations',
    'Minimum number of operations: lower bound plus matching construction',
    'Operations on differences: thinking in the difference array'],
   [('Sum and parity invariants', []), ('Swaps and reversals', []), ('Operations on the difference array', []), ('Lower bound plus construction', []), ('Monovariants', [])])

ch(27, 'interactive', 'Interactive Problems on Arrays', 5, '1500', '2500',
   'The array is hidden behind a judge: binary search, tournaments and adaptive strategies within a query budget.',
   ['The protocol: flushing, reading verdicts, local interactors',
    'Binary search over a hidden array',
    'Finding the maximum, the minimum or both with few comparisons',
    'Sum and XOR queries: recovering an array',
    'Adaptive adversaries and lower bounds on the number of queries'],
   [('Binary search with queries', []), ('Tournaments', []), ('Recovering an array', []), ('Comparisons budget', []), ('Adaptive judges', [])])

ch(28, 'games', 'Games on Arrays', 5, '1100', '2300',
   'Two players, one array: find the winning invariant --- parity, symmetry, the first decisive element, or a Nim sum --- instead of searching the game tree.',
   ['Winning and losing positions; why the game tree is too large to search',
    'Parity and counting arguments',
    'Symmetry and mirror strategies',
    'Games decided by the first element that gives a real choice',
    'Nim and the XOR rule as a tool (Sprague--Grundy theory is the math notes; interval games are the DP book)'],
   [('Parity arguments', []),
    ('Mirror strategies', []),
    ('Decided by the first real choice', [('Sequential Nim', 'CF', '1382B')]),
    ('Nim sums on arrays', []),
    ('Games on operations', [])])

ch(28, 'circular', 'Circular Arrays \\& Rotations', 5, '1400', '2300',
   'An array whose end touches its start: doubling, the complement window, and rotations as shifts of the start index.',
   ['Indexing modulo $n$ and the doubling trick',
    'Circular subarrays: total minus the complementary window',
    'Rotations: the best rotation and incremental rotation costs',
    'Circular two pointers and windows',
    'Cyclic shifts of permutations'],
   [('Doubling', []), ('Complement window', []), ('Best rotation', []), ('Circular two pointers', []), ('Cyclic shifts', [])])

ch(29, 'linked_arrays', 'Deletions, Next Pointers \\& Simulation', 5, '1500', '2400',
   'Arrays that shrink: linked lists inside arrays, skip pointers with path compression, and simulations that would be too slow done naively.',
   ['Doubly linked lists stored in arrays: $\\bigO{1}$ deletion with \\texttt{prev}/\\texttt{next}',
    'Next-free-index with path compression',
    'Simulation with events: only the neighbours of a change can change',
    'Josephus-type eliminations',
    'Amortised arguments: every element is deleted once'],
   [('Linked lists in arrays', []), ('Next free index', []), ('Event-driven simulation', []), ('Eliminations', []), ('Amortised deletions', [])])

# ------------------------------------------------------------------ PART VI
ch(30, 'scanline', 'Scanline over the Right Endpoint', 6, '1900', '2600',
   'For every right endpoint, keep the answer for all left endpoints at once --- the standard way to count or sum over all subarrays with a property.',
   ['The scanline invariant: an array indexed by the left endpoint',
    'Range updates when the right endpoint moves (data structures are the tree book)',
    'Last occurrences and distinct-value scanlines',
    'Offline queries $[l, r]$ answered when the sweep reaches $r$',
    'Summing answers over all subarrays with historic sums'],
   [('Distinct-value scanline', []), ('Range updates per step', []), ('Queries at their right end', []), ('Historic sums', []), ('Two-dimensional scanlines', [])])

ch(31, 'permutations', 'Permutations: Cycles, Inverses \\& Powers', 6, '1800', '2800',
   'A permutation is a function: compose it, invert it, raise it to huge powers, and count arrangements by cycle type.',
   ['Composition and inverses; the permutation group in practice',
    'Cycle decomposition and the order of a permutation',
    'The $k$-th power of a permutation in $\\bigO{n}$',
    'Square roots and other equations in permutations',
    'Counting permutations by cycle structure'],
   [('Composition and inverse', []), ('Powers of a permutation', []), ('Roots of a permutation', []), ('Cycle-structure counting', []), ('Permutation parity', [])])

ch(32, 'bitset', 'Bitsets \\& Word-Level Parallelism', 6, '2000', '2900',
   'Sixty-four booleans per machine word: subset sums, reachability over values and pair counting divided by 64.',
   ['\\texttt{std::bitset} and dynamic bitsets; the $n^2/64$ cost model',
    'Subset sums and reachable values with shifts',
    'Counting pairs and triples with bitset intersections',
    'Bitsets over positions and over values',
    'When $n^2/64$ fits and when it does not'],
   [('Subset sums with shifts', []), ('Bitset intersections', []), ('Bitsets over values', []), ('Bitset over positions', []), ('Mixed bitset tricks', [])])

ch(33, 'heavy_light_values', 'Heavy/Light Splits \\& Root Thresholds', 6, '2100', '2900',
   'Treat frequent values and rare values differently: the threshold $\\sqrt{n}$ balances two algorithms that each fail alone.',
   ['Frequent versus rare values: two algorithms, one threshold',
    'Small versus large step sizes (queries with a stride)',
    'Pairs of values: heavy--heavy, heavy--light and light--light',
    'Choosing the threshold from the cost formula',
    'Sqrt tricks combined with prefix sums'],
   [('Frequent and rare values', []), ('Small and large strides', []), ('Pairs of values', []), ('Threshold tuning', []), ('Sqrt plus prefix sums', [])])

ch(34, 'expert_greedy', 'Expert Greedy \\& Exchange Arguments', 6, '2100', '3000',
   'Hard greedy problems on arrays: regret, potential functions, ordering by ratios and greedy with data structures.',
   ['Ordering by a ratio or a pairwise comparison (exchange of neighbours)',
    'Greedy with regret (undoing a previous choice)',
    'Potential functions and amortised greedy proofs',
    'Greedy on prefix constraints and Hall-type conditions',
    'Proving optimality by a matching lower bound'],
   [('Ordering by exchange', []), ('Greedy with regret', []), ('Prefix constraints', []), ('Hall-type conditions', []), ('Lower bounds', [])])

ch(35, 'expert_mixed', 'Expert Mixed Array Problems', 6, '2400', '3300',
   'Problems that combine three or four ideas from this book; no chapter title tells you which.',
   ['A recognition guide: from the statement and the limits to the technique',
    'Combining scanlines, hashing and sqrt decomposition',
    'Combining binary search on the answer with sweeps and greedy checks',
    'Counting arguments combined with data structures',
    'Writing and testing long array solutions'],
   [('Mixed set 2400', []), ('Mixed set 2600', []), ('Mixed set 2800', []), ('Mixed set 3000', []), ('Mixed set 3200+', [])])

ch(36, 'mastery', 'Contest Mastery', 6, '2600', '3500',
   'No hints about the technique: recognise the model, pick the tool, and combine chapters under contest pressure.',
   ['Reading limits like a contestant: the constraint table revisited',
    'Proof habits: invariants, exchange arguments and counting',
    'Implementation templates and pitfalls (multi-tests, overflow, fast I/O)',
    'Stress testing array solutions',
    'Mixed contest sets by rating'],
   [('Mixed set 2600', []), ('Mixed set 2800', []), ('Mixed set 3000', []), ('Mixed set 3200', []), ('Mixed set 3400+', [])])


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
    L.append('\\chapterfacts{%s}{%s}{%d}{%d}{30}' % (c['lo'], c['hi'], len(c['patterns']), n_solved(c)))
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


def n_solved(c):
    """Planned solved problems: the listed ones, at least two per pattern."""
    return sum(max(2, len(probs)) for _, probs in c['patterns'])


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
             '\\dppartstats{%d}{%d}{%d}{%d}' % (len(cs), n_pat, sum(n_solved(c) for c in cs), 30 * len(cs)), '',
             '\\begin{dpjourney}']
        for c in cs:
            o.append('\\dpstep{%d}{%s}{%s}{%s}{%s}' % (
                c['num'], c['title'], ' $\\cdot$ '.join(name for name, _ in c['patterns']),
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
    short = {1: 'Foundations', 2: 'Core', 3: 'Counting', 4: 'Queries', 5: 'Ad-hoc', 6: 'Advanced'}
    for P in PARTS:
        cs = [c['num'] for c in CH if c['part'] == P['n']]
        lv = P['level'].replace('$\\to$', '--').replace('CF ', '')
        R.append('    \\node[stair=%s] at (%d,%d) {\\textbf{Part %s}\\\\%s\\\\Ch %d--%d\\\\{\\tiny %s}};' % (
            P['color'], P['n'] - 1, P['n'] - 1, ['I', 'II', 'III', 'IV', 'V', 'VI'][P['n'] - 1], short[P['n']],
            cs[0], cs[-1], lv))
    R += ['    \\draw[flow, line width=1.5pt] (-0.45,-1.1) -- (5.45,4.1);', '\\end{tikzpicture}',
          '\\figcap{Each part assumes fluency with everything below it. Do not skip steps.}', '\\end{diagrambox}', '',
          '\\section*{Chapter List}', '',
          'Status: \\yes~written \\quad $\\circ$~planned. For Part~I the \\emph{solved} problems of every pattern are '
          'listed; every problem appears in exactly \\textbf{one} place in the book, and no problem used here is used in '
          'any of the six companion books (DP, graph, tree, string, recursion, stack--queue--heap).', '',
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
