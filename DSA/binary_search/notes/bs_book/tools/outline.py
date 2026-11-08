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
APIs and against the eight companion books (tools/verify_outline.py); later parts are candidates and are
re-checked when the chapter is written.

This book owns binary search on arrays, on functions and on the answer, ternary search, parametric search,
interactive binary search and offline (batched) binary search. Chapters 2--3 and 6--8 contain the 22 problems that
were planned in the array book's former Chapters 7--8 (moved here on 2026-10-26).

Topics owned by a companion book are NOT taught here (the text points to them instead):
    one-call recursion view of binary search, merge sort, quickselect ... recursion book (Ch 13)
    parallel binary search written as a recursion on time ............... recursion book (Ch 29)
    LIS in O(n log n) and other DP ....................................... DP book (Ch 4)
    the Aliens trick (WQS binary search / Lagrangian relaxation) ......... DP book (Ch 24)
    binary search + 2-SAT / flow / minimum ratio cycles .................. graph book
    binary lifting search, Fenwick / segment-tree descent, BSTs .......... tree book
    greedy over bits (deciding the answer bit by bit) .................... bit book (Ch 7)
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

PARTS = [
    dict(n=1, key='foundations', title='Foundations', level=r'Beginner $\to$ CF 1900', color='successgreen',
         tagline='One comparison halves the search space: monotone predicates, invariants, bounds, rotated arrays, functions and matrices.',
         goals=[r'write a correct binary search every time from a monotone predicate and a loop invariant;',
                r'use lower and upper bounds to search, count and answer floor/ceiling queries;',
                r'search arrays that are sorted only in pieces: rotated, bitonic, with peaks;',
                r'search implicit sequences, functions of the index and sorted matrices.']),
    dict(n=2, key='answer', title='Binary Search on the Answer', level=r'CF 1300 $\to$ CF 2300', color='primaryblue',
         tagline='When checking a candidate answer is easy and monotone, search the answer: minimise the maximum, maximise the minimum, find the $k$-th.',
         goals=[r'prove that a feasibility check is monotone and choose safe bounds;',
                r'solve allocation problems (minimise the maximum) with a greedy check;',
                r'solve placement problems (maximise the minimum) with a greedy check;',
                r'find medians and $k$-th values by counting instead of listing.']),
    dict(n=3, key='advanced', title='Advanced Techniques', level=r'CF 1600 $\to$ CF 2600', color='softpurple',
         tagline='Harder checks, real numbers, unimodal functions, ratios, and checks that are graphs or DPs.',
         goals=[r'design greedy, prefix-sum and difference-array checks;',
                r'binary search on real numbers with a fixed number of iterations;',
                r'find the extremum of a unimodal function with ternary or golden-section search;',
                r'optimise ratios and averages by parametric search.']),
    dict(n=4, key='beyond', title='Searching Beyond Arrays', level=r'CF 1500 $\to$ CF 2800', color='accentorange',
         tagline='Binary search inside sorted containers and timelines, unbounded and interpolation search, geometry and many lists at once.',
         goals=[r'search sorted containers and per-key timelines with \texttt{lower\_bound};',
                r'search unbounded and skewed data with exponential and interpolation search;',
                r'binary search on convex polygons and angles;',
                r'search many sorted lists at once with fractional cascading.']),
    dict(n=5, key='expert', title='Interactive, Offline \\& Expert', level=r'CF 1300 $\to$ CF 3500', color='deepblue',
         tagline='Binary search against a judge, many binary searches at once, nested searches, and mixed expert problems.',
         goals=[r'find hidden values within a query budget;',
                r'answer many binary-search queries together offline;',
                r'nest binary searches and combine them with data structures;',
                r'recognise a hidden binary search under contest pressure.']),
]

# (num, slug, title, part, lo, hi, promise, lessons, patterns, connections)
# patterns: list of (name, [(title, judge, id), ...])
CH = []


def ch(num, slug, title, part, lo, hi, promise, lessons, patterns, connections=''):
    num = len(CH) + 1   # chapters are numbered by their order in this file (the first argument is a reminder only)
    CH.append(dict(num=num, slug=slug, title=title, part=part, lo=lo, hi=hi, promise=promise,
                   lessons=lessons, patterns=patterns, connections=connections))


# ------------------------------------------------------------------ PART I
ch(1, 'introduction', 'Introduction to Binary Search', 1, 'Beginner', '1200',
   'Halve the search space with every comparison --- and prove, with an invariant, that you never lose the answer.',
   ['The idea: a sorted array, a guess in the middle, and $\\lceil \\log_2(n+1) \\rceil$ comparisons',
    'The predicate view: every binary search finds the first index where a monotone predicate becomes true',
    'Loop invariants: closed $[lo, hi]$, half-open $[lo, hi)$ and the ``lo is false, hi is true\'\' form; why loops terminate',
    'Overflow in $mid$, infinite loops, and the off-by-one errors you will make once',
    'Linear search versus binary search; \\texttt{std::lower\\_bound} and \\texttt{std::binary\\_search} as the library form'],
   [('Classic search in a sorted array', [('Search Insert Position', 'LC', '35'), ('Guess Number Higher or Lower', 'LC', '374')]),
    ('First true: the predicate view', [('First Bad Version', 'LC', '278'), ('Find Smallest Letter Greater Than Target', 'LC', '744')]),
    ('Integer roots', [('Sqrt(x)', 'LC', '69'), ('Valid Perfect Square', 'LC', '367')])],
   r'The one-recursive-call view of binary search is the recursion book (Ch 13).')

ch(2, 'bounds', 'Lower Bound, Upper Bound and Counting', 1, '900', '1600',
   'Two functions answer almost every question about a sorted array: where the first $\\ge x$ is, and where the first $> x$ is.',
   ['\\texttt{lower\\_bound} and \\texttt{upper\\_bound}: definitions, return values at the ends, custom comparators',
    'Counting elements equal to $x$, in $[a, b]$, below $x$: differences of bounds',
    'Floor and ceiling of $x$ in a sorted array',
    'Bounds on sorted vectors of pairs and on reversed orders',
    'Counting pairs: sort one side, binary search for each element of the other'],
   [('First and last position', [('Find First and Last Position of Element in Sorted Array', 'LC', '34'), ('Interesting drink', 'CF', '706B')]),
    ('Counting pairs with bounds', [('Successful Pairs of Spells and Potions', 'LC', '2300'), ('Find the Distance Value Between Two Arrays', 'LC', '1385')])])

ch(3, 'sorted_arrays', 'Binary Search on Partly Sorted Arrays', 1, '1100', '1900',
   'The array is not sorted, but each half tells you which way to go: rotated arrays, peaks, bitonic arrays and paired structures.',
   ['Rotated sorted arrays: one half is always sorted; finding the minimum and the rotation point',
    'Duplicates in rotated arrays and the $\\bigO{n}$ worst case',
    'Peaks: why a local maximum always exists and how the slope tells you where',
    'Bitonic (mountain) arrays: find the peak, then search both sides',
    'Structure hidden in indices: pairs, parity of positions and missing values'],
   [('Rotated arrays', [('Find Minimum in Rotated Sorted Array', 'LC', '153'), ('Search in Rotated Sorted Array II', 'LC', '81')]),
    ('Peaks and mountains', [('Find Peak Element', 'LC', '162'), ('Find in Mountain Array', 'LC', '1095')]),
    ('Structure hidden in indices', [('Single Element in a Sorted Array', 'LC', '540'), ('Kth Missing Positive Number', 'LC', '1539')])],
   r'Search in Rotated Sorted Array (LC 33) is solved in the recursion book.')

ch(4, 'predicates', 'Binary Search over Functions and Implicit Sequences', 1, '1100', '1900',
   'The array never exists: binary search over a function of the index, an implicit sequence, or prefix sums.',
   ['Monotone functions of an integer: $f(x) \\ge t$ as the predicate',
    'Implicit sequences: the $k$-th element of a sequence you cannot store',
    'Binary search on prefix sums: which block contains position $x$, weighted random choice',
    'Choosing the search range: proving $lo$ is false and $hi$ is true',
    'When the function is not monotone: what goes wrong and how to detect it'],
   [('Implicit sequences', [('K-th Not Divisible by n', 'CF', '1352C'), ('Maximum Number That Sum of the Prices Is Less Than or Equal to K', 'LC', '3007')]),
    ('Monotone functions of the answer', [('Building an Aquarium', 'CF', '1873E'), ('Maximum Median', 'CF', '1201C')]),
    ('Binary search on prefix sums', [('Worms', 'CF', '474B'), ('Random Pick with Weight', 'LC', '528')])])

ch(5, 'matrices', 'Binary Search on 2D Arrays', 1, '1200', '1900',
   'Sorted rows, sorted columns or both: flatten the matrix, walk a staircase, or binary search row by row.',
   ['A fully sorted matrix as a flattened array: $k \\mapsto (k / m, k \\bmod m)$',
    'Rows and columns sorted: the staircase walk from a corner in $\\bigO{n + m}$',
    'Counting elements $\\le x$ in a sorted matrix (the counting step of $k$-th smallest searches, Part II)',
    'Peaks in 2D: binary search on the column of maxima',
    'Binary search per row versus the staircase: which is faster when'],
   [('Sorted matrices', [('Search a 2D Matrix', 'LC', '74'), ('Search a 2D Matrix II', 'LC', '240')]),
    ('Counting and peaks in matrices', [('Count Negative Numbers in a Sorted Matrix', 'LC', '1351'), ('Find a Peak Element II', 'LC', '1901')])])

# ------------------------------------------------------------------ PART II
ch(6, 'answer_intro', 'Binary Search on the Answer', 2, '1300', '1900',
   'Guess the answer, check it, and halve: the feasibility function, its monotonicity, and the bounds that make it safe.',
   ['Feasibility functions: ``can it be done with value $v$?\'\' and why the answers form a prefix or a suffix',
    'Proving monotonicity before coding',
    'Bounds: the smallest and largest possible answers, and overflow inside the check',
    'First-true versus last-true forms and the matching loops',
    'Complexity: $\\bigO{\\text{check} \\cdot \\log(\\text{range})}$'],
   [('Divisors and rates', [('Find the Smallest Divisor Given a Threshold', 'LC', '1283'), ('Koko Eating Bananas', 'LC', '875')]),
    ('Capacities and trips', [('Capacity To Ship Packages Within D Days', 'LC', '1011'), ('Minimum Time to Complete Trips', 'LC', '2187')]),
    ('Days and speeds', [('Minimum Number of Days to Make m Bouquets', 'LC', '1482'), ('Minimum Speed to Arrive on Time', 'LC', '1870')])])

ch(7, 'minimax', 'Minimise the Maximum: Allocation and Distribution', 2, '1400', '2000',
   'Split work, items or arrays so that the largest part is as small as possible: the greedy check that fills each part to the limit.',
   ['The greedy check: fill a part until the limit, then open a new one',
    'Splitting an array into $k$ contiguous parts',
    'Distributing items among stores, bags or machines',
    'Machines working in parallel: total output by time $t$',
    'Running many devices on shared batteries: the sum-of-minimum check'],
   [('Splitting into parts', [('Array Division', 'CSES', '1085'), ('Minimum Limit of Balls in a Bag', 'LC', '1760')]),
    ('Machines and stores', [('Factory Machines', 'CSES', '1620'), ('Minimized Maximum of Products Distributed to Any Store', 'LC', '2064')]),
    ('Shared resources', [('Maximum Running Time of N Computers', 'LC', '2141')])])

ch(8, 'maximin', 'Maximise the Minimum: Placement Problems', 2, '1400', '2000',
   'Place objects so that the closest pair is as far apart as possible, or share so that the smallest share is as large as possible.',
   ['The greedy placement check: place as early as possible',
    'Why sorting positions first is necessary',
    'Sharing piles among children: count how many shares of size $v$ exist',
    'Maximise the minimum difference in a chosen subset',
    'Mirror symmetry with Chapter~\\ref{ch:minimax}'],
   [('Placement with gaps', [('Magnetic Force Between Two Balls', 'LC', '1552'), ('Maximum Tastiness of Candy Basket', 'LC', '2517')]),
    ('Equal shares', [('Maximum Candies Allocated to K Children', 'LC', '2226')])])

ch(9, 'time_budget', 'Time, Budget and Resource Problems', 2, '1500', '2100',
   'How long until\\ldots, how many can we make with budget\\ldots: answers measured in time or money, checked by simulation.',
   ['Time as the answer: simulate ``by time $t$\'\' in $\\bigO{n}$',
    'Budgets: how many units can be produced with money $B$',
    'Checks with difference arrays: covering a timeline by time $t$',
    'Parity and alternating days in time checks',
    'Large answers: bounds up to $10^{18}$ and \\texttt{\\_\\_int128} in the check'],
   [('Budgets', [('Hamburgers', 'CF', '371C')]),
    ('Timelines with traps', [('A Game with Traps', 'CF', '1260D')]),
    ('Alternating days', [('Water the Trees', 'CF', '1661C')])])

ch(10, 'kth', 'Median and k-th Element Search', 2, '1600', '2300',
   'Find the $k$-th smallest of $n^2$ products, pair distances or table entries without listing them: binary search the value, count below it.',
   ['$k$-th smallest by value search: count how many are $\\le v$',
    'The multiplication table and pair distances: counting in $\\bigO{n}$ or $\\bigO{n\\log n}$',
    'Median of two sorted arrays by partitioning',
    'Products of two sorted arrays with negative numbers',
    'Median via $\\pm 1$ transform: ``median $\\ge v$\'\' as a prefix-sum check'],
   [('k-th by counting', [('Multiplication Table', 'CF', '448D'), ('Find K-th Smallest Pair Distance', 'LC', '719')]),
    ('Two sorted arrays', [('Median of Two Sorted Arrays', 'LC', '4'), ('Kth Smallest Product of Two Sorted Arrays', 'LC', '2040')]),
    ('Medians by a plus-minus transform', [('Max Median', 'CF', '1486D')])])

# ------------------------------------------------------------------ PART III
ch(11, 'checks', 'Designing Checks: Greedy, Prefix Sums and Sweeps', 3, '1600', '2400',
   'The binary search is two lines; the check is the problem. Greedy, prefix-sum, difference-array and sweep checks.',
   ['Greedy checks and their exchange-argument proofs',
    'Prefix-sum checks: ``is there a segment with property $\\ge v$?\'\'',
    'Difference-array checks over many intervals',
    'Answer on the time axis: the first moment something becomes true',
    'Making the check fast enough: $\\log$ factors you can afford'],
   [('Greedy checks', [('Keshi Is Throwing a Party', 'CF', '1610C'), ('Set or Decrease', 'CF', '1622C')]),
    ('Prefix-sum checks over time', [('Tracking Segments', 'CF', '1843E')])])

ch(12, 'real', 'Binary Search on Real Numbers', 3, '1600', '2400',
   'When the answer is a real number: iterate a fixed number of times, compare with care, and print with the right precision.',
   ['A fixed number of iterations (60--100) instead of $hi - lo > \\varepsilon$',
    'Absolute versus relative error, and output precision',
    'Meeting points and time on a line',
    'Roots of monotone functions; bisection versus Newton',
    'Avoiding real numbers: scaling to integers when possible'],
   [('Meeting on a line', [('The Meeting Place Cannot Be Changed', 'CF', '780B')]),
    ('Roots of monotone functions', []),
    ('Scaling to integers', [])])

ch(13, 'ternary', 'Ternary Search and Unimodal Functions', 3, '1700', '2400',
   'A function that decreases and then increases has one minimum: find it by comparing two inner points, or by binary search on the slope.',
   ['Unimodal and convex functions; why ternary search needs strict unimodality',
    'Real ternary search and golden-section search',
    'Integer ternary search and binary search on $f(x+1) - f(x)$',
    'Plateaus and when ternary search fails',
    'Convex cost functions in contest problems (sums of absolute values, maxima of lines)'],
   [('Integer unimodal costs', [('Devu and his Brother', 'CF', '439D'), ('Restorer Distance', 'CF', '1355E')]),
    ('Real-valued ternary search', [('Weakness and Poorness', 'CF', '578C')])])

ch(14, 'parametric', 'Parametric Search and Fractional Programming', 3, '2000', '2600',
   'Maximise a ratio or an average: guess $\\lambda$, subtract it, and ask whether the best sum is non-negative.',
   ['Maximum average subarray of length $\\ge k$ by binary search on the average',
    '0/1 fractional programming: maximise $\\sum a_i / \\sum b_i$',
    "Dinkelbach's iteration as the faster alternative",
    'Medians and averages: the same transformation trick',
    'Minimum ratio cycles are the graph book'],
   [('Maximum average', []), ('Fractional programming', []), ("Dinkelbach's method", [])])

ch(15, 'graph_dp_checks', 'Binary Search with Graph and DP Checks', 3, '1800', '2600',
   'Sometimes the check is a BFS, a DSU or a DP: binary search a threshold, build the graph or table, test it.',
   ['Thresholds on edges: BFS or DSU over the edges that pass',
    'DP checks: ``is value $v$ achievable?\'\' as a boolean DP (the DP itself is the DP book)',
    'Bottleneck answers: binary search versus sorting edges once',
    'Binary search + 2-SAT and binary search + flow are the graph book',
    'Avoiding the $\\log$: when a direct algorithm exists'],
   [('Graph checks', []), ('DP checks', []), ('Bottlenecks', [])])

# ------------------------------------------------------------------ PART IV
ch(16, 'containers', 'Binary Search in Sorted Containers and Timelines', 4, '1500', '2200',
   'Sorted vectors per key, timestamps and positions: \\texttt{lower\\_bound} on a container answers ``what was true at time $t$?\'\'.',
   ['\\texttt{set::lower\\_bound} versus \\texttt{std::lower\\_bound} on a set (the $\\bigO{n}$ trap)',
    'Timelines: a sorted list of (time, value) per key, queried by \\texttt{upper\\_bound}',
    'Positions of each value: counting occurrences of $x$ in $[l, r]$ in $\\bigO{\\log n}$',
    'Versioned data without persistence: binary search over snapshots',
    'Balanced BSTs and order statistics are the tree book'],
   [('Timelines per key', [('Time Based Key-Value Store', 'LC', '981'), ('Online Election', 'LC', '911')]),
    ('Positions of values', [('Robot Queries', 'CF', '1902D')])])

ch(17, 'search_variants', 'Exponential, Interpolation and Bitwise Search', 4, '1500', '2300',
   'Other ways to search a sorted space: gallop to find the range, interpolate on uniform data, or build the index bit by bit.',
   ['Exponential (galloping) search for unbounded or huge ranges',
    'Interpolation search: $\\bigO{\\log\\log n}$ on uniform data and its worst case',
    'Fibonacci and golden-ratio search',
    'Bitwise binary search: build the answer from the highest power of two (Fenwick descent is the tree book)',
    'Choosing among them'],
   [('Unbounded search', []), ('Interpolation', []), ('Bitwise binary search', [])])

ch(18, 'geometry', 'Binary Search in Geometry', 4, '1900', '2700',
   'Convexity is monotonicity in disguise: locate points in convex polygons, find tangents and search on angles and radii.',
   ['Point in a convex polygon in $\\bigO{\\log n}$ by binary search on the fan of triangles',
    'Tangents from a point to a convex polygon',
    'Binary search on an angle or a radius',
    'Extreme points in a direction',
    'Geometry primitives are assumed (cross products, orientation)'],
   [('Point in a convex polygon', [('Polygons', 'CF', '166B')]), ('Tangents', []), ('Angles and radii', [])])

ch(19, 'cascading', 'Fractional Cascading and Searching Many Lists', 4, '2200', '2800',
   'Search the same value in $k$ sorted lists in $\\bigO{k + \\log n}$ instead of $\\bigO{k \\log n}$.',
   ['The problem: $k$ binary searches with the same key',
    'Fractional cascading: merging every second element into the previous list',
    'Bridges and the $\\bigO{\\log n + k}$ query',
    'Use in segment trees (merge-sort trees) --- the trees themselves are the tree book',
    'When $\\bigO{k\\log n}$ is good enough'],
   [('Many lists', []), ('Cascading in trees', [])])

# ------------------------------------------------------------------ PART V
ch(20, 'interactive', 'Interactive Binary Search', 5, '1300', '2300',
   'The array is hidden behind a judge: binary search with queries, count your query budget, and handle adaptive answers.',
   ['The protocol: flushing, reading answers, local interactors',
    'Query budgets: $\\lceil \\log_2 n \\rceil$ and when you have only $\\log_2 n + c$',
    'Binary search on prefix sums of a hidden array',
    'Repeated searches: reusing what earlier queries told you',
    'Fixed points and other hidden structures'],
   [('Prefix sums of a hidden array', [('Interview', 'CF', '1807E')]),
    ('Hidden zeros', [('Guess the K-th Zero (Easy version)', 'CF', '1520F1')]),
    ('Hidden fixed points', [('Fixed Point Guessing', 'CF', '1698D')])])

ch(21, 'offline', 'Offline and Parallel Binary Search', 5, '2100', '2900',
   'Many queries, each needing its own binary search over the same process: run all of them together, round by round.',
   ['The batched form: every query keeps $[lo, hi]$; each round replays the process once',
    '$\\bigO{(n + q)\\log V}$ rounds and the data structure per round',
    'Typical processes: adding updates in time order, adding edges by weight',
    'The recursive (divide and conquer on time) form is the recursion book (Ch 29)',
    'When a persistent structure or an offline sweep is simpler'],
   [('Batched binary search', []), ('Updates in time order', []), ('Thresholds on weights', [])])

ch(22, 'expert_answer', 'Expert Binary Search on the Answer', 5, '2200', '3000',
   'Nested binary searches, searches over two parameters, and checks that need a data structure or a clever counting argument.',
   ['Nested binary searches and their cost',
    'Two-parameter answers: fix one, search the other',
    'Checks with data structures (sets, Fenwick trees from the tree book)',
    'Monotone in a hidden quantity: transform the statement first',
    'Proving monotonicity in non-obvious settings'],
   [('Nested searches', []), ('Two parameters', []), ('Data-structure checks', [])])

ch(23, 'expert_mixed', 'Expert Mixed Binary Search Problems', 5, '2400', '3300',
   'Problems where binary search is one of three or four ideas, and nothing in the statement says ``binary search\'\'.',
   ['A recognition guide: monotone answers hidden in the statement',
    'Binary search combined with greedy, DP and graphs',
    'Binary search combined with geometry and number theory',
    'Testing binary searches: stress against linear search',
    'Mixed sets by rating'],
   [('Mixed set 2400', []), ('Mixed set 2700', []), ('Mixed set 3000+', [])])

ch(24, 'mastery', 'Contest Mastery', 5, '2600', '3500',
   'No hints about the technique: recognise the monotone structure, pick the search, and combine chapters under contest pressure.',
   ['Reading limits: when $\\log$ factors fit',
    'Proof habits: invariants and monotonicity',
    'Implementation templates and pitfalls',
    'Stress testing',
    'Mixed contest sets by rating'],
   [('Mixed set 2800', []), ('Mixed set 3100', []), ('Mixed set 3400+', [])])


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
         '\\chapter{Roadmap of the Book}', '', '\\section*{Five Parts, One Staircase}', '',
         '\\begin{diagrambox}',
         '\\begin{tikzpicture}[x=2.45cm, y=1.05cm,',
         '    stair/.style={rectangle, rounded corners=2pt, draw=#1, fill=#1!12, line width=1pt,',
         '                  align=center, font=\\scriptsize, minimum width=2.3cm, minimum height=1.6cm}]']
    short = {1: 'Foundations', 2: 'On the answer', 3: 'Advanced', 4: 'Beyond arrays', 5: 'Expert'}
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
          'any of the eight companion books (array, bit, DP, graph, tree, string, recursion, stack--queue--heap).', '',
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
