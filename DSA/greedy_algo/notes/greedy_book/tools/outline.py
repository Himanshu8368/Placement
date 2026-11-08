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
APIs and against the nine companion books (tools/verify_outline.py); later parts are candidates and are
re-checked when the chapter is written.

This book owns greedy algorithms as a method: the greedy choice, proofs (stays ahead, exchange, cut, potential),
sorting-based and interval greedy, scheduling, lexicographic greedy with feasibility checks, matroids, regret,
Hall-type conditions, online and approximation algorithms.
Topic overlap with the array book's Ch 14 "Greedy on Arrays" and Ch 34 "Expert Greedy" is intended; no problem is
shared (tools/check_cross_book.py).

Topics owned by a companion book are NOT taught here (the text points to them instead):
    greedy with a heap, Huffman, heap scheduling, regret with heaps ... stack-queue-heap book (Ch 19, 23)
    remove-k-digits and other monotonic-stack greedies ................ stack-queue-heap book (Ch 11)
    minimum spanning trees, Dijkstra .................................. graph book
    greedy on trees ................................................... tree book (Ch 33)
    greedy over bits .................................................. bit book (Ch 7)
    greedy checks inside binary search on the answer .................. binary search book (Ch 11)
    jump game, gas station, candy and other array greedies ............ array book (Ch 14)
    interval merging and sweeps ....................................... array book (Ch 8)
    every problem where greedy fails and DP is needed ................. DP book
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

PARTS = [
    dict(n=1, key='foundations', title='Foundations', level=r'Beginner $\to$ CF 1600', color='successgreen',
         tagline='Take the best choice now --- and prove it is safe: the greedy choice, first proofs, sorting, intervals, digits and strings.',
         goals=[r'recognise a greedy choice and break a wrong one with a small counter-example;',
                r"prove simple greedies by ``stays ahead'' and by exchange;",
                r'sort by the right key and pair or schedule after sorting;',
                r'make greedy choices on intervals, digits and strings.']),
    dict(n=2, key='core', title='Core Techniques', level=r'CF 1300 $\to$ CF 2200', color='primaryblue',
         tagline='Matching two lists, deadlines, running balances, reachable sums, and building the lexicographically best answer one position at a time.',
         goals=[r'match two sorted lists and assign tasks greedily;',
                r'schedule jobs with deadlines and weights by the right rule;',
                r'keep a running balance and fix it greedily;',
                r'decide when greedy works and when DP is required, and build lexicographic answers with feasibility checks.']),
    dict(n=3, key='proofs', title='Proofs \\& Structure', level=r'CF 1700 $\to$ CF 2600', color='softpurple',
         tagline='Why greedy works: exchange arguments and comparators, matroids, regret, potential functions and Hall-type conditions.',
         goals=[r'derive a sorting comparator from an adjacent exchange;',
                r'recognise matroids and apply the weighted matroid greedy;',
                r'repair an earlier choice with regret;',
                r'prove feasibility with Hall-type prefix conditions and potentials.']),
    dict(n=4, key='tools', title='Greedy with Tools', level=r'CF 1600 $\to$ CF 2600', color='accentorange',
         tagline='Greedy choices that need a data structure, many machines, an opponent, several criteria, or a construction.',
         goals=[r"make ``smallest available that fits'' choices with sorted containers;",
                r'schedule intervals on several machines and answer interval queries offline;',
                r'play games by taking the best available move;',
                r'combine criteria and construct answers greedily.']),
    dict(n=5, key='beyond', title='Beyond Exact Greedy', level=r'CF 1800 $\to$ CF 2800', color='warningred',
         tagline='When greedy is not optimal but provably close: online algorithms, approximation guarantees and submodular maximisation.',
         goals=[r'analyse online algorithms by their competitive ratio (paging, ski rental);',
                r'prove approximation ratios for greedy set cover, vertex cover and $k$-center;',
                r'maximise submodular functions greedily within $1 - 1/e$;',
                r'use randomised greedy choices.']),
    dict(n=6, key='expert', title='Expert \\& Contest Mastery', level=r'CF 2400 $\to$ CF 3500', color='deepblue',
         tagline='Hard exchange and potential arguments, and unlabeled greedy problems under contest pressure.',
         goals=[r'find the exchange or potential argument behind a hard greedy;',
                r'combine greedy with other techniques;',
                r'stress-test a greedy against a brute force before trusting it;',
                r'recognise a greedy problem without a label.']),
]

# (num, slug, title, part, lo, hi, promise, lessons, patterns, connections)
# patterns: list of (name, [(title, judge, id), ...])
CH = []


def ch(num, slug, title, part, lo, hi, promise, lessons, patterns, connections=''):
    num = len(CH) + 1   # chapters are numbered by their order in this file (the first argument is a reminder only)
    CH.append(dict(num=num, slug=slug, title=title, part=part, lo=lo, hi=hi, promise=promise,
                   lessons=lessons, patterns=patterns, connections=connections))


# ------------------------------------------------------------------ PART I
ch(1, 'introduction', 'Introduction to Greedy Algorithms', 1, 'Beginner', '1200',
   'Make the locally best choice and never look back --- when that is safe, greedy is the simplest and fastest algorithm there is.',
   ['What a greedy algorithm is: a sequence of irrevocable, locally best choices',
    'The two properties: the greedy-choice property and optimal substructure',
    'Breaking a wrong greedy: small counter-examples and brute-force stress tests',
    'Greedy versus DP versus search: the coin-change example (the DP is the DP book)',
    'The greedy workflow: guess the rule, test it, prove it, implement it'],
   [('Take the best available', [('Assign Cookies', 'LC', '455'), ('Maximum Units on a Truck', 'LC', '1710')]),
    ('Greedy with a running state', [('Lemonade Change', 'LC', '860'), ('Minimum Cost of Buying Candies With Discount', 'LC', '2144')]),
    ('Greedy grouping and counting', [('Taxi', 'CF', '158B'), ('Maximum Ice Cream Bars', 'LC', '1833')])])

ch(2, 'proofs_basic', 'First Proofs: Stays Ahead and Exchange', 1, '900', '1500',
   'A greedy is only as good as its proof: show that it stays ahead of every other solution, or that any optimum can be exchanged into it.',
   ['``Greedy stays ahead\'\': compare the greedy and an optimal solution step by step',
    'The exchange argument: turn an optimal solution into the greedy one without making it worse',
    'Local repairs: fix the first violation in the cheapest way',
    'Writing the proof in five lines',
    'Counter-example hunting as the other half of a proof'],
   [('Greedy stays ahead', [('Maximize Sum Of Array After K Negations', 'LC', '1005'), ('Maximum Element After Decreasing and Rearranging', 'LC', '1846')]),
    ('Exchange arguments', [('Minimum Increment to Make Array Unique', 'LC', '945'), ('Minimum Deletions to Make Character Frequencies Unique', 'LC', '1647')]),
    ('Local repairs', [('Minimum Time to Make Rope Colorful', 'LC', '1578'), ('Minimum Replacements to Sort the Array', 'LC', '2366')])])

ch(3, 'sorting_greedy', 'Sorting-Based Greedy', 1, '900', '1600',
   'Most greedies start with a sort: by value, by deadline, by a difference or by a ratio --- the key is the whole algorithm.',
   ['Sorting by one key and processing in order',
    'Sorting by a difference $a_i - b_i$ or a ratio $a_i / b_i$',
    'Pairing after sorting: smallest with smallest, smallest with largest',
    'Sorting both sides of an assignment',
    'Ties and stability: when the order of equal keys matters'],
   [('Sort by one key', [('Advantage Shuffle', 'LC', '870'), ('Minimum Number of Moves to Seat Everyone', 'LC', '2037')]),
    ('Sort by a difference or position', [('Two City Scheduling', 'LC', '1029'), ('Maximum Number of Coins You Can Get', 'LC', '1561')])])

ch(4, 'intervals', 'Interval Greedy', 1, '1100', '1700',
   'Intervals are the classic greedy domain: choose by the earliest end, stab with the latest point, and group with the fewest machines.',
   ['Activity selection: earliest end first, and the stays-ahead proof',
    'Stabbing intervals with points: the latest possible point',
    'Covering requirements of several points per interval',
    'Attending events one per day: earliest deadline among the available',
    'Interval partitioning: the minimum number of groups equals the maximum overlap (sweeps are the array book)'],
   [('Stabbing intervals with points', [('Set Intersection Size At Least Two', 'LC', '757'), ('Minimum Time to Complete All Tasks', 'LC', '2589')]),
    ('Events and groups', [('Maximum Number of Events That Can Be Attended', 'LC', '1353'), ('Divide Intervals Into Minimum Number of Groups', 'LC', '2406')])])

ch(5, 'digits', 'Greedy on Numbers and Digits', 1, '900', '1600',
   'The most significant digit decides: make it as large (or small) as possible, then repair the rest.',
   ['The leftmost difference decides the comparison of two numbers',
    'One swap: the best digit to move and where',
    'Digit rewrites under rules: monotone digits, replacements',
    'Previous and next numbers with constraints',
    'Leading zeros and other edge cases'],
   [('Digit swaps', [('Maximum Swap', 'LC', '670'), ('Previous Permutation With One Swap', 'LC', '1053')]),
    ('Digit rewrites', [('Maximum 69 Number', 'LC', '1323'), ('Monotone Increasing Digits', 'LC', '738')])])

ch(6, 'strings', 'Greedy on Strings', 1, '1000', '1600',
   'Lexicographic order rewards the first position: decide each character as early and as greedily as the constraints allow.',
   ['Lexicographic order and why the first differing position decides',
    'Changing one character for the smallest result',
    'Building strings under limits on runs and counts',
    'Choosing the next character with a lookahead (the full feasibility-check technique is Part II)',
    'String-specific algorithms are the strings book'],
   [('Lexicographic choices', [('Break a Palindrome', 'LC', '1328'), ('Lexicographically Smallest Palindrome', 'LC', '2697')]),
    ('Building strings under constraints', [('String Without AAA or BBB', 'LC', '984'), ('Using a Robot to Print the Lexicographically Smallest String', 'LC', '2434')])])

# ------------------------------------------------------------------ PART II
ch(7, 'matching', 'Pairing and Matching Two Lists', 2, '1300', '2000',
   'Workers and jobs, players and trainers, boxes and items: sort both lists and match with two pointers --- or prove you cannot.',
   ['Matching two sorted lists with two pointers: who gets matched first',
    'Maximising the number of matched pairs versus the total value',
    'Assigning the best job each worker can do',
    'Matching with a threshold or a budget of boosts',
    'When greedy matching fails and general matching is needed (the graph book)'],
   [('Two sorted lists', [('BerSU Ball', 'CF', '489B'), ('Most Profit Assigning Work', 'LC', '826')]),
    ('Matching with a budget', []),
    ('Maximum versus total value', [])])

ch(8, 'scheduling', 'Scheduling and Deadlines', 2, '1400', '2200',
   'Order jobs by the right rule: earliest deadline first, shortest processing time, Smith\'s ratio rule --- each with an exchange proof.',
   ['Minimising total completion time: shortest processing time first',
    'Minimising maximum lateness: earliest deadline first',
    'Weighted completion time: Smith\'s rule $p_i / w_i$',
    'Maximising jobs done before deadlines (Moore--Hodgson; the heap is the stack-queue-heap book)',
    'Job sequencing with profits and unit times'],
   [('Rewards from deadlines', [('Tasks and Deadlines', 'CSES', '1630')]),
    ('Smith\'s rule', []),
    ('Lateness', []),
    ('Profits and unit jobs', [])])

ch(9, 'balances', 'Running Balances and Prefix Constraints', 2, '1300', '2100',
   'Keep a running balance --- of brackets, fuel, money --- and repair it greedily the moment it goes wrong.',
   ['The running balance and its minimum: the classic bracket argument',
    'Fix it when it breaks: insertions, deletions and swaps at the first violation',
    'Prefix constraints $\\sum_{i \\le k} \\ge 0$ and why local fixes are optimal',
    'Two-sided balances (both prefix and suffix)',
    'Bracket sequences with two kinds of closing rules'],
   [('Bracket balances', [('Minimum Add to Make Parentheses Valid', 'LC', '921'), ('Minimum Insertions to Balance a Parentheses String', 'LC', '1541')]),
    ('Swaps that restore balance', [('Minimum Number of Swaps to Make the String Balanced', 'LC', '1963')]),
    ('Prefix constraints', [])])

ch(10, 'math_greedy', 'Mathematical Greedy', 2, '1400', '2200',
   'Greedy meets number theory: canonical coin systems, reachable sums, Egyptian fractions and greedy representations.',
   ['Canonical coin systems: when greedy change-making is optimal',
    'Reachable sums: if $[1, s]$ is reachable and the next coin is $\\le s+1$, then $[1, s+c]$ is',
    'Patching: the cheapest additions to make every sum reachable',
    'Greedy representations: Zeckendorf, factorial base, Egyptian fractions',
    'Greedy with divisibility and parity'],
   [('Reachable sums', [('Maximum Number of Consecutive Values You Can Make', 'LC', '1798'), ('Patching Array', 'LC', '330')]),
    ('Canonical coin systems', []),
    ('Greedy representations', [])])

ch(11, 'greedy_vs_dp', 'Greedy versus DP: When Greedy Fails', 2, '1400', '2200',
   'The most important greedy skill is saying no: recognise the problems where a local choice can be wrong, and hand them to DP.',
   ['Counter-example families: coin change, 0/1 knapsack, weighted intervals',
    'Fractional versus 0/1: why fractional knapsack is greedy',
    'Greedy as a DP whose choice is always the same',
    'Greedy with a small DP fallback',
    'A refusal table: greedy, DP, flow, or search (the DP book owns the DPs)'],
   [('Fractional versus 0/1', []), ('Greedy that is secretly DP', []), ('Refusals', [])])

ch(12, 'lexicographic', 'Lexicographic Greedy with Feasibility Checks', 2, '1500', '2300',
   'Build the lexicographically best answer one position at a time: try the best symbol, check that the rest can still be completed, commit.',
   ['The template: for each position try candidates in order; keep the first that leaves a feasible remainder',
    'Designing the feasibility check (counts, matchings, flows in small cases)',
    'Smallest permutation, sequence or string under constraints',
    'Complexity: positions $\\times$ candidates $\\times$ check',
    'Speeding up the check incrementally'],
   [('Feasibility by counting', []), ('Smallest sequence under constraints', []), ('Incremental checks', [])])

# ------------------------------------------------------------------ PART III
ch(13, 'exchange', 'Exchange Arguments and Comparators', 3, '1700', '2500',
   'Compare two neighbours, swap them, and read the sorting comparator off the inequality --- the most reused proof in greedy.',
   ['Adjacent exchange: when does putting $i$ before $j$ never hurt?',
    'From the inequality to a comparator; checking that it is a strict weak order',
    'Orders of tasks with requirements and rewards (positive and negative tasks)',
    'Order of removal operations',
    'When the exchange order is not transitive: fallbacks'],
   [('Tasks with requirements', [('Complete the Projects (easy version)', 'CF', '1203F1')]),
    ('Order of removals', [('Maximum Score From Removing Substrings', 'LC', '1717')]),
    ('Comparators from inequalities', [])])

ch(14, 'matroids', 'Matroids and the Greedy Algorithm', 3, '1900', '2600',
   'Why greedy works on spanning trees, independent vectors and schedules: they are matroids, and greedy is optimal on every matroid.',
   ['Independence systems and the matroid axioms',
    'The weighted matroid greedy and its proof',
    'Examples: graphic (MST, the graph book), linear (XOR basis, the bit book), partition and scheduling matroids',
    'When greedy fails: independence systems that are not matroids',
    'Matroid intersection as a going-further note'],
   [('Scheduling matroids', []), ('Linear matroids', []), ('Partition matroids', [])])

ch(15, 'regret', 'Greedy with Regret', 3, '1800', '2600',
   'Take the greedy choice now and keep the option to undo it later: regret turns many wrong greedies into right ones.',
   ['Regret: record what each choice could be traded back for',
    'Undo the worst earlier choice when a better one appears (heaps are the stack-queue-heap book)',
    'Refuelling and stops: take fuel only when you must, from the best station passed',
    'Bracket sequences with costs: start cheap, swap when needed',
    'Proving regret greedies with exchange arguments'],
   [('Take only when needed', [('Minimum Number of Refueling Stops', 'LC', '871')]),
    ('Swap a choice for a cheaper one', [('Least Cost Bracket Sequence', 'CF', '3D')]),
    ('Regret with profits', [])])

ch(16, 'potentials', 'Potentials and Amortised Greedy', 3, '1900', '2600',
   'Measure progress with a potential: every greedy step decreases it, so the process ends --- and often ends optimally.',
   ['Monovariants and termination of greedy processes',
    'Potential functions for counting steps',
    'Amortised greedy: expensive steps are rare',
    'Invariants that certify optimality',
    'From a potential to a lower bound that matches the greedy'],
   [('Termination', []), ('Counting steps', []), ('Matching lower bounds', [])])

ch(17, 'hall', 'Hall-Type Conditions and Greedy Feasibility', 3, '2000', '2600',
   'A greedy assignment exists exactly when no group of requests is too greedy itself: Hall\'s condition on prefixes, intervals and supplies.',
   ["Hall's theorem and its interval and prefix forms",
    'Checking supply--demand feasibility with sorted prefixes',
    'Greedy assignment that succeeds iff Hall holds',
    'Deficiency: how many requests can be served',
    'Flows as the general tool (the graph book)'],
   [('Prefix Hall conditions', []), ('Interval supplies', []), ('Deficiency', [])])

# ------------------------------------------------------------------ PART IV
ch(18, 'containers', 'Greedy with Sorted Containers', 4, '1600', '2400',
   'Give each request the smallest available item that fits: a multiset makes ``best available\'\' a logarithmic query.',
   ['Smallest item that fits: \\texttt{multiset::lower\\_bound}',
    'Largest item below a limit and deleting it',
    'Greedy pairing with priorities (heaps are the stack-queue-heap book)',
    'Binary search plus a greedy multiset check',
    'Proof: the exchange behind ``smallest that fits\'\''],
   [('Smallest that fits', [('Maximum Number of Tasks You Can Assign', 'LC', '2071')]),
    ('Pairing by priority', [('Productive Meeting', 'CF', '1579D')]),
    ('Deleting the best', [])])

ch(19, 'adv_intervals', 'Advanced Interval Greedy and Multi-Machine Scheduling', 4, '1800', '2600',
   'Intervals on $k$ machines, queries answered offline by the smallest covering interval, and weighted variants that need care.',
   ['$k$ machines: reuse the machine that frees up just in time',
    'Offline interval queries: sort queries and intervals together',
    'Covering a segment with the fewest intervals',
    'Intervals with demands and capacities',
    'Weighted intervals are the DP book'],
   [('Offline interval queries', [('Minimum Interval to Include Each Query', 'LC', '1851')]),
    ('k machines', []),
    ('Capacities', [])])

ch(20, 'games', 'Greedy in Games', 4, '1200', '2300',
   'When both players take the best available item, sorting decides the game: greedy strategies and their optimality proofs.',
   ['Both players greedy: why taking the largest is optimal in symmetric take-games',
    'Asymmetric values: sort by the sum of both players\' gains',
    'Greedy moves versus full game search (the recursion book) and Nim (the bit book)',
    'Proving a greedy strategy by a pairing argument',
    'Games on arrays and trees are the array and tree books'],
   [('Take the best available item', [('Even-Odd Game', 'CF', '1472D')]),
    ('Sorting by combined value', []),
    ('Pairing strategies', [])])

ch(21, 'multi_criteria', 'Multi-Criteria and Two-Phase Greedy', 4, '1700', '2500',
   'Two keys at once: fix one criterion by enumeration or sorting, then be greedy in the other.',
   ['Lexicographic objectives: optimise the first criterion, then the second',
    'Enumerate one parameter, greedy on the rest',
    'Two-phase greedy: a feasible solution first, then improvement',
    'Pareto fronts and dominance pruning',
    'When the criteria conflict and DP or flow is needed'],
   [('Enumerate one, greedy the other', []), ('Two phases', []), ('Dominance', [])])

ch(22, 'constructive', 'Constructive Greedy', 4, '1600', '2600',
   'Build the required object greedily --- an arrangement, a sequence, an assignment --- and prove that the greedy never gets stuck.',
   ['Greedy constructions that never get stuck: the invariant that guarantees a next move',
    'Placing the most frequent item first',
    'Greedy sequences with forbidden neighbours',
    'Greedy constructions with a final repair step',
    'Validating constructions with a checker'],
   [('Most frequent first', []), ('Forbidden neighbours', []), ('Construct and repair', [])])

# ------------------------------------------------------------------ PART V
ch(23, 'online', 'Online Algorithms', 5, '1700', '2600',
   'Decide without knowing the future: paging, caching and ski rental, measured by how far the greedy can fall behind the optimum.',
   ['Online versus offline; the competitive ratio',
    'Ski rental and the break-even rule',
    "Paging: LRU, FIFO and Belady's offline optimum (farthest in the future)",
    'Caches as data structures: LRU and LFU in $\\bigO{1}$',
    'Randomised online algorithms (marking)'],
   [('Caching', [('LRU Cache', 'LC', '146'), ('LFU Cache', 'LC', '460')]),
    ('Offline optimum: farthest in future', []),
    ('Rent or buy', [])])

ch(24, 'approximation', 'Greedy Approximation Algorithms', 5, '1900', '2800',
   'When the problem is NP-hard, greedy can still guarantee a factor: set cover within $\\ln n$, vertex cover within 2, $k$-center within 2.',
   ['Approximation ratios and how to prove them',
    'Greedy set cover and the $H_n$ bound',
    'Vertex cover by maximal matching (2-approximation)',
    '$k$-center by farthest-point traversal',
    'Load balancing: list scheduling and LPT'],
   [('Set cover', []), ('Vertex cover', []), ('k-center', []), ('Load balancing', [])])

ch(25, 'submodular', 'Submodular and Randomised Greedy', 5, '2100', '2800',
   'Diminishing returns make greedy provably good: the $1 - 1/e$ guarantee, and when random choices help a greedy.',
   ['Submodular functions: diminishing returns',
    'Greedy maximisation under a cardinality constraint: $1 - 1/e$',
    'Coverage and facility-location examples',
    'Randomised greedy and random orders',
    'Lazy evaluation to speed up greedy'],
   [('Coverage', []), ('Random orders', [])])

# ------------------------------------------------------------------ PART VI
ch(26, 'expert_exchange', 'Expert Exchange Arguments', 6, '2400', '3100',
   'Hard greedy problems where the rule is non-obvious: multi-element exchanges, global potentials and proofs by contradiction.',
   ['Exchanges of more than two elements',
    'Global potentials and invariants',
    'Greedy combined with binary search on the answer',
    'Greedy combined with data structures',
    'Proof by contradiction on the first difference'],
   [('Multi-element exchanges', []), ('Global potentials', []), ('Greedy plus data structures', [])])

ch(27, 'expert_mixed', 'Expert Mixed Greedy Problems', 6, '2400', '3300',
   'Problems that combine three or four ideas; no chapter title tells you that the core is greedy.',
   ['A recognition guide: the signals of a greedy problem',
    'Greedy hidden behind DP-looking statements',
    'Greedy hidden behind graph statements',
    'Stress testing every greedy',
    'Mixed sets by rating'],
   [('Mixed set 2400', []), ('Mixed set 2700', []), ('Mixed set 3000+', [])])

ch(28, 'mastery', 'Contest Mastery', 6, '2600', '3500',
   'No hints about the technique: guess the rule, break it, prove it, and implement it under contest pressure.',
   ['Reading limits like a contestant',
    'Proof habits for greedy: the five-line proof',
    'Implementation pitfalls: ties, overflow, comparators',
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
         '\\chapter{Roadmap of the Book}', '', '\\section*{Six Parts, One Staircase}', '',
         '\\begin{diagrambox}',
         '\\begin{tikzpicture}[x=2.45cm, y=1.05cm,',
         '    stair/.style={rectangle, rounded corners=2pt, draw=#1, fill=#1!12, line width=1pt,',
         '                  align=center, font=\\scriptsize, minimum width=2.3cm, minimum height=1.6cm}]']
    short = {1: 'Foundations', 2: 'Core', 3: 'Proofs', 4: 'With tools', 5: 'Beyond exact', 6: 'Expert'}
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
          'any of the nine companion books (array, binary search, bit, DP, graph, tree, string, recursion, stack--queue--heap).', '',
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
