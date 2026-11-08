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
APIs and against the seven companion books (tools/verify_outline.py); later parts are candidates and are
re-checked when the chapter is written.

Topics owned by a companion book are NOT taught here (the text points to them instead):
    bitmask DP, digit DP, SOS DP, subset convolution ................ DP book
    binary tries, max-XOR with a trie, XOR-MST ...................... string book (first tries) + tree book (Ch 25)
    bit-parallel string matching (Shift-And, alphabet masks) ........ string book
    bitmask search vs DP, exact cover / DLX, meet in the middle ..... recursion book
    XOR / AND / OR over subarrays, per-bit sums over subarrays ...... array book (Ch 15)
    XOR paths and the cycle space of a graph, Hamiltonian bitmasks .. graph book
    Fenwick trees (lowbit as an index), sparse tables ............... tree book
This book owns: binary representation and operators, bit tricks, popcount and parity, XOR algebra, the XOR
linear basis, Gaussian elimination over GF(2), Walsh--Hadamard transforms, Nim as XOR, Gray codes, bitwise
equations and constructions, mask enumeration, bitsets as a data structure, word-level parallelism, 64-ary bit
trees, bit-mixing hashes and randomisation, interactive and constructive bit problems, bit-level optimisation.
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

PARTS = [
    dict(n=1, key='foundations', title='Foundations', level=r'Beginner $\to$ CF 1700', color='successgreen',
         tagline='Numbers as strings of bits: representation, the six operators, the classic tricks, popcount, XOR and masks as sets.',
         goals=[r'read and write numbers in binary, hexadecimal and two\textquotesingle s complement, and know every width and overflow trap;',
                r'use AND, OR, XOR, NOT and shifts fluently, with their algebraic identities and precedence pitfalls;',
                r'apply the classic tricks: powers of two, lowest and highest set bit, popcount, parity, XOR cancellation;',
                r'represent sets of up to 64 elements as integers and enumerate all subsets.']),
    dict(n=2, key='core', title='Core Techniques', level=r'CF 1300 $\to$ CF 2400', color='primaryblue',
         tagline='Decide the answer bit by bit, split problems into independent bits, and exploit the order structure of AND and OR.',
         goals=[r'build the best answer greedily from the highest bit, with a feasibility check per bit;',
                r'decompose sums and counts over pairs into independent bits;',
                r'use the monotonicity of AND and OR and the submask partial order;',
                r'solve bitwise equations, use Gray codes and hypercubes, and apply bits in number theory.']),
    dict(n=3, key='xor_algebra', title='XOR Algebra', level=r'CF 1500 $\to$ CF 3200', color='softpurple',
         tagline='XOR is addition over GF(2): prefix XOR, linear bases, Gaussian elimination, Walsh--Hadamard transforms and Nim.',
         goals=[r'answer XOR questions with prefix XOR and random XOR hashing;',
                r'build and query a linear basis: maximum, $k$-th and number of representable XORs;',
                r'solve linear systems over GF(2) and count their solutions;',
                r'convolve with XOR, AND and OR in $\bigO{n \log n}$, and win Nim-type games.']),
    dict(n=4, key='masks_words', title='Bitmask Techniques \\& Word Parallelism', level=r'CF 1600 $\to$ CF 2800', color='accentorange',
         tagline='Masks as sets, submask enumeration, bitsets, SWAR tricks and bit trees: 64 operations for the price of one.',
         goals=[r'enumerate submasks in $\bigO{3^n}$, supersets and $k$-subsets, and prune mask searches;',
                r'encode compatibility and conflicts as masks and count intersections;',
                r'speed up quadratic algorithms by 64 with bitsets;',
                r'write word-parallel code and 64-ary bit trees for fast predecessor queries.']),
    dict(n=5, key='problem_solving', title='Bits in Problem Solving', level=r'CF 1500 $\to$ CF 2600', color='warningred',
         tagline='Randomised hashing, counting over binary prefixes, carries, interactive and constructive problems built on bits.',
         goals=[r'hash states and multisets with random 64-bit values and robust mixers;',
                r'count numbers up to $n$ with a bitwise property by fixing a binary prefix;',
                r'reason about carries when arithmetic and bitwise operations meet;',
                r'recover hidden values with bitwise queries, and construct arrays with XOR, AND and OR constraints.']),
    dict(n=6, key='advanced', title='Advanced \\& Expert Topics', level=r'CF 2200 $\to$ CF 3500', color='deepblue',
         tagline='Linear bases over ranges and time, bit-level performance engineering, and mixed expert problems.',
         goals=[r'maintain linear bases over ranges, prefixes and time;',
                r'make bit-heavy code fast: built-ins, branchless code, memory packing;',
                r'combine several bit techniques in one solution;',
                r'recognise the bit technique of an unlabeled hard problem under contest pressure.']),
]

# (num, slug, title, part, lo, hi, promise, lessons, patterns, connections)
# patterns: list of (name, [(title, judge, id), ...])
CH = []


def ch(num, slug, title, part, lo, hi, promise, lessons, patterns, connections=''):
    num = len(CH) + 1   # chapters are numbered by their order in this file (the first argument is a reminder only)
    CH.append(dict(num=num, slug=slug, title=title, part=part, lo=lo, hi=hi, promise=promise,
                   lessons=lessons, patterns=patterns, connections=connections))


# ------------------------------------------------------------------ PART I
ch(1, 'introduction', 'Binary Representation', 1, 'Beginner', '1200',
   'Every integer is a string of bits: learn to read it, convert it, fix its width and survive signed overflow.',
   ['Positional systems: binary, octal, hexadecimal and base $-2$; converting by repeated division and by grouping bits',
    'Fixed-width integers: \\texttt{int}, \\texttt{long long}, \\texttt{unsigned}, \\texttt{\\_\\_int128}; the range of each',
    "Two's complement: why $-x = \\sim x + 1$, the asymmetric range, and signed overflow as undefined behaviour",
    'Bit numbering and printing: bit $i$ has value $2^i$, \\texttt{std::bitset} for printing, endianness as a footnote',
    'Binary strings as numbers: reading a bit string modulo $m$, and when the number does not fit in 64 bits'],
   [('Bases and conversions', [('Convert a Number to Hexadecimal', 'LC', '405'), ('Convert to Base -2', 'LC', '1017')]),
    ('Reading the bits of a number', [('Binary Gap', 'LC', '868'), ('Binary Number with Alternating Bits', 'LC', '693')]),
    ('Fixed width: complement and reversal', [('Number Complement', 'LC', '476'), ('Reverse Bits', 'LC', '190')]),
    ('Signed integers and overflow', [('Divide Two Integers', 'LC', '29'), ('Integer Replacement', 'LC', '397')]),
    ('Binary strings as numbers', [('Binary Prefix Divisible By 5', 'LC', '1018'), ('Concatenation of Consecutive Binary Numbers', 'LC', '1680')])])

ch(2, 'operators', 'Bitwise Operators and Their Algebra', 1, '800', '1500',
   'Six operators and a handful of identities: AND, OR, XOR, NOT and the two shifts, with the algebra that links them to $+$ and $-$.',
   ['AND, OR, XOR and NOT bit by bit; truth tables and the laws (commutative, associative, distributive, De Morgan)',
    'Shifts: $x \\ll k = x\\cdot 2^k$, arithmetic versus logical right shift, and the undefined shifts ($k \\ge$ width, negative $x$)',
    'Testing, setting, clearing and toggling bit $i$; \\texttt{1 << i} versus \\texttt{1LL << i}',
    'Identities: $a + b = (a \\oplus b) + 2(a \\,\\&\\, b)$, $a + b = (a \\mid b) + (a \\,\\&\\, b)$, and $a \\oplus b \\le a + b$',
    'Operator precedence in C++ (\\texttt{a \\& b == c} is a bug) and how to read bitwise code safely'],
   [('Test, set, clear and toggle', [('Minimum Bit Flips to Convert Number', 'LC', '2220'), ('Minimum Flips to Make a OR b Equal to c', 'LC', '1318')]),
    ('Shifts', [('Number of Steps to Reduce a Number to Zero', 'LC', '1342'), ('Number of Even and Odd Bits', 'LC', '2595')]),
    ('Identities between arithmetic and bitwise operators', [('Sum of Two Integers', 'LC', '371'), ('XORwice', 'CF', '1421A')]),
    ('Operators over a range of numbers', [('Bitwise AND of Numbers Range', 'LC', '201'), ('And Then There Were K', 'CF', '1527A')]),
    ('Choosing a number under bitwise constraints', [('Minimize XOR', 'LC', '2429'), ('Boboniu and Bit Operations', 'CF', '1395C')])])

ch(3, 'tricks', 'Classic Bit Tricks', 1, '900', '1500',
   'The one-line tricks every programmer reuses: powers of two, the lowest and highest set bit, and the next mask with the same popcount.',
   ['Powers of two: $x \\,\\&\\, (x-1) = 0$, rounding up and down to a power of two',
    'The lowest set bit: $x \\,\\&\\, (-x)$, clearing it with $x \\,\\&\\, (x-1)$, counting trailing zeros',
    'The highest set bit and the bit length: \\texttt{\\_\\_lg}, \\texttt{\\_\\_builtin\\_clz}, and the all-ones mask below it',
    'Masks of bit ranges: extracting, inserting and clearing bits $[l, r)$',
    "Gosper's hack: the next mask with the same number of set bits"],
   [('Powers of two', [('Power of Four', 'LC', '342'), ('Odd Divisor', 'CF', '1475A')]),
    ('The lowest set bit and carries', [('Minimum Operations to Reduce an Integer to 0', 'LC', '2571'), ('Johnny and Another Rating Drop', 'CF', '1362C')]),
    ('Highest and lowest bit positions', [('Smallest Number With All Set Bits', 'LC', '3370'), ('Check if Bitwise OR Has Trailing Zeros', 'LC', '2980')])])

ch(4, 'popcount', 'Counting Bits and Parity', 1, '900', '1600',
   'How many bits are set, how many differ, how many are set at each position: popcount and parity in one, many and all numbers.',
   ['Popcount: the naive loop, Kernighan\'s loop, \\texttt{\\_\\_builtin\\_popcount(ll)} and its cost',
    'Parity: XOR of all bits, and parity arguments as proofs',
    'Hamming distance: popcount of the XOR, and total distance over all pairs by counting per bit',
    'Counting set bits over $[1, n]$: the per-position formula',
    'Counting how many numbers have bit $i$ set, and using those counts'],
   [('Popcount', [('Number of 1 Bits', 'LC', '191'), ('Sort Integers by The Number of 1 Bits', 'LC', '1356')]),
    ('Hamming distance', [('Hamming Distance', 'LC', '461'), ('Total Hamming Distance', 'LC', '477')]),
    ('Set bits over a range of numbers', [('Counting Bits', 'CSES', '1146'), ('Prime Number of Set Bits in Binary Representation', 'LC', '762')]),
    ('Counts per bit position', [('Largest Combination With Bitwise AND Greater Than Zero', 'LC', '2275'), ("And It's Non-Zero", 'CF', '1615B')])])

ch(5, 'xor', 'XOR Fundamentals', 1, '900', '1700',
   'XOR is addition without carries: it cancels pairs, undoes itself and has a closed form over every range of integers.',
   ['The XOR laws: $x \\oplus x = 0$, $x \\oplus 0 = x$, associativity, and XOR as addition modulo 2 per bit',
    'XOR of $0..n$ in $\\bigO{1}$ (period 4) and of any range $[l, r]$',
    'Undoing XOR: recovering an array from adjacent or prefix XORs',
    'XOR of pairs and triplets: when an equation $a \\oplus b = c$ fixes the third value',
    'XOR and parity: what an XOR of everything tells you, and what it hides'],
   [('XOR of a range of integers', [('XOR Operation in an Array', 'LC', '1486'), ('MEXor Mixup', 'CF', '1567B')]),
    ('Undoing XOR', [('Decode XORed Array', 'LC', '1720'), ('Find The Original Array of Prefix Xor', 'LC', '2433')]),
    ('XOR equations on neighbours', [('Neighboring Bitwise XOR', 'LC', '2683'), ('Decode XORed Permutation', 'LC', '1734')]),
    ('XOR of pairs and triplets', [('Count Triplets That Can Form Two Arrays of Equal XOR', 'LC', '1442'), ('Bitwise XOR of All Pairings', 'LC', '2425')]),
    ('Cancellation arguments', [('AGAGA XOOORRR', 'CF', '1516B'), ('Find Xor-Beauty of Array', 'LC', '2527')])])

ch(6, 'subsets', 'Sets and Subsets as Bitmasks', 1, 'Beginner', '1500',
   'A set of up to 64 elements is one integer: union is OR, intersection is AND, and all $2^n$ subsets are the numbers $0..2^n-1$.',
   ['Sets as integers: membership, insert, erase, union, intersection, difference, complement within $n$ bits',
    'Enumerating all subsets with a counter; the order and the cost $\\bigO{2^n \\cdot n}$',
    'Iterating over the set bits of a mask',
    'Masks of letters, positions and items as compact signatures',
    'When bitmask enumeration is enough and when it becomes DP (bitmask DP is the DP book)'],
   [('Enumerating all subsets', [('Petr and a Combination Lock', 'CF', '1097B'), ('Sum of All Subset XOR Totals', 'LC', '1863')]),
    ('Masks of chosen items', [('Maximum Rows Covered by Columns', 'LC', '2397'), ('Maximum Length of a Concatenated String with Unique Characters', 'LC', '1239')]),
    ('Masks of letters and positions', [('Maximum Product of the Length of Two Palindromic Subsequences', 'LC', '2002'), ('Maximum Score Words Formed by Letters', 'LC', '1255')])])

# ------------------------------------------------------------------ PART II
ch(7, 'greedy_bits', 'Greedy over Bits: Deciding from the Highest Bit', 2, '1400', '2200',
   'Bit $k$ outweighs all lower bits together, so the best answer is built from the top: fix a bit, check feasibility, go down.',
   ['Why the highest bit decides: $2^k > 2^k - 1 = \\sum_{i<k} 2^i$',
    'Building the answer bit by bit with a feasibility check',
    'Maximum AND and maximum XOR of a pair by prefix filtering (tries are the string and tree books)',
    'Minimising OR / XOR under constraints from the top bit down',
    'Binary search versus bit-by-bit construction of the answer'],
   [('Maximum AND or XOR of a pair', [('Maximum AND', 'CF', '1721D'), ('Maximum XOR of Two Numbers in an Array', 'LC', '421')]),
    ('Answer bit by bit with a check', []),
    ('Minimising under constraints', []),
    ('Lexicographic bit choices', [])])

ch(8, 'per_bit', 'Per-Bit Independence and Contribution', 2, '1400', '2200',
   'Bitwise operations act on each bit separately: solve 30 one-bit problems and add $2^i$ times each answer.',
   ['Independence of bits under AND, OR and XOR (and why $+$ breaks it)',
    'Sum of XOR, AND or OR over all pairs by counting ones and zeros per bit',
    'Sums of products of bitwise expressions (bit $i$ times bit $j$)',
    'Choosing $x$ to optimise $\\sum (a_i \\oplus x)$ bit by bit',
    'Over subarrays the same idea is the array book (Chapter 15)'],
   [('Sums over pairs', [('Apollo versus Pan', 'CF', '1466E'), ('Find XOR Sum of All Pairs Bitwise AND', 'LC', '1835')]),
    ('Choosing x bit by bit', []),
    ('Products of bitwise expressions', []),
    ('Independent one-bit subproblems', [])])

ch(9, 'and_or', 'AND and OR: Monotonicity, Submasks and Closure', 2, '1400', '2300',
   'AND only loses bits and OR only gains them: monotone chains, the submask order, and the few distinct values they reach.',
   ['Monotonicity: $a \\,\\&\\, b \\le \\min(a, b)$, $a \\mid b \\ge \\max(a, b)$, and at most $\\log V$ changes along a chain',
    'The submask partial order: $a \\subseteq b \\iff a \\,\\&\\, b = a \\iff a \\mid b = b$',
    'AND and OR of a range of integers $[l, r]$: the common prefix',
    'Raising AND with a budget of increments; making OR reach a target',
    'Closure under OR: the set of values reachable by OR-ing elements'],
   [('Raising AND with a budget', [('Maximum And Queries (easy version)', 'CF', '1903D1')]),
    ('Submask relations', []),
    ('OR closure', []),
    ('AND chains', [])])

ch(10, 'equations', 'Bitwise Equations and Identities', 2, '1500', '2300',
   'Given $a + b$ and $a \\oplus b$, or $a \\mid b$ and $a \\,\\&\\, b$, find $a$ and $b$ --- or prove they cannot exist.',
   ['The four basic identities and what they determine',
    'Solving $a + b = s$, $a \\oplus b = x$: existence and the free bits',
    'Arrays with given sum and XOR: the 1--2--3 element constructions',
    'Equations bit by bit with carries (a carry chain as a tiny automaton)',
    'Counting solutions: $2^{\\text{free bits}}$ and corrections for zero'],
   [('Sum and XOR given', [('Plus and xor', 'CF', '76D'), ('Ehab the Xorcist', 'CF', '1325D')]),
    ('Equations with carries', []),
    ('Counting solutions', []),
    ('OR and AND given', [])])

ch(11, 'gray', 'Gray Codes, Hypercubes and Bit Orders', 2, '1500', '2400',
   'Order the $2^n$ masks so that neighbours differ in one bit: Gray codes, walks on the hypercube and other useful bit orders.',
   ['The reflected Gray code $g = x \\oplus (x \\gg 1)$ and its inverse',
    'The hypercube $Q_n$: neighbours differ in one bit; Hamiltonian cycles from Gray codes',
    'Operations that flip one bit under rules: the Gray-code rank as the answer',
    'Bit-reversal permutation (the FFT order) and other orders of masks',
    'Hamming balls and distances on the hypercube'],
   [('Gray-code ranks', [('Minimum One Bit Operations to Make Integers Zero', 'LC', '1611')]),
    ('Walks on the hypercube', [('Magical Permutation', 'CF', '1163E')]),
    ('Bit-reversal and other orders', []),
    ('Hamming balls', [])])

ch(12, 'number_theory', 'Bits in Number Theory', 2, '1500', '2400',
   'Binary meets arithmetic: parity of binomial coefficients, powers of two, the 2-adic valuation and binary algorithms for gcd and powers.',
   ['The 2-adic valuation $\\nu_2(n)$, trailing zeros, and odd parts',
    "Lucas mod 2 and Kummer: $\\binom{n}{k}$ is odd iff $k \\subseteq n$; carries count the factors of 2",
    "Legendre's formula in base 2: $\\nu_2(n!) = n - \\operatorname{popcount}(n)$",
    'Binary exponentiation and binary multiplication (the algebra is the math notes)',
    'The binary GCD (Stein) and why it uses only shifts and subtraction'],
   [('Parity of binomial coefficients', []),
    ('Powers of two and odd parts', []),
    ('Factorials in base 2', []),
    ('Binary algorithms', [])])

# ------------------------------------------------------------------ PART III
ch(13, 'prefix_xor', 'Prefix XOR and XOR Hashing', 3, '1500', '2300',
   'Prefix XOR turns every subarray XOR into one XOR of two prefixes, and random values turn parity questions into XOR checks.',
   ['Prefix XOR: $x(l..r) = P_r \\oplus P_{l-1}$; counting equal prefixes',
    'XOR along tree paths: $d(u) \\oplus d(v)$ (the tree itself is the tree book)',
    'Zobrist hashing: a random 64-bit value per element, XOR for sets with parity',
    'Every element appears an even number of times: XOR hashing of multisets (array multisets are the array book)',
    'Collision probability and choosing the generator'],
   [('Equal prefix XORs', [('Vampiric Powers, anyone?', 'CF', '1847C'), ('Maximum Xor Subarray', 'CSES', '1655')]),
    ('XOR on tree paths', [("SlavicG's Favorite Problem", 'CF', '1760G')]),
    ('Zobrist hashing', []),
    ('Parity of occurrences', [])])

ch(14, 'linear_basis', 'Linear Basis over GF(2) (XOR Basis)', 3, '1900', '2700',
   'Every set of integers spans a vector space over GF(2): a basis of at most 60 numbers answers every XOR-subset question.',
   ['Integers as vectors over GF(2); span, independence and rank',
    'Inserting into a basis by Gaussian reduction from the top bit; the reduced (row echelon) form',
    'Maximum XOR of a subset, $k$-th smallest XOR, and whether $x$ is representable',
    'Counting subsets with a given XOR: $2^{n - r}$',
    'Prefix bases with positions (the latest-index trick) for range queries; merging bases'],
   [('Maximum and representable XOR', [('Mahmoud and Ehab and yet another xor task', 'CF', '959F')]),
    ('Rank and counting', [('(Zero XOR Subset)-less', 'CF', '1101G')]),
    ('Bases of prefixes and ranges', [('Ivan and Burgers', 'CF', '1100F')]),
    ('Merging bases', []),
    ('k-th smallest XOR', [])],
   r'XOR paths and the cycle space of a graph are the graph book; this chapter owns the basis itself.')

ch(15, 'gf2', 'Gaussian Elimination and Matrices over GF(2)', 3, '2000', '2800',
   'Systems of XOR equations are linear algebra mod 2: eliminate with bitsets, count solutions, solve Lights Out.',
   ['Linear systems over GF(2); augmented matrices stored as bitsets',
    'Gaussian elimination in $\\bigO{n^3/64}$; rank, free variables and the number of solutions',
    'Lights Out and other toggle puzzles as linear systems',
    'Matrix multiplication and powers over GF(2) with bitsets',
    'Determinants and invertibility mod 2'],
   [('XOR systems', []), ('Toggle puzzles', []), ('Counting solutions', []), ('Matrix powers over GF(2)', [])])

ch(16, 'fwht', 'Walsh--Hadamard and Bitwise Convolutions', 3, '2200', '3200',
   'Convolve two arrays indexed by masks with XOR, AND or OR in $\\bigO{n \\log n}$: the Walsh--Hadamard, zeta and M\\"obius transforms.',
   ['XOR convolution: the Walsh--Hadamard transform and its inverse',
    'AND and OR convolutions: superset and subset sums as transforms (SOS DP is the DP book)',
    'Working modulo a prime, and the division by $n$',
    'Counting pairs and triples with a given XOR over huge multiplicities',
    'Subset convolution is the DP book; this chapter owns the three bitwise transforms'],
   [('XOR convolution', [('Bitwise Xor Convolution', 'LIB', 'bitwise_xor_convolution')]),
    ('AND and OR convolutions', [('Bitwise And Convolution', 'LIB', 'bitwise_and_convolution')]),
    ('Transforms in counting problems', [('Triple', 'CF', '1119H')]),
    ('Powers of a transform', [])])

ch(17, 'nim', 'Nim and XOR Games', 3, '1500', '2600',
   'A Nim position is lost exactly when the XOR of its piles is zero: Nim, its variants, and why games add by XOR.',
   ['Nim: the XOR rule and its proof',
    'Staircase Nim, Moore\'s Nim$_k$ (sums of bits modulo $k+1$) and misere Nim',
    'Sums of games: Grundy values combine by XOR (Sprague--Grundy theory is the math notes)',
    'Computing Grundy values for small games and spotting their bit patterns',
    'Games on arrays without Nim structure are the array book'],
   [('Nim sums', []), ('Staircase Nim', []), ("Moore's Nim", []), ('Grundy values with bit patterns', [])])

# ------------------------------------------------------------------ PART IV
ch(18, 'mask_enum', 'Advanced Mask Enumeration', 4, '1600', '2400',
   'All submasks of all masks in $\\bigO{3^n}$, supersets, $k$-subsets and fast loops over set bits: enumeration done right.',
   ['Submask enumeration \\texttt{for (s = m; s; s = (s-1) \\& m)} and the $3^n$ bound',
    'Supersets and $k$-subsets (Gosper\'s hack)',
    'Iterating set bits with \\texttt{\\_\\_builtin\\_ctz}',
    'Brute force over masks with pruning for $n \\le 20$--$25$ (meet in the middle is the recursion book)',
    'When enumeration should become SOS or bitmask DP (the DP book)'],
   [('Submasks', [('Triples with Bitwise AND Equal To Zero', 'LC', '982')]),
    ('Brute force over masks', [('Preparing Olympiad', 'CF', '550B')]),
    ('k-subsets', []),
    ('Supersets', [])])

ch(19, 'masks_sets', 'Masks as Sets: Compatibility and Intersections', 4, '1600', '2500',
   'Conflicts, requirements and features as masks: compatibility tests in $\\bigO{1}$ and counting pairs by their intersection.',
   ['Compatibility masks: $a \\,\\&\\, b = 0$ means no conflict',
    'Requirement masks: $a \\subseteq b$ means every need is met',
    'Counting compatible pairs over a small universe by frequency of masks',
    'Feature masks of strings, positions and items',
    'Adjacency masks of small graphs (hard graph problems are the graph book)'],
   [('Compatibility', []), ('Requirements', []), ('Counting pairs by masks', []), ('Feature masks', [])])

ch(20, 'bitsets', 'Bitsets', 4, '1800', '2700',
   'Sixty-four booleans per machine word: \\texttt{std::bitset}, dynamic bitsets, and algorithms that run 64 times faster.',
   ['\\texttt{std::bitset}: shifts, \\texttt{count}, \\texttt{\\_Find\\_first}, \\texttt{\\_Find\\_next}; templated sizes',
    'Dynamic bitsets on \\texttt{vector<uint64\\_t>}',
    'Reachable sums with shifts; transitive closure with row ORs',
    'Counting by intersections of bitsets',
    'The cost model: when $n^2/64$ or $n^3/64$ fits (applications on arrays and strings are those books)'],
   [('Reachable sums', []), ('Intersections', [('Alex and a TV Show', 'CF', '1097F')]), ('Transitive closure', []), ('Dynamic bitsets', [])])

ch(21, 'swar', 'Word-Level Parallelism and Low-Level Bit Hacks', 4, '1800', '2800',
   'Treat one 64-bit word as eight bytes or 32 pairs: parallel popcount, broadcasts, byte search and bit interleaving.',
   ['SWAR popcount and parity',
    'Broadcast, compare and find-zero-byte tricks',
    'Bit interleaving: Morton (Z-order) codes and their uses',
    'Bit deposit and extract (\\texttt{pdep}/\\texttt{pext}) and portable fallbacks',
    'Bit-sliced arithmetic: adding 64 small numbers at once'],
   [('Parallel counting', []), ('Byte tricks', []), ('Morton codes', []), ('Bit-sliced arithmetic', [])])

ch(22, 'bit_trees', 'Bit Trees: 64-ary Fast Sets and Packed Structures', 4, '1900', '2700',
   'A tree of 64-bit words answers insert, erase, successor and predecessor in $\\bigO{\\log_{64} n}$: faster than any balanced tree.',
   ['The 64-ary bit tree: one bit per child, \\texttt{ctz}/\\texttt{clz} to descend',
    'Successor and predecessor queries; comparison with \\texttt{std::set}',
    'Packed boolean arrays and bit-packed small integers',
    'Van Emde Boas trees as the idea behind it',
    'Lowbit inside Fenwick trees is the tree book'],
   [('Successor and predecessor', [('Predecessor Problem', 'LIB', 'predecessor_problem')]), ('Packed arrays', []), ('Fast sets in sweeps', []), ('Bit-tree tricks', [])])

# ------------------------------------------------------------------ PART V
ch(23, 'randomized', 'Randomisation and Hashing with Bits', 5, '1700', '2600',
   'Random 64-bit values, xorshift and splitmix: hash states and multisets, defeat anti-hash tests and sample with bits.',
   ['Pseudo-random generators: xorshift, splitmix64, \\texttt{mt19937\\_64}',
    'Custom hashes for \\texttt{unordered\\_map} that survive anti-hash tests',
    'Random masks: colour coding and random subsets',
    'Probabilistic counting and Bloom filters (the ideas)',
    'Collision bounds for $k$ random 64-bit values'],
   [('Robust hashing', []), ('Random masks', []), ('Hashing states', []), ('Probabilistic structures', [])])

ch(24, 'binary_prefix', 'Counting over Binary Prefixes', 5, '1600', '2500',
   'Count the numbers in $[0, n]$ with a bitwise property: walk the bits of $n$, fix a prefix, and count the free suffix in closed form.',
   ['Splitting $[0, n]$ into $\\operatorname{popcount}(n)+1$ blocks of free suffixes',
    'Counting $x \\le n$ with $x \\,\\&\\, k = 0$, given popcount, or given XOR with $m$',
    'Counting arrays or pairs whose AND/XOR compares to a bound, bit by bit',
    'Sum of popcounts and XORs over $[0, n]$',
    'When the counting needs a state (digit DP is the DP book)'],
   [('Free suffixes', [('Moamen and XOR', 'CF', '1557C')]), ('Bounded XOR', []), ('Sums over ranges', []), ('Counting pairs under a bound', [])])

ch(25, 'carries', 'Arithmetic Meets Bitwise: Carries and Borrows', 5, '1700', '2600',
   'When $+$, $-$ and bitwise operators meet, the carries decide everything: when $a + b = a \\mid b$, how $x - 1$ flips bits, and more.',
   ['Carry propagation and when $a + b = a \\oplus b$ (no carries: $a \\,\\&\\, b = 0$)',
    'Subtraction and borrows: $x - 1$, $x - y$ when $y \\subseteq x$',
    'Increment and decrement as bit patterns',
    'Inequalities between sums and bitwise results',
    'Problems mixing $\\max$, $+$ and XOR'],
   [('No-carry additions', []), ('Borrows', []), ('Mixed inequalities', []), ('Carry chains', [])])

ch(26, 'interactive', 'Interactive Bit Problems', 5, '1700', '2600',
   'Recover a hidden array or number with AND, OR, XOR or sum queries --- one bit of information at a time.',
   ['Query budgets in bits of information',
    'Recovering values from $a+b$, $a \\mid b$ and $a \\,\\&\\, b$ ($a + b = (a \\mid b) + (a \\,\\&\\, b)$)',
    'Finding a hidden number bit by bit',
    'XOR queries and parity',
    'Adaptive judges and checking the protocol locally'],
   [('Sums from OR and AND', [('Take a Guess', 'CF', '1556D')]), ('XOR queries', [('Bitwise Queries (Easy Version)', 'CF', '1451E1')]), ('Bit by bit', []), ('Budgets and adversaries', [])])

ch(27, 'constructive', 'Constructive Bit Problems', 5, '1500', '2600',
   'Build arrays, permutations or numbers with prescribed XOR, AND or OR values --- and prove the construction always works.',
   ['Reserve high bits to keep values distinct',
    'Fixing the total XOR with one or two extra elements',
    'Permutations with bitwise conditions on neighbours',
    'Impossibility from parity and the highest bit',
    'Checking constructions with a validator'],
   [('Fixing the total XOR', [('Even-Odd XOR', 'CF', '1722G')]), ('High-bit reservations', []), ('Bitwise permutations', []), ('Impossibility proofs', [])])

# ------------------------------------------------------------------ PART VI
ch(28, 'expert_basis', 'Expert XOR Algebra', 6, '2400', '3100',
   'Linear bases over ranges, trees and time, bases with deletions, and XOR problems that combine a basis with other structures.',
   ['Range bases online with the latest-position trick',
    'Bases on tree paths (with LCA from the tree book)',
    'Deletions: offline segment tree over time with a basis',
    'Bases combined with greedy and with counting',
    'Linear algebra over GF(2) beyond XOR bases'],
   [('Range bases', []), ('Bases on paths', []), ('Bases over time', []), ('Bases plus counting', [])])

ch(29, 'optimization', 'Bit-Level Optimisation and Performance', 6, '1800', '3000',
   'Make bit-heavy code fast: built-ins and target pragmas, branchless code, masks instead of modulo, and memory packing.',
   ['Compiler built-ins and \\texttt{\\#pragma GCC target("popcnt,bmi2")}',
    'Branchless min, max, abs and select with masks',
    'Power-of-two sizes: masking instead of modulo, ring buffers',
    'Packing data into bits to fit memory and cache',
    'Measuring: when a bit trick is worth it'],
   [('Built-ins', []), ('Branchless code', []), ('Packing', []), ('Measuring', [])])

ch(30, 'expert_mixed', 'Expert Mixed Bit Problems', 6, '2400', '3300',
   'Problems that combine three or four bit ideas; no chapter title tells you which.',
   ['A recognition guide: from the statement to the bit technique',
    'Greedy over bits combined with a basis or a trie',
    'Transforms combined with counting',
    'Bit tricks inside other algorithms',
    'Testing bit-heavy code'],
   [('Mixed set 2400', []), ('Mixed set 2600', []), ('Mixed set 2800', []), ('Mixed set 3000+', [])])

ch(31, 'mastery', 'Contest Mastery', 6, '2600', '3500',
   'No hints about the technique: recognise the bit structure, pick the tool, and combine chapters under contest pressure.',
   ['Reading limits like a contestant: $n \\le 20$, $V \\le 2^{20}$, $V \\le 2^{60}$',
    'Proof habits: highest-bit arguments, independence, linear algebra',
    'Implementation pitfalls: \\texttt{1 << 31}, signed shifts, precedence',
    'Stress testing bit solutions',
    'Mixed contest sets by rating'],
   [('Mixed set 2800', []), ('Mixed set 3000', []), ('Mixed set 3200', []), ('Mixed set 3400+', [])])


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
    short = {1: 'Foundations', 2: 'Core', 3: 'XOR algebra', 4: 'Masks \\& words', 5: 'Problem solving', 6: 'Advanced'}
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
          'any of the seven companion books (array, DP, graph, tree, string, recursion, stack--queue--heap).', '',
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
