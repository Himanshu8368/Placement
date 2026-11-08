"""Generate appendix/problem_index.tex from the problems linked in the chapters.

Every solved problem (\\problemcard), exercise (\\exprob) and practice problem (\\problembox)
is listed once, grouped by judge, with its difficulty badges, its place in the book and a
clickable page reference (the macros set \\label{pb:<url>}).

Usage (from the book root):
    python tools/gen_problem_index.py                       -> appendix/problem_index.tex (all chapters)
    python tools/gen_problem_index.py 1 6 problem_index_part1   -> chapters 1..6 only (for a volume)
Re-run it whenever problems are added; then rebuild the book (tools/build.sh does both).
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRST, LAST = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) >= 3 else (0, 999)
OUT = os.path.join(ROOT, 'appendix', (sys.argv[3] if len(sys.argv) >= 4 else 'problem_index') + '.tex')


def macro_args(text, pos, n):
    """Read n brace-delimited arguments starting at text[pos] (skipping spaces)."""
    args = []
    for _ in range(n):
        while pos < len(text) and text[pos] in ' \n\t':
            pos += 1
        if pos >= len(text) or text[pos] != '{':
            return None, pos
        depth, start = 0, pos
        while pos < len(text):
            c = text[pos]
            if c == '\\':
                pos += 2
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        args.append(text[start + 1:pos])
        pos += 1
    return args, pos


def badges(text):
    lc = re.findall(r'\\lc\{(Easy|Medium|Hard)\}', text)
    cf = re.findall(r'\\cf\{(\d+)\}', text)
    book = re.findall(r'\\ex(easy|medium|hard|expert)\b', text)
    return (lc[0] if lc else None), (int(cf[0]) if cf else None), (book[0] if book else None)


def judge_of(url, source):
    m = re.search(r'codeforces\.com/problemset/problem/(\d+)/(\w+)', url)
    if m:
        return 'Codeforces', m.group(1) + m.group(2), (int(m.group(1)), m.group(2))
    m = re.search(r'leetcode\.com/problems/([\w-]+)', url)
    if m:
        num = re.search(r'LeetCode\s+(\d+)', source)
        n = int(num.group(1)) if num else 99999
        return 'LeetCode', str(n) if num else m.group(1), (n, m.group(1))
    m = re.search(r'atcoder\.jp/contests/(\w+)/tasks/(\w+)', url)
    if m:
        ident = m.group(2).replace('_', ' ').upper().replace('TYPICAL90', 'T90')
        return 'AtCoder', ident, (m.group(1), m.group(2))
    m = re.search(r'cses\.fi/problemset/task/(\d+)', url)
    if m:
        return 'CSES', m.group(1), (int(m.group(1)),)
    m = re.search(r'geeksforgeeks\.org/problems/([\w-]+?)(\d*)/', url)
    if m:
        return 'GeeksforGeeks', m.group(1).replace('-problem', ''), (m.group(1),)
    m = re.search(r'onlinejudge\.org/external/\d+/(\d+)\.pdf', url)
    if m:
        return 'Other', 'UVa ' + m.group(1), (url,)
    m = re.search(r'spoj\.com/problems/([A-Za-z0-9_]+)', url)
    if m:
        return 'Other', 'SPOJ ' + m.group(1).replace('_', ' '), (url,)
    return 'Other', url, (url,)


def chapter_of(path):
    m = re.search(r'ch0*(\d+)_', path.replace('\\', '/'))
    return int(m.group(1)) if m else 0


entries = []
for f in sorted(glob.glob(os.path.join(ROOT, 'chapters', '**', '*.tex'), recursive=True)):
    text = open(f, encoding='utf8').read()
    ch = chapter_of(f)
    if not FIRST <= ch <= LAST:
        continue
    rel = f.replace('\\', '/')
    # solved problems
    for m in re.finditer(r'\\problemcard(?=\{)', text):
        a, _ = macro_args(text, m.end(), 6)
        if not a:
            continue
        num, title, source, url, diff, _tech = a
        lc, cf, book = badges(diff)
        entries.append(dict(ch=ch, role='Problem ' + num, kind=0, title=title, source=source, url=url,
                            lc=lc, cf=cf, book=book))
    # practice problems
    for m in re.finditer(r'\\problembox(?=\{)', text):
        a, _ = macro_args(text, m.end(), 4)
        if not a:
            continue
        pid, head, _hint, url = a
        title = re.sub(r'\\ex(easy|medium|hard|expert)\\ ', '', head).split(' --- ')[0].strip()
        lc, cf, book = badges(head)
        entries.append(dict(ch=ch, role='Practice ' + pid, kind=2, title=title, source=head, url=url,
                            lc=lc, cf=cf, book=book))
    # variants of solved problems: \varlink{title}{source}{url} inside \variant{...}{<where>}...
    for m in re.finditer(r'\\varlink(?=\{)', text):
        a, _ = macro_args(text, m.end(), 3)
        if not a:
            continue
        title, source, url = a
        cards = [macro_args(text, c.end(), 1)[0] for c in re.finditer(r'\\problemcard(?=\{)', text[:m.start()])]
        parent = cards[-1][0] if cards and cards[-1] else '?'
        line = text[text.rfind('\n', 0, m.start()) + 1:m.start()]
        lc, cf, book = badges(line)
        entries.append(dict(ch=ch, role='Variant of P' + parent, kind=3, title=title, source=source, url=url,
                            lc=lc, cf=cf, book=book))
    # exercises: number them in order of \item ... \exprob / \exown inside the exercise list
    if rel.endswith('/exercises.tex'):
        pm = re.search(r'/p(\d+)_', rel)
        pat = int(pm.group(1)) if pm else 0
        k = 0
        for line in text.split('\n'):
            if re.match(r'\s*\\item', line) and ('\\exprob' in line or '\\exown' in line):
                k += 1
                if '\\exprob' in line:
                    i = line.index('\\exprob') + len('\\exprob')
                    a, _ = macro_args(line, i, 3)
                    if not a:
                        continue
                    title, source, url = a
                    lc, cf, book = badges(line)
                    entries.append(dict(ch=ch, role='Exercise E%d.%d.%d' % (ch, pat, k), kind=1, title=title,
                                        source=source, url=url, lc=lc, cf=cf, book=book))

CF_RANKS = [(1200, 'cfgray'), (1400, 'cfgreen'), (1600, 'cfcyan'), (1900, 'cfblue'),
            (2100, 'cfviolet'), (2400, 'cforange'), (10**9, 'cfred')]


def diff_cell(e):
    parts = []
    if e['book']:
        parts.append('\\ex' + e['book'])
    if e['cf']:
        parts.append('\\cf{%d}' % e['cf'])
    if e['lc']:
        parts.append('\\lc{%s}' % e['lc'])
    return '\\,'.join(parts)


def tex_escape_title(t):
    t = re.sub(r'\\p(classic|concept|recent|tricky)\s*', '', t)
    return t.replace('&', '\\&').replace('#', '\\#')


groups = {}
for e in entries:
    judge, ident, key = judge_of(e['url'], e['source'])
    e['judge'], e['ident'], e['key'] = judge, ident, key
    groups.setdefault(judge, []).append(e)

seen = {}
for e in entries:
    seen.setdefault(e['url'], []).append(e['role'])
dups = {u: r for u, r in seen.items() if len(r) > 1}

order = ['LeetCode', 'Codeforces', 'AtCoder', 'CSES', 'GeeksforGeeks', 'Other']
lines = []
lines.append('%' + '=' * 60)
lines.append('% APPENDIX — PROBLEM INDEX  (generated by tools/gen_problem_index.py; do not edit by hand)')
lines.append('%' + '=' * 60)
lines.append('\\chapter{Problem Index}')
lines.append('\\label{ch:problem-index}')
lines.append('')
total = len(entries)
counts = {g: len(groups.get(g, [])) for g in order}
lines.append('Every problem in the book appears exactly once: as a fully solved problem, an exercise of a pattern, '
             'a practice-set problem, or a variant of a solved problem. %d problems in total --- ' % total +
             ', '.join('%d from %s' % (counts[g], g) for g in order if counts[g]) + '. '
             'Click a title to open the problem on its judge; click the page number to jump to it in the book. '
             'Codeforces problems are sorted by rating, the others by their number.')
lines.append('')
lines.append('\\begin{center}\\small')
lines.append('\\textbf{Where:} \\textit{Problem} = fully solved (four approaches) \\quad '
             '\\textit{Exercise} = end of a pattern, full solution \\quad '
             '\\textit{Practice} = chapter problem set, design sketch \\\\ '
             '\\textit{Variant} = one change to a solved problem: subproblem and recurrence in the Variants box')
lines.append('\\end{center}')
for g in order:
    es = groups.get(g)
    if not es:
        continue
    if g == 'Codeforces':
        es.sort(key=lambda e: (e['cf'] or 0, e['key']))
    else:
        es.sort(key=lambda e: e['key'])
    lines.append('')
    lines.append('\\section*{%s \\normalsize\\mdseries(%d problems)}' % (g, len(es)))
    lines.append('\\addcontentsline{toc}{section}{%s}' % g)
    lines.append('{\\small')
    lines.append('\\begin{longtable}{@{}p{1.5cm} p{5.6cm} p{3.4cm} p{3.0cm} r@{}}')
    lines.append('\\toprule')
    head = '\\textbf{ID} & \\textbf{Problem} & \\textbf{Difficulty} & \\textbf{Where} & \\textbf{Page} \\\\'
    lines.append(head)
    lines.append('\\midrule\\endfirsthead')
    lines.append('\\toprule ' + head + ' \\midrule\\endhead')
    lines.append('\\bottomrule\\endlastfoot')
    for e in es:
        title = '\\href{%s}{%s}' % (e['url'], tex_escape_title(e['title']))
        lines.append('%s & %s & %s & {\\scriptsize Ch.~%d $\\cdot$ %s} & \\pageref{pb:%s} \\\\' % (
            e['ident'], title, diff_cell(e), e['ch'], e['role'], e['url']))
    lines.append('\\end{longtable}}')
lines.append('')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w', encoding='utf8').write('\n'.join(lines))
print('wrote', OUT, '-', total, 'problems:', counts)
if dups:
    print('WARNING: problems listed more than once:')
    for u, r in dups.items():
        print('  ', u, r)

# Same task on different judges (tools/known_duplicates.txt): at most one member of each group may appear.
KD = os.path.join(ROOT, 'tools', 'known_duplicates.txt')
if os.path.exists(KD):
    for line in open(KD, encoding='utf8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        members = [m.strip() for m in line.split(' = ')]
        present = sorted({m for m in members for u in seen if m in u})
        if len(present) > 1:
            print('WARNING: same task under different names:', ' = '.join(present))
