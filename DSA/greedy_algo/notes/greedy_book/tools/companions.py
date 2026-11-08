"""companions.py -- where the nine companion books live, and every problem URL they use.

A problem is "used" by a companion book if its URL appears in the book's chapters, or if it is planned in the
book's tools/outline.py (planned problems are reserved, so two books never pick the same one later).
"""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DSA = os.path.join(HERE, '..', '..', '..', '..')
BOOKS = {
    'array': os.path.join(DSA, 'array', 'notes', 'array_book'),
    'bs': os.path.join(DSA, 'binary_search', 'notes', 'bs_book'),
    'bit': os.path.join(DSA, 'bit_manipulation', 'notes', 'bit_book'),
    'dp': os.path.join(DSA, 'dp', 'notes', 'dp_book'),
    'graph': os.path.join(DSA, 'graph', 'notes', 'graph_book'),
    'tree': os.path.join(DSA, 'tree', 'notes', 'tree_book'),
    'string': os.path.join(DSA, 'string', 'notes', 'string_book'),
    'recursion': os.path.join(DSA, 'recursion', 'notes', 'recursion_book'),
    'sqh': os.path.join(DSA, 'stack_queue_heap', 'notes', 'sqh_book'),
}
URL = re.compile(r'https?://(?:www\.)?((?:leetcode\.com/problems/[\w-]+|codeforces\.com/problemset/problem/\d+/\w+|'
                 r'cses\.fi/problemset/task/\d+|atcoder\.jp/contests/\w+/tasks/\w+|judge\.yosupo\.jp/problem/\w+|'
                 r'spoj\.com/problems/\w+))')
# outline tuples: ('Title', 'LC', '123')
TUPLE = re.compile(r"\(\s*['\"][^'\"]*['\"]\s*,\s*['\"](CF|LC|CSES|AC|SPOJ|LIB)['\"]\s*,\s*['\"]([\w-]+)['\"]\s*\)")


def key_of(judge, pid):
    """Normalised key of a problem: 'LC:123', 'CF:1760C', 'CSES:1646', 'AC:abc138_d', 'SPOJ:AGGRCOW'."""
    return '%s:%s' % (judge, pid.upper() if judge in ('CF', 'SPOJ') else pid)


def key_of_url(u, lc_slug_to_id=None):
    m = re.match(r'codeforces\.com/problemset/problem/(\d+)/(\w+)', u)
    if m:
        return key_of('CF', m.group(1) + m.group(2))
    m = re.match(r'cses\.fi/problemset/task/(\d+)', u)
    if m:
        return key_of('CSES', m.group(1))
    m = re.match(r'atcoder\.jp/contests/\w+/tasks/(\w+)', u)
    if m:
        return key_of('AC', m.group(1))
    m = re.match(r'judge\.yosupo\.jp/problem/(\w+)', u)
    if m:
        return key_of('LIB', m.group(1))
    m = re.match(r'spoj\.com/problems/(\w+)', u)
    if m:
        return key_of('SPOJ', m.group(1))
    m = re.match(r'leetcode\.com/problems/([\w-]+)', u)
    if m:
        slug = m.group(1)
        if lc_slug_to_id and slug in lc_slug_to_id:
            return key_of('LC', lc_slug_to_id[slug])
        return 'LCSLUG:' + slug
    return 'URL:' + u


def lc_slug_map():
    """{leetcode slug: frontend id}, from the LeetCode API, cached in tools/.lc_slugs.json."""
    import json
    import urllib.request
    cache = os.path.join(HERE, '.lc_slugs.json')
    try:
        req = urllib.request.Request('https://leetcode.com/api/problems/all/', headers={'User-Agent': 'Mozilla/5.0'})
        data = json.loads(urllib.request.urlopen(req, timeout=30).read().decode('utf8'))['stat_status_pairs']
        m = {p['stat']['question__title_slug']: str(p['stat']['frontend_question_id']) for p in data}
        json.dump(m, open(cache, 'w'))
        return m
    except Exception:
        return json.load(open(cache)) if os.path.exists(cache) else {}


def chapter_urls(root):
    s = set()
    for f in glob.glob(os.path.join(root, 'chapters', '**', '*.tex'), recursive=True):
        s |= set(URL.findall(open(f, encoding='utf8', errors='replace').read()))
    return s


def used_keys(lc_slug_to_id=None):
    """{book: set of problem keys} for every companion book (chapters + planned outline)."""
    out = {}
    for name, root in BOOKS.items():
        keys = {key_of_url(u, lc_slug_to_id) for u in chapter_urls(root)}
        for extra in ('tools/outline.py', 'frontmatter/roadmap.tex'):
            p = os.path.join(root, extra)
            if os.path.exists(p):
                txt = open(p, encoding='utf8', errors='replace').read()
                keys |= {key_of(j, i) for j, i in TUPLE.findall(txt)}
                keys |= {key_of_url(u, lc_slug_to_id) for u in URL.findall(txt)}
        out[name] = keys
    return out
