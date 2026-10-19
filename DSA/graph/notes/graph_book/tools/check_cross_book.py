"""check_cross_book.py -- no problem may be used in both this book and the companion DP book.
Lists every problem URL that appears in the chapters of both books. Run from the graph_book folder."""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BOOKS = {'graph': os.path.join(HERE, '..'),
         'dp': os.path.join(HERE, '..', '..', '..', '..', 'dp', 'notes', 'dp_book')}
URL = re.compile(r'https?://(?:www\.)?((?:leetcode\.com/problems/[\w-]+|codeforces\.com/problemset/problem/\d+/\w+|'
                 r'cses\.fi/problemset/task/\d+|atcoder\.jp/contests/\w+/tasks/\w+))')

urls = {}
for name, root in BOOKS.items():
    s = set()
    for f in glob.glob(os.path.join(root, 'chapters', '**', '*.tex'), recursive=True):
        s |= set(URL.findall(open(f, encoding='utf8').read()))
    urls[name] = s
both = sorted(urls['graph'] & urls['dp'])
for u in both:
    print('IN BOTH BOOKS:', u)
print('%d graph-book URLs, %d DP-book URLs, %d shared' % (len(urls['graph']), len(urls['dp']), len(both)))
