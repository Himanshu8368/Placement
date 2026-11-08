"""verify_outline.py -- check every problem named in tools/outline.py against its judge
(name, rating / difficulty, premium flag), against the seven companion books (no problem in two books),
and for duplicates inside this book.
Run from the bit_book folder:  python tools/verify_outline.py [first_chapter last_chapter]"""
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outline  # noqa: E402
import companions  # noqa: E402

UA = {'User-Agent': 'Mozilla/5.0'}
lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (1, 99)


def get(url):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=30).read().decode('utf8', 'replace')


cf = {str(p['contestId']) + p['index']: p for p in
      json.loads(get('https://codeforces.com/api/problemset.problems?lang=en'))['result']['problems']}
lc_all = json.loads(get('https://leetcode.com/api/problems/all/'))['stat_status_pairs']
lc = {str(p['stat']['frontend_question_id']): p for p in lc_all}
slug2id = {p['stat']['question__title_slug']: str(p['stat']['frontend_question_id']) for p in lc_all}
used = companions.used_keys(slug2id)
# the DP book has no outline.py: its planned problems are named in its roadmap
dp_road = open(os.path.join(companions.BOOKS['dp'], 'frontmatter', 'roadmap.tex'), encoding='utf8').read().lower()

bad = 0
seen = {}
for c in outline.CH:
    if not lo <= c['num'] <= hi:
        continue
    for name, probs in c['patterns']:
        for t, j, i in probs:
            note = ''
            if j == 'CF':
                p = cf.get(i)
                real = p['name'] if p else 'MISSING'
                note = 'rating %s' % (p.get('rating') if p else '-')
            elif j == 'LC':
                p = lc.get(i)
                real = p['stat']['question__title'] if p else 'MISSING'
                note = ['', 'Easy', 'Medium', 'Hard'][p['difficulty']['level']] if p else ''
                if p and p['paid_only']:
                    note += ' PREMIUM'
            elif j == 'CSES':
                m = re.search(r'<title>CSES - ([^<]*)', get('https://cses.fi/problemset/task/' + i))
                real = m.group(1) if m else 'MISSING'
            elif j == 'AC':
                m = re.search(r'<title>[A-Z0-9]+ - ([^<]*)', get('https://atcoder.jp/contests/%s/tasks/%s' % (i.split('_')[0], i)))
                real = m.group(1).strip() if m else 'MISSING'
            else:
                real = t
            k = companions.key_of(j, i)
            books = [b for b, s in used.items() if k in s]
            if t.lower() in dp_road and 'dp' not in books:
                books.append('dp?(name in roadmap)')
            if books:
                note += ' IN-BOOK:' + ','.join(books)
            if k in seen:
                note += ' DUPLICATE of Ch%d' % seen[k]
            seen[k] = c['num']
            flag = '' if real.strip().lower() == t.strip().lower() else '   <-- name differs'
            if flag or 'MISSING' in real or 'PREMIUM' in note or 'IN-BOOK' in note or 'DUPLICATE' in note:
                bad += 1
                mark = '!!'
            else:
                mark = '  '
            print('%s Ch%-2d %-5s %-8s %-45s | %s %s%s' % (mark, c['num'], j, i, t[:45], real, note, flag))
print('\n%d problem(s) need attention' % bad)
