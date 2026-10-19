"""verify_outline.py -- check every problem named in tools/outline.py against its judge
(name, rating / difficulty, premium flag) and against the companion DP book (no problem in both books).
Run from the graph_book folder:  python tools/verify_outline.py"""
import glob
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outline  # noqa: E402

DP_BOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'dp', 'notes', 'dp_book')
UA = {'User-Agent': 'Mozilla/5.0'}


def get(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=dict(UA, **(headers or {})))
    return urllib.request.urlopen(req, timeout=30).read().decode('utf8', 'replace')


def url_of(j, i):
    if j == 'CF':
        return 'codeforces.com/problemset/problem/%s/%s' % (re.match(r'\d+', i).group(), i[len(re.match(r'\d+', i).group()):])
    if j == 'LC':
        return None
    if j == 'CSES':
        return 'cses.fi/problemset/task/' + i
    if j == 'AC':
        return 'atcoder.jp/contests/%s/tasks/%s' % (i.split('_')[0], i)
    return None


cf = {str(p['contestId']) + p['index']: p for p in
      json.loads(get('https://codeforces.com/api/problemset.problems'))['result']['problems']}
lc_all = json.loads(get('https://leetcode.com/api/problems/all/'))['stat_status_pairs']
lc = {str(p['stat']['frontend_question_id']): p for p in lc_all}
dp_text = ''.join(open(f, encoding='utf8').read() for f in glob.glob(os.path.join(DP_BOOK, 'chapters', '**', '*.tex'), recursive=True))

bad = 0
for c in outline.CH:
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
                slug = p['stat']['question__title_slug'] if p else ''
                note = ['', 'Easy', 'Medium', 'Hard'][p['difficulty']['level']] if p else ''
                if p and p['paid_only']:
                    note += ' PREMIUM'
                if slug and ('leetcode.com/problems/%s/' % slug) in dp_text:
                    note += ' IN-DP-BOOK'
            elif j == 'CSES':
                m = re.search(r'<title>CSES - ([^<]*)', get('https://cses.fi/problemset/task/' + i))
                real = m.group(1) if m else 'MISSING'
            elif j == 'AC':
                m = re.search(r'<title>[A-Z0-9]+ - ([^<]*)', get('https://' + url_of(j, i)))
                real = m.group(1).strip() if m else 'MISSING'
            else:
                real = t
            u = url_of(j, i)
            if u and u in dp_text:
                note += ' IN-DP-BOOK'
            flag = '' if real.strip().lower() == t.strip().lower() else '   <-- name differs'
            if flag or 'MISSING' in real or 'PREMIUM' in note or 'IN-DP-BOOK' in note:
                bad += 1
            print('Ch%-2d %-5s %-12s %-45s | %s %s%s' % (c['num'], j, i, t[:45], real, note, flag))
print('\n%d problem(s) need attention' % bad)
