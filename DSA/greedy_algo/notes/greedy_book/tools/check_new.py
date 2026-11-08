"""check_new.py -- vet candidate problems before putting them in the book.

For each candidate prints the judge's official name, the CF rating or LeetCode difficulty, PREMIUM, and whether the
problem is already used in this book (chapters or outline) or in a companion book (chapters or outline).

Usage (from the greedy_book folder):
    python tools/check_new.py LC:1480 CF:1520D CSES:1660 AC:abc138_d SPOJ:AGGRCOW
"""
import glob
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import companions  # noqa: E402
import outline  # noqa: E402

UA = {'User-Agent': 'Mozilla/5.0'}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read().decode('utf8', 'replace')


def main():
    cands = [a.split(':', 1) for a in sys.argv[1:]]
    need_cf = any(j == 'CF' for j, _ in cands)
    cf = {}
    if need_cf:
        cf = {str(p['contestId']) + p['index']: p for p in
              json.loads(get('https://codeforces.com/api/problemset.problems?lang=en'))['result']['problems']}
    lc_all = json.loads(get('https://leetcode.com/api/problems/all/'))['stat_status_pairs']
    lc = {str(p['stat']['frontend_question_id']): p for p in lc_all}
    slugs = {p['stat']['question__title_slug']: str(p['stat']['frontend_question_id']) for p in lc_all}
    used = companions.used_keys(slugs)
    here = {companions.key_of_url(u, slugs) for u in companions.chapter_urls(os.path.join(companions.HERE, '..'))}
    planned = {companions.key_of(j, i): c['num'] for c in outline.CH for _, ps in c['patterns'] for _, j, i in ps}
    for j, i in cands:
        j = j.upper()
        info, url = '', ''
        if j == 'CF':
            p = cf.get(i.upper())
            info = '%s | rating %s' % (p['name'], p.get('rating', '-')) if p else 'MISSING'
            m = re.match(r'(\d+)(\w+)', i)
            url = 'https://codeforces.com/problemset/problem/%s/%s' % (m.group(1), m.group(2).upper())
        elif j == 'LC':
            p = lc.get(i)
            if p:
                info = '%s | %s%s' % (p['stat']['question__title'], ['', 'Easy', 'Medium', 'Hard'][p['difficulty']['level']],
                                      ' | PREMIUM' if p['paid_only'] else '')
                url = 'https://leetcode.com/problems/%s/' % p['stat']['question__title_slug']
            else:
                info = 'MISSING'
        elif j == 'CSES':
            m = re.search(r'<title>CSES - ([^<]*)', get('https://cses.fi/problemset/task/' + i))
            info = m.group(1) if m else 'MISSING'
            url = 'https://cses.fi/problemset/task/' + i
        elif j == 'AC':
            url = 'https://atcoder.jp/contests/%s/tasks/%s' % (i.split('_')[0], i)
            m = re.search(r'<title>[A-Z0-9]+ - ([^<]*)', get(url))
            info = m.group(1).strip() if m else 'MISSING'
        elif j == 'SPOJ':
            url = 'https://www.spoj.com/problems/%s/' % i.upper()
        k = companions.key_of(j, i)
        flags = [b.upper() + '-BOOK' for b, s in used.items() if k in s]
        if k in here:
            flags.append('ALREADY-IN-THIS-BOOK')
        if k in planned:
            flags.append('PLANNED-SOLVED-CH%d' % planned[k])
        print('%-14s %s  %s  %s' % (k, info, url, ('<-- ' + ' '.join(flags)) if flags else 'ok'))


if __name__ == '__main__':
    main()
