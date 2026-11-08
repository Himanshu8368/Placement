import re, glob
urls = []
for f in glob.glob(r'd:\Prep\DSA\tree\tree_notes\chapters\**\*.tex', recursive=True):
    for m in re.findall(r'https?://[^\s}]+', open(f, encoding='utf-8', errors='replace').read()):
        urls.append((m.rstrip('\\'), f.split('\\')[-2] + '/' + f.split('\\')[-1]))
cf = sorted(set(u for u, _ in urls if 'codeforces' in u))
lc = sorted(set(u for u, _ in urls if 'leetcode' in u))
print('TOTAL matches:', len(urls), '| unique CF:', len(cf), '| unique LC:', len(lc))
print('--- CF ---')
for u in cf: print(u)
print('--- LC ---')
for u in lc: print(u)
# duplicate check
from collections import Counter
c = Counter(u for u, _ in urls)
dups = {k: v for k, v in c.items() if v > 1}
print('--- DUPLICATES ---')
for k, v in dups.items(): print(v, k)
