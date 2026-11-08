import json
d = json.load(open('scripts/.cache/cf_problemset.json', encoding='utf-8'))
ps = d['result']['problems']
cands = {(868, 'F'), (812, 'E'), (220, 'B'), (13, 'E'), (246, 'E'), (375, 'D'),
         (617, 'E'), (86, 'D'), (940, 'F'), (1000, 'F'), (600, 'E'), (208, 'E'),
         (522, 'D'), (797, 'E'), (2134, 'C'), (2141, 'D'), (1900, 'E'),
         (1418, 'G'), (985, 'F'), (703, 'D'), (1284, 'E'), (692, 'E'),
         (547, 'E'), (981, 'E'), (1114, 'D'), (1527, 'F'), (570, 'G'),
         (1070, 'H'), (612, 'E'), (342, 'E'), (1076, 'F'), (1638, 'E')}
for p in ps:
    k = (p['contestId'], p['index'])
    if k in cands:
        print(k, repr(p['name']), p.get('rating'), p.get('tags'))
