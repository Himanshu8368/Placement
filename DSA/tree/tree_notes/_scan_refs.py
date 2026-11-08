import os, re, sys
sys.stdout.reconfigure(encoding='utf-8')
pat = re.compile(r'(Chapter|Ch\.)\s*~?\s*[0-9]{1,2}')
for d in ['chapters', 'frontmatter', 'appendices', 'backmatter']:
    for root, _, fs in os.walk(d):
        for f in sorted(fs):
            if not f.endswith('.tex'):
                continue
            fn = os.path.join(root, f)
            for i, ln in enumerate(open(fn, encoding='utf-8'), 1):
                s = ln.strip()
                if s.startswith('%') or s.startswith('//'):
                    continue
                if pat.search(ln) and '\\chref' not in ln:
                    print(f'{fn}:{i}: {s}')
