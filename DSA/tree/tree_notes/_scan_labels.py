import os, re, sys
sys.stdout.reconfigure(encoding='utf-8')
pat = re.compile(r'\\bookchapter\s*(\[[^\]]*\])?\s*\{[^}]*\}\s*\{([^}]*)\}')
rows = []
for root, _, fs in os.walk('chapters'):
    for f in fs:
        if not f.endswith('.tex'):
            continue
        p = os.path.join(root, f)
        for ln in open(p, encoding='utf-8'):
            m = pat.search(ln)
            if m:
                rows.append((m.group(2), p))
for tag, p in sorted(rows):
    print(f'{tag:>6}  <-  {p}')
