"""check_varin.py -- every \\varin{url} must point to a problem that is in the book
(as a solved problem, exercise, practice problem or linked variant). Run from the dp_book folder."""
import glob
import re

texts = {f: open(f, encoding='utf8').read() for f in glob.glob('chapters/**/*.tex', recursive=True)}
VARIN = re.compile(r'\\varin\{([^}]*)\}')
URL = re.compile(r'\{(https?://[^}\s]+)\}')
targets = set()
for t in texts.values():
    targets |= set(URL.findall(VARIN.sub('', t)))
bad = 0
for f, t in sorted(texts.items()):
    for u in VARIN.findall(t):
        if u not in targets:
            bad += 1
            print('UNRESOLVED', f, u)
print('%d \\varin references checked, %d unresolved' % (sum(len(VARIN.findall(t)) for t in texts.values()), bad))
