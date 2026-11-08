"""check_problem_mix.py -- is a chapter's practice set a good mix?

For every \\problembox in chapters/<chapter>/problem_set.tex it reads the level badge (\\exeasy, \\exmedium, \\exhard,
\\exstretch) and the kind tag (\\pclassic, \\pconcept, \\precent, \\ptricky), then reports
  * the number of problems of each kind, per tier and in total (AUTHORING.md 6b: >= 5 of each kind,
    >= 1 Recent and >= 1 Tricky per tier),
  * problems without exactly one kind tag,
  * Codeforces problems rated above the chapter's upper level (\\chapterfacts) that are NOT in the Stretch group,
    and Stretch problems that are not above it.
Usage (from array_book):  python tools/check_problem_mix.py chapters/ch01_introduction
"""
import os
import re
import sys

KINDS = ['pclassic', 'pconcept', 'precent', 'ptricky']
TIERS = ['exeasy', 'exmedium', 'exhard', 'exstretch']


def args(text, pos, n):
    out = []
    for _ in range(n):
        while text[pos] in ' \n\t':
            pos += 1
        depth, start = 0, pos
        while True:
            c = text[pos]
            if c == '\\':
                pos += 2
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        out.append(text[start + 1:pos])
        pos += 1
    return out


def main(folder):
    entry = [f for f in os.listdir(folder) if re.match(r'ch\d+\.tex$', f)][0]
    facts = re.search(r'\\chapterfacts\{([^}]*)\}\{([^}]*)\}', open(os.path.join(folder, entry), encoding='utf8').read())
    hi = int(facts.group(2)) if facts and facts.group(2).isdigit() else 10 ** 9
    text = open(os.path.join(folder, 'problem_set.tex'), encoding='utf8').read()
    table = {t: {k: 0 for k in KINDS} for t in TIERS}
    problems = 0
    issues = []
    for m in re.finditer(r'\\problembox(?=\{)', text):
        pid, head, _hint, url = args(text, m.end(), 4)
        problems += 1
        tier = [t for t in TIERS if '\\' + t in head]
        kinds = [k for k in KINDS if '\\' + k in head]
        cf = re.search(r'\\cf\{(\d+)\}', head)
        if len(tier) != 1:
            issues.append('%s: level badge missing or ambiguous' % pid)
            continue
        if len(kinds) != 1:
            issues.append('%s: needs exactly one kind tag (found %s)' % (pid, kinds or 'none'))
        else:
            table[tier[0]][kinds[0]] += 1
        if cf:
            r = int(cf.group(1))
            if tier[0] != 'exstretch' and r > hi:
                issues.append('%s: CF %d is above the chapter level %d -> move it to Stretch' % (pid, r, hi))
            if tier[0] == 'exstretch' and r <= hi:
                issues.append('%s: Stretch but CF %d is within the chapter level %d' % (pid, r, hi))
    print('%s: %d problems, chapter level up to %s' % (folder, problems, hi))
    print('%-10s' % 'tier' + ''.join('%10s' % k[1:] for k in KINDS))
    for t in TIERS:
        print('%-10s' % t[2:] + ''.join('%10d' % table[t][k] for k in KINDS))
    tot = {k: sum(table[t][k] for t in TIERS) for k in KINDS}
    print('%-10s' % 'total' + ''.join('%10d' % tot[k] for k in KINDS))
    for k in KINDS:
        if tot[k] < 5:
            issues.append('only %d %s problems (want >= 5)' % (tot[k], k[1:]))
    for t in TIERS[:3]:
        for k in ('precent', 'ptricky'):
            if sum(table[t].values()) and table[t][k] == 0:
                issues.append('tier %s has no %s problem' % (t[2:], k[1:]))
    for i in issues:
        print('  !', i)
    return 1 if issues else 0


if __name__ == '__main__':
    sys.exit(max(main(f) for f in sys.argv[1:]))
