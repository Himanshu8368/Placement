"""Check every judge link in the book.

Link format (see STYLE.md):
    [CF 279B · Books](https://codeforces.com/problemset/problem/279/B){rating="1400"}
    [LC 167 · Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/){difficulty="Medium"}

Codeforces: the problem must exist, and the id, name and rating in the link must match the
problemset (cached at scripts/.cache/cf_problemset.json; --refresh downloads a new copy).
LeetCode: with --online, the slug must exist and the number, title and difficulty must match.
CSES and AtCoder: with --online, the task page must exist and its title must match the link name
(cached at scripts/.cache/judge_names.json).
Without --online only the format is checked.

Usage: python scripts/check_links.py [--online] [--refresh] [files...]
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmdlib import PROBLEMS, chapter_files, expand, problem_key_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "scripts" / ".cache" / "cf_problemset.json"
CF_API = "https://codeforces.com/api/problemset.problems"
LC_API = "https://leetcode.com/graphql"

LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)(?:\{([^}]*)\})?")
CF = re.compile(r"codeforces\.com/(?:problemset/problem|contest)/(\d+)/(?:problem/)?(\w+)")
LC = re.compile(r"leetcode\.com/problems/([a-z0-9-]+)/?$")
TEXT = re.compile(r"^(CF|LC|CSES|AC) (\S+) · (.+)$")
ATTR = re.compile(r'(\w+)="?([^"\s}]+)"?')


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def cf_problems(refresh: bool) -> dict:
    if refresh or not CACHE.exists():
        with urllib.request.urlopen(CF_API, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(data), encoding="utf-8")
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    probs = data["result"]["problems"] if "result" in data else data["problems"]
    return {(p["contestId"], p["index"]): p for p in probs}


def lc_question(slug: str) -> dict | None:
    body = json.dumps({
        "query": "query q($s: String!) { question(titleSlug: $s) { questionFrontendId title difficulty } }",
        "variables": {"s": slug},
    }).encode()
    req = urllib.request.Request(LC_API, data=body, headers={
        "Content-Type": "application/json", "Referer": "https://leetcode.com", "User-Agent": "Mozilla/5.0"})
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))["data"]["question"]
        except (TimeoutError, urllib.error.URLError) as exc:
            last = exc
    raise last


CSES = re.compile(r"cses\.fi/problemset/task/(\d+)")
AC = re.compile(r"atcoder\.jp/contests/(\w+)/tasks/(\w+)")
NAMES = ROOT / "scripts" / ".cache" / "judge_names.json"
TITLE = re.compile(r"<title>\s*(.*?)\s*</title>", re.S)


def page_name(url: str) -> str | None:
    """Problem name from a CSES or AtCoder task page (cached in scripts/.cache/judge_names.json).
    CSES titles read "CSES - Name"; AtCoder titles read "A - Name"."""
    cache = json.loads(NAMES.read_text(encoding="utf-8")) if NAMES.exists() else {}
    if url not in cache:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                page = r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        t = TITLE.search(page)
        title = t[1] if t else ""
        cache[url] = title.split(" - ", 1)[1].strip() if " - " in title else title
        NAMES.parent.mkdir(parents=True, exist_ok=True)
        NAMES.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache[url]


def links(files: list[Path]):
    for f in files:
        in_code = False
        # Registry files are checked on their own; chapters are not expanded here to avoid
        # reporting the same link twice (once in the problem file, once in its chapter).
        text = f.read_text(encoding="utf-8")
        for no, line in enumerate(text.splitlines(), 1):
            if line.startswith("```"):
                in_code = not in_code
            if in_code:
                continue
            for m in LINK.finditer(line):
                # Only CSES *problemset* links are judge links; the handbook PDF
                # (cses.fi/book/book.pdf) is reading material and is not name-checked.
                if (CF.search(m[2]) or "leetcode.com/problems" in m[2]
                        or "cses.fi/problemset" in m[2] or "atcoder.jp" in m[2]):
                    yield f, no, m[1], m[2], dict(ATTR.findall(m[3] or ""))


def duplicate_solved() -> list[str]:
    """Each problem is solved (marked `.solved`) in exactly one chapter, and each registry problem
    (`books/problems/<key>/problem.qmd`) is included in full by at most one chapter; other chapters
    use its card or its solution. Chapters are read with includes expanded. The appendix sample page
    is exempt, since it only demonstrates the layout."""
    errors: list[str] = []
    seen: dict[str, list[str]] = {}
    homes: dict[str, list[str]] = {}
    for f in chapter_files():
        text, included = expand(f)
        for inc in included:
            if inc.name == "problem.qmd":
                homes.setdefault(problem_key_of(inc), []).append(str(f.relative_to(ROOT)))
        for no, line in enumerate(text.splitlines(), 1):
            # A `Source:` line opens a full write-up, solved or illustrative: one per problem.
            if not line.startswith("Source: "):
                continue
            for m in LINK.finditer(line):
                seen.setdefault(m[2].rstrip("/").lower(), []).append(f"{f.relative_to(ROOT)} (line {no} expanded)")
                break
    errors += [f"{url} is written up in full more than once: {', '.join(where)} "
               f"(keep one write-up in the registry; include its card elsewhere)"
               for url, where in seen.items() if len(where) > 1]
    errors += [f"problem '{key}' is included in full by several chapters: {', '.join(where)} "
               f"(include its card.qmd or solution.qmd instead)" for key, where in homes.items() if len(where) > 1]
    return errors


def default_files() -> list[Path]:
    """Chapters (expanded later by `links`) plus registry problems and other pages."""
    out = chapter_files()
    out += sorted(PROBLEMS.glob("*/problem.qmd")) if PROBLEMS.exists() else []
    out += sorted((ROOT / "books" / "appendix").glob("*.qmd"))
    return [p for p in out if p.name != "problem_index.qmd"]


def main(argv: list[str]) -> int:
    online = "--online" in argv
    refresh = "--refresh" in argv
    args = [a for a in argv if not a.startswith("--")]
    files = [Path(a) for a in args] or default_files()
    cf = cf_problems(refresh)
    errors, checked = [], 0
    for f, no, text, url, attrs in links(files):
        checked += 1
        where = f"{f.relative_to(ROOT) if f.is_relative_to(ROOT) else f}:{no}"
        tm = TEXT.match(text)
        if not tm:
            errors.append(f"{where}: link text must be '<JUDGE> <id> · <name>': {text!r}")
            continue
        judge, pid, name = tm.groups()
        if m := CF.search(url):
            key = (int(m[1]), m[2])
            p = cf.get(key)
            if judge != "CF" or pid != f"{key[0]}{key[1]}":
                errors.append(f"{where}: text says {judge} {pid} but URL is CF {key[0]}{key[1]}")
            if p is None:
                errors.append(f"{where}: CF {key[0]}{key[1]} not found in the problemset (try --refresh)")
                continue
            if norm(p["name"]) != norm(name):
                errors.append(f"{where}: CF {pid} name is {p['name']!r}, link says {name!r}")
            want = str(p.get("rating", ""))
            if want and attrs.get("rating") != want:
                errors.append(f"{where}: CF {pid} rating is {want}, link says {attrs.get('rating')!r}")
        elif m := LC.search(url):
            if judge != "LC":
                errors.append(f"{where}: LeetCode URL but text says {judge}")
            if attrs.get("difficulty") not in ("Easy", "Medium", "Hard"):
                errors.append(f'{where}: LC link needs difficulty="Easy|Medium|Hard"')
            if online:
                q = lc_question(m[1])
                if q is None:
                    errors.append(f"{where}: LC slug {m[1]!r} does not exist")
                else:
                    if q["questionFrontendId"] != pid:
                        errors.append(f"{where}: LC {m[1]} is #{q['questionFrontendId']}, link says {pid}")
                    if norm(q["title"]) != norm(name):
                        errors.append(f"{where}: LC title is {q['title']!r}, link says {name!r}")
                    if q["difficulty"] != attrs.get("difficulty"):
                        errors.append(f"{where}: LC difficulty is {q['difficulty']}, link says {attrs.get('difficulty')}")
        elif "leetcode.com" in url:
            errors.append(f"{where}: LeetCode URL must be https://leetcode.com/problems/<slug>/")
        elif m := (CSES.search(url) or AC.search(url)):
            want_judge = "CSES" if "cses.fi" in url else "AC"
            if judge != want_judge:
                errors.append(f"{where}: URL is {want_judge} but text says {judge}")
            if online:
                real = page_name(url)
                if real is None:
                    errors.append(f"{where}: {url} does not exist")
                elif norm(real) != norm(name):
                    errors.append(f"{where}: {want_judge} name is {real!r}, link says {name!r}")
    errors += duplicate_solved()
    for e in errors:
        print("ERROR", e)
    print(f"{'OK  ' if not errors else 'FAIL'} {checked} judge links checked, {len(errors)} problems"
          + ("" if online else " (LeetCode format only; use --online to verify)"))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
