"""Compile every C++ listing in the book (```cpp blocks), so the C++ tabs are checked like the Python.

    python scripts/check_cpp.py                 # every chapter (includes expanded) and registry problem
    python scripts/check_cpp.py books/dp/229_dp_intro.qmd
    python scripts/check_cpp.py --jobs 8

A listing with `int main` is compiled as a full program; any other listing is compiled as a
translation unit (`-c`), after a prelude with <bits/stdc++.h>, `using namespace std;` and the
LeetCode node types (ListNode, TreeNode, Node). Listings that are deliberately not standalone
(fragments, pseudo-code) are skipped when their first line is `// fragment`.

Results are cached by content hash in scripts/.cache/cpp_ok.json, so only changed listings recompile.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmdlib import PROBLEMS, ROOT, chapter_files, expand  # noqa: E402

CPP = re.compile(r"^```cpp[^\n]*\n(.*?)^```\s*$", re.S | re.M)
CACHE = ROOT / "scripts" / ".cache" / "cpp_ok.json"
PRELUDE = r"""#include <bits/stdc++.h>
using namespace std;
using ll = long long;
struct ListNode { int val; ListNode *next; ListNode(int x = 0, ListNode* n = nullptr) : val(x), next(n) {} };
struct TreeNode { int val; TreeNode *left, *right;
  TreeNode(int x = 0, TreeNode* l = nullptr, TreeNode* r = nullptr) : val(x), left(l), right(r) {} };
"""


def compiler() -> str | None:
    for c in ("g++", "clang++"):
        if shutil.which(c):
            return shutil.which(c)
    for p in (r"C:\MinGW\bin\g++.exe", r"C:\msys64\mingw64\bin\g++.exe"):
        if Path(p).exists():
            return p
    return None


def listings(files: list[Path]):
    for f in files:
        text = expand(f)[0] if f.parent.parent != PROBLEMS else f.read_text(encoding="utf-8")
        for k, m in enumerate(CPP.finditer(text), 1):
            code = m.group(1)
            line = text[: m.start()].count("\n") + 1
            yield f, k, line, code


STRUCTURED_BINDING = re.compile(r"\bauto\s*&{0,2}\s*\[")


def gcc_major(cxx: str) -> int:
    try:
        out = subprocess.run([cxx, "-dumpversion"], capture_output=True, text=True).stdout.strip()
        return int(out.split(".")[0])
    except (OSError, ValueError):
        return 99


def compile_one(cxx: str, code: str, major: int = 99) -> tuple[bool | None, str]:
    """True = compiles, False = error, None = cannot be judged by this (old) compiler."""
    if code.lstrip().startswith("// fragment"):
        return True, "skipped (fragment)"
    if major < 7 and STRUCTURED_BINDING.search(code):
        return None, "uses C++17 structured bindings; needs GCC >= 7"
    has_main = re.search(r"\bint\s+main\s*\(", code) is not None
    src = code if "#include" in code else PRELUDE + code
    if "#include" in code and not has_main:
        src = PRELUDE + code
    with tempfile.TemporaryDirectory() as d:
        cpp = Path(d) / "x.cpp"
        cpp.write_text(src, encoding="utf-8")
        args = [cxx, "-std=c++17", "-O0", "-w", "-fsyntax-only" if not has_main else "-c", str(cpp)]
        if has_main:
            args += ["-o", str(Path(d) / "x.o")]
        r = subprocess.run(args, capture_output=True, text=True, timeout=120)
        msg = (r.stderr or r.stdout).replace(str(cpp), "listing.cpp")
        return r.returncode == 0, msg


def main(argv: list[str]) -> int:
    jobs = 6
    if "--jobs" in argv:
        jobs = int(argv[argv.index("--jobs") + 1])
        argv = [a for i, a in enumerate(argv) if a != "--jobs" and (i == 0 or argv[i - 1] != "--jobs")]
    files = [Path(a).resolve() for a in argv if not a.startswith("--")] or chapter_files()
    cxx = compiler()
    if not cxx:
        print("SKIP no C++ compiler found (install g++ / MinGW)")
        return 0
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    work, failures, total = [], [], 0
    seen = set()
    for f, k, line, code in listings(files):
        total += 1
        h = hashlib.sha1((PRELUDE + code).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        if cache.get(h):
            continue
        work.append((f, k, line, code, h))
    major = gcc_major(cxx)
    unverified = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for (f, k, line, code, h), (ok, msg) in zip(work, ex.map(lambda w: compile_one(cxx, w[3], major), work)):
            if ok:
                cache[h] = True
            elif ok is None:
                unverified += 1
            else:
                first = "\n".join(l for l in msg.splitlines() if "error" in l)[:600]
                failures.append(f"{f.relative_to(ROOT)} listing {k} (line {line} of the expanded text):\n{first}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache))
    for fl in failures:
        print("FAIL", fl, "\n")
    note = (f", {unverified} unverified (C++17 structured bindings need GCC >= 7; this is GCC {major})"
            if unverified else "")
    print(f"{'OK  ' if not failures else 'FAIL'} {total} C++ listings, {len(work)} compiled now, "
          f"{len(failures)} failed{note}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
