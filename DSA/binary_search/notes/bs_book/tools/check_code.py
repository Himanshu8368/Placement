"""
check_code.py -- compile every complete C++ listing in the book.

Every  \\begin{lstlisting}[style=cpp] ... \\end{lstlisting}  block is extracted
and compiled with  g++ -std=c++17 -fsyntax-only  (falls back to c++14 if the
compiler is old). Fragments must use [style=snippet] instead, so they are skipped.

Usage (from the dp_book folder):
    python tools/check_code.py                 # all chapters
    python tools/check_code.py chapters/ch01_introduction
"""
import glob
import os
import re
import subprocess
import sys
import tempfile

ROOT = sys.argv[1] if len(sys.argv) > 1 else "chapters"
BLOCK = re.compile(r"\\begin\{lstlisting\}\[style=cpp\](.*?)\\end\{lstlisting\}", re.S)


def pick_std():
    for std in ("c++17", "c++14"):
        r = subprocess.run(["g++", f"-std={std}", "-x", "c++", "-fsyntax-only", "-"],
                           input="int main(){}", capture_output=True, text=True)
        if r.returncode == 0:
            return std
    return "c++11"


def main():
    std = pick_std()
    files = sorted(glob.glob(os.path.join(ROOT, "**", "*.tex"), recursive=True))
    total = failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            text = open(f, encoding="utf8").read()
            for k, m in enumerate(BLOCK.finditer(text), 1):
                total += 1
                line = text[:m.start()].count("\n") + 1
                src = os.path.join(tmp, "a.cpp")
                with open(src, "w", encoding="utf8") as out:
                    out.write(m.group(1))
                r = subprocess.run(["g++", f"-std={std}", "-fsyntax-only", "-Wall", src],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    failed += 1
                    print(f"FAIL {f}:{line} (block {k})")
                    print("   " + r.stderr.strip().replace("\n", "\n   ")[:1500])
    print(f"\n{total - failed}/{total} listings compile ({std}).")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
