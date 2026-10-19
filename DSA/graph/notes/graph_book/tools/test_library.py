"""
test_library.py -- compile and run every test of the code library (code/tests/test_*.cpp).

Each test checks one header of code/lib against a brute-force reference on thousands of
random small graphs from code/gen/gen.hpp and prints "OK <component> <cases>" or the first
failing input. This script prints a summary and exits with 1 if anything fails.

Usage (from the book root):
    python tools/test_library.py              # all tests
    python tools/test_library.py dsu bfs      # only tests whose name contains a word
"""
import glob
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "code", "tests")


def main():
    words = sys.argv[1:]
    files = sorted(glob.glob(os.path.join(TESTS, "test_*.cpp")))
    if words:
        files = [f for f in files if any(w in os.path.basename(f) for w in words)]
    passed, failed = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            name = os.path.basename(f)[5:-4]
            exe = os.path.join(tmp, name + ".exe")
            t0 = time.time()
            c = subprocess.run(["g++", "-std=c++14", "-O2", "-Wall", "-o", exe, f],
                               capture_output=True, text=True)
            if c.returncode != 0:
                failed.append(name)
                print("COMPILE ERROR %s\n%s" % (name, c.stderr[:3000]))
                continue
            try:
                r = subprocess.run([exe], capture_output=True, text=True, timeout=300)
                out, code = r.stdout.strip(), r.returncode
            except subprocess.TimeoutExpired:
                out, code = "TIMEOUT", 1
            dt = time.time() - t0
            if code == 0 and out.startswith("OK"):
                passed.append(name)
                print("%-55s %6.1fs" % (out, dt))
            else:
                failed.append(name)
                print("FAILED %s (exit code %s)\n%s" % (name, code, out[:3000]))
    print("\n%d/%d library tests passed." % (len(passed), len(passed) + len(failed)))
    if failed:
        print("failing: " + ", ".join(failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
