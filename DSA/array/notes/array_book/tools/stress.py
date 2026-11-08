"""
stress.py -- generic stress tester: run a solution and a brute force on many random inputs
and stop at the first input on which their outputs differ.

Usage:
    python tools/stress.py <solution.cpp> <brute.cpp> <generator.cpp|generator.py> [runs] [first_seed]

  * both programs read the test from stdin and write the answer to stdout;
  * the generator receives the seed as its only argument (argv[1]) and prints one test;
  * outputs are compared token by token (whitespace differences are ignored);
  * a crash (non-zero exit code) or a timeout (10 s) of either program also counts as a failure.
On a failure the input, both outputs and the seed are printed and the input is saved to
stress_fail.txt in the current folder; exit code 1. Otherwise "OK: <runs> tests passed".
"""
import os
import subprocess
import sys
import tempfile


def compile_cpp(src, exe):
    r = subprocess.run(["g++", "-std=c++14", "-O2", "-o", exe, src], capture_output=True, text=True)
    if r.returncode != 0:
        print("compile error in %s:\n%s" % (src, r.stderr[:3000]))
        sys.exit(2)


def command(path, tmp, tag):
    """Return the command line that runs `path` (compiling C++ first)."""
    if path.endswith(".py"):
        return [sys.executable, path]
    exe = os.path.join(tmp, tag + ".exe")
    compile_cpp(path, exe)
    return [exe]


def run(cmd, inp, args=()):
    try:
        r = subprocess.run(list(cmd) + list(args), input=inp, capture_output=True, text=True, timeout=10)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return "TIMEOUT", ""


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    sol, brute, gen = sys.argv[1:4]
    runs = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    first = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    with tempfile.TemporaryDirectory() as tmp:
        c_sol = command(sol, tmp, "sol")
        c_bru = command(brute, tmp, "brute")
        c_gen = command(gen, tmp, "gen")
        for seed in range(first, first + runs):
            code, inp = run(c_gen, "", [str(seed)])
            if code != 0:
                print("generator failed on seed %d (exit %s)" % (seed, code))
                sys.exit(2)
            c1, out1 = run(c_sol, inp)
            c2, out2 = run(c_bru, inp)
            if c1 != 0 or c2 != 0 or out1.split() != out2.split():
                print("MISMATCH on test %d (seed %d)" % (seed - first + 1, seed))
                print("---- input ----")
                print(inp.rstrip())
                print("---- solution (exit %s) ----" % c1)
                print(out1.rstrip())
                print("---- brute force (exit %s) ----" % c2)
                print(out2.rstrip())
                with open("stress_fail.txt", "w") as f:
                    f.write(inp)
                sys.exit(1)
        print("OK: %d tests passed" % runs)


if __name__ == "__main__":
    main()
