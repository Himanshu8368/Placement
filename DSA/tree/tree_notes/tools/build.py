"""Build the whole book into build/ and copy the PDF to tree_notes.pdf.

Run from DSA/tree/tree_notes:
    python tools/build.py              # full build (3 pdflatex passes + index)
    python tools/build.py --only 1 2   # typeset only chapters 1 and 2; the rest keep
                                       # the page numbers and stats of the last full build
    python tools/build.py --quick      # one pass, no index (fast check)
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latexlog import ROOT, run_pdflatex, summarise_log, print_summary  # noqa: E402

JOB = "tree_notes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None, help="chapter numbers to typeset")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    root_tex = (ROOT / f"{JOB}.tex").read_text(encoding="utf-8")
    includes = re.findall(r"^\s*\\include\{([^}]+)\}", root_tex, re.M)
    out = ROOT / "build"
    for inc in includes:
        (out / Path(inc).parent).mkdir(parents=True, exist_ok=True)

    tex_arg = f"{JOB}.tex"
    if a.only:
        wanted = []
        for n in a.only:
            hits = [i for i in includes if re.search(rf"chapters/ch{n:02d}_[^/]+/ch{n:02d}$", i)]
            if not hits:
                sys.exit(f"chapter {n} not found in {JOB}.tex")
            wanted += hits
        tex_arg = "\\def\\BuildOnly{" + ",".join(wanted) + "}\\input{" + JOB + ".tex}"

    passes = 1 if a.quick else 2
    for _ in range(passes):
        run_pdflatex(tex_arg, str(out), jobname=JOB)
    idx = out / f"{JOB}.idx"
    if not a.quick and idx.exists():
        subprocess.run(["makeindex", "-q", idx.name], cwd=out)
        run_pdflatex(tex_arg, str(out), jobname=JOB)

    s = summarise_log(out / f"{JOB}.log")
    pdf = out / f"{JOB}.pdf"
    target = ROOT / f"{JOB}.pdf"
    if pdf.exists():
        try:
            shutil.copyfile(pdf, target)
        except PermissionError:
            print(f"NOTE      {target} is locked by a viewer; the fresh PDF is {pdf}")
    print_summary("BUILD " + ("chapters " + " ".join(map(str, a.only)) if a.only else "full book"),
                  target, s, show_undefined=60)
    sys.exit(1 if s["errors"] else 0)


if __name__ == "__main__":
    main()
