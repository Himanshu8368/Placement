"""Compile ONE file of the book in isolation (fast, parallel-safe).

Run from DSA/tree/tree_notes:
    python tools/preview.py chapters/ch01_tree_basics/02_anatomy.tex
    python tools/preview.py chapters/ch01_tree_basics/ch01.tex        # whole chapter
    python tools/preview.py frontmatter/how_to_read.tex
    python tools/preview.py <file> --png 3-5    # also render pages 3..5 to PNG

A topic file NN_name.tex inside chapters/chXX_*/ is numbered as topic XX.NN,
so theorem/figure/problem numbers match the full book. References to other
files show as "??" and are listed as UNDEFINED REFS -- that is expected.
Output (PDF, log, PNGs) goes to %TEMP%/treebook_preview/<file>/.
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latexlog import ROOT, run_pdflatex, summarise_log, print_summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--png", default="", help="pages to render, e.g. 3 or 2-5")
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()

    src = Path(a.file)
    if not src.is_absolute():
        src = (Path.cwd() / src).resolve()
        if not src.exists():
            src = (ROOT / a.file).resolve()
    rel = src.relative_to(ROOT).as_posix()
    rel_noext = rel[:-4] if rel.endswith(".tex") else rel

    m = re.search(r"chapters/ch(\d\d)_", rel)
    chapter = int(m.group(1)) if m else 0
    stem = src.stem
    body = []
    if m and re.fullmatch(r"ch\d\d", stem):
        body += [f"\\setcounter{{chapter}}{{{chapter - 1}}}", f"\\input{{{rel_noext}}}"]
    elif m:
        sm = re.match(r"(\d\d)_", stem)
        sec = int(sm.group(1)) if sm else 0
        body += [f"\\setcounter{{chapter}}{{{chapter - 1}}}",
                 "\\bookchapter{Preview}{preview}"]
        if 1 <= sec <= 98:
            body.append(f"\\setcounter{{section}}{{{sec - 1}}}")
        body.append(f"\\input{{{rel_noext}}}")
    else:
        body.append(f"\\input{{{rel_noext}}}")

    outdir = Path(tempfile.gettempdir()) / "treebook_preview" / re.sub(r"[^A-Za-z0-9]+", "_", rel_noext)
    outdir.mkdir(parents=True, exist_ok=True)
    driver = outdir / "preview.tex"
    driver.write_text(
        "\\documentclass[11pt,a4paper,oneside]{book}\n\\input{setup/preamble}\n"
        "\\begin{document}\n" + "\n".join(body) + "\n\\end{document}\n", encoding="utf-8")

    for _ in range(max(1, a.runs)):
        run_pdflatex(str(driver), str(outdir))
    s = summarise_log(outdir / "preview.log")
    pdf = outdir / "preview.pdf"
    print_summary(f"PREVIEW {rel}", pdf, s)

    if a.png and pdf.exists():
        first, _, last = a.png.partition("-")
        last = last or first
        prefix = outdir / "page"
        for old in outdir.glob("page-*.png"):
            old.unlink()
        subprocess.run(["pdftoppm", "-f", first, "-l", last, "-r", str(a.dpi), "-png",
                        str(pdf), str(prefix)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for p in sorted(outdir.glob("page-*.png")):
            print(f"PNG       {p}")
    sys.exit(1 if s["errors"] else 0)


if __name__ == "__main__":
    main()
