"""Shared helpers for tools/preview.py and tools/build.py: run pdflatex and
summarise its log (errors, undefined references, overfull boxes, pages)."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # DSA/tree/tree_notes

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_pdflatex(tex_arg, outdir, jobname=None, timeout=1800):
    cmd = ["pdflatex", "-interaction=nonstopmode", "-file-line-error",
           f"-output-directory={outdir}"]
    if jobname:
        cmd.append(f"-jobname={jobname}")
    cmd.append(tex_arg)
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout)
    return proc.returncode


def summarise_log(log_path, overfull_min=2.0):
    text = Path(log_path).read_text(encoding="latin-1", errors="replace")
    lines = text.splitlines()
    errors = []
    for i, line in enumerate(lines):
        if re.match(r"^(\S.*?\.tex):(\d+): ", line) or line.startswith("! "):
            ctx = [l for l in lines[i + 1:i + 4] if l.strip()]
            errors.append(line.strip() + ("\n      " + "\n      ".join(ctx) if ctx else ""))
    undefined = sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", text)))
    multiply = sorted(set(re.findall(r"Label `([^']+)' multiply defined", text)))
    overfull = []
    for m in re.finditer(r"Overfull \\hbox \((\d+(?:\.\d+)?)pt too wide\) (.*?lines? [\d\-]+)", text):
        if float(m.group(1)) >= overfull_min:
            overfull.append(f"{float(m.group(1)):.1f}pt {m.group(2)}")
    missing_chars = len(re.findall(r"Missing character: There is no", text))
    pages = None
    m = re.search(r"Output written on .*?\((\d+) pages?", text, re.S)
    if m:
        pages = int(m.group(1))
    return {"errors": errors, "undefined": undefined, "multiply": multiply,
            "overfull": overfull, "missing_chars": missing_chars, "pages": pages}


def print_summary(title, pdf_path, s, show_undefined=40):
    print(f"== {title}")
    print(f"PDF       {pdf_path}  ({s['pages']} pages)")
    print(f"ERRORS    {len(s['errors'])}")
    for e in s["errors"][:15]:
        print("  - " + e)
    if len(s["errors"]) > 15:
        print(f"  ... {len(s['errors']) - 15} more")
    if s["multiply"]:
        print(f"MULTIPLY-DEFINED LABELS {len(s['multiply'])}: " + ", ".join(s["multiply"][:30]))
    if s["undefined"]:
        shown = ", ".join(s["undefined"][:show_undefined])
        more = "" if len(s["undefined"]) <= show_undefined else f" ... (+{len(s['undefined']) - show_undefined})"
        print(f"UNDEFINED REFS {len(s['undefined'])}: {shown}{more}")
    print(f"OVERFULL  {len(s['overfull'])} boxes >= 2pt")
    for o in s["overfull"][:12]:
        print("  - " + o)
    if s["missing_chars"]:
        print(f"MISSING CHARACTERS {s['missing_chars']} (non-ASCII glyph not in font)")
