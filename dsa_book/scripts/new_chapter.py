"""Create a chapter file from outline.yml and templates/chapter.qmd.

    python scripts/new_chapter.py 230 --prefix dp1d [--slug one_d] [--owner my-session]

Writes books/<folder>/<num:03d>_<slug>.qmd with the title from outline.yml, the label prefix filled
in, and the chapter's patterns (outline "covers") listed in a comment as the plan to follow; marks the
chapter `status: drafting` (and `owner:`) in outline.yml.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmdlib import BOOKS, ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("num", type=int)
    ap.add_argument("--prefix", required=True, help="label prefix, e.g. dp1d -> sec-dp1d-…")
    ap.add_argument("--slug", help="file slug (default: from the title)")
    ap.add_argument("--owner", default="", help="who edits it now; recorded in outline.yml (STYLE.md, Coordination)")
    a = ap.parse_args()

    data = yaml.safe_load((ROOT / "outline.yml").read_text(encoding="utf-8"))
    for v in data["volumes"]:
        for p in v["parts"]:
            for c in p["chapters"]:
                if c["num"] == a.num:
                    folder, title, covers = p["folder"], c["title"], c.get("covers", "")
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        raise SystemExit(f"chapter {a.num} is not in outline.yml")

    slug = a.slug or re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]
    path = BOOKS / folder / f"{a.num:03d}_{slug}.qmd"
    if list((BOOKS / folder).glob(f"{a.num:03d}_*.qmd")):
        raise SystemExit(f"a file for chapter {a.num} already exists in books/{folder}/")
    text = (ROOT / "templates" / "chapter.qmd").read_text(encoding="utf-8")
    text = text.replace("# Chapter Title {#sec-SLUG}", f"# {title} {{#sec-{a.prefix}}}").replace("SLUG", a.prefix)
    plan = "\n".join(f"  - {x.strip()}" for x in covers.split(";") if x.strip())
    text = text.replace("<!--\n", f"<!--\nOutline (outline.yml) for chapter {a.num}:\n{plan}\n\n", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    claim(a.num, a.owner)
    print(f"created {path.relative_to(ROOT)} — status: drafting" + (f", owner: {a.owner}" if a.owner else "")
          + " — run scripts/build.ps1 to add it to the sidebar")


def claim(num: int, owner: str) -> None:
    """Mark the chapter `status: drafting` (and its owner) in outline.yml, editing the text in place so
    comments and layout survive."""
    path = ROOT / "outline.yml"
    text = path.read_text(encoding="utf-8")
    m = re.search(rf"^( *)- num: {num}\n((?:\1  .*\n)*)", text, re.M)
    if not m:
        return
    ind, body = m[1] + "  ", m[2]
    body = re.sub(rf"^{ind}(status|owner): .*\n", "", body, flags=re.M)
    body += f"{ind}status: drafting\n" + (f"{ind}owner: {owner}\n" if owner else "")
    path.write_text(text[: m.start(2)] + body + text[m.end(2):], encoding="utf-8")


if __name__ == "__main__":
    main()
