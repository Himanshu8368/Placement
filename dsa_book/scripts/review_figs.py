"""Contact sheets of every figure in the chapters you name: one sheet per theme, all figures at a glance.

    python scripts/review_figs.py books/dp/229_dp_intro.qmd [more chapters]
    -> scripts/.cache/figs/<chapter>-light.png and <chapter>-dark.png

Each chapter's cells run as in check_chapter.py (includes expanded); every figure that V.show draws
is captured instead of displayed (animations are skipped: step through them in the browser). Use it to catch
clipped labels, colours that vanish in one theme, and figures that are far too big or small.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmdlib import ROOT, expand  # noqa: E402

CELL = re.compile(r"^```\{python\}[^\n]*\n(.*?)^```\s*$", re.S | re.M)
OUT = ROOT / "scripts" / ".cache" / "figs"
THUMB_W = 520


def capture(path: Path) -> dict[str, list[tuple[str, Image.Image]]]:
    shots: dict[str, list] = {"light": [], "dark": []}
    cell_no = [0]
    calls = [0]       # V.show and V.plot draw light then dark, one plt.show() each

    def grab(*_a, **_k):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            theme = "light" if calls[0] % 2 == 0 else "dark"
            calls[0] += 1
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=90, transparent=True)
            im = Image.open(io.BytesIO(buf.getvalue())).convert("RGBA")
            page = Image.new("RGBA", im.size, "#ffffff" if theme == "light" else "#0b1020")
            shots[theme].append((f"cell {cell_no[0]}", Image.alpha_composite(page, im).convert("RGB")))
        plt.close("all")

    text, _ = expand(path)
    env: dict = {"__name__": "__chapter__"}
    real_show = plt.show
    plt.show = grab
    try:
        with contextlib.chdir(path.resolve().parent), contextlib.redirect_stdout(io.StringIO()):
            for i, cell in enumerate(CELL.findall(text), 1):
                cell_no[0] = i
                try:
                    exec(compile(cell, f"{path.name}:cell{i}", "exec"), env)
                except Exception as exc:  # keep going: a broken cell is check_chapter's business
                    print(f"  cell {i} failed: {exc}", file=sys.stderr)
                plt.close("all")          # leftovers (animation frames) are not V.show figures
                calls[0] = 0
    finally:
        plt.show = real_show
    return shots


def sheet(items: list[tuple[str, Image.Image]], bg: str, fg: str) -> Image.Image:
    cols = 3
    thumbs = []
    for label, im in items:
        if im.width > THUMB_W:
            im = im.resize((THUMB_W, max(1, round(im.height * THUMB_W / im.width))))
        thumbs.append((label, im))
    rows = [thumbs[i:i + cols] for i in range(0, len(thumbs), cols)]
    heights = [max(im.height for _, im in r) + 28 for r in rows]
    out = Image.new("RGB", (cols * (THUMB_W + 16) + 16, sum(heights) + 16), bg)
    d = ImageDraw.Draw(out)
    y = 16
    for r, h in zip(rows, heights):
        for c, (label, im) in enumerate(r):
            x = 16 + c * (THUMB_W + 16)
            d.text((x, y), label, fill=fg)
            out.paste(im, (x, y + 18))
            d.rectangle([x - 1, y + 17, x + im.width, y + 18 + im.height], outline="#94a3b8")
        y += h
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    for a in argv:
        path = Path(a)
        shots = capture(path)
        for theme, (bg, fg) in (("light", ("#ffffff", "#1e293b")), ("dark", ("#0b1020", "#e2e8f0"))):
            if not shots[theme]:
                continue
            dest = OUT / f"{path.stem}-{theme}.png"
            sheet(shots[theme], bg, fg).save(dest)
            print(f"{dest.relative_to(ROOT)}: {len(shots[theme])} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
