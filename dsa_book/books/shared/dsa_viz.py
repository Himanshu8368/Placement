"""Diagrams for the DSA book: arrays with pointers and regions, traces, trees, intervals, bars, flows.

Every figure is drawn twice, once per site theme, so it reads well in light and dark mode.
In a chapter:

    ```{python}
    #| label: fig-xyz
    #| fig-cap: "..."
    #| renderings: [light, dark]
    def draw(ax, p):
        V.array(ax, p, [3, 1, 4], pointers=[(0, "lo", "indigo")])
    V.show(draw)
    ```

`p` is the palette (LIGHT or DARK). Colours are referred to by name: "indigo", "cyan", "green",
"amber", "pink", "violet", "red", "muted". Coordinates are in "cell" units: an array cell is 1 x 1,
and figures are sized from their content so cells look the same size in every figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ACCENTS = ("indigo", "cyan", "green", "amber", "pink", "violet", "red")


def _mix(c1: str, c2: str, t: float) -> str:
    """Blend hex colour c1 towards c2 by t (0 = c1, 1 = c2)."""
    a = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def _palette(name, bg, text, muted, line, cell, accents):
    p = dict(name=name, bg=bg, text=text, muted=muted, line=line, cell=cell, **accents)
    # Soft fills: the accent blended into the page background, readable under text in both themes.
    p["soft"] = {k: _mix(bg, v, 0.22 if name == "light" else 0.38) for k, v in accents.items()}
    p["soft"]["muted"] = _mix(bg, muted, 0.18 if name == "light" else 0.30)
    p["muted_fill"] = p["soft"]["muted"]
    return p


LIGHT = _palette(
    "light", bg="#f8fafc", text="#1e293b", muted="#64748b", line="#94a3b8", cell="#ffffff",
    accents=dict(indigo="#4f46e5", cyan="#0891b2", green="#059669", amber="#d97706",
                 pink="#db2777", violet="#7c3aed", red="#e11d48"),
)
DARK = _palette(
    "dark", bg="#0b1020", text="#e2e8f0", muted="#94a3b8", line="#64748b", cell="#111827",
    accents=dict(indigo="#818cf8", cyan="#22d3ee", green="#34d399", amber="#fbbf24",
                 pink="#f472b6", violet="#a78bfa", red="#fb7185"),
)
for _p in (LIGHT, DARK):
    _p["muted_color"] = _p["muted"]

INCH_PER_UNIT = 0.42
MAX_WIDTH_IN = 7.4
FONT = 11


def col(p, name):
    """Colour by name: an accent, "text", "muted" or "line"."""
    return p[name] if name in p and isinstance(p[name], str) else name


def soft(p, name):
    return p["soft"].get(name, p["muted_fill"])


# ---------------------------------------------------------------- figure plumbing

def _extend(ax, x0, y0, x1, y1):
    b = getattr(ax, "_dsa_bounds", None)
    ax._dsa_bounds = (x0, y0, x1, y1) if b is None else (min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1))


def _fit(fig, ax, pad, scale, max_width, fixed=None):
    """Size the figure to its content (shapes, then measured text) and return the bounds used.
    With `fixed` bounds (animations), every frame gets exactly the same limits and size."""
    bounds = fixed or getattr(ax, "_dsa_bounds", (0, 0, 1, 1))
    # Size from shapes first, then grow to fit the measured text, three times (text size is in
    # points, so its extent in data units depends on the figure scale).
    for _ in range(1 if fixed else 3):
        x0, y0, x1, y1 = bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        s = min(scale, max_width / (x1 - x0))
        fig.set_size_inches(max((x1 - x0) * s, 1.0), max((y1 - y0) * s, 0.5))
        fig.subplots_adjust(0, 0, 1, 1)
        if fixed:
            break
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        for t in ax.texts:
            if not t.get_text():
                continue
            (a, b), (c, d) = inv.transform(t.get_window_extent(renderer).get_points())
            bounds = (min(bounds[0], a), min(bounds[1], b), max(bounds[2], c), max(bounds[3], d))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    return bounds


def show(draw, *, pad=0.35, scale=INCH_PER_UNIT, max_width=MAX_WIDTH_IN):
    """Draw once per theme (light, then dark) and show both; pair with `#| renderings: [light, dark]`."""
    for p in (LIGHT, DARK):
        fig, ax = plt.subplots()
        ax.set_axis_off()
        draw(ax, p)
        _fit(fig, ax, pad, scale, max_width)
        plt.show()


MAX_FRAMES = 15


def animate(draw_frame, n_frames, *, captions=None, interval=900, pad=0.35, scale=INCH_PER_UNIT,
            max_width=MAX_WIDTH_IN, dpi=110):
    """A small animation (2–15 frames) that plays inline, in both themes.

    draw_frame(ax, p, k) draws frame k (0-based). `captions[k]` (optional) is shown under frame k.
    Every frame is measured first and drawn with the union of the bounds, so nothing jumps between
    frames. The output is HTML (one <img> per frame and theme); books/shared/site.js adds the
    play / pause / step controls and shows the frame set that matches the current theme.

        ```{python}
        #| echo: false
        V.animate(lambda ax, p, k: V.array(ax, p, dp[: k + 1]), len(dp), captions=[...])
        ```
    """
    import base64
    import html as _html
    import io

    if not 2 <= n_frames <= MAX_FRAMES:
        raise ValueError(f"an animation has 2..{MAX_FRAMES} frames, got {n_frames}")
    if captions is not None and len(captions) != n_frames:
        raise ValueError("captions must have one entry per frame")

    union = None
    for k in range(n_frames):
        fig, ax = plt.subplots()
        ax.set_axis_off()
        draw_frame(ax, LIGHT, k)
        b = _fit(fig, ax, pad, scale, max_width)
        union = b if union is None else (min(union[0], b[0]), min(union[1], b[1]), max(union[2], b[2]), max(union[3], b[3]))
        plt.close(fig)

    import hashlib
    from pathlib import Path

    # Frames are written next to the chapter as content-addressed WebP files (anim/<hash>.webp) and
    # lazy-loaded, so a page carries links, not megabytes of inline images. _quarto.yml lists
    # "*/anim/*.webp" as resources so Quarto copies them into the site.
    out_dir = Path("anim")
    out_dir.mkdir(exist_ok=True)
    last_png = None
    parts = [f'<div class="dsa-anim" data-interval="{int(interval)}" tabindex="0" role="group" '
             f'aria-label="animation, {n_frames} frames; left/right arrows step, space plays">']
    for p in (LIGHT, DARK):
        imgs = []
        for k in range(n_frames):
            fig, ax = plt.subplots()
            ax.set_axis_off()
            draw_frame(ax, p, k)
            _fit(fig, ax, pad, scale, max_width, fixed=union)
            w = fig.get_size_inches()[0]
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, transparent=True)
            plt.close(fig)
            png = buf.getvalue()
            if p is LIGHT and k == n_frames - 1:
                last_png = png
            name = hashlib.sha1(png).hexdigest()[:16] + ".webp"
            target = out_dir / name
            if not target.exists():
                try:
                    from PIL import Image
                    Image.open(io.BytesIO(png)).save(target, "WEBP", lossless=True, method=6)
                except Exception:            # no WebP support: fall back to the PNG bytes
                    name = name.replace(".webp", ".png")
                    target = out_dir / name
                    target.write_bytes(png)
            alt = _html.escape(str(captions[k]) if captions else f"frame {k + 1} of {n_frames}", quote=True)
            imgs.append(f'<img class="anim-frame{" on" if k == 0 else ""}" alt="{alt}" loading="lazy" decoding="async" '
                        f'style="width:{w * 96:.0f}px" src="anim/{name}">')
        parts.append(f'<div class="anim-theme anim-{p["name"]}">' + "".join(imgs) + "</div>")
    if captions:
        parts.append('<div class="anim-captions" aria-live="polite">' + "".join(
            f'<div class="anim-caption{" on" if k == 0 else ""}">{_html.escape(str(c))}</div>'
            for k, c in enumerate(captions)) + "</div>")
    parts.append("</div>")
    out = "".join(parts)
    try:
        from IPython.display import display
        # HTML for the web; the last frame as a plain image for PDF/EPUB, where scripts do not run.
        display({"text/html": out, "image/png": base64.b64encode(last_png).decode("ascii")}, raw=True)
    except ImportError:          # plain Python (no IPython): nothing to display
        pass
    # Return nothing: a returned string would be echoed as a second, raw-text output when the call
    # is the last line of a cell.


def text(ax, p, x, y, s, *, color="text", size=FONT, ha="center", va="center", weight="normal", style="normal"):
    ax.text(x, y, s, color=col(p, color), fontsize=size, ha=ha, va=va, weight=weight, style=style,
            clip_on=False)
    _extend(ax, x, y, x, y)


def arrow(ax, p, x0, y0, x1, y1, *, color="muted", lw=1.4, style="-|>", rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=col(p, color), lw=lw, shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))
    bulge = abs(rad) * ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / 2   # curved arrows bow out by about this
    _extend(ax, min(x0, x1) - bulge, min(y0, y1) - bulge, max(x0, x1) + bulge, max(y0, y1) + bulge)


def box(ax, p, x, y, w, h, label="", *, color="indigo", filled=True, size=FONT, weight="normal",
        round_=True, dashed=False, label_color="text"):
    """Box with lower-left corner (x, y)."""
    face = soft(p, color) if filled else p["cell"]
    if round_:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15", fc=face,
                               ec=col(p, color), lw=1.4, ls="--" if dashed else "-")
    else:
        patch = Rectangle((x, y), w, h, fc=face, ec=col(p, color), lw=1.2, ls="--" if dashed else "-")
    ax.add_patch(patch)
    _extend(ax, x, y, x + w, y + h)
    if label != "":
        text(ax, p, x + w / 2, y + h / 2, label, size=size, weight=weight, color=label_color)


# ---------------------------------------------------------------- arrays and traces

def array(ax, p, values, x=0.0, y=0.0, *, regions=(), pointers=(), colors=None, index=True, label=None,
          note=None, cell_w=1.0, dim=(), size=FONT):
    """One array row with its lower-left corner at (x, y).

    regions:  (lo, hi, colour, text)      cells lo..hi-1 shaded, text under a bracket below the row
    pointers: (i, text, colour[, "top"])   arrow to cell i (i may equal len(values): one past the end)
    colors:   {i: colour}                  fill individual cells
    dim:      indices drawn greyed out (discarded part of the array)
    """
    colors = colors or {}
    n = len(values)
    for lo, hi, c, _ in regions:
        for i in range(lo, hi):
            colors.setdefault(i, c)
    for i, v in enumerate(values):
        cx = x + i * cell_w
        c = colors.get(i)
        face = soft(p, c) if c else (p["muted_fill"] if i in dim else p["cell"])
        ax.add_patch(Rectangle((cx, y), cell_w, 1, fc=face, ec=col(p, c) if c else p["line"], lw=1.2))
        text(ax, p, cx + cell_w / 2, y + 0.5, "" if v is None else str(v), size=size,
             color="muted" if i in dim else "text")
        if index:
            text(ax, p, cx + cell_w / 2, y + 1.22, str(i), size=size - 3, color="muted")
    _extend(ax, x, y, x + max(n, 1) * cell_w, y + 1)
    for lo, hi, c, t in regions:
        if hi <= lo or not t:
            continue
        x0, x1 = x + lo * cell_w + 0.08, x + hi * cell_w - 0.08
        ax.plot([x0, x0, x1, x1], [y - 0.12, y - 0.25, y - 0.25, y - 0.12], color=col(p, c), lw=1.3)
        text(ax, p, (x0 + x1) / 2, y - 0.55, t, color=c, size=size - 1)
    for ptr in pointers:
        i, t, c = ptr[:3]
        top = len(ptr) > 3 and ptr[3] == "top"
        cx = x + i * cell_w + cell_w / 2
        if top:
            base = y + 1.5 if index else y + 1.2
            arrow(ax, p, cx, base + 0.7, cx, base + 0.05, color=c)
            text(ax, p, cx, base + 0.95, t, color=c, size=size - 1, weight="bold")
        else:
            low = y - (0.95 if any(r[3] for r in regions) else 0.15)
            arrow(ax, p, cx, low - 0.6, cx, low - 0.05, color=c)
            text(ax, p, cx, low - 0.85, t, color=c, size=size - 1, weight="bold")
    if label:
        text(ax, p, x - 0.3, y + 0.5, label, ha="right", color="muted", size=size - 1)
    if note:
        text(ax, p, x + n * cell_w + 0.4, y + 0.5, note, ha="left", color="muted", size=size - 1)


def trace(ax, p, rows, *, gap=3.9, x=0.0):
    """Several array rows stacked top to bottom; each row is a dict of `array` keyword arguments
    plus `values` (and optionally `label`, `note`)."""
    for k, r in enumerate(rows):
        r = dict(r)
        values = r.pop("values")
        array(ax, p, values, x, -k * gap, **r)


def stack(ax, p, items, x=0.0, y=0.0, *, title=None, color="indigo", w=1.6, top_label="top"):
    """Vertical stack, bottom item at (x, y); the last item is the top."""
    for k, v in enumerate(items):
        box(ax, p, x, y + k, w, 1, str(v), color=color, round_=False, filled=k == len(items) - 1)
    if not items:
        ax.plot([x, x + w], [y, y], color=p["line"], lw=1.2)
        _extend(ax, x, y, x + w, y + 0.1)
    ax.plot([x, x, x + w, x + w], [y + max(len(items), 1) + 0.3, y, y, y + max(len(items), 1) + 0.3],
            color=p["line"], lw=1.6)
    if title:
        text(ax, p, x + w / 2, y - 0.45, title, color="muted", size=FONT - 1)
    if items and top_label:
        text(ax, p, x + w + 0.2, y + len(items) - 0.5, f"← {top_label}", ha="left", color=color, size=FONT - 2)


# ---------------------------------------------------------------- trees

class Node:
    def __init__(self, label, children=(), *, color="indigo", filled=True, note=None, edge=None, dashed=False):
        self.label, self.children = str(label), list(children)
        self.color, self.filled, self.note, self.edge, self.dashed = color, filled, note, edge, dashed
        self.x = self.y = 0.0


def _layout(node, depth, next_x, dx):
    if not node.children:
        node.x = next_x[0]
        next_x[0] += dx
    else:
        for ch in node.children:
            _layout(ch, depth + 1, next_x, dx)
        node.x = (node.children[0].x + node.children[-1].x) / 2
    node.depth = depth


def tree(ax, p, root, x=0.0, y=0.0, *, dx=1.6, dy=1.5, node_w=None, node_h=0.75, size=FONT,
         level_notes=None):
    """Draw a tree with the root at the top. Leaves are spaced dx apart; levels dy apart.
    level_notes: list of strings written to the right of each level (e.g. work per level)."""
    _layout(root, 0, [0.0], dx)
    nodes, stack_ = [], [root]
    while stack_:
        v = stack_.pop()
        nodes.append(v)
        stack_.extend(v.children)
    def width(v):
        return node_w or max(0.7, 0.21 * len(v.label) * size / FONT + 0.4)
    for v in nodes:
        v.px, v.py = x + v.x, y - v.depth * dy
    for v in nodes:
        for ch in v.children:
            ax.plot([v.px, ch.px], [v.py - node_h / 2, ch.py + node_h / 2], color=p["line"], lw=1.3,
                    ls="--" if ch.dashed else "-", zorder=1)
            if ch.edge:
                text(ax, p, (v.px + ch.px) / 2 + (0.18 if ch.px >= v.px else -0.18),
                     (v.py + ch.py) / 2, ch.edge, size=size - 2, color="muted",
                     ha="left" if ch.px >= v.px else "right")
    for v in nodes:
        w = width(v)
        box(ax, p, v.px - w / 2, v.py - node_h / 2, w, node_h, v.label, color=v.color, filled=v.filled,
            size=size, dashed=v.dashed)
        if v.note:
            text(ax, p, v.px, v.py - node_h / 2 - 0.3, v.note, size=size - 2, color="muted")
    if level_notes:
        right = max(v.px + width(v) / 2 for v in nodes) + 0.8
        for d, s in enumerate(level_notes):
            text(ax, p, right, y - d * dy, s, ha="left", color="violet", size=size - 1)
    return nodes


# ---------------------------------------------------------------- intervals, bars, labelled grids

def intervals(ax, p, items, *, y=0.0, row_h=0.8, scale=1.0, axis=True):
    """items: (start, end, label, colour[, dashed]) drawn one per row, top to bottom, with an axis."""
    if not items:
        return
    lo = min(s for s, *_ in items)
    hi = max(e for _, e, *_ in items)
    for k, it in enumerate(items):
        s, e, lab, c = it[:4]
        dashed = len(it) > 4 and it[4]
        yy = y - k * (row_h + 0.35)
        box(ax, p, s * scale, yy, (e - s) * scale, row_h, lab, color=c, dashed=dashed, size=FONT - 1,
            filled=not dashed)
    if axis:
        ya = y - len(items) * (row_h + 0.35) + 0.1
        ax.plot([lo * scale, hi * scale], [ya, ya], color=p["line"], lw=1.2)
        for t in range(int(lo), int(hi) + 1):
            ax.plot([t * scale] * 2, [ya - 0.08, ya + 0.08], color=p["line"], lw=1)
            text(ax, p, t * scale, ya - 0.4, str(t), size=FONT - 3, color="muted")


def bars(ax, p, values, x=0.0, y=0.0, *, height=3.0, colors=None, labels=True, bar_w=0.8, index=True,
         pointers=(), baseline=True):
    """Vertical bars scaled so the largest is `height` units tall. colors: {i: colour}."""
    colors = colors or {}
    top = max(max(values), 1e-9)
    for i, v in enumerate(values):
        h = height * v / top
        c = colors.get(i, "muted")
        ax.add_patch(Rectangle((x + i + (1 - bar_w) / 2, y), bar_w, h, fc=soft(p, c), ec=col(p, c), lw=1.2))
        if labels:
            text(ax, p, x + i + 0.5, y + h + 0.3, f"{v:g}" if isinstance(v, float) else str(v), size=FONT - 2)
        if index:
            text(ax, p, x + i + 0.5, y - 0.35, str(i), size=FONT - 3, color="muted")
    _extend(ax, x, y, x + len(values), y + height)
    if baseline:
        ax.plot([x, x + len(values)], [y, y], color=p["line"], lw=1.2)
    for ptr in pointers:
        i, t, c = ptr[:3]
        cx = x + i + 0.5
        arrow(ax, p, cx, y - 1.4, cx, y - 0.65, color=c)
        text(ax, p, cx, y - 1.7, t, color=c, size=FONT - 1, weight="bold")


def grid(ax, p, cells, x=0.0, y=0.0, *, row_labels=None, col_labels=None, colors=None, cell_w=1.0, size=FONT):
    """Table of cells; cells[r][c] is a string, row 0 at the top. colors: {(r, c): colour}.

    ``cell_w`` is the *minimum* column width: a column whose widest string would not fit is grown
    to fit it, so text never spills over the neighbour. Widths only ever grow, never shrink.
    """
    colors = colors or {}
    cols = max(len(r) for r in cells)
    # ~0.21 data units per character at FONT, plus padding on both sides (same estimate as tree()).
    need = [0.21 * max((len(str(r[c])) for r in cells if c < len(r)), default=0) * size / FONT + 0.3
            for c in range(cols)]
    cw = [max(cell_w, w) for w in need]
    xs = [x]
    for w in cw:
        xs.append(xs[-1] + w)
    for r, row in enumerate(cells):
        for c, v in enumerate(row):
            cc = colors.get((r, c))
            yy = y - r
            ax.add_patch(Rectangle((xs[c], yy), cw[c], 1, fc=soft(p, cc) if cc else p["cell"],
                                   ec=col(p, cc) if cc else p["line"], lw=1.1))
            text(ax, p, xs[c] + cw[c] / 2, yy + 0.5, str(v), size=size)
    _extend(ax, x, y - len(cells) + 1, xs[-1], y + 1)
    if row_labels:
        for r, s in enumerate(row_labels):
            text(ax, p, x - 0.25, y - r + 0.5, s, ha="right", color="muted", size=size - 1)
    if col_labels:
        for c, s in enumerate(col_labels):
            text(ax, p, xs[c] + cw[c] / 2, y + 1.35, s, color="muted", size=size - 2)


def flow(ax, p, nodes, edges, *, w=2.6, h=0.9, size=FONT):
    """Boxes-and-arrows diagram. nodes: name -> (cx, cy, label, colour); edges: (a, b, label)."""
    for name, (cx, cy, lab, c) in nodes.items():
        box(ax, p, cx - w / 2, cy - h / 2, w, h, lab, color=c, size=size)
    for a, b, lab in edges:
        ax_, ay, *_ = nodes[a]
        bx, by, *_ = nodes[b]
        if abs(bx - ax_) >= abs(by - ay):      # mostly sideways: side to side
            sgn = 1 if bx > ax_ else -1
            x0, y0, x1, y1 = ax_ + sgn * w / 2, ay, bx - sgn * w / 2, by
        else:                                   # mostly vertical: bottom/top to top/bottom
            sgn = 1 if by > ay else -1
            x0, y0, x1, y1 = ax_, ay + sgn * h / 2, bx, by - sgn * h / 2
        arrow(ax, p, x0, y0, x1, y1, color="muted")
        if lab:
            text(ax, p, (x0 + x1) / 2, (y0 + y1) / 2 + 0.3, lab, size=size - 2, color="muted")


def iso_grid(ax, p, shape, x=0.0, y=0.0, *, values=None, colors=None, layer_gap=None, cell=1.0,
             axis_labels=("i", "j", "k"), layer_names=None, size=FONT):
    """A three-index table dp[i][j][k] in 3D isometric view: one tilted slab per k, stacked upward.

    shape = (ni, nj, nk). values: {(i, j, k): str} written on the cell. colors: {(i, j, k): colour}
    highlights a cell (e.g. the cell being filled in one colour, the cells it reads in another).
    Each slab is drawn above (and on top of) the one below it, so the stack reads as solid plates.
    """
    from matplotlib.patches import Polygon
    values, colors = values or {}, colors or {}
    ni, nj, nk = shape
    cx, cy = 0.866 * cell, 0.5 * cell            # cos 30°, sin 30°
    gap = layer_gap if layer_gap is not None else 0.6 * (ni + nj) * cy + 0.7

    def proj(i, j, k):                            # i runs down-right, j runs up-right, k is height
        return x + (i + j) * cx, y + (j - i) * cy + k * gap

    for k in range(nk):
        z = 10 + 3 * k                            # a higher slab covers the text of a lower one
        a, b, d = proj(0, 0, k), proj(ni, 0, k), proj(ni, nj, k)
        t = 0.16                                  # plate thickness
        ax.add_patch(Polygon([a, b, d, (d[0], d[1] - t), (b[0], b[1] - t), (a[0], a[1] - t)], closed=True,
                             fc=p["muted_fill"], ec=p["line"], lw=0.8, zorder=z))
        for i in range(ni):
            for j in range(nj):
                c = colors.get((i, j, k))
                pts = [proj(i, j, k), proj(i + 1, j, k), proj(i + 1, j + 1, k), proj(i, j + 1, k)]
                ax.add_patch(Polygon(pts, closed=True, fc=soft(p, c) if c else p["cell"],
                                     ec=col(p, c) if c else p["line"], lw=1.6 if c else 0.9,
                                     zorder=z + (1 if c else 0.5)))
                if (i, j, k) in values:
                    mx, my = proj(i + 0.5, j + 0.5, k)
                    ax.text(mx, my, str(values[(i, j, k)]), color=p["text"], fontsize=size - 1, ha="center",
                            va="center", weight="bold" if c else "normal", zorder=z + 2, clip_on=False)
        name = layer_names[k] if layer_names else f"{axis_labels[2]} = {k}"
        text(ax, p, a[0] - 0.3, a[1], name, ha="right", color="muted", size=size - 1)
    # name the two horizontal axes on the front edges of the bottom slab
    for (i0, j0, i1, j1, nx, ny, lab) in ((0, 0, ni, 0, -cy, -cx, axis_labels[0]),
                                          (ni, 0, ni, nj, cy, -cx, axis_labels[1])):
        (x0, y0), (x1, y1) = proj(i0, j0, 0), proj(i1, j1, 0)
        o = 0.55
        arrow(ax, p, x0 + o * nx, y0 + o * ny, x1 + o * nx, y1 + o * ny, color="muted", lw=1.1)
        text(ax, p, (x0 + x1) / 2 + 1.1 * nx, (y0 + y1) / 2 + 1.1 * ny, lab, color="muted", size=size - 1)
    _extend(ax, x - 1.8, proj(ni, 0, 0)[1] - 1.2, proj(ni, nj, 0)[0] + 0.6, proj(0, nj, nk - 1)[1] + 0.3)


def plot(draw_axes, *, size=(6.4, 3.4)):
    """Ordinary data plots (axes, ticks) in both themes. draw_axes(ax, p) plots with colours from p."""
    for p in (LIGHT, DARK):
        fig, ax = plt.subplots(figsize=size)
        draw_axes(ax, p)
        for s in ax.spines.values():
            s.set_color(p["line"])
        ax.tick_params(colors=p["muted"], labelcolor=p["muted"])
        ax.xaxis.label.set_color(p["text"])
        ax.yaxis.label.set_color(p["text"])
        ax.title.set_color(p["text"])
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_alpha(0)
            for t in leg.get_texts():
                t.set_color(p["text"])
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------- step boards (recursion, memo, table, optimal)

def _as_node(spec):
    kids = [_as_node(k) for k in (spec.get("kids") or ())]
    note = spec.get("note") or None
    return Node(spec.get("label", ""), kids, color=spec.get("color") or "indigo", note=note)


def _label_width(label, size):
    return max(0.82, 0.155 * len(str(label)) * size / FONT + 0.38)


def _tree_span(root, dx, dy, node_h, size):
    """Width and height of a tree laid out the same way as ``tree``."""
    _layout(root, 0, [0.0], dx)

    def width(v):
        return _label_width(v.label, size)

    nodes = []
    stack = [root]
    while stack:
        v = stack.pop()
        nodes.append(v)
        stack.extend(v.children)
    left = min(v.x - width(v) / 2 for v in nodes)
    right = max(v.x + width(v) / 2 for v in nodes)
    depth = max(v.depth for v in nodes)
    extra = 0.5 if any(v.note for v in nodes) else 0.06
    return left, right, depth * dy + node_h + extra


def _fit_tree(spec, inner):
    node = _as_node(spec)
    dx, dy, node_h, size = 1.48, 1.46, 0.56, 8
    left, right, height = 0.0, inner, 1.2
    for _ in range(7):
        left, right, height = _tree_span(node, dx, dy, node_h, size)
        if right - left <= inner or dx <= 0.82:
            break
        dx *= max(0.62, inner / (right - left)) * 0.94
    return node, dx, dy, node_h, size, left, right, height


def _hi_of(step):
    hi = step.get("hi")
    if hi is None:
        return None
    return (int(hi[0]), int(hi[1]))


def _marks_of(step):
    marks = {}
    for key, color in (step.get("marks") or {}).items():
        if isinstance(key, str) and "," in key:
            r, c = key.split(",")
            marks[(int(r), int(c))] = color
        else:
            marks[(int(key[0]), int(key[1]))] = color
    return marks


def _wrap(s, width, size=8):
    """Split s into lines that fit `width` data units at font `size` (same 0.21-units-per-character
    estimate at FONT as tree() and grid(), plus a margin), breaking at spaces; a single word longer
    than the width is broken where needed rather than allowed to spill into the next card."""
    import textwrap
    per_char = 0.21 * size / FONT * 1.05
    chars = max(8, int(width / per_char))
    return textwrap.wrap(str(s), chars, break_long_words=True, break_on_hyphens=False) or [""]


def _content_height(step, inner):
    """Height of the drawing inside a card, under the step title."""
    h = 0.15
    h += 0.34 * (len(_wrap(f"Step 00 — {step.get('title', '')}", inner)) - 1)   # extra title lines
    if step.get("tree"):
        *_, height = _fit_tree(step["tree"], inner)
        h += height
    grid = step.get("grid")
    if grid:
        h += len(grid) * 0.78 + 0.15
    for _row in step.get("rows") or []:
        h += 1.05
    if step.get("chips"):
        h += 1.35
    if step.get("lines"):
        n = sum(len(_wrap(line, inner)) for line in step["lines"])
        h += 0.42 * n + 0.08
    if step.get("eq"):
        h += 0.48 + 0.36 * (len(_wrap(step["eq"], inner)) - 1)
    return max(h, 0.9)


def _draw_tree(ax, p, spec, cx, top, inner):
    """Draw a call tree whose top sits at ``top``. Return the y under it."""
    node, dx, dy, node_h, size, left, right, height = _fit_tree(spec, inner)
    root_y = top - node_h / 2
    tree(ax, p, node, x=cx - (left + right) / 2, y=root_y, dx=dx, dy=dy, node_h=node_h, size=size)
    return top - height


def _draw_grid(ax, p, step, x, top, inner):
    grid = step["grid"]
    if not grid:
        return top
    cols = max(len(r) for r in grid)
    cell = min(0.92, inner / max(cols, 1))
    width = cols * cell
    left = x + (inner - width) / 2
    hi = _hi_of(step)
    marks = _marks_of(step)
    y = top
    for r, row in enumerate(grid):
        for c, raw in enumerate(row):
            val = "" if raw is None else str(raw)
            color = marks.get((r, c))
            if hi == (r, c) and color != "red":
                color = "green"
            if val in {"·", "?", ""} and not color:
                color = "muted"
            face = soft(p, color) if color and color != "muted" else (p["muted_fill"] if color == "muted" else p["cell"])
            edge = col(p, color) if color and color != "muted" else p["line"]
            ax.add_patch(Rectangle((left + c * cell, y - 0.72), cell - 0.06, 0.66, fc=face, ec=edge, lw=1.05))
            text(ax, p, left + c * cell + (cell - 0.06) / 2, y - 0.39, val, size=8)
        y -= 0.78
    return y


def _draw_rows(ax, p, step, x, top, inner):
    y = top
    for row in step.get("rows") or []:
        cells = ["" if v is None else str(v) for v in row.get("cells") or []]
        name = row.get("name") or ""
        hi = row.get("hi")
        text(ax, p, x + 0.15, y - 0.42, name, ha="left", size=8, color="muted")
        name_w = 1.15
        cell = min(0.78, (inner - name_w - 0.2) / max(len(cells), 1))
        for i, val in enumerate(cells):
            color = "green" if hi == i else None
            if val in {"·", "?"}:
                color = "muted"
            face = soft(p, color) if color and color != "muted" else (p["muted_fill"] if color == "muted" else p["cell"])
            edge = col(p, color) if color and color != "muted" else p["line"]
            cx = x + name_w + i * cell
            ax.add_patch(Rectangle((cx, y - 0.7), cell - 0.06, 0.62, fc=face, ec=edge, lw=1.05))
            text(ax, p, cx + (cell - 0.06) / 2, y - 0.39, val, size=8)
        y -= 1.05
    return y


def _draw_chips(ax, p, step, x, top, inner):
    chips = step.get("chips") or []
    if not chips:
        return top
    widths = [max(1.25, 0.13 * len(str(c.get("value", ""))) + 0.85) for c in chips]
    gap = 0.22
    total = sum(widths) + gap * (len(chips) - 1)
    scale = min(1.0, (inner - 0.2) / total) if total else 1.0
    cursor = x + (inner - total * scale) / 2
    for chip, w in zip(chips, widths):
        w *= scale
        color = chip.get("color") or "indigo"
        box(ax, p, cursor, top - 1.05, w, 0.62, str(chip.get("value", "")), color=color, size=8)
        text(ax, p, cursor + w / 2, top - 0.18, str(chip.get("name", "")), size=7, color="muted")
        cursor += w + gap * scale
    return top - 1.35


def _draw_lines(ax, p, step, cx, top, inner):
    y = top - 0.12
    for line in step.get("lines") or []:
        for part in _wrap(line, inner):
            text(ax, p, cx, y, part, size=8)
            y -= 0.42
    return y


def story(ax, p, title, input_line, steps, final):
    """A board of step cards: call trees, tables, or rolling variables.

    ``steps`` items are the dicts produced by ``trace_log``. Two cards per row.
    Active calls are indigo, resolved calls green, bases amber, memo hits violet.
    """
    steps = list(steps or [])
    card_w, gap_x, gap_y = 8.15, 0.42, 0.4
    inner = card_w - 0.7
    # A very wide call tree gets the whole row, so the labels stay readable.
    wide = False
    for step in steps:
        if step.get("tree"):
            *_rest, left, right, _h = _fit_tree(step["tree"], 15)
            if right - left > 9:
                wide = True
                break
    ncols = 1 if wide else 2
    board_w = ncols * card_w + (ncols - 1) * gap_x
    # A long input wraps inside the pill. Extra lines grow the pill upward and lift the title by the
    # same amount, so the legend and the cards below keep their places.
    in_lines = _wrap(input_line, board_w - 1.1)
    extra = 0.32 * (len(in_lines) - 1)
    text(ax, p, board_w / 2, 0.15 + extra, title, size=13, weight="bold")
    pill_w = min(board_w, max(2.4, 0.13 * max(len(s) for s in in_lines) + 1.1))
    box(ax, p, (board_w - pill_w) / 2, -0.85, pill_w, 0.55 + extra, "\n".join(in_lines), color="indigo", size=8)
    legend = (("active", "indigo"), ("resolved", "green"), ("base", "amber"), ("repeated", "violet"))
    lw = 1.55
    lx = (board_w - (len(legend) * (lw + 0.85))) / 2
    for name, color in legend:
        box(ax, p, lx, -1.7, lw, 0.42, name, color=color, size=7)
        lx += lw + 0.85
    heights = [_content_height(s, inner) + 0.85 for s in steps]
    y = -2.35
    for i, step in enumerate(steps):
        col = 0 if ncols == 1 else i % ncols
        if col == 0 and i:
            row_h = max(heights[i - ncols:i]) if ncols > 1 else heights[i - 1]
            # previous row height was applied when that row started; recompute from the row
        if col == 0:
            if i:
                prev = heights[i - ncols:i] if ncols > 1 else [heights[i - 1]]
                y -= max(prev) + gap_y
            row_top = y
        else:
            row_top = y
        x = col * (card_w + gap_x)
        h = heights[i]
        # The row is as tall as its taller card, so neighbours line up.
        if ncols == 2 and i + 1 < len(steps) and i % 2 == 0:
            h = max(h, heights[i + 1])
        elif ncols == 2 and i % 2 == 1:
            h = max(h, heights[i - 1])
        ax.add_patch(FancyBboxPatch(
            (x, row_top - h), card_w, h, boxstyle="round,pad=0,rounding_size=0.16",
            fc=p["cell"], ec=p["line"], lw=1.15, zorder=0,
        ))
        _extend(ax, x, row_top - h, x + card_w, row_top)
        title_lines = _wrap(f"Step {i + 1} — {step.get('title', '')}", inner)
        for k, part in enumerate(title_lines):
            text(ax, p, x + 0.28, row_top - 0.38 - 0.34 * k, part,
                 ha="left", size=8, color="indigo", weight="bold")
        cursor = row_top - 0.72 - 0.34 * (len(title_lines) - 1)
        cx = x + card_w / 2
        if step.get("tree"):
            cursor = _draw_tree(ax, p, step["tree"], cx, cursor, inner)
        if step.get("grid"):
            cursor = _draw_grid(ax, p, step, x + 0.35, cursor, inner)
        if step.get("rows"):
            cursor = _draw_rows(ax, p, step, x + 0.28, cursor, inner)
        if step.get("chips"):
            cursor = _draw_chips(ax, p, step, x + 0.28, cursor, inner)
        if step.get("lines"):
            cursor = _draw_lines(ax, p, step, cx, cursor, inner)
        if step.get("eq"):
            for k, part in enumerate(_wrap(step["eq"], inner)):
                text(ax, p, cx, cursor - 0.08 - 0.36 * k, part, size=8, color="muted", style="italic")
    if steps:
        last_row = (len(steps) - 1) // ncols
        # y is the top of the current (last) row; drop by that row's height
        start = last_row * ncols
        y -= max(heights[start:]) + 0.55
    else:
        y -= 0.4
    # A long final output wraps inside its box, which grows downward.
    per_char = 0.21 * 11.5 / FONT * 1.05          # bold 10 pt is about as wide as regular 11.5 pt
    fin_lines = _wrap(f"Final output: {final}", board_w - 0.8, size=11.5)
    fw = min(board_w, max(3.2, per_char * max(len(s) for s in fin_lines) + 0.8))
    fh = 0.62 + 0.4 * (len(fin_lines) - 1)
    box(ax, p, (board_w - fw) / 2, y - 0.08 - fh, fw, fh, "\n".join(fin_lines), color="amber", size=10,
        weight="bold", dashed=True)
