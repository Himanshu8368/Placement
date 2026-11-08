"""Regenerate everything that depends on outline.yml and on which chapters / problems exist.

1. books/_quarto.yml          the sidebar chapter list (only parts with a written chapter appear)
2. books/index.qmd            home page: stat line, volume cards, collapsible outline
3. books/appendix/problem_index.qmd   every judge link in chapters from Part II on, by judge
4. books/problems/<key>/card.qmd      a short card for every registry problem (for reuse)
5. books/shared/site_after.html       page data (numbers, status, prerequisites, next) + books/shared/site.js
6. books/_freeze/...                  cleared for chapters whose includes or trace modules changed
7. books/parts/part-<roman>.qmd       a part opener page: the chapter ladder and what to read first
8. books/appendix/{cheatsheets,question_bank,glossary}.qmd   revision pages (scripts/gen_revise.py)

Chapters live at books/<folder>/NNN_<slug>.qmd; NNN and <folder> come from outline.yml.

Usage: python scripts/gen_book.py
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmdlib import BOOKS, PROBLEMS, ROOT, chapter_files, deps_hash, expand, problem_key_of  # noqa: E402
import gen_revise  # noqa: E402

OUTLINE = ROOT / "outline.yml"
INDEX_FROM_PART = "II"      # the problem index starts with Part II (arrays and strings)

H1 = re.compile(r"^# (.+?)\s*\{#(sec-[\w-]+)[^}]*\}", re.M)
HEADING = re.compile(r"^#{1,6} .*?\{#(sec-[\w-]+)[^}]*\}\s*$")
LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)(?:\{([^}]*)\})?")
IMPORT = re.compile(r"^\s*(?:import|from)\s+(\w+)", re.M)

JUDGES = [
    ("Codeforces", re.compile(r"codeforces\.com/(?:problemset/problem|contest)/(\d+)/(?:problem/)?(\w+)")),
    ("LeetCode", re.compile(r"leetcode\.com/problems/([\w-]+)")),
    ("CSES", re.compile(r"cses\.fi/problemset/task/(\d+)")),
    ("AtCoder", re.compile(r"atcoder\.jp/contests/(\w+)/tasks/(\w+)")),
]


# ---------------------------------------------------------------- outline

def parse_outline() -> list[dict]:
    """volumes -> parts -> chapters from outline.yml, with the file of each written chapter."""
    data = yaml.safe_load(OUTLINE.read_text(encoding="utf-8"))
    volumes, prev_last = [], None       # prev_last: last chapter of the previous part
    for v in data["volumes"]:
        prev_last = None                # a volume's first chapter needs only what outline.yml says
        vol = {"id": f"Volume {v['num']}", "num": v["num"], "name": v["title"], "intro": v.get("intro", ""), "parts": []}
        for p in v["parts"]:
            part = {"roman": p["roman"], "name": p["title"], "folder": p["folder"], "intro": p.get("intro", ""),
                    "volume": v["num"], "chapters": []}
            for k, c in enumerate(p["chapters"]):
                hits = sorted((BOOKS / p["folder"]).glob(f"{c['num']:03d}_*.qmd"))
                file = hits[0].relative_to(BOOKS).as_posix() if hits else None
                # needs: explicit list, else the previous chapter (for a part's first chapter: the last
                # chapter of the previous part)
                default = [p["chapters"][k - 1]["num"]] if k else ([prev_last] if prev_last else [])
                needs = c.get("needs", default)
                part["chapters"].append({"num": c["num"], "title": c["title"], "covers": c.get("covers", ""),
                                         "file": file, "needs": needs, "owner": c.get("owner", ""),
                                         "status": c.get("status", "written" if file else "planned")})
            vol["parts"].append(part)
            prev_last = p["chapters"][-1]["num"] if p["chapters"] else prev_last
        volumes.append(vol)
    return volumes


# ---------------------------------------------------------------- part openers

def part_page(part: dict) -> str:
    return f"parts/part-{part['roman'].lower()}.qmd"


def chapter_index(volumes: list[dict]) -> dict[int, dict]:
    return {c["num"]: c for v in volumes for p in v["parts"] for c in p["chapters"]}


def _ref(c: dict, num: bool = True) -> str:
    text = f"{c['num']} · {c['title']}" if num else c["title"]
    return f"[{text}](../{c['file']})" if c["file"] else text


def write_part_pages(volumes: list[dict]) -> int:
    """One opener page per part that has a written chapter (Quarto shows it before the part's chapters):
    where the part sits in its volume, what to read first, and the chapter ladder."""
    by_num = chapter_index(volumes)
    folder = BOOKS / "parts"
    folder.mkdir(exist_ok=True)
    keep, flat = set(), [p for v in volumes for p in v["parts"]]
    for vol in volumes:
        for part in vol["parts"]:
            if not any(c["file"] for c in part["chapters"]):
                continue
            path = BOOKS / part_page(part)
            keep.add(path.name)
            nums = {c["num"] for c in part["chapters"]}
            before = sorted({n for c in part["chapters"] for n in c["needs"] if n not in nums})
            k = flat.index(part)
            nxt = flat[k + 1] if k + 1 < len(flat) else None
            out = [f"# {part['name']} {{.unnumbered .part-opener}}", "",
                   "<!-- Generated by scripts/gen_book.py from outline.yml. Do not edit. -->", "",
                   f"[Volume {vol['num']} · {vol['name']} · Part {part['roman']}]{{.part-kicker}}", ""]
            # A volume's `intro` in outline.yml is a note for maintainers (source books), not reader text;
            # a part's own `intro`, if given, is shown.
            chs = part["chapters"]
            written = sum(1 for c in chs if c["file"])
            out += [part["intro"] or (f"{len(chs)} chapters, from {chs[0]['num']} to {chs[-1]['num']}; "
                                      f"{written} written so far. Read them in order: each one adds a single "
                                      "idea to the one before."), ""]
            if before:
                out += ["::: {.callout-tip title=\"Read first\"}",
                        *[f"- {_ref(by_num[n])}" for n in before if n in by_num], ":::", ""]
            out += ["## The chapters {.unnumbered}", "", "::: {.part-ladder}"]
            for c in part["chapters"]:
                st = c["status"]
                out.append(f"{c['num']}. **{_ref(c, num=False)}** [{st}]{{.status .st-{st}}} [{c['covers']}]{{.covers}}")
            out += [":::", ""]
            if nxt:
                out += [f"[**Next part.** {nxt['name']} (Part {nxt['roman']}).]{{.flow-bridge}}", ""]
            path.write_text("\n".join(out), encoding="utf-8")
    for old in folder.glob("part-*.qmd"):
        if old.name not in keep:
            old.unlink()
    return len(keep)


# ---------------------------------------------------------------- _quarto.yml

def write_quarto_yml(volumes: list[dict]) -> int:
    lines, written = [], 0
    for vol in volumes:
        for part in vol["parts"]:
            files = [c["file"] for c in part["chapters"] if c["file"]]
            if not files:
                continue
            written += len(files)
            lines.append(f'    - part: {part_page(part)}')
            lines.append("      chapters:")
            lines += [f"        - {f}" for f in files]
    path = BOOKS / "_quarto.yml"
    text = path.read_text(encoding="utf-8")
    block = "\n".join(lines) + ("\n" if lines else "")
    text = re.sub(r"(    # >>> generated chapters[^\n]*\n).*?(    # <<< generated chapters)",
                  lambda m: m[1] + block + m[2], text, flags=re.S)
    path.write_text(text, encoding="utf-8")
    return written


# ---------------------------------------------------------------- home page

VOLUME_COLOURS = ["#4f46e5", "#0891b2", "#059669", "#d97706", "#db2777", "#7c3aed", "#0f766e",
                  "#b45309", "#be185d", "#4338ca", "#047857", "#9333ea", "#0e7490"]


def _short_part(name: str) -> str:
    return name.split(": ", 1)[-1]


def write_outline(volumes: list[dict]) -> None:
    """Home page: a stat line, one 3D card per volume, and a collapsible outline per volume."""
    total = sum(len(p["chapters"]) for v in volumes for p in v["parts"])
    done = sum(1 for v in volumes for p in v["parts"] for c in p["chapters"] if c["file"])
    nparts = sum(len(v["parts"]) for v in volumes)
    nprob = len(list(PROBLEMS.glob("*/problem.qmd"))) if PROBLEMS.exists() else 0

    cards = ['```{=html}', '<p class="home-stats">'
             f'<span><b>{len(volumes)}</b> volumes</span><span><b>{nparts}</b> parts</span>'
             f'<span><b>{total}</b> chapters</span><span><b>{done}</b> written</span>'
             f'<span><b>{nprob}</b> problems in the registry</span></p>',
             '<div class="vol-grid">']
    outline = []
    for k, vol in enumerate(volumes):
        num = vol["num"]
        colour = VOLUME_COLOURS[k % len(VOLUME_COLOURS)]
        chs = [c for p in vol["parts"] for c in p["chapters"]]
        written = sum(1 for c in chs if c["file"])
        first, last = chs[0]["num"], chs[-1]["num"]
        parts_txt = html.escape(" · ".join(_short_part(p["name"]) for p in vol["parts"]))
        name = html.escape(vol["name"])
        pct = round(100 * written / len(chs))
        first_file = next((c["file"] for c in chs if c["file"]), None)
        href = first_file.replace(".qmd", ".html") if first_file else f"#vol-{num}"
        opened = " open" if written else ""
        cards += [
            f'<a class="vol-card tilt" href="{href}" style="--vol:{colour}">',
            f'<span class="vol-num">{num:02d}</span>',
            f'<span class="vol-title">{name}</span>',
            f'<span class="vol-parts">{parts_txt}</span>',
            f'<span class="vol-foot">Ch. {first}–{last} · {written}/{len(chs)} written</span>',
            f'<span class="vol-bar"><i style="width:{pct}%"></i></span>',
            '</a>']
        outline += [f'<details class="vol" id="vol-{num}" style="--vol:{colour}"{opened}>',
                    f'<summary><span class="vol-num">{num:02d}</span> {name}'
                    f'<span class="vol-meta">{len(chs)} chapters · {written} written</span></summary>', ""]
        for part in vol["parts"]:
            outline += [f"[Part {part['roman']} · {part['name']}]{{.part-title}}", ""]
            for c in part["chapters"]:
                title = f"[{c['title']}]({c['file']})" if c["file"] else c["title"]
                st = c["status"]
                pill = f" [{st}]{{.status .st-{st}}}" if c["file"] or st != "planned" else ""
                outline.append(f"{c['num']}. **{title}**{pill} [{c['covers']}]{{.covers}}")
            outline.append("")
        outline += ["</details>", ""]
    cards += ["</div>", "```"]

    path = BOOKS / "index.qmd"
    text = path.read_text(encoding="utf-8")
    for marker, body in (("volumes", cards), ("outline", outline)):
        text = re.sub(rf"(<!-- {marker}:start -->\n).*?(<!-- {marker}:end -->)",
                      lambda m: m[1] + "\n".join(body) + "\n" + m[2], text, flags=re.S)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- problem registry: cards

DIV = re.compile(r"^::: \{\.(p-intuition)\}\n(.*?)^:::\s*$", re.S | re.M)
SOURCE = re.compile(r"^Source: (.+)$", re.M)
PROBLEM_H = re.compile(r"^#{2,4} Problem: (.+?)\s*\{#(sec-[\w-]+)\}", re.M)


def homes() -> dict[str, Path]:
    """registry key -> the chapter that includes its problem.qmd (its home)."""
    out = {}
    for ch in chapter_files():
        for inc in expand(ch)[1]:
            if inc.name == "problem.qmd":
                out[problem_key_of(inc)] = ch
    return out


def write_cards() -> int:
    """card.qmd = a compact pointer to a registry problem: title, source badge, the key idea, and a
    cross-reference to its full write-up. Include it in any chapter that reuses the problem."""
    if not PROBLEMS.exists():
        return 0
    home = homes()
    n = 0
    for prob in sorted(PROBLEMS.glob("*/problem.qmd")):
        key = prob.parent.name
        text = prob.read_text(encoding="utf-8")
        h = PROBLEM_H.search(text)
        if not h:
            continue
        title, label = h.group(1), h.group(2)
        src = SOURCE.search(text)
        source = src.group(1).replace(".solved", "").replace("{ ", "{").strip() if src else ""
        source = re.sub(r"\{\s*\}", "", source)
        idea = DIV.search(text)
        idea_txt = idea.group(2).strip() if idea else ""
        where = f"Solved in full in @{label}." if key in home else "Not yet solved in any chapter."
        card = ["<!-- Generated by scripts/gen_book.py from problem.qmd. Do not edit. -->",
                "", "::: {.problem-card}", "", f"[Problem]{{.card-kicker}} **{title}** {source}", "",
                idea_txt, "", f"[{where}]{{.card-link}}", "", ":::", ""]
        (prob.parent / "card.qmd").write_text("\n".join(card), encoding="utf-8")
        n += 1
    return n


# ---------------------------------------------------------------- freeze invalidation

def invalidate_freeze() -> list[str]:
    """Quarto's freeze only watches the chapter file itself. Hash the chapter, every file it includes,
    and every local module it imports from books/shared or books/traces; when the hash changes,
    delete the chapter's frozen results so the next render executes it again."""
    state_file = BOOKS / ".deps.json"
    old = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    new, cleared = {}, []
    for ch in chapter_files():
        rel = ch.relative_to(BOOKS).as_posix()
        new[rel] = deps_hash(ch)
        if rel in old and old[rel] != new[rel]:
            frozen = BOOKS / "_freeze" / rel[:-4]
            if frozen.exists():
                shutil.rmtree(frozen)
                cleared.append(rel)
    state_file.write_text(json.dumps(new, indent=1), encoding="utf-8")
    return cleared


# ---------------------------------------------------------------- site script + page data

def write_site_script(volumes: list[dict]) -> None:
    """Inline the chapter map and books/shared/site.js into shared/site_after.html (Quarto's
    include-after-body), so every page knows its real chapter number, volume and part."""
    pages, by_num = {}, chapter_index(volumes)
    written = [c for v in volumes for p in v["parts"] for c in p["chapters"] if c["file"]]
    link = lambda c: {"num": c["num"], "title": c["title"],
                      "href": c["file"].replace(".qmd", ".html") if c["file"] else None}
    for vol in volumes:
        for part in vol["parts"]:
            for c in part["chapters"]:
                if c["file"]:
                    k = written.index(c)
                    text = expand(BOOKS / c["file"])[0]
                    prose = re.sub(r"```.*?```", " ", text, flags=re.S)
                    words = len(re.findall(r"[A-Za-z]{2,}", prose))
                    pages[c["file"].replace(".qmd", ".html")] = {
                        "num": c["num"], "title": c["title"], "volume": vol["num"], "volumeTitle": vol["name"],
                        "part": part["roman"], "partTitle": part["name"], "status": c["status"],
                        "owner": c["owner"], "minutes": max(1, round(words / 200)),
                        "needs": [link(by_num[n]) for n in c["needs"] if n in by_num],
                        "next": link(written[k + 1]) if k + 1 < len(written) else None}
    js = (BOOKS / "shared" / "site.js").read_text(encoding="utf-8") if (BOOKS / "shared" / "site.js").exists() else ""
    out = ["<!-- Generated by scripts/gen_book.py: page data + shared/site.js. Do not edit. -->",
           "<script>", "window.DSA_PAGES = " + json.dumps(pages, ensure_ascii=False) + ";", js, "</script>", ""]
    (BOOKS / "shared" / "site_after.html").write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------- problem index

def judge_of(url: str):
    for name, rx in JUDGES:
        if m := rx.search(url):
            return name, m
    return None, None


def sort_key(judge: str, m: re.Match) -> tuple:
    if judge == "Codeforces":
        return (int(m[1]), m[2])
    if judge == "CSES":
        return (int(m[1]), "")
    return (0, m.group(0))


def indexed_chapters(volumes: list[dict]) -> list[Path]:
    files, started = [], False
    for vol in volumes:
        for part in vol["parts"]:
            started = started or part["roman"] == INDEX_FROM_PART
            if started:
                files += [BOOKS / c["file"] for c in part["chapters"] if c["file"]]
    return files


def scan_problems(files: list[Path]) -> dict:
    """judge -> url -> {md, key, solved, where: [labels]}; chapters are read with includes expanded."""
    found: dict = defaultdict(dict)
    for qmd in files:
        label, in_code = None, False
        for line in expand(qmd)[0].splitlines():
            if line.startswith("```"):
                in_code = not in_code
            if in_code:
                continue
            if m := HEADING.match(line):
                label = m[1]
            for lm in LINK.finditer(line):
                judge, jm = judge_of(lm[2])
                if not judge:
                    continue
                attrs = lm[3] or ""
                entry = found[judge].setdefault(lm[2], {"md": lm[0], "key": sort_key(judge, jm),
                                                        "solved": False, "where": []})
                if ".solved" in attrs.split():
                    entry["solved"] = True
                    entry["md"] = lm[0]
                if label and label not in entry["where"]:
                    entry["where"].append(label)
    return found


def write_problem_index(found: dict) -> None:
    n_all = sum(len(v) for v in found.values())
    n_solved = sum(e["solved"] for v in found.values() for e in v.values())
    out = ["# Problem Index {#sec-problem-index}", "", "<!-- Generated by scripts/gen_book.py. Do not edit by hand. -->", ""]
    if found:
        out += [f"Every problem from Part {INDEX_FROM_PART} (arrays) onwards: **{n_all}** in total, "
                f"**{n_solved}** solved in full and **{n_all - n_solved}** for practice. "
                "*Where* links to the section that uses it. Tick the box next to a practice problem on its "
                "chapter page to track what you have solved (stored in this browser).", "",
                "```{=html}", '<div class="pb-bar" role="search">'
                '<input class="pb-q" type="search" placeholder="Filter by name, id or chapter" aria-label="Filter problems">'
                '<select class="pb-role" aria-label="Role"><option value="">Any role</option>'
                '<option>Solved</option><option>Practice</option></select>'
                '<select class="pb-mine" aria-label="Your progress"><option value="">All</option>'
                '<option value="todo">Not done yet</option><option value="done">Done</option></select>'
                '<label>Rating <input class="pb-min" type="number" step="100" placeholder="min" aria-label="Minimum rating">'
                '<input class="pb-max" type="number" step="100" placeholder="max" aria-label="Maximum rating"></label>'
                '<span class="pb-count" aria-live="polite"></span></div>', "```", ""]
    else:
        out += ["The problem list starts with Part II (arrays and strings).", ""]
    for judge, _ in JUDGES:
        entries = found.get(judge)
        if not entries:
            continue
        out += [f"## {judge} {{.unnumbered}}", "", "| Problem | Level | Role | Where |", "|---|---|---|---|"]
        for url, e in sorted(entries.items(), key=lambda kv: kv[1]["key"]):
            md = e["md"].replace("|", "\\|")
            lv = re.search(r'(?:rating|difficulty)="?(\w+)', e["md"])
            role = "**Solved**" if e["solved"] else "Practice"
            where = ", ".join(f"@{w}" for w in e["where"]) or "—"
            out.append(f"| {md} | {lv[1] if lv else '—'} | {role} | {where} |")
        out.append("")
    (BOOKS / "appendix").mkdir(exist_ok=True)
    (BOOKS / "appendix" / "problem_index.qmd").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    vols = parse_outline()
    n = write_quarto_yml(vols)
    write_outline(vols)
    nparts = write_part_pages(vols)
    ncards = write_cards()
    ncs = gen_revise.write_cheatsheets(vols)
    nq = gen_revise.write_question_bank(vols)
    ng, missing = gen_revise.write_glossary()
    for m in missing:
        print("WARN glossary label not found:", m)
    write_site_script(vols)
    cleared = invalidate_freeze()
    probs = scan_problems(indexed_chapters(vols))
    write_problem_index(probs)
    print(f"part openers: {nparts}; cheat sheets: {ncs}; questions: {nq}; glossary terms: {ng}")
    print(f"chapters written: {n}; registry cards: {ncards}; problems indexed: "
          f"{sum(len(v) for v in probs.values())}; re-run (deps changed): {', '.join(cleared) or 'none'}")
