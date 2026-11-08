"""Put the book's chapter numbers into the rendered search index.

Quarto numbers chapters in the order they exist (1, 2, 3, ...), while the book uses the global
numbers from outline.yml (229 for the first DP chapter). The pages are renumbered in the browser by
site.js; this script does the same for books/_build/search.json, so search results show
"229.3 Lesson 3" rather than "12.3 Lesson 3". Run after `quarto render` (build.ps1 does).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmdlib import BOOKS  # noqa: E402

SITE = BOOKS / "shared" / "site_after.html"
LEAD = re.compile(r"^(\d+)(?=[ . ]|$)")


def main() -> int:
    search = BOOKS / "_build" / "search.json"
    if not search.exists():
        print("SKIP no _build/search.json (render the HTML first)")
        return 0
    m = re.search(r"window\.DSA_PAGES = (\{.*?\});\n", SITE.read_text(encoding="utf-8"))
    pages = json.loads(m[1]) if m else {}
    entries = json.loads(search.read_text(encoding="utf-8"))
    # Quarto's number for a page = the leading number of its own title entry (the one without a '#').
    qnum = {}
    for e in entries:
        page = e.get("href", "").split("#")[0]
        lm = LEAD.match(e.get("title", ""))
        if page in pages and lm and "#" not in e.get("href", ""):
            qnum[page] = lm[1]
    fixed = 0
    for e in entries:
        page = e.get("href", "").split("#")[0]
        if page not in pages or page not in qnum:
            continue
        old, new = qnum[page], str(pages[page]["num"])
        swap = lambda s: LEAD.sub(lambda m: new if m[1] == old else m[1], s)
        for k in ("title", "section"):
            if isinstance(e.get(k), str) and swap(e[k]) != e[k]:
                e[k] = swap(e[k])
                fixed += 1
        if isinstance(e.get("crumbs"), list):
            e["crumbs"] = [swap(c) if isinstance(c, str) else c for c in e["crumbs"]]
    search.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    print(f"fix_search: {fixed} titles renumbered across {len(qnum)} chapters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
