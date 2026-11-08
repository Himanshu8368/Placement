"""Run check_chapter.py only on chapters whose content changed since they last passed.

"Changed" means the hash of the chapter, its includes, and the local modules it imports
(qmdlib.deps_hash) differs from the one stored when the chapter last passed, in
scripts/.cache/checked.json. Use it after an edit instead of re-running every chapter.

    python scripts/check_changed.py          # changed chapters only
    python scripts/check_changed.py --all    # every chapter, and refresh the record
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_chapter import check  # noqa: E402
from qmdlib import BOOKS, ROOT, chapter_files, deps_hash  # noqa: E402

STATE = ROOT / "scripts" / ".cache" / "checked.json"


def main(argv: list[str]) -> int:
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    todo = []
    for ch in chapter_files():
        rel, h = ch.relative_to(BOOKS).as_posix(), deps_hash(ch)
        if "--all" in argv or state.get(rel) != h:
            todo.append((ch, rel, h))
    if not todo:
        print("OK   nothing changed since the last passing check")
        return 0
    failed = 0
    for ch, rel, h in todo:
        if check(ch):
            state[rel] = h
        else:
            failed += 1
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(f"{'OK  ' if not failed else 'FAIL'} {len(todo)} changed chapter(s) checked, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
