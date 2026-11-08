"""Generate the Quarto book from the volume modules."""

from __future__ import annotations

import content_01
import content_03
import content_05
import content_07
import content_09
import content_11
import content_13
import content_15
import content_17
from booklib import VOLUMES, write_all

def main() -> None:
    # The content modules register volumes on import.
    assert content_01 and content_17
    write_all()
    print(f"volumes: {len(VOLUMES)}")
    for vol in VOLUMES:
        print(f"  {vol['id']}  {len(vol['chapters']):2d}  {vol['title']}")

if __name__ == "__main__":
    main()
