#!/bin/bash
# Download the official CS249r release PDFs/EPUBs bundled with Part XV (books/cs249r/).
# Vol 3 and Vol 4 are still being written; add their release tags here when they ship.
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/books/cs249r"
mkdir -p "$DEST"
BASE=https://github.com/harvard-edge/cs249r_book/releases/download
for f in \
  vol1-v0.8.0/Machine-Learning-Systems-Vol1.pdf \
  vol1-v0.8.0/Machine-Learning-Systems-Vol1.epub \
  vol2-v0.3.0/Machine-Learning-Systems-Vol2.pdf \
  vol2-v0.3.0/Machine-Learning-Systems-Vol2.epub; do
  name=$(basename "$f")
  if [ -s "$DEST/$name" ]; then echo "SKIP $name"; continue; fi
  curl -sSL --fail -o "$DEST/$name" "$BASE/$f" && echo "OK   $name"
done
