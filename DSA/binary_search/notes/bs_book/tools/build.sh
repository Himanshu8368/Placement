#!/bin/bash
#============================================================
# Build the book or one volume.
#   bash tools/build.sh            whole book      -> main.pdf
#   bash tools/build.sh main       same
#   bash tools/build.sh vol1       Part I only     -> vol1_foundations.pdf
# Regenerates the problem index first, then runs pdflatex twice
# (three times on a clean folder) and prints errors and the page count.
#============================================================
cd "$(dirname "$0")/.." || exit 1
target=${1:-main}
case "$target" in
  main) file=main;             python tools/gen_problem_index.py ;;
  vol1) file=vol1_foundations; python tools/gen_problem_index.py 1 5 problem_index_part1 ;;
  *) echo "unknown target: $target (use main or vol1)"; exit 1 ;;
esac
passes=2
[ -f "$file.aux" ] || passes=3
for i in $(seq 1 $passes); do
  pdflatex -interaction=nonstopmode -halt-on-error "$file.tex" > /dev/null 2>&1
  rc=$?
  echo "pass $i: exit $rc"
  [ $rc -ne 0 ] && break
done
grep -n -A5 '^!' "$file.log" | head -20
grep -o "Output written on $file.pdf ([0-9]* pages" "$file.log"
echo "undefined references: $(grep -c 'Reference.*undefined' "$file.log")"
