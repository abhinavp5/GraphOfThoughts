#!/usr/bin/env bash
# Build GraphOfThoughts_paper_Overleaf.zip from paper/ with a flat root (Overleaf-friendly).
# Do not use "zip -r … ../existing.zip" to merge updates—that duplicates paths such as paper/FinalReport.tex.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/GraphOfThoughts_paper_Overleaf.zip"
rm -f "$OUT"
cd "$REPO/paper"
zip -r "$OUT" \
  FinalReport.tex \
  acmart.cls \
  ACM-Reference-Format.bst \
  sample-base.bib \
  acmnumeric.bbx \
  acmnumeric.cbx \
  acmauthoryear.bbx \
  acmauthoryear.cbx \
  acmdatamodel.dbx \
  acm-jdslogo.png \
  README.txt \
  figures \
  cross_algo_summaries \
  r1_n10_metrics
echo "Wrote $OUT"
unzip -l "$OUT" | grep -i '[Ff]inal[Rr]eport' || true
