#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# GenomicRecursiveFormer: one command to reproduce everything.
#
#   ./run_all.sh                 # default (tractable) scale
#   RMT_EPOCHS=20 ./run_all.sh   # scale up
#   RMT_ONLY=main,random_markers,independent ./run_all.sh   # subset of the suite
#
# Steps: (1) run experiments -> results/*.json
#        (2) generate paper   -> paper/genomicrecursiveformer.tex + refs.bib
#        (3) compile PDF       -> paper/genomicrecursiveformer.pdf
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

# --- resolve a Python with torch (prefer the project venv) -----------------
if [ -x "./.venv/bin/python" ]; then
  PY="./.venv/bin/python"
else
  PY="python3"
fi
echo "[run_all] python: $PY"
$PY -c "import torch" 2>/dev/null || {
  echo "[run_all] ERROR: torch not importable by $PY."
  echo "          Create the env:  python3.11 -m venv .venv && ./.venv/bin/pip install torch 'numpy<2' scikit-learn pandas scipy"
  exit 1
}

# --- experiment scale (env-overridable) ------------------------------------
EPOCHS="${RMT_EPOCHS:-12}"
NHVG="${RMT_NHVG:-2000}"
DMODEL="${RMT_DMODEL:-96}"
NMARKERS="${RMT_NMARKERS:-200}"
DEPTH="${RMT_DEPTH:-4}"
HEADS="${RMT_HEADS:-cancer_type}"
ONLY="${RMT_ONLY:-}"
RESULTS="${RMT_RESULTS:-results}"
PAPER="${RMT_PAPER:-paper}"

echo "[run_all] step 1/3: experiments (epochs=$EPOCHS n_hvg=$NHVG d_model=$DMODEL)"
ONLY_ARG=()
[ -n "$ONLY" ] && ONLY_ARG=(--only "$ONLY")
# bash 3.2 (macOS) + set -u: guard empty-array expansion
$PY -m recursive_marker_transformer.experiments \
  --epochs "$EPOCHS" --n_hvg "$NHVG" --d_model "$DMODEL" \
  --n_markers "$NMARKERS" --depth "$DEPTH" --heads "$HEADS" \
  --outdir "$RESULTS" ${ONLY_ARG[@]+"${ONLY_ARG[@]}"}

echo "[run_all] step 2/3: generate paper"
$PY -m recursive_marker_transformer.make_paper --results "$RESULTS" --outdir "$PAPER"

echo "[run_all] step 3/3: compile PDF"
if command -v pdflatex >/dev/null 2>&1; then
  ( cd "$PAPER"
    pdflatex -interaction=nonstopmode genomicrecursiveformer.tex >/dev/null 2>&1 || true
    if command -v bibtex >/dev/null 2>&1; then bibtex genomicrecursiveformer >/dev/null 2>&1 || true; fi
    pdflatex -interaction=nonstopmode genomicrecursiveformer.tex >/dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode genomicrecursiveformer.tex >/dev/null 2>&1 || true )
  echo "[run_all] DONE -> $PAPER/genomicrecursiveformer.pdf"
else
  echo "[run_all] pdflatex not found; wrote $PAPER/genomicrecursiveformer.tex (compile elsewhere)."
fi
