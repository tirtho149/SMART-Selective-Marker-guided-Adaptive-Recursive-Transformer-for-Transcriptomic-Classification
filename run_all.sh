#!/usr/bin/env bash
# ============================================================================
# SMART: Selective Marker-guided Adaptive Recursive Transformer
#        for Transcriptomic Classification
#
# Authors:
#   Koushik Howlader   - Iowa State University
#   Tirtho Roy         - Iowa State University
#   Md Tauhidul Islam  - Stanford University
#   Wei Le             - Iowa State University
#
# Copyright (c) 2026 The SMART Authors. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, without the express written
# permission of the authors is STRICTLY PROHIBITED and will be prosecuted to
# the fullest extent permitted by law. See the LICENSE file for full terms.
# ============================================================================

# ---------------------------------------------------------------------------
# SMART: one command to reproduce everything (SINGLE-CELL ONLY -- Tabula Muris
# + pancreas, genomap-native; there is NO TCGA / bulk content).
#
#   ./run_all.sh                 # full genomap protocol (slow on CPU; use a GPU)
#   RMT_EPOCHS=10 ./run_all.sh   # quick smoke
#
# On a Slurm cluster, prefer the array jobs (one GPU each):
#   sbatch run_sc_interaction.sbatch   # biology-informed-router ablation (headline)
#   sbatch run_sc_arch.sbatch          # architecture / marker-selection ablation
# then just step 4-5 below (make_paper + compile).
#
# Steps: (1) param-efficiency table     -> results_sc/param_efficiency.json
#        (2) biology-informed router     -> results_sc_interaction/ (headline)
#        (3) architecture ablation       -> results_singlecell_arch/
#        (4) generate paper              -> paper/genomicrecursiveformer.tex + refs.bib
#        (5) compile PDF                 -> paper/genomicrecursiveformer.pdf
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

# --- resolve a Python with torch (prefer the shared venv) ------------------
if [ -x "/work/mech-ai-scratch/tirtho/.venv/bin/python" ]; then
  PY="/work/mech-ai-scratch/tirtho/.venv/bin/python"
elif [ -x "./.venv/bin/python" ]; then
  PY="./.venv/bin/python"
else
  PY="python3"
fi
echo "[run_all] python: $PY"
$PY -c "import torch" 2>/dev/null || {
  echo "[run_all] ERROR: torch not importable by $PY."; exit 1; }

# --- experiment scale (env-overridable) ------------------------------------
EPOCHS="${RMT_EPOCHS:-150}"
DMODEL="${RMT_DMODEL:-96}"
NMARKERS="${RMT_NMARKERS:-128}"
SEEDS="${RMT_SEEDS:-0 1 2}"
DEVICE="${RMT_DEVICE:-auto}"
PAPER="${RMT_PAPER:-paper}"
COMMON="--epochs $EPOCHS --d_model $DMODEL --n_markers $NMARKERS \
  --batch_size 128 --lr 0.001 --weight_decay 0.00001 --patience 15 --device $DEVICE"

echo "[run_all] step 1/5: parameter-efficiency table"
mkdir -p results_sc
$PY - <<PYEOF
import json
from recursive_marker_transformer.recursion import RecursiveStack
rows=[]
for K in (1,2,4,6,8):
    sh=sum(p.numel() for p in RecursiveStack($DMODEL,4,2*$DMODEL,0.1,depth=K,share_weights=True).parameters())
    ind=sum(p.numel() for p in RecursiveStack($DMODEL,4,2*$DMODEL,0.1,depth=K,share_weights=False).parameters())
    rows.append({"depth":K,"shared_params":sh,"independent_params":ind,"ratio":ind/sh})
json.dump(rows, open("results_sc/param_efficiency.json","w"), indent=1)
print("  wrote results_sc/param_efficiency.json")
PYEOF

echo "[run_all] step 2/5: biology-informed-router ablation (none/coexpr/random)"
$PY -m recursive_marker_transformer.sc_interaction \
  --datasets tabula_muris pancreas --modes none coexpr random --seeds $SEEDS \
  --out results_sc_interaction $COMMON

echo "[run_all] step 3/5: architecture / marker-selection ablation"
for SEED in $SEEDS; do
  for spec in "shared:--recursion_mode expert" \
              "independent:--recursion_mode expert --no_share_weights" \
              "token:--recursion_mode token" \
              "fixed:--recursion_mode fixed" \
              "depth1:--recursion_mode expert --recursion_depth 1" \
              "marker_random:--recursion_mode expert --marker_mode random" \
              "marker_var:--recursion_mode expert --marker_mode variance"; do
    V="${spec%%:*}"; F="${spec#*:}"
    $PY -m recursive_marker_transformer.singlecell \
      --data data/singlecell --out "results_singlecell_arch/$V/s$SEED" \
      --datasets tabula_muris pancreas --seed $SEED $COMMON $F
  done
done

echo "[run_all] step 4/5: generate paper"
$PY -m recursive_marker_transformer.make_paper --outdir "$PAPER"

echo "[run_all] step 5/5: compile PDF"
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
