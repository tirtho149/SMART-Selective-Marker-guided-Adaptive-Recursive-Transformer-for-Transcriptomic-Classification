#!/usr/bin/env bash
#SBATCH --job-name=mogat-build
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_MOGAT/work_dirs/build_%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_MOGAT/work_dirs/build_%j.out
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_MOGAT
export PYTHONUNBUFFERED=1

# Strategy: reuse the shared venv (torch 2.10+cu128) for torch/numpy/sklearn via
# PYTHONPATH, and install ONLY the pure-python torch_geometric into a light
# isolated venv (no deps, so numpy is NOT upgraded/shadowed).
SHARED=/work/mech-ai-scratch/tirtho/.venv
SHARED_SP=$SHARED/lib/python3.11/site-packages
VENV=/work/mech-ai-scratch/tirtho/.venvs_baselines/mogat
if [ ! -f "$VENV/bin/activate" ]; then
  echo "[mogat] building light isolated venv at $VENV"
  mkdir -p /work/mech-ai-scratch/tirtho/.venvs_baselines
  "$SHARED/bin/python" -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --no-input --upgrade pip
  # torch_geometric 2.6 is pure-python; install its light deps but NOT numpy/torch.
  pip install --no-input --no-deps torch_geometric==2.6.1
  pip install --no-input openpyxl jinja2 fsspec aiohttp psutil tqdm requests pyparsing
else
  echo "[mogat] reusing venv $VENV"
  source "$VENV/bin/activate"
fi
# Make the shared venv's heavy libs (torch, numpy, sklearn, pandas, anndata)
# importable, but AFTER the local venv so torch_geometric resolves locally.
export PYTHONPATH="$SHARED_SP:${PYTHONPATH:-}"

python -c "import torch,torch_geometric;print('torch',torch.__version__,'pyg',torch_geometric.__version__,'cuda',torch.cuda.is_available())"
echo "[mogat] === SMOKE prostate ==="
COHORT=prostate SMOKE=1 python scripts/mogat_cv.py
echo "[mogat-smoke] rc=$?"
