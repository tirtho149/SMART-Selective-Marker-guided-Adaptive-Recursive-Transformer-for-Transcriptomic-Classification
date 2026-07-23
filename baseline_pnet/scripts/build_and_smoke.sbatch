#!/usr/bin/env bash
#SBATCH --job-name=pnet-build
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_pnet/work_dirs/build_%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_pnet/work_dirs/build_%j.out
set -uo pipefail
BASE=/work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_pnet
cd "$BASE"
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE

VENV=/work/mech-ai-scratch/tirtho/.venvs_baselines/pnet
if [ ! -f "$VENV/bin/activate" ]; then
  echo "[pnet] building isolated venv (TF) at $VENV"
  mkdir -p /work/mech-ai-scratch/tirtho/.venvs_baselines
  module load python/3.11.11-dnlm5qp 2>/dev/null || true
  PYBIN=$(command -v python3.11 || command -v python3)
  echo "[pnet] base python = $PYBIN ($($PYBIN --version))"
  "$PYBIN" -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --no-input --upgrade pip
  pip install --no-input "tensorflow[and-cuda]>=2.16,<2.20" \
      numpy pandas scipy scikit-learn h5py pyyaml matplotlib seaborn \
      networkx tqdm openpyxl xlrd lifelines psutil
else
  echo "[pnet] reusing venv $VENV"
  source "$VENV/bin/activate"
fi

python -c "import tensorflow as tf; print('TF', tf.__version__, 'GPUs', tf.config.list_physical_devices('GPU'))"

# Build Reactome files for the smoke cohort (prostate) from its filtered_pathways.csv.
COHORT=${PNET_COHORT:-prostate}
DATA_ROOT=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/data
export PNET_REACTOME_DIR="$BASE/_database/pathways/Reactome_$COHORT"
echo "[pnet] building Reactome files for $COHORT (flat topology) -> $PNET_REACTOME_DIR"
python preprocessing/build_reactome_files.py \
    --pathways "$DATA_ROOT/$COHORT/filtered_pathways.csv" \
    --out      "$PNET_REACTOME_DIR"

echo "[pnet] === SMOKE $COHORT (1 fold via PNET_SMOKE) ==="
export PNET_COHORT=$COHORT
export BIOMOR_DATA_ROOT=$DATA_ROOT
export PNET_PARAMS_FILE="./pnet/crossvalidation_average_reg_10_tanh_prostate"
export PNET_SMOKE=1
python train/run_me.py
echo "[pnet-smoke] rc=$?"
