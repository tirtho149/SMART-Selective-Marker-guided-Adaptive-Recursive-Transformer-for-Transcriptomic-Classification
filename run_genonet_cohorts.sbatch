#!/usr/bin/env bash
#SBATCH --job-name=smart-cohorts
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cohorts-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cohorts-%j.out
# =============================================================================
# Cohort-wise modeling: train each clinical/outcome task SEPARATELY within each
# cancer cohort (breast / lung / thyroid / head_neck), so staging/survival are
# learned without cross-cohort confounding. cancer_type is omitted (it is the
# cohort identity). Shared recursive model.
#   sbatch run_genonet_cohorts.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[cohorts] host=$(hostname) date=$(date)"
nvidia-smi -L
$PY -c "import torch;print('[cohorts] torch',torch.__version__,'cuda',torch.cuda.is_available())"

COMMON="--csv data/tcga/unified_bio5.csv --epochs 40 --d_model 128 --n_markers 256 \
        --batch_size 32 --lr 3e-4 --patience 8 --device cuda \
        --tasks pathologic_stage pathologic_T pathologic_N os_binary tumor_status"

for COH in breast lung thyroid head_neck; do
  echo "==================== COHORT: $COH ===================="
  $PY -m recursive_marker_transformer.genonet_tasks $COMMON \
      --cohort "$COH" --out "results_cohorts/$COH"
  echo "[cohorts] $COH rc=$? date=$(date)"
done

echo "[cohorts] DONE date=$(date)"
