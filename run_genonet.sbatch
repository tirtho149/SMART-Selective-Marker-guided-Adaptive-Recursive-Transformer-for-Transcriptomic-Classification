#!/usr/bin/env bash
#SBATCH --job-name=smart-genonet
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/genonet-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/genonet-%j.out
# =============================================================================
# SMART (MoR) on genomap's genoNet classification tasks, run directly on the
# raw full gene-expression vectors (all data).
#   - 6 BIO5 TCGA tasks from data/tcga/unified_bio5.csv (2738 x 20530 genes)
#     cancer_type, pathologic_stage, pathologic_T, pathologic_N,
#     os_binary, tumor_status
# The 4 single-cell genomap-converted datasets (tabula_muris/common_class/
# prototype/pancreas) were already run -> results_singlecell/.
#   sbatch run_genonet.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[genonet] host=$(hostname) date=$(date)"
nvidia-smi -L
$PY -c "import torch;print('[genonet] torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

$PY -m recursive_marker_transformer.genonet_tasks \
  --csv data/tcga/unified_bio5.csv \
  --out results_genonet \
  --epochs 40 --d_model 128 --n_markers 256 --batch_size 32 \
  --lr 3e-4 --patience 8 --device cuda
RC=$?
echo "[genonet] DONE rc=$RC date=$(date)"
