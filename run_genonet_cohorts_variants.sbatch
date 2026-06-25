#!/usr/bin/env bash
#SBATCH --job-name=smart-cohort-var
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cohort-var-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cohort-var-%j.out
# =============================================================================
# Cohort-wise modeling for the Independent (untied) and MoR (token) variants,
# so Table 4 can compare all THREE model types per (cohort x task). The shared
# recursive model is produced by run_genonet_cohorts.sbatch.
#   sbatch run_genonet_cohorts_variants.sbatch
# =============================================================================
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[cohort-var] host=$(hostname) date=$(date)"
nvidia-smi -L

COMMON="--csv data/tcga/unified_bio5.csv --epochs 40 --d_model 128 --n_markers 256 \
        --batch_size 32 --lr 3e-4 --patience 8 --device cuda \
        --tasks pathologic_stage pathologic_T pathologic_N os_binary tumor_status"

for COH in breast lung thyroid head_neck; do
  echo "########## INDEPENDENT cohort=$COH ##########"
  $PY -m recursive_marker_transformer.genonet_tasks $COMMON \
      --cohort "$COH" --recursion_mode expert --no_share_weights \
      --out "results_cohorts_independent/$COH"
  echo "[cohort-var] indep $COH rc=$? date=$(date)"

  echo "########## MoR cohort=$COH ##########"
  $PY -m recursive_marker_transformer.genonet_tasks $COMMON \
      --cohort "$COH" --recursion_mode token \
      --out "results_cohorts_mor/$COH"
  echo "[cohort-var] mor $COH rc=$? date=$(date)"
done

echo "[cohort-var] DONE date=$(date)"
