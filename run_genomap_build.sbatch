#!/usr/bin/env bash
# ============================================================================
# Tier-2: build the raw genomap-paper pancreas datasets into genomap features on a
# compute node (the donor concat + GW construction OOMs the login node). Each array
# task builds one dataset -> data/singlecell/<name>/ (capsule format SMART consumes).
#   baron       GSE84133 (human, inline labels)
#   segerstolpe E-MTAB-5061 (labels in sdrf 'inferred cell type')
#   sbatch run_genomap_build.sbatch
# ============================================================================
#SBATCH --job-name=genomap-build
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --array=0-1
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/genomapbuild-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/genomapbuild-%A_%a.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DS=(baron segerstolpe)
D=${DS[$SLURM_ARRAY_TASK_ID]}
echo "[genomap-build] dataset=$D host=$(hostname) date=$(date)"
$PY tools/build_genomap_raw.py --dataset "$D" --hvg 1936 --grid 44 --num_iter 1000
echo "[genomap-build] $D DONE rc=$? date=$(date)"
