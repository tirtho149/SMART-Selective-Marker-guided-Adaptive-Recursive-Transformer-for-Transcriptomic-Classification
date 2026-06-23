#!/usr/bin/env bash
#SBATCH --job-name=smart-sweeps
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
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sweeps-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/sweeps-%j.out
# Multi-seed ablations (5 seeds) + recursion-depth sweep + marker-count (M) sweep
# on the HARD genoNet tasks (pathologic stage/T/N), all 20530 genes.
# Resumable: each (exp,task,variant,seed) cell is a separate JSON, finished cells
# are skipped on requeue -> results_sweeps/{ablate,depth,markers}/
#   sbatch run_sweeps.sbatch
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[sweeps] host=$(hostname) date=$(date)"; nvidia-smi -L
$PY -m recursive_marker_transformer.sweeps \
  --csv data/tcga/unified_bio5.csv --out results_sweeps \
  --exp ablate depth markers --device cuda
echo "[sweeps] DONE rc=$? date=$(date)"
