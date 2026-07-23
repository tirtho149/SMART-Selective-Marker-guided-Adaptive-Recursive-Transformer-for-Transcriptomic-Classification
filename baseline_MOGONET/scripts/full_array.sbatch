#!/usr/bin/env bash
#SBATCH --job-name=mogonet-full
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_MOGONET/work_dirs/full_%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_MOGONET/work_dirs/full_%A_%a.out
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_MOGONET
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1

COHORTS=(prostate blca stad brca pan_meta_pri pan_meta_pri_3modal)
MODS=(mutation,cnv mutation,cnv mutation,cnv mutation,cnv mutation,cnv mutation,cnv,expression)
C=${COHORTS[$SLURM_ARRAY_TASK_ID]}
M=${MODS[$SLURM_ARRAY_TASK_ID]}
echo "[mogonet-full] cohort=$C mods=$M"
COHORT=$C MODALITIES=$M python scripts/mogonet_cv.py
echo "[mogonet-full] cohort=$C rc=$?"
