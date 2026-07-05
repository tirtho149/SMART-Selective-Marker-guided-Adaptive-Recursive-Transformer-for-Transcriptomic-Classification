#!/usr/bin/env bash
#SBATCH --job-name=fm-geneformer-pnet
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200|rtxpro6000|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/fm-geneformer-pnet-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/fm-geneformer-pnet-%j.out
# Geneformer FM baseline on the CURRENT P-NET cohorts (prostate, blca, stad),
# comparable to SMART: verbatim pathway_tasks split, macro-F1 x100, seeds 0/1/2.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv_geneformer/bin/python
RUN=$PROJ/lit_pipeline/baseline_repos/run_geneformer_pnet.py
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
echo "[fm-geneformer-pnet] start $(date)"; nvidia-smi -L || true

# 1) smoke: 1 cohort, 1 seed, 1 epoch, subsampled -- abort the full run if it breaks
echo "[fm-geneformer-pnet] SMOKE ..."
$PY "$RUN" --cohorts prostate --seeds 0 --smoke || { echo "SMOKE FAILED, aborting"; exit 1; }
echo "[fm-geneformer-pnet] smoke OK; full run ..."

# 2) full: 3 cohorts x 3 seeds
$PY "$RUN" --cohorts prostate blca stad --seeds 0 1 2 --epochs 8
echo "[fm-geneformer-pnet] DONE rc=$? $(date)"
