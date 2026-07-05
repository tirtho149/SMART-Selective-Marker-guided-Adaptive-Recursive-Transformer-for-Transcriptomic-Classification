#!/usr/bin/env bash
#SBATCH --job-name=fm-scgpt-pnet
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/fm-scgpt-pnet-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/fm-scgpt-pnet-%j.out
# scGPT FM baseline on the CURRENT P-NET cohorts (prostate, blca, stad),
# comparable to SMART: verbatim pathway_tasks split, macro-F1 x100, seeds 0/1/2.
# NOTE: A100/H200-class GPU only (V100/CC7.0 broke scGPT); no V100 in --constraint.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv_scgpt/bin/python
RUN=$PROJ/lit_pipeline/baseline_repos/run_scgpt_pnet.py
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[fm-scgpt-pnet] start $(date)"; nvidia-smi -L || true

# 1) smoke: 1 cohort, 1 seed, subsampled probe -- abort the full run if it breaks
echo "[fm-scgpt-pnet] SMOKE ..."
$PY "$RUN" --cohorts prostate --seeds 0 --smoke || { echo "SMOKE FAILED, aborting"; exit 1; }
echo "[fm-scgpt-pnet] smoke OK; full run ..."

# 2) full: 3 cohorts x 3 seeds
$PY "$RUN" --cohorts prostate blca stad --seeds 0 1 2
echo "[fm-scgpt-pnet] DONE rc=$? $(date)"
