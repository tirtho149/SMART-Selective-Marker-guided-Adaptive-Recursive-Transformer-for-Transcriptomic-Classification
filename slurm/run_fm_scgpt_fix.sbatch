#!/usr/bin/env bash
#SBATCH --job-name=fm-scgpt-fix
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/fm-scgpt-fix-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/fm-scgpt-fix-%j.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv_scgpt/bin/python
RUN=$PROJ/lit_pipeline/baseline_repos/run_scgpt_pnet.py
cd "$PROJ"; export PYTHONUNBUFFERED=1
echo "[scgpt-fix] $(date)"; nvidia-smi -L || true
$PY "$RUN" --cohorts prostate stad --seeds 0 1 2
echo "[scgpt-fix] DONE rc=$? $(date)"
