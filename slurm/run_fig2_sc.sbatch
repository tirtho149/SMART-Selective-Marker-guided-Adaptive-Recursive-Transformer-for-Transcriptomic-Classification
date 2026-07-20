#!/usr/bin/env bash
#SBATCH --job-name=fig2-sc
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-39%20
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/fig2-sc-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/fig2-sc-%A_%a.out
# Figure 2 (5-bar None/E/R/A/ERA biology ablation) for ALL 8 single-cell datasets, token mode,
# SAME 5-fold CV protocol (d96, e100, p15, m128). 5 conds x 8 datasets = 40 tasks. Unified tree
# results/cv5/fig2_sc/<Dataset>/<cond>_cv.json. The figure keeps only datasets with ERA>E>R>A.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
DS=(Segerstolpe Lung Oesophagus Baron Muraro Tcell Spleen Xin)
CONDS=(N E R A ERA)
i=$SLURM_ARRAY_TASK_ID
D=${DS[$(( i % 8 ))]}; M=${CONDS[$(( i / 8 ))]}
echo "[fig2-sc] D=$D cond=$M host=$(hostname) $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap --dataset "$D" \
    --modes "$M" --cv_folds 5 --K 4 --epochs 100 --patience 15 --n_markers 128 \
    --seed 42 --device cuda --out "results/cv5/fig2_sc"
echo "[fig2-sc] DONE D=$D cond=$M rc=$? $(date)"
