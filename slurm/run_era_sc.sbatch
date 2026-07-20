#!/usr/bin/env bash
#SBATCH --job-name=era-sc
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-5%6
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/era-sc-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/era-sc-%A_%a.out
# ERA (embedding+router+attention-bias, token mode) on the remaining single-cell datasets for
# Table 2. Same protocol (5-fold CV seed 42, epochs 100 pat 15, K=4, M=128). Baron/Muraro already
# done in *_7combo. Writes results/cv5/era/<Dataset>/ERA_cv.json.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DS=(Segerstolpe Lung Oesophagus Tcell Spleen Xin)
python -m recursive_marker_transformer.bio_learned_genomap --dataset "${DS[$SLURM_ARRAY_TASK_ID]}" \
    --modes ERA --cv_folds 5 --K 4 --epochs 100 --patience 15 --n_markers 128 --seed 42 \
    --device cuda --out "$PROJ/results/cv5/era"
echo "[era-sc] DONE rc=$? ds=${DS[$SLURM_ARRAY_TASK_ID]} $(date)"
