#!/usr/bin/env bash
#SBATCH --job-name=smoke-era
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/smoke-era-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/smoke-era-%j.out
# CPU smoke: verify the NEW canonical bioMoR (bio_era = E+R+A) runs end-to-end in BOTH
# expert- and token-choice routing on a small single-cell dataset. Exercises the expert-choice
# per-step (k,k) attention sub-mask gather (router.py / recursion.py). 2 epochs, no CV.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[smoke] EXPERT bio_era ==============================="
python -m recursive_marker_transformer.bio_learned_genomap --dataset Xin \
    --modes bio_era --seeds 0 --K 4 --epochs 2 --n_markers 64 --patience 5 \
    --recursion_mode expert --device cpu --out "$PROJ/results/cv5/_smoke_era/expert" --force
echo "[smoke] rc_expert=$?"
echo "[smoke] TOKEN bio_era ==============================="
python -m recursive_marker_transformer.bio_learned_genomap --dataset Xin \
    --modes bio_era --seeds 0 --K 4 --epochs 2 --n_markers 64 --patience 5 \
    --recursion_mode token --device cpu --out "$PROJ/results/cv5/_smoke_era/token" --force
echo "[smoke] rc_token=$? DONE $(date)"
