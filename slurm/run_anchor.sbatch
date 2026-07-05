#!/usr/bin/env bash
#SBATCH --job-name=smart-anchor
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/anchor-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/anchor-%A_%a.out
# Bio-anchor ablation: does an ANNEALED ||A_learned - A_bio||^2 penalty make the bio
# warm-start actually beat random init? Four arms on the 8 SC suites, 5 seeds, M=128:
#   learned         random init (control)
#   learned_bio     bio warm-start, weak init (0.01/0.01), NO anchor  (current null result)
#   learned_bigbio  bio warm-start, strong init (0.05/0.005), NO anchor (init-only ablation)
#   learned_anchor  bio warm-start, strong init + annealed anchor      (the fix)
# Output: results_anchor/<Dataset>/<mode>_s<seed>.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; PY=$PROJ/../.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATASETS=(Lung Oesophagus Spleen Tcell Baron Muraro Segerstolpe Xin)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[anchor] $DS $(date)"; nvidia-smi -L || true
$PY -m recursive_marker_transformer.bio_learned_genomap --dataset "$DS" \
    --modes learned learned_bio learned_bigbio learned_anchor \
    --seeds 0 1 2 3 4 --K 4 --epochs 50 --device cuda \
    --out results_anchor
echo "[anchor] DONE rc=$? $(date)"
