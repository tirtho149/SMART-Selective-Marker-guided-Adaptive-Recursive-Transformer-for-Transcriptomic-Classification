#!/usr/bin/env bash
#SBATCH --job-name=verify-pipeline
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --array=0-5%6
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/verify-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/verify-%A_%a.out
# GPU VERIFICATION PASS: run ONE reduced-scope task per distinct training entry point
# (cv_folds=2, epochs=3) to prove every README training command runs end-to-end on GPU.
# Writes to results/_verify/ ONLY -- never touches the committed results/ the paper uses.
#   0 singlecell            (SC ladder)
#   1 pathway_tasks         (MO ladder / injection)
#   2 bio_learned_genomap   (bioMoR single-cell)
#   3 bio_redesign_curated  (bioMoR multi-omics / biorouter)
#   4 baselines11           (Table 3 classical baselines)
#   5 reproduce_path        (supplementary pos-F1)
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd "$PROJ"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
V="$PROJ/results/_verify"
COMMON="--cv_folds 2 --seed 42 --epochs 3 --patience 2 --device cuda"

echo "[verify] $(date) task=$SLURM_ARRAY_TASK_ID host=$(hostname)"; nvidia-smi -L || true
case "$SLURM_ARRAY_TASK_ID" in
  0) python -m recursive_marker_transformer.singlecell --data data/singlecell --datasets baron \
        --d_model 96 --n_markers 128 --batch_size 128 --lr 0.001 --weight_decay 0.00001 \
        --recursion_mode expert --recursion_depth 4 $COMMON --out "$V/singlecell" ;;
  1) python -m recursive_marker_transformer.pathway_tasks --task prostate --channels mut_cnv \
        --marker_mode pathway --n_markers 256 --d_model 128 --batch_size 32 --lr 3e-4 \
        --gene_interaction reactome $COMMON --out "$V/pathway_tasks" ;;
  2) python -m recursive_marker_transformer.bio_learned_genomap --dataset Baron --modes bio_both \
        --K 4 --n_markers 128 $COMMON --out "$V/biomor_sc" ;;
  3) python -m recursive_marker_transformer.bio_redesign_curated --family pnet --cohort prostate \
        --task response --modes learned --K 4 --cap_genes 3000 $COMMON --out "$V/biomor_mo" ;;
  4) python -m recursive_marker_transformer.baselines11 --datasets Lung --cv_folds 2 \
        --out "$V/baselines" ;;
  5) python scripts/reproduce_path.py --task prostate --arm vanilla --channels mut_cnv \
        --pos_class 1 --path_protocol --seeds 42 --d_model 128 $COMMON --out "$V/repro" ;;
esac
echo "[verify] DONE rc=$? $(date) task=$SLURM_ARRAY_TASK_ID"
