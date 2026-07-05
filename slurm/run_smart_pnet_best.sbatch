#!/usr/bin/env bash
#SBATCH --job-name=smart-pnetbest
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pnetbest-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/pnetbest-%A_%a.out
# SMART best config on P-NET, FULL multimodality (mut_cnv, N,G,2). Two MoR arms (expert,
# token), Reactome routing, pathway markers, 3 SEEDS via per-seed output dirs (the tag
# has no seed, so seeds MUST go to separate dirs). Arm chosen later by validation F1.
# Output: results_smart_pnet_best/<arm>/s<seed>/<coh>__...json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; PY=$PROJ/../.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
TASKS=(prostate blca stad); TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "[pnetbest] $TASK $(date)"; nvidia-smi -L || true
for SEED in 0 1 2; do
  for ARM in expert token; do
    echo "==== $TASK arm=$ARM seed=$SEED ===="
    $PY -m recursive_marker_transformer.pathway_tasks --task "$TASK" \
        --channels mut_cnv --marker_mode pathway --recursion_mode $ARM \
        --gene_interaction reactome --n_markers 256 --d_model 128 \
        --epochs 60 --lr 3e-4 --patience 8 --seed $SEED --device cuda \
        --out results_smart_pnet_best/$ARM/s$SEED || echo "  arm=$ARM seed=$SEED FAILED"
  done
done
echo "[pnetbest] DONE rc=$? $(date)"
