#!/usr/bin/env bash
#SBATCH --job-name=smart-msweep
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/msweep-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/msweep-%A_%a.out
# Close the gap to linear baselines: SMART learned graph at larger marker budgets
# M in {256,512,1024}. Output: results_learnedM/M<M>/<Dataset>/learned_s<seed>.json
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; PY=$PROJ/../.venv/bin/python
cd "$PROJ"; export PYTHONUNBUFFERED=1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATASETS=(Lung Oesophagus Spleen Tcell Baron Muraro Segerstolpe Xin)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[msweep] $DS $(date)"; nvidia-smi -L || true
for M in 256 512 1024; do
  echo "==== $DS M=$M ===="
  $PY -m recursive_marker_transformer.bio_learned_genomap --dataset "$DS" \
      --modes learned --n_markers $M --seeds 0 1 2 --K 4 --epochs 50 --device cuda \
      --out results_learnedM/M$M
done
echo "[msweep] DONE rc=$? $(date)"
