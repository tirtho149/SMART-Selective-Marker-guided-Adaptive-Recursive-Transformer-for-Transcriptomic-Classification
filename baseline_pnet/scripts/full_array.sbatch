#!/usr/bin/env bash
#SBATCH --job-name=pnet-full
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_pnet/work_dirs/full_%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_pnet/work_dirs/full_%A_%a.out
set -uo pipefail
BASE=/work/mech-ai-scratch/tirtho/bioMOR-baselines/baseline_pnet
cd "$BASE"
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE
source /work/mech-ai-scratch/tirtho/.venvs_baselines/pnet/bin/activate

DATA_ROOT=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/data
export BIOMOR_DATA_ROOT=$DATA_ROOT

COHORTS=(prostate blca stad pan_meta_pri)
PFILES=(crossvalidation_average_reg_10_tanh_prostate \
        crossvalidation_average_reg_10_tanh_blca \
        crossvalidation_average_reg_10_tanh_stad \
        crossvalidation_average_reg_10_tanh_pan_meta_pri)
C=${COHORTS[$SLURM_ARRAY_TASK_ID]}
PF=${PFILES[$SLURM_ARRAY_TASK_ID]}

# Per-cohort Reactome dir so concurrent tasks don't clobber each other.
export PNET_REACTOME_DIR="$BASE/_database/pathways/Reactome_$C"
echo "[pnet-full] cohort=$C reactome=$PNET_REACTOME_DIR"
python preprocessing/build_reactome_files.py \
    --pathways "$DATA_ROOT/$C/filtered_pathways.csv" \
    --out      "$PNET_REACTOME_DIR"

export PNET_COHORT=$C
export PNET_PARAMS_FILE="./pnet/$PF"
export PNET_SMOKE=0
python train/run_me.py
echo "[pnet-full] cohort=$C rc=$?"
