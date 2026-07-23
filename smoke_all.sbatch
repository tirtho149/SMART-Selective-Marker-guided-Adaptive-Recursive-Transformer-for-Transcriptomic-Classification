#!/usr/bin/env bash
#SBATCH --job-name=biomor-smoke
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|l40s|h200|rtxpro6000"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/smoke_%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/bioMOR-baselines/logs/smoke_%j.out
set -uo pipefail
ROOT=/work/mech-ai-scratch/tirtho/bioMOR-baselines
cd "$ROOT"
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONUNBUFFERED=1 SMOKE=1

echo "############ CNN ei (prostate) ############"
python baseline_CNN/scripts/cnn_cv.py --model ei --cohort prostate; echo "rc_cnn_ei=$?"
echo "############ CNN li (prostate) ############"
python baseline_CNN/scripts/cnn_cv.py --model li --cohort prostate; echo "rc_cnn_li=$?"
echo "############ DeePathNet (prostate) ############"
python baseline_DeePathNet/scripts/deepathnet_biomor_cv.py baseline_DeePathNet/configs/biomor/prostate.json; echo "rc_dpn=$?"
echo "############ PathCNN (prostate) ############"
python baseline_PathCNN/scripts/pathcnn_cv.py --cohort prostate; echo "rc_pathcnn=$?"
echo "############ Pathformer (prostate) ############"
python baseline_Pathformer/scripts/pathformer_cv.py --cohort prostate; echo "rc_pathformer=$?"
echo "############ SMOKE DONE ############"
