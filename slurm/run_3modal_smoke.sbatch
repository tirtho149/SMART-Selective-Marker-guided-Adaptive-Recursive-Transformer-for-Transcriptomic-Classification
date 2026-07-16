#!/usr/bin/env bash
#SBATCH --job-name=3modal-smoke
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=2:00:00
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/3modal-smoke-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/3modal-smoke-%j.out
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"; source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1; export PYTHONPATH="$PROJ"
echo "[3modal] loading tri-modal cohort $(date) host=$(hostname)"
$PY -c "
from recursive_marker_transformer.pathway_data import load_cohort
import numpy as np
d=load_cohort('pan_meta_pri_3modal', channels='mut_cnv_expr')
print('LOADED X=%s (N,G,C) y=%s K=%d G=%d M=%d'%(d.X.shape,d.y.shape,len(d.classes),len(d.genes),len(d.pathways)))
u,c=np.unique(d.y,return_counts=True); print('dist',dict(zip(u.tolist(),c.tolist())),'channels',d.channels)
"
echo "[3modal] rc=$? $(date)"
