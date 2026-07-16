#!/usr/bin/env bash
#SBATCH --job-name=cv5-refresh
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --requeue
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/cv5_refresh_%j.log
# Persistent table-refresh: runs the rebuild loop for the walltime; RESUBMITS itself
# unless the loop reported completion. Independent of any chat session.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
cd "$PROJ"
RUN_SECONDS=$(( 24*3600 - 300 ))
timeout --signal=TERM "${RUN_SECONDS}s" bash "$PROJ/refresh_loop.sh"
rc=$?
echo "[sbatch] refresh loop exited rc=$rc"
# rc 0 => loop finished (all done) => do NOT resubmit. Otherwise (timeout=124) resubmit.
if [ "$rc" -ne 0 ]; then
  echo "[sbatch] resubmitting for continued refresh"; sbatch "$PROJ/refresh_loop.sbatch"
fi
