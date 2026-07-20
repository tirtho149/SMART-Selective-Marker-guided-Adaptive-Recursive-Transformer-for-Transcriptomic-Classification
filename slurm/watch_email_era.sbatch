#!/usr/bin/env bash
#SBATCH --job-name=watch-era-mail
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --requeue
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/watch-era-mail-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/logs/watch-era-mail-%j.out
# Runs AFTER finalize-era (afterany dep). Emails the completion summary of the bioMoR-ERA
# Table 2 + Figure 2 rebuild. Native SBATCH --mail-type is a backup; the sendmail body carries
# the actual RUN_COMPLETE_ERA.txt content (kept Fig-2 datasets + table rebuild log).
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer; cd "$PROJ"
MARK="$PROJ/RUN_COMPLETE_ERA.txt"
TO=tirtho@iastate.edu
if [ -f "$MARK" ]; then
  SUBJ="[bioMoR] ERA Table 2 + Figure 2 COMPLETE"
  BODY=$(cat "$MARK")
else
  SUBJ="[bioMoR] ERA finalize ran but RUN_COMPLETE_ERA.txt MISSING (check jobs)"
  BODY="finalize-era completed but $MARK is absent. Some array tasks may have failed/requeued. Check squeue -u tirtho and logs/finalize-era-*.out"
fi
{
  echo "To: $TO"
  echo "Subject: $SUBJ"
  echo "From: tirtho-bioMoR@nova.hpc"
  echo "Content-Type: text/plain; charset=utf-8"
  echo
  echo "bioMoR ERA pipeline finished on $(date)."
  echo "Repo: $PROJ"
  echo
  echo "===== RUN_COMPLETE_ERA.txt ====="
  echo "$BODY"
} | /usr/sbin/sendmail -t
echo "[watch-era-mail] emailed $TO subj='$SUBJ' rc=$? $(date)"
