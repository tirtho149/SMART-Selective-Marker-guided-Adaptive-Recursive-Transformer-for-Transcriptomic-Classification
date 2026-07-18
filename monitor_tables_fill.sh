#!/usr/bin/env bash
# Poll until the Table 2 fill jobs (pancan PM/PC + 3M apple-to-apple) are all done,
# then do a final refresh and print the resulting PM/PC/3M cells. Exits -> re-invokes Claude.
set -uo pipefail
PROJ=/work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
cd "$PROJ"
for t in $(seq 1 240); do   # up to ~20h at 300s
  n=$(squeue -u tirtho -h -o "%j" 2>/dev/null | grep -cE 'bmboth|3m-cv5')
  runs=$(grep -o 'run.ldots' paper/cv5_main_table.tex 2>/dev/null | wc -l)
  echo "[mon $(date +%H:%M:%S)] training_jobs_left=$n run_cells=$runs"
  [ "$n" -eq 0 ] && break
  sleep 300
done
# final regen
bash refresh_cv5.sh >/dev/null 2>&1 || true
echo "==================== TABLES FILLED ===================="
echo "run... cells remaining: $(grep -o 'run.ldots' paper/cv5_main_table.tex 2>/dev/null | wc -l)"
echo "---- PM/PC/3M landed ----"
echo "PM arms: $(ls results_cv5/biomor_ladder_mo/*/pan_meta_pri__*_cv.json 2>/dev/null | wc -l) + inject_mo/both"
echo "PC arms: $(ls results_cv5/biomor_ladder_mo/*/panmeta_response__*_cv.json 2>/dev/null | wc -l) + inject_mo/both"
echo "3M baseline: $(ls results_cv5/mo/*/pan_meta_pri_3modal__*_cv.json 2>/dev/null | wc -l)/10  3M bioMoR: $(ls results_cv5/biomor_ladder_mo/*/pan_meta_pri_3modal__*_cv.json 2>/dev/null | wc -l)/5"
echo "---- bioMoR rows (PM PC 3M columns) ----"
grep -E 'textbf\{bioMoR\}|quad . Token' paper/cv5_main_table.tex | sed 's/{\\scriptsize\$\\pm\$[0-9.]*}//g' | awk -F'&' '{print $1" | PM="$14" PC="$15" 3M="$16}'
