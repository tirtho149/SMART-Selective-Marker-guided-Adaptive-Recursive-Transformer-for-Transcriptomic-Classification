#!/usr/bin/env bash
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
echo "==================== REPRODUCTION: positive-class F1 vs macro-F1 (all cohorts) ===================="
printf "%-18s %-8s %-22s %-14s %-8s\n" cohort arm "POS-F1 ± sd (P/R)" "macro-F1 ± sd" "N/K"
for t in prostate blca stad pan_meta_pri panmeta_response brca; do
 for a in vanilla biomor; do
  f="results_repro/all_${t}_${a}/${t}__${a}.json"
  if [ -f "$f" ]; then
    $PY -c "
import json;d=json.load(open('$f'))
print('%-18s %-8s POS %5.1f ± %4.1f (P%.2f R%.2f)  macro %5.1f ± %4.1f'%('$t','$a',d['pos_f1'][0]*100,d['pos_f1'][1]*100,d['pos_precision'][0],d['pos_recall'][0],d['macro_f1'][0]*100,d['macro_f1'][1]*100))"
  else printf "%-18s %-8s %s\n" "$t" "$a" "... running"; fi
 done
done
echo "PATH(pos-F1): pancan .88 | prostate .81 | blca .81   baselines span ~.53-.81"
echo "done: $(ls results_repro/all_*/*.json 2>/dev/null | wc -l)/12"
