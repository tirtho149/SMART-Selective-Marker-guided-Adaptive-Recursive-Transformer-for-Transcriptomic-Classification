#!/usr/bin/env bash
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
echo "======= TABLE 2 LADDER: threshold-tuned positive-class F1 (macro-F1) ======="
printf "%-11s | %-13s %-13s %-13s %-13s %-13s\n" "row" prostate blca stad pan_meta_pri panmeta_resp
for arm in vanilla fixed_k2 fixed_k3 fixed_k4 expert_k2 expert_k3 expert_k4 token_k2 token_k3 token_k4; do
  printf "%-11s |" "$arm"
  for t in prostate blca stad pan_meta_pri panmeta_response; do
    f="results_repro/ladder/${t}_${arm}/${t}__${arm}.json"
    if [ -f "$f" ]; then
      $PY -c "import json;d=json.load(open('$f'));print(' %5.1f(%4.1f)   '%(d['pos_f1'][0]*100,d['macro_f1'][0]*100),end='')" 2>/dev/null
    else printf " %-13s" " ...";  fi
  done; echo
done
echo "PATH(pos-F1): prostate .81 | blca .81 | pancan .88   (macro in parens)"
echo "done: $(ls results_repro/ladder/*/*.json 2>/dev/null|wc -l)/50"
