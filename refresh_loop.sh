#!/usr/bin/env bash
# CV5_REFRESH_LOOP: rebuild macro + pos-F1 tables so Overleaf (pushed by overleaf-sync
# daemon) updates near real-time. Exits 0 when every run has landed.
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer
PY=/work/mech-ai-scratch/tirtho/.venv/bin/python
while true; do
  $PY build_cv5_tex.py     >/dev/null 2>&1
  $PY build_posf1_table.py >/dev/null 2>&1
  lad=$(ls results_repro/ladder/*/*.json 2>/dev/null | wc -l)
  pm=$(find results_cv5/mo -name 'pan_meta_pri__*_cv.json' 2>/dev/null | wc -l)
  bio=$(ls results_repro/biomor_learned/*/*.json 2>/dev/null | wc -l)
  # 3M tri-modal ladder (10 arms -> fills Table 3 3M col + Table 5 3M bioMoR)
  lad3=$(ls results_repro/ladder/pan_meta_pri_3modal_*/*.json 2>/dev/null | wc -l)
  # 7 missing Table 2 token-k cells (bioMoR +Token N_R=2/3)
  tk=$(ls results_cv5/biomor_sc_token_k3/Segerstolpe/learned_cv.json \
         results_cv5/biomor_sc_token_k3/Spleen/learned_cv.json \
         results_cv5/biomor_sc_token_k3/Tcell/learned_cv.json \
         results_cv5/biomor_sc_token_k3/Xin/learned_cv.json \
         results_cv5/biomor_mo_token_k2/pnet/prostate__response/learned_cv.json \
         results_cv5/biomor_mo_token_k2/pnet/stad__response/learned_cv.json \
         results_cv5/biomor_mo_token_k3/pnet/blca__response/learned_cv.json 2>/dev/null | wc -l)
  echo "[$(date +%FT%T)] ladder $lad/50 | pan_meta macro $pm/10 | 3M ladder $lad3/10 | token-k $tk/7 | biomor-learned $bio"
  { [ "$lad" -ge 50 ] && [ "$pm" -ge 10 ] && [ "$lad3" -ge 10 ] && [ "$tk" -ge 7 ]; } && { echo "ALL DONE $(date)"; exit 0; }
  sleep 120
done
