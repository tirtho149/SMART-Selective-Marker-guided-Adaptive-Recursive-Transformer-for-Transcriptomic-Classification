# SMART review — action items & resolution status

Review: `paperreview_ai_2026-06-23.md`. Status: [x] done · [~] partial/future-work · [ ] todo.
Resolved 2026-06-23 in `make_paper.py` (source of truth); paper rebuilt, 0 unresolved
tokens, 12 pages, clean compile.

## Correctness bugs (must-fix; reviewer caught real errors)
- [x] **B1 PRAD->THCA.** Already fixed: `train.py _CANCER_RAW_NAMES` is 4-cohort
  (3=Thyroid/THCA), `results/main.json` persists THCA, and `make_paper._class_name`
  overrides positionally. Paper renders Thyroid (THCA) throughout (4 hits).
- [x] **B2 Routing text vs Table 4.** Reframed: fixed 98.41 / expert 98.71 / token
  98.92 all within ~0.5 F1; routing's value is FLOPs (mean depth 4.0->~2.7, 0.60x).
  Removed the false "token saves less / higher depth" claim (token actually saves
  marginally more here). Expert kept as headline for being balanced-by-construction.
- [x] **B3 Selection text vs Table 5.** Reframed + table reordered: Concrete 99.13 is
  best, router 98.41 second, both learned beat variance 97.70 / random 97.45;
  "Concrete edges the router." Prose + caption fixed.
- [x] **B4 Gene-validation "pending".** `gene_validation_table` -> `gene_identification_table`:
  always renders top-16 genes by recursion depth (two-column descriptive table; no
  [pending]). Removed false "recovers KRT14/TRH/..." prose (none are in the panel);
  now names only table-present genes (ROS1, IL1A) descriptively. Softened two
  downstream "recover known biology" claims (abstract + intro).

## Under-specified / definitions
- [x] **D1** Defined s_i (per-gene importance logit from the marker head) + gradient
  flow + noted it is off (beta=0) for the soft selectors used in the headline.
- [x] **D2** Added Appendix "Effective-FLOPs Accounting": phi(a)=4a^2 d + 4 a d d_ff,
  components counted, empirical per-step survivor depth aggregation, saving ratio.
- [x] **D3** Added marker-token construction note: identity+value summed w/o intermediate
  LayerNorm; pre-norm LayerNorm at block entry places them on a common scale.
- [x] **D4** Router z-loss / load-balancing forms already written in the objective.

## Reporting additions (data already exists)
- [x] **R1** Added Table (tab:cost): stack params, total params (incl. embeddings),
  train wall-clock per config + prose (712,457 total, 43 s headline).
- [x] **R2** Clarified selection protocol: variance/random see only chosen genes
  (drop); router/Concrete attend softly over all genes (gradient everywhere), hard
  arg-max at eval; all feed the same fixed-depth recursion.

## New experiments (GPU) — data in results_extra/
- [x] **E1** Multi-seed (5) on MAIN cohort task: 96.7+/-1.4 macro-F1 reported in Main Results.
- [x] **E2** Init/anneal 2x2x3 ablation -> new table (tab:initanneal) + subsection:
  peaked init decisive (~57%->~96%, +39 pts), annealing <1 pt.
- [x] **E3** Reframed `independent` ablation as the matched-budget standard $K$-layer
  transformer over the M markers; efficient-attention cited as complementary future work.

## Related work / framing
- [x] **W1** Cited GexBERT (Jiang & Hassanpour, Sci. Rep. 2025; arXiv:2504.09704) in
  related work + bib (web-verified).
- [x] **W2** Pancreas on/off-distribution already separated (genomap-image stress test).
- [x] **W3** Added honest depth-vs-variance note (limitation v): possible entanglement
  with statistical prominence; rank correlation vs variance/mean|z| left to future work.

## Deferred (stated as future work, honestly)
- [~] Reactome/GO enrichment table (needs gene-set DB + network on compute node).
- [~] Full efficient-attention (Linformer/Performer) re-implementation under matched budget.
