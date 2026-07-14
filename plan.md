# SMART 13-Dataset Rebuild — Plan

**Date:** 2026-07-03
**Goal:** Rebuild the SMART paper, slides, and code around exactly **13 datasets**, with
full (real) results, clean reproducible code, and old material archived locally.

## The 13 datasets
- **9 single-cell (genomap):** Baron, Lung, Muraro, Oesophagus, Segerstolpe, Spleen,
  Tcell, Wang, Xin
- **4 P-NET multi-omics:** prostate, blca, stad, brca

Dropped from the old paper: tabula_muris, pancreas, common_class, prototype (single-cell)
and panmeta_subtype (P-NET → replaced by brca).

## Paper table structure (per user)
- **Table 1 — REPLACED:** the new **learned bio-router ablation** — none / coexpr / random /
  **learned** (general MoR), macro-F1 + accuracy, across all 13. *Data already exists*
  (`results_learned_genomap/` for 9 SC, `results_bio_curated/pnet/` for 4 P-NET) → no new GPU.
- **Table 2 — param:** shared vs independent parameter counts, K× reduction. No GPU
  (dataset-agnostic); regenerate `results_sc/param_efficiency.json`.
- **Table 3 — token:** macro-F1 vs marker budget M ∈ {16,32,64,128,256}. Needs GPU sweep on 13.
- **Table 4 — uq:** calibration NLL / ECE across configs. Needs GPU sweep on 13.
- **Table 5 — ADDED:** the slide-2 bottom-left **Vanilla / Recursive / MoR** ladder
  (params · FLOPs · compute-saved · macro-F1 · accuracy), reproduced on the 13. Needs the
  arch ablation sweep (independent/fixed/token/shared) on 13.

## Figures (per user)
Keep **only the 2 method figures** (system overview Fig 1, MoR diagram Fig 2). Remove all
result figures (mor_figures PNGs, param_reduction_tradeoff).

## Key technical finding (loader bridge)
Arch/token/uq runners read `data/singlecell/<name>/{expression.csv.gz,labels.csv}` (CSV).
Muraro/Wang/Xin exist only as `.mat` in `genomap_data/`. The learned Table-1 sweep used a
separate loader (`bio_learned_genomap.load_genomap`) + the SAME training core
(`singlecell._fit_eval`, `_make_splits`). **Bridge:** materialize all 9 SC datasets from
`genomap_data/` into `data/singlecell/<lower>/` CSVs (no split.csv → stratified 70/30, exactly
what the learned sweep used), so arch/token/uq train on identical arrays to Table 1.
P-NET: swap `panmeta_subtype → brca` in sbatch/uq (module already supports `--task brca`).

## Phases / tasks
1. **[done]** Map runner/table/data code.
2. **Loader bridge:** materialize 9 SC datasets → `data/singlecell/`.
3. **GPU sweeps (13 datasets):** arch ablation (Table 5 + significance), token sweep (Table 3),
   uq/calibration (Table 4). Param table = CPU. Multi-seed.
4. **FULL PAPER REWRITE (user 2026-07-04: "update the full paper and title and all")** — done
   THROUGH THE GENERATORS, not by hand-editing .tex (every table + the main doc are
   "% Auto-generated -- do not edit" and get overwritten). Scope = title, abstract, all prose
   sections, Tables 1-5, table PLACEMENT, figures. Rewrite make_paper.py / consolidated_table.py /
   stats_tests.py to the 13 datasets. Table 1 = learned bio-router ablation, Tables 2-4 updated,
   Table 5 = ladder (NEW - must be generated from results_arch13 + results_pw13; mor_tables.tex
   today is a STALE stub = orphan paragraphs + 3 result figures, NOT the ladder). Rewrite
   abstract/setup/dataset lists off the old 10-SC+PanCan roster; swap exemplars off dropped
   Tabula Muris/pancreas. Keep ONLY the 2 method figures (Fig 1 overview, Fig 2 MoR); DELETE the
   3 result figures (fig_scaling/fig_depth/fig_param) mor_tables injects.
   - PLACEMENT (user "tables in exact place"): each table floats at the section that discusses it
     (T1 main-results/learned, T5 ladder after H1, T4 after H4, T2 after H2, T3 after H3), not
     lumped in a trailing "Result Tables" section; use [t] (not full-page [p]) so they anchor there.
   - Abstract currently frames "four hypotheses, four tables" + only-negatives-plus-2-efficiency;
     the learned-graph POSITIVE (Table 1) is now a headline result and must enter title/abstract.

## Paper generation facts (make_paper.py)
- `python -m recursive_marker_transformer.make_paper --outdir paper` writes EVERYTHING:
  `build_tex()` fills a template with `@@TOKEN@@` placeholders -> `main.tex`;
  it also (re)writes consolidated_table.tex, param_table.tex, token_table.tex, uq_table.tex and
  copies mor_tables.tex. A post-check greps for unresolved `@@...@@` tokens -> fix in make_paper.py.
- So: edit the TEMPLATE + table-builder fns in make_paper.py (main_sc_table etc.) and the
  per-table generators; the .tex are outputs. Hand edits to .tex are throwaway.
5. **Reproduce slide-2 bottom-left** as Table 5 (paper) + update slides_smart.tex numbers.
6. **Refactor into clean package** + full reproduction run to validate numbers.
7. **Archive** old paper variants, stale result dirs, superseded code → `archive/`.
   **DIRECTORY REORG (user 2026-07-04: "clean proper organized directory").** Do this in the
   post-jobs refactor pass — NOT now: moving results_*/ data/ or the package would break the 3
   running nova jobs and every hardcoded path in the generators + sbatch. Target tree:
   ```
   RecusrsiveQFormer/
     README.md  LICENSE  requirements.txt  plan.md
     recursive_marker_transformer/   # THE package (keep import path stable)
     tools/                          # data builders (materialize_sc.py, build_genomap_*.py)
     slurm/                          # keep ONLY the 13-ds jobs (*_13/nova, learned, bio_curated)
     data/                           # singlecell/<ds>/ + cohort dirs prostate|blca|stad|brca
     genomap_data/                   # raw .mat SC sources
     results/                        # KEEP ONLY today's 7: arch13 learned_genomap bio_curated
                                     #   token13 pwtoken13 uq13 pw13  (rename results_X -> results/X,
                                     #   then update ROOT/paths in make_paper.py + generators + sbatch)
     paper/                          # tex + generated pdf (+ figs/: keep only 2 method figs)
     docs/                           # design notes: BIO_ROUTER_*.md/.txt, PATHWAY_PLAN.md,
                                     #   GENOMAP_MOR_REPRODUCTION_PLAN.md, old PLAN.md
     archive/                        # everything superseded (below)
     logs/
   ```
   Move to archive/: superseded result dirs (results_bio_redesign*, results_depth*, results_k4_arch,
   results_k32_arch, results_learned_genomap_quick, results_learned_smoke, results_msweep_ms,
   results_pathformer, results_pathway_ms, results_pathway_msweep_ms, results_scaling,
   results_sc_interaction, results_singlecell_arch, results_uq), stray one-offs (Screenshot*.png,
   xena_dataset, genomap_demo.py, run_all.sh if unused), and stale sbatch (non-13 variants).
   Move design .md/.txt to docs/. Root ends with only README/LICENSE/requirements/plan + the
   dir tree above. VALIDATE after moving: re-run make_paper.py (paths updated) + a smoke train to
   confirm nothing path-broke; `git status` reviewed before commit.
8. **Build + validate** paper + slides: no unresolved tokens, all 13 present, 2 figures, numbers consistent.

## Runner facts (CLIs)
- Interpreter: `/work/mech-ai-scratch/tirtho/.venv/bin/python`
- Arch (Table 5 + Table 1 accuracy): `python -m recursive_marker_transformer.singlecell --data data/singlecell --out results_singlecell_arch/<variant>/s<seed> --datasets <...> ...` ; variants via flags (shared/independent/token/fixed/depth1/marker_*).
- Token (Table 3): same `singlecell` module, sweep `--n_markers`, out `results_msweep_ms/M<M>/s<seed>`.
- UQ (Table 4): `python -m recursive_marker_transformer.uq_sweep --config <c> --seeds ... --cohorts` → `results_uq/<config>/s<seed>`.
- P-NET arch: `python -m recursive_marker_transformer.pathway_tasks --task <cohort> ...` (supports brca).
- Param (Table 2): `experiments.param_efficiency_table` → `results_sc/param_efficiency.json` (CPU).
- Learned (Table 1, already done): `bio_learned_genomap` (SC) + `bio_redesign_curated` (P-NET).

## Status log
- 2026-07-03: plan created; mapping complete; materialization next.
- 2026-07-03: Phase 2 DONE (all 9 SC datasets materialized to data/singlecell/, incl.
  muraro/wang/xin from .mat). Phase 3 arch13 (SC half of Table 5, 7 variants x 3 seeds x 9
  ds = 189 jsons) DONE rc=0 -> results_arch13/. Verified all 4 P-NET cohorts (incl. brca w/
  CNV) load with mut_cnv; origin/data-branch holds all cohort CSVs. Submitted remaining
  Phase-3 GPU jobs: token13 (Table 3, job 11418866), uq13 (Table 4, 11418867), pwarch13
  (Table 5 P-NET, 11418868) -> results_token13/, results_pwtoken13/, results_uq13/,
  results_pw13/. NEXT: Phase 4-5 paper regen once these land.
- 2026-07-04: SCOPING DECISION (user): paper tables use ONLY today's experiments.
  T1=results_learned_genomap + results_bio_curated/pnet; T2=fresh analytic param counts
  (DROP the Jun-30 results_depthsweep K=1..100 depth x F1 grid -> Table 2 = param counts /
  Kx reduction only); T3=results_token13 + results_pwtoken13; T4=results_uq13;
  T5=results_arch13 + results_pw13. All pre-Jul-3 sweep dirs (results_msweep_ms,
  results_pathway_msweep_ms, results_uq, results_sc_interaction, results_singlecell_arch,
  results_depthsweep, etc.) are superseded -> archive in Phase 7, not sourced for tables.
- 2026-07-04: SCOPE = full-paper rewrite via generators (see expanded Phase 4). Confirmed the
  paper is NOT ready for the 13-ds tables: 36 stale old-roster mentions (Tabula Muris/pancreas/
  common/prototype/PanCan) incl. title-line + abstract "ten datasets..."; Muraro/Xin/brca absent
  from prose; "four hypotheses/four tables" framing; mor_tables.tex is a stale figure/stub file,
  Table-5 ladder does not exist. Moved sweeps to NOVA (jobs 11418873 token13 / 11418874 uq13 /
  11418875 pwarch13; scavenger ones cancelled); poller bv81q0szx re-invokes when all 3 finish.
  PLAN: do the whole make_paper.py rewrite + regenerate in ONE pass once numbers land (T1/T2 data
  ready now; T3/T4/T5 gated on the nova jobs). Hand-editing .tex deferred/avoided.
- 2026-07-04: Added DIRECTORY REORG to scope (Phase 7, target tree + move-list above). Sequenced
  AFTER jobs+rewrite so path moves don't break running nova jobs / generator + sbatch paths.
  One combined post-jobs execution pass: (a) make_paper.py full rewrite + regenerate on 13 ds,
  (b) results_X -> results/X + update all path refs, (c) archive superseded, (d) docs/ for notes,
  (e) build paper + smoke-validate + git review.
- 2026-07-04: DONE (partial Phase 7): moved 16 superseded result dirs -> archive/superseded_results/
  (bio_redesign*, depth*, k4/k32_arch, learned_*_quick/smoke, msweep_ms, pathformer, pathway_ms*,
  scaling, sc_interaction, singlecell_arch, uq). Top level now = only the 7 active/kept results
  (arch13, bio_curated, learned_genomap, token13, pwtoken13, uq13, +pw13 when it appears). NOTE:
  current (unrewritten) make_paper.py/consolidated_table.py still point at some now-archived dirs
  -> will be repointed to today's dirs during the rewrite; don't run the old generator meanwhile.
  Nova jobs actively writing (token13/uq13/pwtoken13 producing jsons).
