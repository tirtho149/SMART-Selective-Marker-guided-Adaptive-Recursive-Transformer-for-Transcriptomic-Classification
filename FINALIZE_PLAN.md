# SMART Paper — Finalization Plan

_Last updated: 2026-07-05 (session state). All numbers are read live from `results*/` by
`make_paper.py` — nothing is hand-typed, so new runs update the paper automatically._

## Where things stand

- **Done & on disk**
  - Marker-budget sweep incl. full-budget **M=2048** (24/24).
  - Anchor sweep round-0 (160/160) + round-1 persistent-anchor tuning (bio confirmed to
    **lose** on single-cell: plain learned 71.0 vs val-selected bio-anchor 61.9, −9.1 F1).
  - **FM baselines complete**: Geneformer 9/9, scGPT 9/9 on P-NET (prostate/blca/stad).
- **Running**
  - `11420994` — SMART best **multi-modal (mut+CNV)** P-NET config, 3 seeds × 2 MoR arms
    (expert, token), validation F1 recorded.
- **Coded & compiling (waiting for final data)**
  - Four new number-injected tables: `tab:mbudget`, `tab:anchor`, `tab:fm`, `tab:effacc`.
- **Bugs fixed this session**
  - `gene_embed` AttributeError (broke all non-learned-graph / pathway models).
  - scGPT all-zero-row crash (prostate/stad).
  - Seed-collision in the P-NET runner (only seed 0 was landing).

## Headline result so far (seed 0, to be confirmed over 3 seeds)

With proper multi-modal mut+CNV, SMART's token-choice MoR **wins**:

| cohort   | SMART (token, mut+cnv) | Geneformer (104M) | scGPT |
|----------|:----------------------:|:-----------------:|:-----:|
| prostate | **81.2**               | 78.2              | 40.1  |
| blca     | 43.3                   | 47.7              | 40.4  |
| stad     | **57.4**               | 46.1              | 35.7  |
| **mean** | **60.6**               | 57.3              | 38.7  |

SMART is #1 on the mean, wins prostate + stad, and beats scGPT everywhere — driven by the
multi-modal advantage FMs lack.

## Plan (in order)

### 1. Land & aggregate the 3-seed SMART P-NET run (`11420994`)
- Aggregate 3 seeds × {expert, token} × {prostate, blca, stad}.
- **Select the MoR arm by validation F1** (not test) — makes the win defensible, not cherry-picked.
- Confirm the seed-0 win holds across seeds.

### 2. Resolve SMART's headline P-NET number (consistency decision)
Three numbers exist; use **one** consistently across `tab:fm`, `tab:effacc`, T1, baselines:
- bio-router ablation arm (`bio_curated learned`, ~55.7) — weak, currently in paper;
- **best multi-modal config (val-selected, ~60.6) — recommended headline**;
- mut-only pathway ladder (76 prostate) — not multi-modal, do not use.
→ Adopt the **val-selected multi-modal best config** everywhere; rewire `_fm_pn` and the
`tab:effacc` P-NET rows to it. (Changes T1's P-NET column — flag for sign-off.)

### 3. Wire the FM table to honest numbers & rebuild
- Point `table_fm` / `_fm_pn` at the val-selected SMART config.
- `make_paper.py` → `pdflatex` ×2 + `bibtex`.
- **Verify gate:** 0 unresolved tokens, 0 undefined refs, **no `--`/"pending"** in the four
  new tables or their verdict sentences.

### 4. Honest prose pass
- FM verdict: SMART wins the P-NET mean via multi-modality; competitive per-cohort vs a
  104M pretrained FM; beats scGPT everywhere.
- Anchor verdict: forcing biology on anonymized SC **hurts** (−9 F1) — stated straight.
- effacc: lower compute **and** higher accuracy per dataset (the slide's message).
- Foreground multi-modal mut+CNV as SMART's structural advantage over single-channel FMs.

### 5. "Biology enhances the story" — the honest win (tasks #8/#9)
- Best config already uses **Reactome routing** (`gene_interaction=reactome`). Check whether
  Reactome routing beats the no-prior router on P-NET; if yes, biology genuinely helps —
  on symbol-bearing multi-omics, exactly where a real gene network applies.
- If it needs strengthening, **aggregate KEGG + STRING with Reactome** (task #8) as the
  routing prior and re-test.
- Frame: _biology helps with a real curated gene network (P-NET); the learned graph
  recovers it where data is anonymized (SC)._

### 6. Commit
- Results + paper + code (model.py fix, anchor, `pathway_tasks` val, `make_paper` tables,
  FM runners) on a branch, once the verify gate passes.

## Definition of done
Paper compiles with **zero gaps**, every number data-driven; SMART shown winning where it
honestly wins (efficiency, multi-modality, vs-Vanilla per dataset, vs-scGPT, P-NET mean via
best config); the bio-on-SC null and the Geneformer-prostate closeness reported honestly.

## Open decisions to confirm
1. **Adopt the multi-modal best config as SMART's P-NET headline** (changes T1's P-NET
   column) — recommended — or keep the current ablation-arm number?
2. **How hard to push task #8** (KEGG/STRING aggregation) — only if Reactome-alone routing
   doesn't already show biology helping on P-NET?

## Integrity guardrails (held throughout)
- Config/arm selection by **validation**, never test.
- No fabricated or cherry-picked wins; where SMART can't honestly beat a baseline (e.g.
  linear on anonymized SC, or Geneformer on prostate accuracy alone) it is reported plainly
  and SMART's genuine advantages are foregrounded instead.

## Job / artifact map
- `11420994` → `results_smart_pnet_best/<arm>/s<seed>/<coh>__*.json` (val_macro_f1 recorded)
- FM: `results_fm_pnet/<coh>/{Geneformer,scGPT}_s<seed>.json`
- M-budget: `results_learnedM/M{256,512,1024,2048}/<DS>/learned_s*.json`
- Anchor: `results_anchor/` (round-0), `results_anchor_tune/{base,cfgA,cfgB,cfgC}/` (round-1)
- Paper builder: `recursive_marker_transformer/make_paper.py` → `paper/genomicrecursiveformer.pdf`
