# SMART — Review-Response Plan (paperreview.ai, AAAI, 2026-07-04)

Verdict: **weak reject, revise-and-resubmit**. Ideas praised; two blockers: (1) the
"learned routing graph" gain is **confounded** (it *smooths inputs*, not just routes),
(2) **no external baselines**. Efficiency claims need end-to-end profiling.

Legend: [reuse]=existing code in repo, [new]=to build, GPU=needs cluster run.

---

## P0 — CRITICAL (blocks acceptance)

### C1. Isolate SMOOTHING vs ROUTING (the central confound)
Reviewer Q1 + main weakness. The learned graph does `x ← (1−λ)x + λ (xẼ)Ẽᵀ` (input
denoising) BEFORE marker selection; fixed/random graphs enter as an additive router
logit (Eq. 1). So gains may be denoising, not routing. **Run a clean factorial:**

| axis | levels |
|---|---|
| **Smoothing** of x | none / fixed-bio graph / learned graph |
| **Routing prior** on depth logit | none / fixed-bio centrality / learned-graph centrality |

- [reuse] mechanisms already exist: `cfg.bio_graph_prop` (smoothing), `bio_prior_gate`/
  `router_prior_beta` (routing), `bio_learned_graph` (learned). Just need mode combos.
- [new] add ablation modes: `smooth_only_learned`, `route_only_learned`,
  `smooth_fixed_route_none`, etc. (a 3×3 grid, or the 4 cells the reviewer named).
- Deliver: a factorial table showing where the gain lives. **Expected honest finding:**
  gain = learned *smoothing* (denoising) >> routing. Then **reframe the paper**: rename
  "learned routing graph" → "learned gene-graph **smoothing**"; keep routing as a
  separate, smaller effect. This turns the confound into a clean, defensible story.
- GPU: ~1 array job over 11 datasets × the grid × 3 seeds. Reuse learned-graph runners.

### C2. External baselines (reviewer's 2nd blocker)
- [reuse] `genonet_tasks.py` = **genoNet** (genomap's own CNN) — run on the 8 SC sets.
- [reuse] `baselines.py`/`dl_baselines.py` — linear + shallow baselines already coded;
  add **ANOVA→PCA→logistic** and **scTOP**-style if not present.
- [reuse] `lit_pipeline/baseline_repos` + `baselines_reproducible.csv` — prior repro of
  foundation-model baselines; check coverage, reuse ckpts (see [[smart-baseline-reproduction]]).
- [new] a **Baselines table**: SMART vs genoNet vs linear vs (where feasible) fine-tuned
  scGPT/Geneformer, on the SAME 11 splits. For P-NET, add **P-NET / PATH** pathway baseline.
- GPU: moderate (linear=CPU fast; genoNet fast; FM fine-tune heavier — scope to a subset).

---

## P1 — HIGH (substantially strengthens)

### H1. End-to-end efficiency profiling [new] GPU-light
Reviewer C3. Current FLOPs are stack-only; cross-attention marker selection is O(MN).
- Report **total FLOPs** (embedding + O(MN) router + recursive stack), **wall-clock/epoch**,
  **peak GPU memory**, batch size, vs a matched vanilla transformer and vs genoNet.
- [reuse] `_phi`/`_flops_ratios` in make_paper; extend to full model. `torch.profiler` /
  `thop`/`fvcore` for measured FLOPs+mem.

### H2. Marker-panel stability + biological validation [new]
Reviewer Q2. Interpretability claim currently qualitative.
- [reuse] `marker.py` (selection). Extract arg-max marker panels per seed/split.
- Compute **Jaccard / Spearman** overlap across seeds; report stability.
- Quantitative overlap with **PanglaoDB / CellMarker 2.0** (need to fetch the DBs).
- Deliver: a stability figure/table + a "markers match known biology" number.

### H3. Fixed-prior sensitivity [reuse] GPU-light
Reviewer Q5. Why does the fixed prior collapse on some suites?
- Sweep k-NN ∈ {8,16,32}, centrality ∈ {eigenvector, degree, betweenness, PPR},
  anneal β schedule. `interaction.py` already parameterises these.
- Add the **degenerate-graph diagnosis** we already found (zero-variance genes → NaN
  co-expression operator on Muraro/Seger/Xin) as the mechanistic explanation.

---

## P2 — MEDIUM

- **R1 routing stability** [reuse recursion.py/router.py]: compare expert-choice hard
  top-k vs soft-capacity / stochastic routing; report gradient-flow / convergence. Q7.
- **R2 rare-cell eval** [reuse per_class in result JSON]: per-class recall on rare
  classes; show smoothing doesn't wash out rare/transitional states.
- **R3 OOD / cross-batch** [new, harder]: a cross-study split on ≥1 suite. Scope to one
  demonstrative case if time-boxed.
- **R4 pooling** [reuse pathway_tasks pool]: mean/sum vs learned/attention pooling per
  channel on P-NET; small ablation. Q6.

---

## P3 — LOW (prose / related work, no GPU)

- Related work + discussion additions: **scTransformer** (TF→TG mask priors; contrast our
  fixed-centrality negative), **DOGMA** (deterministic ontology topology), **GeneMamba**
  (linear-time SSM efficiency comparator), **BMFM-RNA/WCED**, **Souza & Mehta** (linear
  baselines rival FMs — motivates C2). One "when do fixed vs learned priors help" paragraph.
- Precise-language fix: separate "smoothing" from "routing" throughout (ties to C1 reframe).
- Centrality-prior construction details (k, sparsity, measure) in appendix.

---

## Sequencing
1. **C1 factorial** (reframes the paper's core claim) — do FIRST; it changes prose everywhere.
2. **C2 baselines** in parallel (independent, reuses old code).
3. Then H1/H2/H3 (profiling, marker stability, sensitivity).
4. P2 as time permits; P3 prose folded into each regeneration.

All results flow through `make_paper.py` (new tables auto-generated); every claim stays
number-injected. Fusion experiment (learned_fused) is DEPRIORITISED — it adds more
smoothing and muddies C1; keep the code but do not feature it.
