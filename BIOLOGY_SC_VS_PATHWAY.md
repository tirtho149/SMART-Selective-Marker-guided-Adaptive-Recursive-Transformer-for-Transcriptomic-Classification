# Is biology used the *same way* in single-cell and pathway (multi-omics)?

**Short answer:** the **mechanism is the same** — a learnable, sigmoid-bounded low-rank graph
smoother at the embedding site plus a **zero-init graph-convolution residual** at the router site,
applied at *both* sites in both regimes. What differs is (1) the **source of the graph** and
(2) two **multi-omics-only additions** (fixed-graph *fusion* and *attention bias*) that exist only
because the multi-omics cohorts ship a real pathway graph while the single-cell features are
anonymized. The model code says this explicitly:

> `recursive_marker_transformer/model.py:164–165` — the pathway→pathway adjacency
> (`adjacency_matrix.csv`) is *"warm-started + propagated + **fused exactly like the gene path** —
> NEVER a co-expression graph computed on the sparse [...]"*.

So it is **not** an inconsistency: it is the *same operator* fed a *different (provided vs learned)
graph*, which is forced by the data.

---

## Where each regime is configured

| Regime | Entry point | Canonical "both" flags |
|---|---|---|
| **Single-cell** | `recursive_marker_transformer/bio_learned_genomap.py` | `--modes bio_both` → sets the cfg at **lines 165–184** |
| **Multi-omics** | `recursive_marker_transformer/pathway_tasks.py` | `--marker_mode pathway --pathway_learned_graph --pathway_learned_fuse --bio_graph_router` (`slurm/run_injection_pancan.sbatch`, cond 3 = "both"); the token-mode canonical adds `--pathway_attn_bias` (`slurm/run_canonical_biomor_mo.sbatch`) |

`bio_both` (single-cell), `bio_learned_genomap.py:165–184`:
```python
cfg.gene_interaction   = "none"      # NO static gene network (genes are anonymized)
cfg.bio_learned_graph  = True; cfg.bio_learned_rank = 16
cfg.bio_learned_init   = "bio"       # warm-start from the CO-EXPRESSION operator
cfg.bio_prop_lambda_init = 0.2       # embedding smoothing (bio_learned_prop defaults True, config.py:208)
cfg.bio_graph_prop     = False       # no FIXED-graph smoothing (there is no provided gene graph)
cfg.bio_graph_router   = True        # zero-init graph-conv router
```

---

## Site-by-site comparison

| Injection site | Single-cell | Multi-omics (pathway) | Same? |
|---|---|---|---|
| **Token interface** | learnable **marker** tokens (`marker_mode="learnable"`, `config.py:52`) | fixed **Reactome pathway** tokens (`marker_mode="pathway"`) | **Different** (data structure) |
| **Graph source** | learned low-rank gene graph **warm-started from co-expression** (`bio_learned_init="bio"`, `bio_learned_genomap.py:177–178`; op built at `model.py:117–118`) | learned pathway graph **warm-started + fused from the provided Reactome `adjacency_matrix.csv`** (`model.py:164–165, 362`) | **Different source, same low-rank machinery** |
| **Embedding smoothing** | `lam = σ(bio_prop_logit)`; `prop = (x·Eₙ)·Eₙᵀ` over the learned graph (`model.py:131, 435–447`) | `lam = σ(pathway_prop_logit)`; propagate pathway tokens over the learned graph (`model.py:175, 487–501`) | **Same operator form** |
| **…plus fixed-graph FUSION** | **absent** (no provided gene graph to fuse) | `--pathway_learned_fuse` → `g = σ(bio_fuse_gate)` blends the **provided Reactome** operator into the learned one (`model.py:149, 180, 439`) | **Multi-omics only** |
| **Router (depth allocation)** | `bio_graph_router` = zero-init graph-conv residual over the **learned gene sub-graph** (`model.py:201, 547–552`; residual in `router.py`) | `bio_graph_router` = zero-init graph-conv residual over the **learned pathway graph** (same code path, `model.py:547–552`) | **Same mechanism** |
| **…plus attention bias** | **absent** | `--pathway_attn_bias` biases self-attention along the Reactome graph: `pathway_attn = λ·(A>0)` (`model.py:110, 335–341, 541`) | **Multi-omics only** |

Notes:
- The router graph *source* is the **learned** graph in both regimes by default; the provided
  Reactome operator is only used directly if `--pathway_fixed_graph` is set (`model.py:549`), an
  optional decoupling switch **not** used in the "both" runs behind the paper's numbers.
- The single-cell "both" run does **not** fuse or bias-attend, because there is no provided gene
  graph — `gene_interaction="none"` (`bio_learned_genomap.py:176`).

---

## Why they differ — the justification

The difference is **entirely explained by data availability**, and is principled:

1. **Single-cell (genomap) features are anonymized** — the columns carry no gene identities
   (see the paper's Limitations and `bio_redesign_curated.py`/`bio_learned_genomap.py`, which never
   map to named genes). A *provided* gene–gene network (e.g. STRING/Reactome) is therefore
   **unusable**: there is nothing to key it on. The only biological structure available is what can
   be **learned** (warm-started from a co-expression operator, `bio_learned_init="bio"`). Hence no
   fixed-graph fusion and no attention bias on the single-cell side.

2. **Multi-omics cohorts ship a curated Reactome pathway graph** (`adjacency_matrix.csv`, one per
   cohort). Because the tokens *are* named pathways, the model can additionally **fuse that fixed
   graph** into the learned one (`pathway_learned_fuse`) and **bias attention** along it
   (`pathway_attn_bias`). These are strict *additions* enabled by having a real graph, not a
   different routing philosophy.

3. **The shared core is identical.** Both regimes use the same two operators — a learnable
   sigmoid-λ low-rank smoother at the embedding and a zero-init graph-conv residual at the router —
   applied at both sites. `model.py:164–165` states the pathway path is propagated/fused *"exactly
   like the gene path,"* differing only in that its source graph is the provided Reactome adjacency
   rather than a co-expression matrix.

**Conclusion:** biology enters through the **same mechanism and at the same two sites** in both
regimes; the only differences are (a) the graph is *learned-from-co-expression* for single-cell vs
*provided-Reactome* for multi-omics, and (b) multi-omics adds fixed-graph fusion and attention bias.
Both differences are forced by whether a named biological graph exists for the data — a data
constraint, not a modelling inconsistency.

---

### Exact references
- `recursive_marker_transformer/config.py`: `marker_mode`:52, `gene_interaction`:92, `bio_graph_prop`:106, `bio_prop_lambda_init`:107, `bio_learned_graph`:125, `bio_learned_rank`:126, `bio_learned_init`:127, `pathway_attn_bias`:159, `pathway_learned_graph`:172, `pathway_learned_fuse`:177, `bio_graph_router`:204, `bio_learned_prop`:208
- `recursive_marker_transformer/bio_learned_genomap.py`: `bio_both` mode 165–184 (gene_interaction=none:176, bio_learned_graph:177, bio_learned_init="bio":178, bio_graph_router:184)
- `recursive_marker_transformer/model.py`: co-expression smoother 117–118, 131, 435–447; pathway smoother 167–201, 487–501; fixed-graph fuse gate 149, 439; "fused exactly like the gene path" 164–165; bio_graph_router 201, 547–552; pathway attention bias 110, 335–341, 541
- `slurm/run_injection_pancan.sbatch` (MO "both" flags), `slurm/run_canonical_biomor_mo.sbatch` (adds `--pathway_attn_bias`)
