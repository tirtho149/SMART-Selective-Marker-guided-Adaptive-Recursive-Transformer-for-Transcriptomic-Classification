# Appendix 1 — Architecture comparison at the selected depth (pancreas)

**Setup.** Vanilla (independent layers) vs. Recursive (one weight-shared block) vs.
MoR (adaptive expert-choice routing) on the **pancreas** single-cell dataset, with
*everything held fixed* except the architecture knob: `d_model=96`, `n_markers=128`,
`d_ff=192`, `epochs=60`, same splits. Depth is set to the **validation-selected best
depth K\*=4** (one-standard-error rule on the depth sweep; see selection curve below).
3 seeds per arm. Per-sample transformer-stack FLOPs Φ = K·φ(M), φ(M)=4M²d+4Mdd_ff =
15.7 MFLOP at M=128, d=96.

Driver: `recursive_marker_transformer/k32_arch.py` (`--K 4 --seeds 0 1 2`);
raw results in `results_k4_arch/pancreas/`.

## Table A1.1 — Vanilla vs. Recursive vs. MoR at K\*=4

| Metric | Vanilla | Recursive | MoR |
|---|--:|--:|--:|
| Unique transformer params | 299,136 | **74,784** | 75,168 |
| Δ vs. Recursive | +224,352 (4×) | — | +384 (router) |
| Total params (incl. head) | 519,295 | 294,943 | 295,327 |
| Nominal FLOPs / sample | 62.9 M | 62.9 M | 62.9 M |
| Effective FLOPs / sample | 62.9 M | 62.9 M | **38.9 M** |
| Compute saving | 0% | 0% | **38.1%** |
| Mean recursion depth | 4 / 4 | 4 / 4 | **2.75 / 4** |
| macro-F1 per seed | 53.4, 55.1, 53.7 | 51.8, 54.7, 65.4 | 53.0, 54.9, 54.9 |
| **macro-F1 mean ± std** | **54.1 ± 0.8** | **57.3 ± 5.8** | **54.2 ± 0.9** |
| accuracy per seed | 91.4, 93.7, 92.0 | 88.2, 93.7, 93.6 | 88.9, 93.4, 91.0 |
| accuracy mean | 92.4 | 91.8 | 91.1 |

### Reading of the exact change
- **Accuracy is a wash** across all three arms (~54 macro-F1, ~92% accuracy). Recursive's
  57.3 mean is inflated by one lucky seed (65.4) and carries the largest spread. No arm wins.
- **Vanilla → Recursive:** 4× fewer unique params (299k → 74.8k) at *identical* FLOPs and
  equal accuracy — the weight-sharing win.
- **Recursive → MoR:** +384 params (4 linear router heads = 4×96), buys a **38.1% FLOP cut**
  (mean depth 2.75/4) at equal accuracy. Higher than the ~31% headline because the quadratic
  attention term saves more than the linear token count implies.

## Table A1.2 — Why depth selection matters (K\*=4 vs. K=32 max)

| | K=4 (selected) | K=32 (max) |
|---|---|---|
| MoR macro-F1 | **54.2 ± 0.9** (stable) | 5.35 (collapsed, majority-class) |
| MoR FLOP saving | 38.1% | 57.3% (hollow — model dead) |
| Recursive macro-F1 | 57.3 ± 5.8 (stable) | 29.8 (one seed collapsed) |
| Vanilla macro-F1 | 54.1 ± 0.8 (stable) | 31.4 (one seed collapsed) |

At K=32, training is past the accuracy peak and unstable: Vanilla/Recursive each keep one
seed alive and collapse the other; MoR collapses both. Its 57% "saving" is meaningless
because the model predicts the majority class (accuracy 41.65% = largest-class fraction).

## Depth-selection curve (recursive arm, validation one-SE rule)

| K | val F1 | val SEM | test F1 |
|--:|--:|--:|--:|
| 1 | 74.7 | 1.4 | 58.8 |
| **4** | **82.5** | 1.2 | **58.6** ← K\* (argmax and one-SE) |
| 8 | 79.3 | 1.3 | 58.9 |
| 10 | 81.8 | 1.1 | 59.5 |
| 12 | 81.6 | 1.2 | 59.2 |
| 24 | 75.6 | 5.0 | 53.4 |
| 32 | 59.8 | 13.8 | 44.8 |
| 100 | 3.8 | 0.2 | 6.4 |

The maximum depth is 32/100, but the parsimonious best depth for pancreas is **K\*=4**:
top validation accuracy at the fewest parameters and FLOPs. Deeper recursion adds cost with
no accuracy gain and eventually collapses. MoR at K\*=4 matches a 4×-larger vanilla
transformer at 38% less compute.

---
*Generated from `results_k4_arch/pancreas/` and `results_depthsweep/pancreas/`.*
