"""Bar chart for the Baron interaction-matrix INJECTION-SITE ablation.

Three conditions, differing only in WHERE the co-expression interaction matrix is
injected (5-fold CV macro-F1, mean +/- SD over the shared folds, seed 42):

  A "Embedding only"  = learned gene-graph warm-started from the interaction matrix;
                        depth router is plain data-driven      (mode: learned_bio)
  B "Router only"     = fixed co-expression centrality prior on the depth router;
                        no learned graph, no smoothing         (mode: route_coexpr)
  C "Both (bioMoR)"   = interaction injected at both sites      (mode: bio_both)

Reads results_cv5/biorouter_ablation/Baron/<mode>_cv.json.
Writes paper/figs/injection_site_bars.pdf (+ .png). Okabe-Ito palette (CB-safe).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results_cv5" / "biorouter_ablation" / "Baron"

# (label, mode-file-stem, colour)  -- Okabe-Ito: orange / sky-blue / green
BARS = [
    ("Embedding only",   "learned_bio",  "#E69F00"),
    ("Router only",      "route_coexpr", "#56B4E9"),
    ("Both (bioMoR)",    "bio_both",     "#009E73"),
]


def rd(stem):
    d = json.load(open(RES / f"{stem}_cv.json"))["cv_macro_f1"]
    return d["mean"], d["std"]


plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(figsize=(3.35, 2.6))

labels, means, stds, colors = [], [], [], []
for lab, stem, c in BARS:
    m, s = rd(stem)
    labels.append(lab); means.append(m); stds.append(s); colors.append(c)

x = np.arange(len(BARS))
ax.bar(x, means, 0.62, yerr=stds, capsize=3, color=colors,
       edgecolor="black", linewidth=0.6, error_kw=dict(lw=0.7))
for xi, m, s in zip(x, means, stds):
    ax.text(xi, m + s + 0.4, f"{m:.1f}", ha="center", va="bottom", fontsize=7)

lo = min(means) - max(stds) - 4
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("macro-F1 (%)")
ax.set_ylim(max(0, lo), max(means) + max(stds) + 3)
ax.set_title("Baron: where the interaction matrix is injected", fontsize=8)
ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
fig.tight_layout()

figs = ROOT / "paper" / "figs"; figs.mkdir(parents=True, exist_ok=True)
fig.savefig(figs / "injection_site_bars.pdf")
fig.savefig(figs / "injection_site_bars.png", dpi=200)
print("wrote", figs / "injection_site_bars.pdf")
for lab, m, s in zip(labels, means, stds):
    print(f"  {lab:16s} {m:5.1f} +/- {s:.1f}")
