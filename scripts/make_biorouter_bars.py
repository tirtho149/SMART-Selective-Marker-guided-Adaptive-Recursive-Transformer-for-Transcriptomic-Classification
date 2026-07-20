"""Injection-site ablation figure (Fig 2) -- 2x2 panel grid, hatched colour-coded bars.

Where should the biological interaction enter? -- None / Router-only / Embedding-only /
Both(=bioMoR) -- on the four ablation cohorts Baron, Muraro (single-cell) and PM, 3M
(pan-cancer multi-omics), one panel each (2 rows x 2 cols, single-column figure). Every value
is read from the SAME 5-fold CV result files as the main table, so the 'Both (bioMoR)' bars
equal Table 2 exactly. Writes paper/figs/biorouter_bars.pdf (600 dpi).
"""
import glob
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "hatch.linewidth": 0.6,
                     "xtick.major.width": 0.7, "ytick.major.width": 0.7})


def rd(pat):
    for f in sorted(glob.glob(str(ROOT / pat))):
        try:
            d = json.load(open(f))["cv_macro_f1"]
            return float(d["mean"]), float(d["std"])
        except Exception:
            pass
    return None


VARIANTS = ["None", "Router only", "Embedding only", "Both (bioMoR)"]
SC_MODE = {"None": "none", "Router only": "route_graph",
           "Embedding only": "learned_bio", "Both (bioMoR)": "bio_both"}
MO_SUB = {"None": "none", "Router only": "router",
          "Embedding only": "embed", "Both (bioMoR)": "both"}
# colour-coded (matching the reference): grey / periwinkle-blue / teal / coral-red
COLORS = ["#8c8c8c", "#8e94d6", "#48a597", "#e0675c"]

# (panel title, kind, key) -- meaningful dataset names, not cryptic codes
PANELS = [("Baron (pancreas)", "sc", "Baron"), ("Muraro (pancreas)", "sc", "Muraro"),
          ("Pan-cancer\n(mutation+CNV)", "mo", "pan_meta_pri__mut_cnv"),
          ("Pan-cancer\n(tri-modal)", "mo", "pan_meta_pri_3modal__mut_cnv_expr")]


def value(kind, key, v):
    if kind == "sc":
        return rd(f"results/cv5/biomor_canonical/{key}/{SC_MODE[v]}_cv.json")
    return rd(f"results/cv5/inject_mo/{MO_SUB[v]}/{key}__*_cv.json")


fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.35))
x = np.arange(len(VARIANTS))
for ax, (title, kind, key) in zip(axes.ravel(), PANELS):
    means, stds = [], []
    for v in VARIANTS:
        t = value(kind, key, v)
        means.append(t[0] if t else np.nan); stds.append(t[1] if t else 0.0)
    means = np.array(means); stds = np.array(stds)
    for xi, m, s, c in zip(x, means, stds, COLORS):
        rgb = matplotlib.colors.to_rgb(c)
        ax.bar(xi, m, 0.78, yerr=s, capsize=2.2,
               facecolor=(*rgb, 0.45), edgecolor=c, linewidth=0.9, hatch="////",
               error_kw=dict(lw=0.8, ecolor="#333333"))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    lo = np.nanmin(means - stds); hi = np.nanmax(means + stds); pad = 0.18 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("F1", fontsize=8)
    ax.grid(True, axis="y", ls="-", lw=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# shared top legend with hatched colour swatches
handles = [Patch(facecolor=(*matplotlib.colors.to_rgb(c), 0.45), edgecolor=c,
                 hatch="////", linewidth=0.9, label=v) for v, c in zip(VARIANTS, COLORS)]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=6.3,
           handlelength=1.0, handletextpad=0.4, columnspacing=0.8, bbox_to_anchor=(0.5, 1.005))
fig.tight_layout(rect=(0, 0, 1, 0.92))
figs = ROOT / "paper" / "figs"; figs.mkdir(parents=True, exist_ok=True)
fig.savefig(figs / "biorouter_bars.pdf", bbox_inches="tight", dpi=600)
fig.savefig(figs / "biorouter_bars.png", bbox_inches="tight", dpi=600)
print("wrote", figs / "biorouter_bars.pdf")
for title, kind, key in PANELS:
    print(title, {v: (round(value(kind, key, v)[0], 1) if value(kind, key, v) else None) for v in VARIANTS})
