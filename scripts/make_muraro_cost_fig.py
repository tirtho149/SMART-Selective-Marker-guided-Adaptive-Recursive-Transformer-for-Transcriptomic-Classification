"""Computational-cost-vs-accuracy figure on the A100, Vanilla vs bioMoR, on ONE dataset
(Muraro). Reads results/cv5/curves/muraro_cost.json (written by make_baron_cost.py on an
A100): per-epoch validation macro-F1 and cumulative GPU-seconds, plus final test-F1 and
parameter count.

Learning curves: x = cumulative training compute (real A100 GPU-seconds), y = validation
macro-F1. bioMoR reaches a higher accuracy; the end markers report final test-F1 and params.
Writes paper/figs/muraro_cost_accuracy.{pdf,png}.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "cv5" / "curves" / "muraro_cost.json"
STYLE = {  # name -> (color, label)
    "Vanilla": ("#999999", "Vanilla Transformer"),
    "bioMoR":  ("#009E73", "bioMoR"),
}
A100_USD_PER_HR = 1.80   # representative A100 80GB on-demand cloud rate ($/GPU-hour)

d = json.load(open(DATA))
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(6.0, 4.2))

summary = []
for name, (color, label) in STYLE.items():
    if name not in d:
        continue
    h = d[name]["history"]
    sec = np.array([e["sec"] for e in h])           # already-cumulative A100 GPU-seconds
    f1 = [e["val_f1"] for e in h]
    ax.plot(sec, f1, color=color, lw=1.8, label=label, alpha=0.9, zorder=3)
    tf1 = d[name].get("test_f1")
    ax.scatter(sec[-1], f1[-1], s=70, color=color, edgecolors="black", linewidths=1.0, zorder=4)
    summary.append((color, f"{label}:  {sec[-1]:.1f} A100 GPU-s  →  test-F1 {tf1:.1f}"))

# clean summary box in the empty lower-middle region (no curve overlap)
y0 = 30
for i, (color, txt) in enumerate(summary):
    ax.text(0.30, y0 - i * 7.5, txt, color=color, fontsize=9.5, fontweight="bold",
            va="center", ha="left", zorder=5)

ax.set_xlabel("Training compute  (cumulative A100 GPU-seconds)", fontsize=11)
ax.set_ylabel("Validation macro-F1 (%)", fontsize=11)
ax.grid(True, ls=":", lw=0.5, alpha=0.5)
ax.legend(fontsize=10, frameon=False, loc="lower right")

# secondary top axis: same compute, expressed in estimated USD cloud cost
from matplotlib.ticker import FuncFormatter
_to_usd = lambda s: s / 3600.0 * A100_USD_PER_HR
_to_sec = lambda u: u * 3600.0 / A100_USD_PER_HR
secax = ax.secondary_xaxis("top", functions=(_to_usd, _to_sec))
secax.set_xlabel(f"Estimated cloud cost  (USD, A100 @ \\${A100_USD_PER_HR:.2f}/hr)".replace("\\", ""),
                 fontsize=10)
secax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.4f}"))
ax.set_title("Compute cost vs. accuracy on Muraro (A100)", fontsize=12, fontweight="bold", pad=28)
fig.tight_layout()
figs = ROOT / "paper" / "figs"; figs.mkdir(parents=True, exist_ok=True)
fig.savefig(figs / "muraro_cost_accuracy.pdf", bbox_inches="tight", dpi=600)
fig.savefig(figs / "muraro_cost_accuracy.png", bbox_inches="tight", dpi=200)
print("wrote", figs / "muraro_cost_accuracy.pdf")
for name in STYLE:
    if name in d:
        h = d[name]["history"]; tot = h[-1]["sec"]   # sec is already cumulative
        print(f"  {name:8s} epochs={len(h)} total_gpu_sec={tot:.1f} "
              f"final_val_f1={h[-1]['val_f1']:.1f} test_f1={d[name].get('test_f1'):.1f} "
              f"params={d[name].get('params')}")
