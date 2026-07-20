"""Two training-dynamics figures on Baron, each overlaying all four architectures
(Vanilla / Recursive / MoR / bioMoR), under the unified 5-fold CV (parity with Table 2):

  paper/figs/baron_val_f1.pdf : validation macro-F1 vs estimated cloud cost (USD)
  paper/figs/baron_loss.pdf   : training loss       vs estimated cloud cost (USD)

x-axis = estimated A100 cloud cost (USD) = mean cumulative A100 GPU-seconds x rate/3600.
Lines are the 5-fold mean; shaded bands are +/- 1 SD across folds. Okabe-Ito colours.
Reads results/cv5/curves/baron_cost_cv5.json (make_baron_cost_cv5.py).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/cv5" / "curves" / "baron_cost_cv5.json"
ORDER = ["Vanilla", "Recursive", "MoR", "bioMoR"]
COL = {"Vanilla": "#999999", "Recursive": "#E69F00", "MoR": "#56B4E9", "bioMoR": "#009E73"}
MK = {"Vanilla": "s", "Recursive": "^", "MoR": "D", "bioMoR": "o"}
A100_USD_PER_HR = 1.80          # representative A100 80GB on-demand cloud rate ($/GPU-hour)


def usd(sec_mean):
    return np.asarray(sec_mean) / 3600.0 * A100_USD_PER_HR


def mark_idx(a):
    return [i for i, e in enumerate(a) if e["epoch"] == 1 or e["epoch"] % 10 == 0]


def _plot(data, mkey, skey, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for name in ORDER:
        if name not in data:
            continue
        a = data[name]["agg"]
        x = usd([e["sec_mean"] for e in a])
        m = np.array([e[mkey] for e in a])
        sd = np.array([e[skey] for e in a])
        ax.fill_between(x, m - sd, m + sd, color=COL[name], alpha=0.15, linewidth=0)
        ax.plot(x, m, color=COL[name], lw=1.6, marker=MK[name], markevery=mark_idx(a),
                ms=4.5, markeredgecolor="black", markeredgewidth=0.4, label=name)
    ax.set_xlabel(f"estimated cloud cost (USD, A100 @ \\${A100_USD_PER_HR:.2f}/hr)".replace("\\", ""))
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax.legend(fontsize=7, frameon=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.3f}"))
    fig.tight_layout()
    figs = ROOT / "paper" / "figs"; figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / fname, bbox_inches="tight", dpi=600)
    fig.savefig(figs / fname.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")


def main():
    if not SRC.exists():
        print(f"[epoch-figs] {SRC} not ready yet."); return
    data = json.load(open(SRC))
    _plot(data, "val_f1_mean", "val_f1_sd", "validation macro-F1 (%)",
          "Baron (pancreas): validation macro-F1 vs cost (5-fold CV)", "baron_val_f1.pdf")
    _plot(data, "loss_mean", "loss_sd", "training loss",
          "Baron (pancreas): training loss vs cost (5-fold CV)", "baron_loss.pdf")
    print("[epoch-figs] wrote paper/figs/baron_val_f1.{pdf,png} and baron_loss.{pdf,png}")
    for n in ORDER:
        if n in data:
            a = data[n]["agg"]
            print(f"  {n:10s} epochs<= {len(a):3d} peakVal={max(e['val_f1_mean'] for e in a):.1f} "
                  f"test-F1={data[n]['test_f1_mean']:.1f}+/-{data[n]['test_f1_sd']:.1f}")


if __name__ == "__main__":
    main()
