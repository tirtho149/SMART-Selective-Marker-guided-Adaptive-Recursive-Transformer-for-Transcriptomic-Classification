"""Build the two companion compute-cost figures from results_cv5/curves/baron_cost.json:

  Fig 2  learning curves : validation macro-F1 vs cumulative training time (per architecture)
  Fig 3  time-to-target  : training time to first reach fixed macro-F1 targets (bar chart)

Both are BRAINSTORM prototypes (written to repo root, not wired into the paper). Colours are
Okabe-Ito (colour-blind safe).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results_cv5" / "curves" / "baron_cost.json"
ORDER = ["Vanilla", "Recursive", "MoR", "bioMoR"]
COL = {"Vanilla": "#999999", "Recursive": "#E69F00", "MoR": "#56B4E9", "bioMoR": "#009E73"}
MK = {"Vanilla": "s", "Recursive": "^", "MoR": "D", "bioMoR": "o"}


def load():
    d = json.load(open(SRC))
    return {k: d[k] for k in ORDER if k in d}


def running_best(vf1):
    b, out = -1e9, []
    for v in vf1:
        b = max(b, v); out.append(b)
    return out


def fig_learning_curves(data):
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for name in ORDER:
        if name not in data:
            continue
        h = data[name]["history"]
        t = [e["sec"] for e in h]
        f = running_best([e["val_f1"] for e in h])  # best-so-far val F1
        ax.plot(t, f, color=COL[name], lw=1.8, marker=MK[name], markevery=max(1, len(t) // 8),
                ms=4, label=f"{name} (test {data[name]['test_f1']:.1f})")
    ax.set_xlabel("cumulative training time (GPU-seconds)")
    ax.set_ylabel("validation macro-F1 (%)")
    ax.set_title("Baron: accuracy vs. training cost (learning curves)", fontsize=9)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "cost_learning_curves.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(ROOT / "cost_learning_curves.png", dpi=600, bbox_inches="tight")


def time_to(h, target):
    for e in h:
        if e["val_f1"] >= target:
            return e["sec"]
    return None


def fig_time_to_target(data):
    # choose targets inside the range that at least two architectures reach
    peak = {n: max(e["val_f1"] for e in data[n]["history"]) for n in data}
    lo = int(min(peak.values()))
    targets = [t for t in (lo - 4, lo, lo + 5) if t > 0][:3]
    if not targets:
        targets = [60, 65, 70]
    names = [n for n in ORDER if n in data]
    x = np.arange(len(targets)); w = 0.2
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for i, n in enumerate(names):
        secs = [time_to(data[n]["history"], t) for t in targets]
        xs = x + (i - (len(names) - 1) / 2) * w
        vals = [s if s is not None else 0 for s in secs]
        ax.bar(xs, vals, w, color=COL[n], edgecolor="black", linewidth=0.5, label=n)
        for xi, s in zip(xs, secs):
            if s is None:
                ax.text(xi, 0.5, "n/a", ha="center", va="bottom", fontsize=5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([f"F1$\\geq${t}" for t in targets])
    ax.set_ylabel("training time to reach target (s)")
    ax.set_title("Baron: compute to reach a target accuracy", fontsize=9)
    ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / "cost_time_to_target.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(ROOT / "cost_time_to_target.png", dpi=600, bbox_inches="tight")


def main():
    if not SRC.exists():
        print(f"[cost-figs] {SRC} not found yet -- run make_baron_cost.py (job 11665654) first.")
        return
    data = load()
    fig_learning_curves(data)
    fig_time_to_target(data)
    print("[cost-figs] wrote cost_learning_curves.{pdf,png} and cost_time_to_target.{pdf,png}")
    for n in ORDER:
        if n in data:
            h = data[n]["history"]
            print(f"  {n:10s} epochs={len(h):3d} time={h[-1]['sec']:6.1f}s "
                  f"peakVal={max(e['val_f1'] for e in h):.1f} test={data[n]['test_f1']:.1f} "
                  f"params={data[n]['params']}")


if __name__ == "__main__":
    main()
