"""Per-pathway recursion-depth figure on PM (pan-cancer, mutation+CNV).

Reads results/cv5/pm_routing/pm_routing.json (pm_routing_experiment.py) and draws, for the
K=4 canonical bioMoR, how deep each Reactome pathway token is routed --- a "recursion track"
of four steps that fills up to the pathway's MEAN depth over the 5 held-out folds. Some
pathways survive all 4 steps (kept), others exit at 3, 2, ... (dropped early). Two panels
side by side compare Expert-choice vs Token-choice routing on the SAME 10 pathways, so the
allocation can be read pathway-by-pathway. Fully data-driven; 600 dpi.

Writes paper/figs/pm_depth_expert_vs_token.{pdf,png}.
"""
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/cv5/pm_routing/pm_routing.json"
K = 4              # depth budget to visualise (canonical bioMoR default)
NSHOW = 10         # number of pathways in the figure


def _cfg(results, routing, k):
    for r in results:
        if r["routing"] == routing and r["K"] == k:
            return r
    return None


def _short(name, n=40):
    """Compact Reactome pathway label. The genomap/Reactome pathway tokens carry only the
    stable Reactome accession (R-HSA-####) with no free-text name, so we label by
    accession, which is what the supplementary depth tables list too."""
    s = re.sub(r"_", " ", str(name)).strip()
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _pick(dep, m):
    """Choose m pathway indices spanning THIS routing's own depth spectrum (deep ->
    shallow), so each panel honestly shows its own 4/3/2 gradient. Expert- and
    token-choice keep different pathways, so each panel is picked independently."""
    order = np.argsort(-dep)                              # deepest first
    pos = np.linspace(0, len(order) - 1, m).round().astype(int)
    pos = sorted(set(pos.tolist()))
    while len(pos) < m:                                   # backfill if collisions
        for c in range(len(order)):
            if c not in pos:
                pos.append(c); break
    return order[np.array(sorted(pos))]


def _panel(ax, depths, names, title, cmap, norm, show_names, cellw=0.82, gap=0.16):
    n = len(depths)
    ax.set_xlim(0.3, K + 2.15)
    ax.set_ylim(-0.6, n - 0.4)
    ax.invert_yaxis()                                     # deepest (row 0) on top
    for row, d in enumerate(depths):
        rc = cmap(norm(d))
        for j in range(1, K + 1):                         # four recursion steps
            x0 = j
            ax.add_patch(FancyBboxPatch((x0, row - cellw / 2), 1 - gap, cellw,
                         boxstyle="round,pad=0.008,rounding_size=0.10",
                         linewidth=0.7, edgecolor="#c9ced6", facecolor="#f2f4f7", zorder=1))
            frac = float(np.clip(d - (j - 1), 0.0, 1.0))  # how much of this step is used
            if frac > 0.02:
                ax.add_patch(FancyBboxPatch((x0, row - cellw / 2), (1 - gap) * frac, cellw,
                             boxstyle="round,pad=0.008,rounding_size=0.10",
                             linewidth=0.0, facecolor=rc, zorder=2))
        ax.text(K + 1.02, row, f"{d:.2f}", va="center", ha="left", fontsize=7.8,
                fontweight="bold", color="#33383f")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names if show_names else [], fontsize=7.4)
    ax.set_xticks([j + (1 - gap) / 2 for j in range(1, K + 1)])
    ax.set_xticklabels([f"{j}" for j in range(1, K + 1)], fontsize=8)
    ax.set_xlabel("recursion step", fontsize=8.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)


def main():
    if not SRC.exists():
        print(f"[pm-depth] {SRC} not ready yet (PM routing job still running)."); return
    out = json.load(open(SRC))
    res = out["results"]; paths = out["pathways"]
    ex, tk = _cfg(res, "expert", K), _cfg(res, "token", K)
    if ex is None or tk is None:
        print(f"[pm-depth] need expert & token K={K} in json; have "
              f"{[(r['routing'], r['K']) for r in res]}"); return
    dep_ex = np.asarray(ex["mean_depth"], float)
    dep_tk = np.asarray(tk["mean_depth"], float)

    # Each routing keeps a DIFFERENT subset of pathways, so each panel is picked and
    # sorted by its own depth: both panels then show the same story (compute
    # concentrates on a small subset) without implying the subsets coincide.
    idx_ex = _pick(dep_ex, NSHOW)
    idx_tk = _pick(dep_tk, NSHOW)
    names_ex = [_short(paths[i], 40) for i in idx_ex]
    names_tk = [_short(paths[i], 40) for i in idx_tk]
    de, dt = dep_ex[idx_ex], dep_tk[idx_tk]

    lo = float(min(de.min(), dt.min())) - 0.15
    norm = plt.Normalize(vmin=max(0.0, lo), vmax=K)
    cmap = plt.cm.YlGnBu

    # two-column (figure*): two panels side by side; each panel labels its own pathways.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35),
                             gridspec_kw=dict(wspace=0.55, left=0.16, right=0.965,
                                              top=0.86, bottom=0.20))
    _panel(axes[0], de, names_ex, f"Expert-choice  (F1 {ex['f1_mean']:.1f})", cmap, norm, show_names=True)
    _panel(axes[1], dt, names_tk, f"Token-choice  (F1 {tk['f1_mean']:.1f})", cmap, norm, show_names=True)

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.045,
                      pad=0.22, aspect=42)
    cb.set_label("mean recursion depth over 5 folds  "
                 "(deeper $\\Rightarrow$ kept;  shallow $\\Rightarrow$ dropped early)",
                 fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    figs = ROOT / "paper" / "figs"; figs.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figs / f"pm_depth_expert_vs_token.{ext}", bbox_inches="tight", dpi=600)
    print("[pm-depth] wrote paper/figs/pm_depth_expert_vs_token.{pdf,png}")
    print(f"[pm-depth] expert panel: {[f'{n}={d:.2f}' for n, d in zip(names_ex, de)]}")
    print(f"[pm-depth] token  panel: {[f'{n}={d:.2f}' for n, d in zip(names_tk, dt)]}")


if __name__ == "__main__":
    main()
