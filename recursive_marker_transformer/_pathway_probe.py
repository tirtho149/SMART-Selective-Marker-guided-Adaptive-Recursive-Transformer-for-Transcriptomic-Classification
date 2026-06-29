#!/usr/bin/env python3
"""Schema probe for the Reactome/P-NET pathway data on data-branch.

Validates the assumptions the loader (pathway_data.py) will rely on, per cohort:
  * mutation and CNV share the same gene order (so channel-stacking is direct);
  * Excel-corrupted gene symbols (1-Mar -> MARCH1, 10-Sep -> SEPT10) are recoverable;
  * how many cohort genes actually land in >=1 Reactome pathway (membership coverage);
  * the pathway-hierarchy adjacency is square and index-aligned to filtered_pathways;
  * label balance per task.

Read-only. Run: python -m recursive_marker_transformer._pathway_probe
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
COHORTS = ["prostate", "blca", "stad", "brca", "pan_meta_pri"]

# Excel mangles a handful of gene symbols into dates: "MARCH1" -> "1-Mar",
# "SEPT10" -> "10-Sep", "DEC1" -> "1-Dec". Reverse the common month forms.
_MONTH = {"Mar": "MARCH", "Sep": "SEPT", "Dec": "DEC", "Sept": "SEPT"}
_DATEY = re.compile(r"^(\d{1,2})-(Mar|Sep|Sept|Dec)$")


def fix_symbol(s: str) -> str:
    m = _DATEY.match(s)
    return f"{_MONTH[m.group(2)]}{int(m.group(1))}" if m else s


def _header(path: Path) -> list[str]:
    with open(path) as f:
        return f.readline().rstrip("\n").split(",")[1:]


def probe(cohort: str) -> None:
    d = DATA / cohort
    print(f"\n===== {cohort} =====")
    if not d.exists():
        print("  (missing)")
        return

    # ---- pathways: membership + hierarchy ----
    pw = pd.read_csv(d / "filtered_pathways.csv")
    pw_ids = pw["Pathway_ID"].tolist()
    pathway_genes = set()
    for g in pw["Genes"].dropna():
        pathway_genes.update(x.strip() for x in str(g).split(","))
    print(f"  pathways={len(pw_ids)}  union_pathway_genes={len(pathway_genes)}")

    adj = pd.read_csv(d / "adjacency_matrix.csv", index_col=0)
    aligned = list(adj.columns) == pw_ids and list(adj.index) == pw_ids
    print(f"  adjacency shape={adj.shape}  index-aligned-to-pathways={aligned}  "
          f"edges={int((adj.values != 0).sum())}")

    # ---- omics genes + alignment + corruption recovery ----
    for omics in ("mutation_data.csv", "cnv_data.csv"):
        p = d / omics
        if not p.exists() or p.stat().st_size < 5:
            print(f"  {omics}: ABSENT")
            continue
        raw = _header(p)
        fixed = [fix_symbol(g) for g in raw]
        n_datey = sum(bool(_DATEY.match(g)) for g in raw)
        in_pw_raw = len(set(raw) & pathway_genes)
        in_pw_fix = len(set(fixed) & pathway_genes)
        print(f"  {omics}: genes={len(raw)}  date-corrupted={n_datey}  "
              f"in_pathway raw={in_pw_raw} -> fixed={in_pw_fix} "
              f"({100*in_pw_fix/max(len(pw_ids),1):.0f}% of pathways covered by genes)")

    # mut/cnv same gene order?
    mp, cp = d / "mutation_data.csv", d / "cnv_data.csv"
    if mp.exists() and cp.exists() and cp.stat().st_size > 5 and mp.stat().st_size > 5:
        same = _header(mp) == _header(cp)
        print(f"  mut/cnv gene order identical={same}")

    # ---- labels ----
    lab = pd.read_csv(d / "patient_labels.csv")
    ycol = lab.columns[-1]
    vc = lab[ycol].value_counts()
    print(f"  patients={len(lab)}  label='{ycol}'  classes={len(vc)}  "
          f"balance={dict(vc.head(6))}")


if __name__ == "__main__":
    for c in COHORTS:
        probe(c)
