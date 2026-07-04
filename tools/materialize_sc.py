# ============================================================================
# Loader bridge: materialize the 9 genomap single-cell datasets from
# genomap_data/ (.mat/.npy) into data/singlecell/<lower>/ CSVs, in the exact
# format singlecell._load_dataset expects, so the arch/token/uq sweeps train on
# IDENTICAL arrays to the learned Table-1 sweep (bio_learned_genomap).
#
# No split.csv is written -> _make_splits falls back to stratified 70/30, which
# is precisely what the learned sweep used (it passed split=None).
#
#   python tools/materialize_sc.py
# ============================================================================
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recursive_marker_transformer.bio_learned_genomap import DATASETS, load_genomap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "singlecell"

# The 13-dataset paper uses these 9 single-cell sets (lowercase dir names to match
# the existing pipeline convention + make_paper globs).
NAMES = list(DATASETS)  # Baron, Lung, Muraro, Oesophagus, Segerstolpe, Spleen, Tcell, Wang, Xin


def materialize(name: str) -> dict:
    X, y = load_genomap(name)                       # X:[N,G] float32, y:[N] int64 0-based
    N, G = X.shape
    lo = name.lower()
    d = OUT / lo
    d.mkdir(parents=True, exist_ok=True)
    cell_ids = [f"{lo}_{i}" for i in range(N)]
    genes = [f"g{j}" for j in range(G)]
    expr = pd.DataFrame(X, index=pd.Index(cell_ids, name="cell_id"), columns=genes)
    expr.to_csv(d / "expression.csv.gz", compression="gzip")
    lab = pd.DataFrame({"cell_id": cell_ids, "label": y.astype(np.int64)}).set_index("cell_id")
    lab.to_csv(d / "labels.csv")
    return {"name": name, "dir": lo, "N": N, "G": G, "C": int(y.max() + 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[materialize] -> {OUT}")
    for name in NAMES:
        info = materialize(name)
        print(f"  {info['name']:12s} dir={info['dir']:12s} "
              f"N={info['N']:>6d} G={info['G']:>5d} C={info['C']:>3d}")
    print("[materialize] done")


if __name__ == "__main__":
    main()
