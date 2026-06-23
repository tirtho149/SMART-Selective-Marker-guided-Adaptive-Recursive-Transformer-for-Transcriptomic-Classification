"""
Genomap demo wired to the unified TCGA dataset.

Data  : genomic_dataloader/data/unified_bio5.csv
         2738 samples × 20530 genes + 5 BIO5 labels
         cancer_type: 0=breast  1=head_neck  2=lung  3=thyroid

Grid size note
--------------
Genomap needs rowNum*colNum >= n_genes.  For 20530 genes the minimum square
grid is 144×143 = 20592.  Larger grids are fine; smaller grids trigger
automatic feature selection inside genoClassification / genoVis.

Practical tip: start with a small grid (e.g. 32×32 = 1024 top genes) so the
Gromov-Wasserstein step is fast.  Scale up when you need all genes.

Usage
-----
    conda run -n ml573 python genomap_demo.py
"""

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import scipy.stats
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── paths ─────────────────────────────────────────────────────────────────────
UNIFIED_CSV = Path("genomic_dataloader/data/unified_bio5.csv")
GENOMAP_DIR = Path("genomap")
sys.path.insert(0, str(GENOMAP_DIR))

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading unified_bio5.csv …", flush=True)
df = pd.read_csv(UNIFIED_CSV, index_col="sample")

META_COLS = {"cancer_type", "cancer_name",
             "os_binary", "pathologic_stage",
             "pathologic_T", "pathologic_N", "tumor_status"}
gene_cols = [c for c in df.columns if c not in META_COLS]

X = df[gene_cols].values.astype(np.float32)           # (2738, 20530)
cancer_type  = df["cancer_type"].values.astype(int)   # 0-3
os_binary    = df["os_binary"].values.astype(int)
tumor_status = df["tumor_status"].values.astype(int)

print(f"  X shape      : {X.shape}")
print(f"  cancer_type  : {np.unique(cancer_type, return_counts=True)}")
print(f"  os_binary    : {np.unique(os_binary,   return_counts=True)}")

# ── z-score normalise (column-wise, ddof=1 matches genomap convention) ────────
X_norm = scipy.stats.zscore(X, axis=0, ddof=1)
X_norm = np.nan_to_num(X_norm, nan=0.0)               # constant genes → 0

# ── grid helper ───────────────────────────────────────────────────────────────
def grid_for_genes(n_genes: int, target_pixels: int | None = None):
    """Return (rowNum, colNum) with rowNum*colNum >= n_genes."""
    if target_pixels is not None:
        import math
        side = math.ceil(math.sqrt(target_pixels))
        return side, side
    import math
    side = math.ceil(math.sqrt(n_genes))
    return side, side          # square grid, slightly > n_genes

# ── Example 1 : cancer-type classification (fast — small grid) ────────────────
def demo_classification(target_label=cancer_type, label_name="cancer_type",
                        row=32, col=32, epochs=50, seed=42):
    """
    Run genoClassification on the TCGA data.
    Default grid 32×32 = 1024 — genomap auto-selects the top 1024 genes.
    Use row=144, col=143 to use all 20530 genes (slow).
    """
    import genomap.genoClassification as gCls

    print(f"\n{'='*60}")
    print(f"genoClassification  label={label_name}  grid={row}×{col}")
    print(f"{'='*60}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_norm, target_label, test_size=0.2, random_state=seed,
        stratify=target_label,
    )
    print(f"  train={len(y_tr)}  test={len(y_te)}", flush=True)

    pred = gCls.genoClassification(
        X_tr, y_tr, X_te,
        rowNum=row, colNum=col, epoch=epochs,
    )
    acc = accuracy_score(y_te, pred)
    print(f"\nAccuracy ({label_name}): {acc:.4f}")
    print(classification_report(y_te, pred))
    return pred, y_te

# ── Example 2 : construct raw genomaps (for custom CNN / RecursiveQFormer) ────
def build_genomaps(row=32, col=32, num_iter=200, epsilon=0.0):
    """
    Returns genomap image tensors: shape (n_samples, row, col, 1).
    Use as input to any CNN — or feed into RecursiveQFormer.
    """
    import genomap as gp

    print(f"\nConstructing genomaps  grid={row}×{col}  n_iter={num_iter} …", flush=True)
    maps = gp.construct_genomap(X_norm, row, col, epsilon=epsilon, num_iter=num_iter)
    print(f"  genomaps shape : {maps.shape}   (samples, rows, cols, 1)")
    return maps                # np.ndarray (N, row, col, 1)

# ── Example 3 : genoVis — 2-D visualisation + clustering ─────────────────────
def demo_vis(row=32, col=32):
    import genomap.genoVis as gVis
    import matplotlib.pyplot as plt

    print(f"\ngenoVis  grid={row}×{col} …", flush=True)
    n_clusters = len(np.unique(cancer_type))
    res = gVis.genoVis(X_norm, n_clusters=n_clusters, colNum=col, rowNum=row)
    emb, clus = res[0], res[1]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=cancer_type, cmap="jet", s=8)
    plt.colorbar(sc, label="cancer_type (0=breast 1=hnsc 2=lung 3=thyroid)")
    plt.xlabel("genoVis1"); plt.ylabel("genoVis2")
    plt.title("genoVis — TCGA 4-cohort coloured by cancer type")
    out = Path("genomap_vis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved → {out}")
    return emb, clus


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task",   default="classify",
                   choices=["classify", "build", "vis"],
                   help="classify | build | vis")
    p.add_argument("--label",  default="cancer_type",
                   choices=["cancer_type", "os_binary", "tumor_status"])
    p.add_argument("--row",    type=int, default=32)
    p.add_argument("--col",    type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()

    label_map = {
        "cancer_type":  cancer_type,
        "os_binary":    os_binary,
        "tumor_status": tumor_status,
    }

    if args.task == "classify":
        demo_classification(label_map[args.label], args.label,
                            args.row, args.col, args.epochs)
    elif args.task == "build":
        maps = build_genomaps(args.row, args.col)
        out = Path(f"genomap_maps_{args.row}x{args.col}.npy")
        np.save(out, maps)
        print(f"Saved → {out}  shape={maps.shape}")
    elif args.task == "vis":
        demo_vis(args.row, args.col)
