# ============================================================================
# SMART: Selective Marker-guided Adaptive Recursive Transformer
#        for Transcriptomic Classification
#
# Authors:
#   Koushik Howlader   - Iowa State University
#   Tirtho Roy         - Iowa State University
#   Md Tauhidul Islam  - Stanford University
#   Wei Le             - Iowa State University
#
# Copyright (c) 2026 The SMART Authors. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, without the express written
# permission of the authors is STRICTLY PROHIBITED and will be prosecuted to
# the fullest extent permitted by law. See the LICENSE file for full terms.
# ============================================================================

"""Smoke test for the pure-gene dynamic loader.

Run with:
    conda run -n ml573 python genomic_dataloader/smoke_test.py

On first run this will auto-download from UCSC Xena (~1-2 GB per cohort).
Subsequent runs load from the local cache and finish in seconds.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from genomic_dataloader import build_loaders, quality_report, ALL_HEADS

print("=" * 60)
print("Available heads:", ALL_HEADS)
print("=" * 60)

# --- Quality report (triggers download if missing) ---
print("\n=== QUALITY REPORT ===")
report = quality_report(verbose=True)

# --- Full 5-cohort loader ---
print("\n=== BUILD LOADERS (all heads) ===")
loaders, meta = build_loaders(
    heads=["tmt", "rt", "os", "stage", "cancer_type"],
    batch_size=64, seed=42,
)
print(f"n_genes       : {meta.n_genes}")
print(f"n_samples     : {meta.n_samples}")
print(f"cohorts       : {meta.cohorts}")
print(f"head_types    : {meta.head_types}")
print(f"head_n_classes: {meta.head_n_classes}")
print(f"split sizes   : train={len(meta.splits['train'])} "
      f"val={len(meta.splits['val'])} test={len(meta.splits['test'])}")
print(f"class_weights[tmt]: {meta.class_weights['tmt'].tolist()}")

# --- Batch shape check ---
print("\n=== ONE BATCH ===")
X, targets = next(iter(loaders["train"]))
print(f"X shape  : {tuple(X.shape)}  dtype={X.dtype}")
for h, t in targets.items():
    print(f"  {h:12s}: shape={tuple(t.shape)} dtype={t.dtype} "
          f"sample={t[:4].tolist()}")

# --- Reproducibility ---
print("\n=== REPRODUCIBILITY ===")
_, meta2 = build_loaders(heads=["os"], seed=42, batch_size=32)
same = (meta.splits["train"] == meta2.splits["train"]).all()
print("Same train indices (seed=42 twice):", same)
assert same, "FAILED: split not reproducible"

# --- Partial cohort ---
print("\n=== PARTIAL COHORT (breast + lung) ===")
loaders2, meta2 = build_loaders(
    heads=["tmt", "cancer_type"],
    cohorts=["breast", "lung"],
    batch_size=32, seed=0,
)
print(f"n_samples={meta2.n_samples}  n_genes={meta2.n_genes}")
X2, t2 = next(iter(loaders2["train"]))
ct = sorted(set(t2["cancer_type"].tolist()))
print(f"cancer_type values in batch: {ct}")

# --- Invalid head error ---
print("\n=== INVALID HEAD → ValueError ===")
try:
    build_loaders(heads=["pathway_score"])
    assert False, "Should have raised"
except ValueError as e:
    print(f"Correctly raised: {e}")

print("\nALL CHECKS PASSED")
