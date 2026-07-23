"""Validate biomor_common against real data + confirm folds match bioMoR exactly."""
import sys
import numpy as np
import biomor_common as bc

# 1) fold parity vs bioMoR's own cv.cv_folds
sys.path.insert(0, str(bc.BIOMOR_ROOT))
from recursive_marker_transformer.cv import cv_folds as ref_folds  # noqa: E402

def check_folds(y, tag):
    a = bc.cv_folds(y); b = ref_folds(y)
    ok = all(np.array_equal(x[0], r[0]) and np.array_equal(x[1], r[1])
             and np.array_equal(x[2], r[2]) for x, r in zip(a, b))
    print(f"  [folds] {tag}: {'IDENTICAL to bioMoR' if ok else 'MISMATCH!!'} "
          f"({len(a)} folds, sizes tr/va/te={len(a[0][0])}/{len(a[0][1])}/{len(a[0][2])})")
    assert ok

# multi-omics
for c, mods in [("prostate", ("mutation", "cnv")), ("brca", ("mutation", "cnv")),
                ("pan_meta_pri_3modal", ("mutation", "cnv", "expression"))]:
    X, y, meta = bc.load_omics(c, modalities=mods)
    print(f"[omics] {c} X={X.shape} C={y.max()+1} dims={meta['modality_dims']} "
          f"labels={np.bincount(y)}")
    check_folds(y, c)

# single-cell (small ones only, to stay light)
for ds in ["Xin", "Tcell"]:
    X, y, sym = bc.load_sc(ds)
    folds = bc.load_sc_folds(ds, y)
    print(f"[sc] {ds} X={X.shape} C={y.max()+1} nsym={len(sym)} folds={len(folds)} "
          f"te0={len(folds[0][2])}")

# score writer
p = bc.write_scores("/tmp/_bc_test", "DUMMY", "Xin", [61.0, 62.0, 60.0, 63.0, 59.0],
                    [70, 71, 69, 72, 68], [300, 300, 300, 300, 292])
print(f"[scores] wrote {p}")
print("ALL BACKBONE CHECKS PASSED")
