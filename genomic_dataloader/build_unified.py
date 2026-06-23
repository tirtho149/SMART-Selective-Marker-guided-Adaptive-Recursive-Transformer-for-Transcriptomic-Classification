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

"""Build the unified genomic CSV from UCSC Xena.

Downloads all 5 TCGA cohorts on first run (~2-8 GB total).
Subsequent runs load from local cache and finish in minutes.

Output
------
genomic_dataloader/data/unified_bio5.csv

Columns
-------
  sample           (index)  TCGA barcode + cohort suffix
  cancer_type      int      0=breast 1=head_neck 2=lung 3=prostate 4=thyroid
  cancer_name      str      cohort name
  os_binary        float    overall survival ≥180 d  (0/1)
  pathologic_stage int      stage I→0  II→1  III→2  IV→3
  pathologic_T     int      T1→1  T2→2  T3→3  T4→4
  pathologic_N     int      N0→0  N1→1  N2→2  N3→3
  tumor_status     float    with-tumour=1 / tumour-free=0
  <gene_1..N>      float    log2(RPKM+1) expression (shared gene intersection)

Usage
-----
    conda run -n ml573 python genomic_dataloader/build_unified.py

Optional flags (edit below or pass as env vars):
    COHORTS  comma-separated cohort names  (default: all five)
    OUT_PATH absolute path for CSV output  (default: data/unified_bio5.csv)
"""
import os, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from pathlib import Path
from genomic_dataloader import build_unified_csv, BIO5_HEADS

COHORTS  = os.environ.get("COHORTS", "").split(",") or None
COHORTS  = [c.strip() for c in COHORTS if c.strip()] or None

OUT_PATH = os.environ.get("OUT_PATH", "")
OUT_PATH = Path(OUT_PATH) if OUT_PATH else None

t0 = time.time()
out = build_unified_csv(
    out_path = OUT_PATH,
    cohorts  = COHORTS,
    heads    = list(BIO5_HEADS),
)
print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
print(f"File size : {out.stat().st_size / 1e6:.1f} MB")
print(f"Path      : {out}")
