"""Pure dynamic genomic data loader — UCSC Xena, all clinical phenotypes.

Auto-downloads raw TCGA HiSeqV2 gene expression + full clinical matrices from
UCSC Xena on first use.  Every phenotyping column that appears in the Xena
clinical matrices is encoded and stored.  You choose which ones to use as
output heads at runtime.

Features
--------
Raw log₂(RPKM+1) gene expression (genes × samples → samples × genes).
When multiple cohorts are loaded the gene-column intersection is kept.

Supported output heads (ALL_HEADS)
-----------------------------------
Universal (all 5 cohorts):
    os_binary         overall survival ≥180 d                  binary float32
    vital_status      dead=1 / alive=0                          binary float32
    tmt               targeted molecular therapy yes=1           binary float32
    rt                radiation therapy yes=1                    binary float32
    additional_rt     additional radiation therapy               binary float32
    additional_pharma additional pharmaceutical therapy          binary float32
    new_tumor_event   recurrence after initial treatment         binary float32
    tumor_status      with tumor=1 / tumor free=0               binary float32
    gender            female=1 / male=0                         binary float32
    age               age at diagnosis (z-scored continuous)     float32
    pathologic_T      T-stage 1-4                               int64
    pathologic_N      N-stage 0-3                               int64
    pathologic_stage  overall stage I-IV → 0-3                  int64
    histological_type cancer subtype (factorised)               int64

Breast-specific:
    er_status         ER positive=1                             binary float32
    pr_status         PR positive=1                             binary float32
    her2_status       HER2 positive=1                           binary float32
    pam50             PAM50 subtype 0-4                         int64

Lung-specific:
    expression_subtype  RNA subtype (factorised)               int64
    kras_mutation       KRAS mutant=1                          binary float32
    egfr_mutation       EGFR mutant=1                          binary float32
    smoking_history     smoking pack-history ordinal 0-4       int64

Head & neck-specific:
    hpv_status          HPV p16-positive=1                     binary float32

Thyroid-specific:
    extrathyroid_extension  extrathyroid spread yes=1          binary float32

Cohort label:
    cancer_type       cohort index 0-3                         int64

Quick start
-----------
    from genomic_dataloader.dataset import build_loaders, quality_report, ALL_HEADS

    print(ALL_HEADS)                # inspect every available head
    report = quality_report()       # auto-downloads on first run
    loaders, meta = build_loaders(
        heads=["os_binary", "er_status", "pam50"],
        cohorts=["breast"],
        batch_size=64, seed=42,
    )
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths & Xena catalogue
# ---------------------------------------------------------------------------

_ROOT      = Path(__file__).resolve().parent.parent / "data" / "tcga"
if not _ROOT.exists():                                    # legacy layout fallback
    _ROOT  = Path(__file__).resolve().parent / "data"
_XENA_BASE = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"

_COHORT_XENA: Dict[str, dict] = {
    "breast": {
        "expr_url": _XENA_BASE + "TCGA.BRCA.sampleMap%2FHiSeqV2.gz",
        "clin_url": _XENA_BASE + "TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix",
    },
    "lung": {
        "expr_url": _XENA_BASE + "TCGA.LUNG.sampleMap%2FHiSeqV2.gz",
        "clin_url": _XENA_BASE + "TCGA.LUNG.sampleMap%2FLUNG_clinicalMatrix",
    },
    "head_neck": {
        "expr_url": _XENA_BASE + "TCGA.HNSC.sampleMap%2FHiSeqV2.gz",
        "clin_url": _XENA_BASE + "TCGA.HNSC.sampleMap%2FHNSC_clinicalMatrix",
    },
    "thyroid": {
        "expr_url": _XENA_BASE + "TCGA.THCA.sampleMap%2FHiSeqV2.gz",
        "clin_url": _XENA_BASE + "TCGA.THCA.sampleMap%2FTHCA_clinicalMatrix",
    },
}

CANCER_TYPE_INDEX: Dict[str, int] = {
    name: i for i, name in enumerate(sorted(_COHORT_XENA))
}

# ---------------------------------------------------------------------------
# Phenotype registry
# ---------------------------------------------------------------------------

@dataclass
class _Spec:
    head:        str
    dtype:       str           # "binary" | "multiclass" | "continuous"
    description: str
    cohorts:     Optional[List[str]]   # None = all five


# Encoding helpers ----------------------------------------------------------

def _yesno(col: pd.Series) -> pd.Series:
    return col.str.strip().str.upper().map({"YES": 1.0, "NO": 0.0})

def _posneg(col: pd.Series) -> pd.Series:
    return col.str.strip().str.upper().map({"POSITIVE": 1.0, "NEGATIVE": 0.0})

def _deadlive(col: pd.Series) -> pd.Series:
    return col.str.strip().str.upper().map({"DEAD": 1.0, "ALIVE": 0.0})

def _mf(col: pd.Series) -> pd.Series:
    return col.str.strip().str.upper().map({"FEMALE": 1.0, "MALE": 0.0})

def _tumor_status(col: pd.Series) -> pd.Series:
    return col.str.strip().str.upper().map(
        {"WITH TUMOR": 1.0, "TUMOR FREE": 0.0}
    )

def _t_stage(col: pd.Series) -> pd.Series:
    def _enc(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip().upper()
        if v.startswith("T1"):  return 1
        if v.startswith("T2"):  return 2
        if v.startswith("T3"):  return 3
        if v.startswith("T4"):  return 4
        return np.nan
    return col.map(_enc).astype(float)

def _n_stage(col: pd.Series) -> pd.Series:
    def _enc(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip().upper()
        if v.startswith("N0"):  return 0
        if v.startswith("N1"):  return 1
        if v.startswith("N2"):  return 2
        if v.startswith("N3"):  return 3
        return np.nan
    return col.map(_enc).astype(float)

def _overall_stage(col: pd.Series) -> pd.Series:
    def _enc(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip().upper().lstrip("STAGE").strip()
        if v.startswith("I") and not v.startswith("II") and not v.startswith("IV"):
            return 0
        if v.startswith("II") and not v.startswith("III") and not v.startswith("IV"):
            return 1
        if v.startswith("III") and not v.startswith("IV"):
            return 2
        if v.startswith("IV"):
            return 3
        return np.nan
    return col.map(_enc).astype(float)

def _factorize(col: pd.Series) -> pd.Series:
    if col.dtype == object:
        col = col.str.strip()
    codes, _ = pd.factorize(col)
    result   = pd.Series(codes, index=col.index, dtype=float)
    result[codes == -1] = np.nan
    return result

def _pam50(col: pd.Series) -> pd.Series:
    mapping = {"LumA": 0, "LumB": 1, "Her2": 2, "Basal": 3, "Normal": 4}
    return col.str.strip().map(mapping).astype(float)

def _smoking(col: pd.Series) -> pd.Series:
    # TCGA codes: 1=never, 2=current, 3=reformed >15y, 4=reformed ≤15y, 5=reformed unknown
    return pd.to_numeric(col, errors="coerce") - 1   # shift to 0-based

def _os_binary(clin: pd.DataFrame) -> pd.Series:
    """Overall survival ≥ 180 days — works for all cohorts."""
    if "OS_Time_nature2012" in clin.columns:
        t = pd.to_numeric(clin["OS_Time_nature2012"], errors="coerce")
    else:
        t = pd.to_numeric(clin.get("days_to_death"), errors="coerce")
        t = t.fillna(pd.to_numeric(clin.get("days_to_last_followup", pd.Series(dtype=float)), errors="coerce"))
    return (t >= 180).where(t.notna()).astype(float)

def _age(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col, errors="coerce")


# Registry: head_name → (dtype, description, cohorts)
# The actual encoding logic is applied in _encode_all_phenotypes()
_PHENOTYPE_REGISTRY: Dict[str, _Spec] = {
    # Universal
    "os_binary":          _Spec("os_binary",          "binary",      "Overall survival ≥180 d",              None),
    "vital_status":       _Spec("vital_status",       "binary",      "Dead=1 / Alive=0",                     None),
    "tmt":                _Spec("tmt",                "binary",      "Targeted molecular therapy yes=1",      None),
    "rt":                 _Spec("rt",                 "binary",      "Radiation therapy yes=1",               None),
    "additional_rt":      _Spec("additional_rt",      "binary",      "Additional radiation therapy",          None),
    "additional_pharma":  _Spec("additional_pharma",  "binary",      "Additional pharmaceutical therapy",     None),
    "new_tumor_event":    _Spec("new_tumor_event",    "binary",      "New tumor / recurrence after tx",       None),
    "tumor_status":       _Spec("tumor_status",       "binary",      "With tumor=1 / Tumor free=0",          None),
    "gender":             _Spec("gender",             "binary",      "Female=1 / Male=0",                    None),
    "age":                _Spec("age",                "continuous",  "Age at diagnosis (continuous)",         None),
    "pathologic_T":       _Spec("pathologic_T",       "multiclass",  "T-stage 1-4",                          None),
    "pathologic_N":       _Spec("pathologic_N",       "multiclass",  "N-stage 0-3",                          None),
    "pathologic_stage":   _Spec("pathologic_stage",   "multiclass",  "Overall stage I→0 … IV→3",             None),
    "histological_type":  _Spec("histological_type",  "multiclass",  "Cancer subtype (per-cohort)",          None),
    # Breast
    "er_status":          _Spec("er_status",          "binary",      "ER positive=1 (breast)",               ["breast"]),
    "pr_status":          _Spec("pr_status",          "binary",      "PR positive=1 (breast)",               ["breast"]),
    "her2_status":        _Spec("her2_status",        "binary",      "HER2 positive=1 (breast)",             ["breast"]),
    "pam50":              _Spec("pam50",              "multiclass",  "PAM50 subtype 0-4 (breast)",           ["breast"]),
    # Lung
    "expression_subtype": _Spec("expression_subtype", "multiclass",  "RNA expression subtype (lung)",        ["lung"]),
    "kras_mutation":      _Spec("kras_mutation",      "binary",      "KRAS mutant=1 (lung)",                 ["lung"]),
    "egfr_mutation":      _Spec("egfr_mutation",      "binary",      "EGFR mutant=1 (lung)",                 ["lung"]),
    "smoking_history":    _Spec("smoking_history",    "multiclass",  "Smoking history ordinal 0-4 (lung/hnsc)",["lung","head_neck"]),
    # Head & neck
    "hpv_status":         _Spec("hpv_status",         "binary",      "HPV p16-positive=1 (head_neck)",       ["head_neck"]),
    # Thyroid
    "extrathyroid_extension": _Spec("extrathyroid_extension","binary","Extrathyroid spread yes=1 (thyroid)", ["thyroid"]),
    # Multi-cohort label
    "cancer_type":        _Spec("cancer_type",        "multiclass",  "Cohort index 0-4",                     None),
}

ALL_HEADS: Tuple[str, ...] = tuple(sorted(_PHENOTYPE_REGISTRY))

# Five biologically meaningful universal heads (present in all 5 cohorts).
# They span four orthogonal axes:
#   os_binary       — patient survival outcome           (prognosis)
#   pathologic_stage— integrated TNM staging I-IV        (disease severity)
#   pathologic_T    — primary tumour local invasion T1-T4 (tumour growth)
#   pathologic_N    — lymph node spread N0-N3             (metastatic capacity)
#   tumor_status    — with-tumour vs. tumour-free         (treatment response)
BIO5_HEADS: Tuple[str, ...] = (
    "os_binary",
    "pathologic_stage",
    "pathologic_T",
    "pathologic_N",
    "tumor_status",
)

_NON_FEATURE_COLS = set(ALL_HEADS)


def describe_heads() -> pd.DataFrame:
    """Return a DataFrame describing every available output head."""
    rows = []
    for h, s in _PHENOTYPE_REGISTRY.items():
        rows.append({
            "head":        h,
            "dtype":       s.dtype,
            "description": s.description,
            "cohorts":     ", ".join(s.cohorts) if s.cohorts else "all",
        })
    return pd.DataFrame(rows).set_index("head").sort_index()


# ---------------------------------------------------------------------------
# Encode all phenotypes from one clinical matrix
# ---------------------------------------------------------------------------

def _encode_all_phenotypes(clin: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Encode every possible phenotype column from a clinical DataFrame.

    Returns a DataFrame indexed by sample_id with one column per head.
    NaN = not available for this sample/cohort combination.
    """
    def _get(col: str) -> pd.Series:
        return clin[col] if col in clin.columns else pd.Series(pd.NA, index=clin.index, dtype=object)

    out: Dict[str, pd.Series] = {}

    # Universal
    out["os_binary"]         = _os_binary(clin)
    out["vital_status"]      = _deadlive(_get("vital_status"))
    out["tmt"]               = _yesno(_get("targeted_molecular_therapy"))
    out["rt"]                = _yesno(_get("radiation_therapy"))
    out["additional_rt"]     = _yesno(_get("additional_radiation_therapy"))
    out["additional_pharma"] = _yesno(_get("additional_pharmaceutical_therapy"))
    out["new_tumor_event"]   = _yesno(_get("new_tumor_event_after_initial_treatment"))
    out["tumor_status"]      = _tumor_status(_get("person_neoplasm_cancer_status"))
    out["gender"]            = _mf(_get("gender"))
    out["age"]               = _age(_get("age_at_initial_pathologic_diagnosis"))
    out["pathologic_T"]      = _t_stage(_get("pathologic_T"))
    out["pathologic_N"]      = _n_stage(_get("pathologic_N"))
    out["pathologic_stage"]  = _overall_stage(_get("pathologic_stage"))
    out["histological_type"] = _factorize(_get("histological_type"))

    # Breast
    er_col = "ER_Status_nature2012" if "ER_Status_nature2012" in clin.columns \
             else "breast_carcinoma_estrogen_receptor_status"
    pr_col = "PR_Status_nature2012" if "PR_Status_nature2012" in clin.columns \
             else "breast_carcinoma_progesterone_receptor_status"
    out["er_status"]   = _posneg(_get(er_col))
    out["pr_status"]   = _posneg(_get(pr_col))
    out["her2_status"] = _posneg(_get("HER2_Final_Status_nature2012"))
    pam50_col = "PAM50Call_RNAseq" if "PAM50Call_RNAseq" in clin.columns \
                else "PAM50_mRNA_nature2012"
    out["pam50"]       = _pam50(_get(pam50_col))

    # Lung
    out["expression_subtype"] = _factorize(_get("Expression_Subtype"))
    out["kras_mutation"]      = _yesno(_get("kras_mutation_found"))
    out["egfr_mutation"]      = _yesno(_get("egfr_mutation_result"))
    out["smoking_history"]    = _smoking(_get("tobacco_smoking_history"))

    # Head & neck
    out["hpv_status"] = _yesno(_get("hpv_status_by_p16_testing"))

    # Thyroid
    out["extrathyroid_extension"] = _yesno(
        _get("extrathyroid_carcinoma_present_extension_status")
    )

    # Cancer type (constant per cohort)
    out["cancer_type"] = pd.Series(
        float(CANCER_TYPE_INDEX[cohort]), index=clin.index
    )

    return pd.DataFrame(out, index=clin.index)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _fetch_bytes(url: str, label: str) -> bytes:
    print(f"  [download] {label} …", flush=True)
    t0   = time.time()
    data = urlopen(url).read()
    print(f"  [download] done  {len(data)/1e6:.1f} MB  ({time.time()-t0:.1f}s)",
          flush=True)
    return data


# ---------------------------------------------------------------------------
# Per-cohort download & cache
# ---------------------------------------------------------------------------

def download_cohort(
    cohort: str,
    data_root: Path = _ROOT,
    force: bool = False,
) -> Tuple[Path, Path]:
    """Download one TCGA cohort from UCSC Xena and cache locally.

    Writes two files:
        <cohort>_genes.csv   – samples × genes (raw log2 expression)
        <cohort>_labels.csv  – samples × all phenotype heads

    Returns (genes_path, labels_path).
    """
    if cohort not in _COHORT_XENA:
        raise ValueError(f"Unknown cohort '{cohort}'. Available: {sorted(_COHORT_XENA)}")

    cfg         = _COHORT_XENA[cohort]
    data_root.mkdir(parents=True, exist_ok=True)
    genes_path  = data_root / f"{cohort}_genes.csv"
    labels_path = data_root / f"{cohort}_labels.csv"

    if genes_path.exists() and labels_path.exists() and not force:
        return genes_path, labels_path

    import io as _io

    print(f"\n{'='*60}", flush=True)
    print(f"Cohort: {cohort.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    # Gene expression → samples × genes
    raw_expr = _fetch_bytes(cfg["expr_url"], "gene expression (HiSeqV2)")
    expr_df  = pd.read_csv(
        _io.BytesIO(raw_expr), sep="\t", index_col=0, compression="gzip"
    )
    expr_df  = expr_df.apply(pd.to_numeric, errors="coerce")
    expr_df  = expr_df.T.fillna(expr_df.median(axis=1)).T
    expr_T   = expr_df.T
    expr_T.index.name = "sample"
    print(f"  Expression: {expr_T.shape[0]} samples × {expr_T.shape[1]} genes",
          flush=True)

    # Clinical → all phenotype labels
    raw_clin = _fetch_bytes(cfg["clin_url"], "clinical matrix")
    clin_df  = pd.read_csv(
        _io.BytesIO(raw_clin), sep="\t", index_col=0, low_memory=False
    )
    labels   = _encode_all_phenotypes(clin_df, cohort)
    labels.index.name = "sample"

    # Intersect samples
    common = sorted(set(expr_T.index) & set(labels.index))
    print(f"  Labeled intersection: {len(common)} samples", flush=True)

    expr_T.loc[common].to_csv(genes_path)
    labels.loc[common].to_csv(labels_path)
    print(f"  Saved → {genes_path.name}  &  {labels_path.name}", flush=True)
    return genes_path, labels_path


def _ensure_cohort(cohort: str, data_root: Path = _ROOT) -> Tuple[Path, Path]:
    gp = data_root / f"{cohort}_genes.csv"
    lp = data_root / f"{cohort}_labels.csv"
    if not (gp.exists() and lp.exists()):
        download_cohort(cohort, data_root)
    return gp, lp


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------

def load_raw(
    cohorts: Optional[List[str]] = None,
    data_root: Path = _ROOT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load gene expression and label frames for all requested cohorts.

    Auto-downloads missing cohorts from UCSC Xena.

    Returns
    -------
    expr_df   – samples × genes  (float32, gene-column intersection)
    labels_df – samples × heads  (float, NaN where not applicable)
    """
    if cohorts is None:
        cohorts = sorted(_COHORT_XENA)
    unknown = set(cohorts) - set(_COHORT_XENA)
    if unknown:
        raise ValueError(f"Unknown cohorts: {unknown}. Available: {sorted(_COHORT_XENA)}")

    expr_parts:   List[pd.DataFrame] = []
    labels_parts: List[pd.DataFrame] = []

    for name in cohorts:
        gp, lp = _ensure_cohort(name, data_root)
        expr   = pd.read_csv(gp,  index_col="sample")
        labels = pd.read_csv(lp,  index_col="sample")
        suffix = f"__{name}"
        expr.index   = [f"{i}{suffix}" for i in expr.index]
        labels.index = [f"{i}{suffix}" for i in labels.index]
        expr_parts.append(expr)
        labels_parts.append(labels)

    # Gene intersection
    gene_sets = [set(e.columns) for e in expr_parts]
    shared    = sorted(gene_sets[0].intersection(*gene_sets[1:]))
    print(f"[load] {len(shared)} shared genes | cohorts: {cohorts}", flush=True)

    expr_df   = pd.concat([e[shared] for e in expr_parts],  axis=0, sort=False)
    labels_df = pd.concat(labels_parts,                      axis=0, sort=False)

    return expr_df.astype(np.float32), labels_df


# ---------------------------------------------------------------------------
# Unified CSV builder
# ---------------------------------------------------------------------------

def build_unified_csv(
    out_path: Optional[Path] = None,
    cohorts: Optional[List[str]] = None,
    heads: Optional[List[str]] = None,
    data_root: Path = _ROOT,
    drop_na: bool = True,
) -> Path:
    """Download all cohorts and write one self-contained unified CSV.

    Schema
    ------
    Index   : sample_id  (format: <TCGA-barcode>__<cohort>)
    Columns :
        cancer_type      int   cohort index 0-4
        cancer_name      str   cohort name (breast/lung/…)
        <head_1..N>      float BIO5 phenotype labels (or custom heads)
        <gene_1..M>      float raw log2 gene expression (shared gene intersection)

    Rows with NaN in any requested label head are dropped when drop_na=True.

    Parameters
    ----------
    out_path  : destination CSV path; defaults to data/unified_bio5.csv
    cohorts   : cohort keys to include; None = all five
    heads     : phenotype label columns; None = BIO5_HEADS
    data_root : data cache directory (downloads go here)
    drop_na   : drop samples missing any label value (default True)

    Returns
    -------
    Path to the written CSV.
    """
    if cohorts is None:
        cohorts = sorted(_COHORT_XENA)
    if heads is None:
        heads = list(BIO5_HEADS)
    if out_path is None:
        head_tag = "_".join(heads) if len(heads) <= 3 else f"bio{len(heads)}"
        out_path = data_root / f"unified_{head_tag}.csv"

    bad = set(heads) - set(ALL_HEADS)
    if bad:
        raise ValueError(f"Unknown heads: {bad}. Choose from: {sorted(ALL_HEADS)}")

    print(f"\n{'='*60}", flush=True)
    print("Building unified CSV", flush=True)
    print(f"  cohorts : {cohorts}", flush=True)
    print(f"  heads   : {heads}", flush=True)
    print(f"  out     : {out_path}", flush=True)
    print(f"{'='*60}", flush=True)

    expr_df, labels_df = load_raw(cohorts, data_root)

    # --- Filter rows: keep only samples that have all requested labels ---
    if drop_na and heads:
        avail_heads = [h for h in heads if h in labels_df.columns]
        missing_heads = set(heads) - set(avail_heads)
        if missing_heads:
            print(f"[warn] heads not found in labels and skipped: {missing_heads}",
                  flush=True)
        mask = labels_df[avail_heads].notna().all(axis=1)
        n_dropped = int((~mask).sum())
        expr_df   = expr_df[mask]
        labels_df = labels_df[mask]
        print(f"[filter] {n_dropped} samples dropped (NaN in labels) → "
              f"{len(expr_df)} retained", flush=True)

    # --- Build cancer_name column from index suffix ---
    cancer_name = pd.Series(
        [idx.rsplit("__", 1)[-1] for idx in expr_df.index],
        index=expr_df.index,
        name="cancer_name",
    )

    # --- Assemble: metadata | labels | genes ---
    meta_cols = pd.DataFrame({
        "cancer_type": labels_df["cancer_type"].astype(int),
        "cancer_name": cancer_name,
    })
    label_cols = labels_df[[h for h in heads if h in labels_df.columns]].copy()
    # Cast multiclass heads to int, binary/continuous to float32
    for h in label_cols.columns:
        spec = _PHENOTYPE_REGISTRY[h]
        if spec.dtype == "multiclass":
            label_cols[h] = label_cols[h].astype(int)
        else:
            label_cols[h] = label_cols[h].astype(np.float32)

    unified = pd.concat([meta_cols, label_cols, expr_df], axis=1)
    unified.index.name = "sample"

    # --- Write ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(out_path)

    n_samples, n_cols = unified.shape
    n_genes = expr_df.shape[1]
    print(f"\n[done] Unified CSV written → {out_path}", flush=True)
    print(f"       {n_samples} samples  ×  "
          f"{2} meta  +  {len(heads)} labels  +  {n_genes} genes  "
          f"=  {n_cols} columns total", flush=True)
    return out_path


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def quality_report(
    cohorts: Optional[List[str]] = None,
    data_root: Path = _ROOT,
    nzv_threshold: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """Quality checks across loaded cohorts (auto-downloads if needed).

    Checks
    ------
    1. Missing gene values
    2. Cross-cohort duplicate sample IDs
    3. Coverage of each phenotype head (% samples with a valid label)
    4. Class distribution for binary/multiclass heads
    5. Near-zero-variance genes
    """
    expr_df, labels_df = load_raw(cohorts, data_root)
    report: dict       = {}

    # 1. Missing gene values
    n_miss = int(expr_df.isnull().sum().sum())
    report["missing_gene_values"] = n_miss
    if verbose:
        print(f"[QC] Missing gene values  : {n_miss}")

    # 2. Duplicate sample IDs
    raw_ids   = [i.rsplit("__", 1)[0] for i in expr_df.index]
    dup_count = int(pd.Series(raw_ids).duplicated().sum())
    report["cross_cohort_duplicates"] = dup_count
    if verbose:
        print(f"[QC] Cross-cohort dups    : {dup_count}")

    # 3. Head coverage + class distribution
    head_stats: dict = {}
    n_total = len(labels_df)
    for head in sorted(ALL_HEADS):
        col        = labels_df[head] if head in labels_df.columns else pd.Series(dtype=float)
        valid_mask = col.notna()
        n_valid    = int(valid_mask.sum())
        coverage   = n_valid / n_total if n_total > 0 else 0.0
        vc         = col.dropna().value_counts().sort_index().to_dict()
        head_stats[head] = {"coverage": round(coverage, 3), "n_valid": n_valid, "counts": vc}
        if verbose:
            spec    = _PHENOTYPE_REGISTRY[head]
            counts  = " | ".join(f"{k:.0f}:{v}" for k, v in list(vc.items())[:6])
            print(f"[QC] {head:28s} cov={coverage:.1%}  [{spec.dtype:11s}]  {counts}")

    report["head_stats"] = head_stats

    # 4. Near-zero-variance genes
    var   = expr_df.var(axis=0)
    nzv   = var[var < nzv_threshold].index.tolist()
    report["nzv_genes"] = len(nzv)
    if verbose:
        print(f"[QC] NZV genes            : {len(nzv)} (var < {nzv_threshold})")

    if verbose:
        print(f"[QC] Summary              : {n_total} samples × "
              f"{expr_df.shape[1]} genes | {len(ALL_HEADS)} phenotype heads")

    return report


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

@dataclass
class Scaler:
    mean: np.ndarray
    std:  np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def fit_scaler(X_train: np.ndarray) -> Scaler:
    mu = X_train.mean(axis=0).astype(np.float32)
    sd = X_train.std(axis=0).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return Scaler(mu, sd)


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def _stratified_split(
    n: int, strat_labels: np.ndarray,
    val_frac: float, test_frac: float, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)

    def _can(labels, frac):
        vc = pd.Series(labels).value_counts()
        return (vc >= 2).all() and int(vc.min() * frac) >= 1

    s1 = strat_labels if _can(strat_labels, test_frac) else None
    if s1 is None:
        warnings.warn("Falling back to non-stratified test split.", UserWarning)
    rest_idx, test_idx = train_test_split(
        idx, test_size=test_frac, random_state=seed, stratify=s1
    )
    val_ratio = val_frac / (1.0 - test_frac)
    rest_lbl  = strat_labels[rest_idx]
    s2        = rest_lbl if _can(rest_lbl, val_ratio) else None
    if s2 is None:
        warnings.warn("Falling back to non-stratified val split.", UserWarning)
    train_idx, val_idx = train_test_split(
        rest_idx, test_size=val_ratio, random_state=seed, stratify=s2
    )
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class GenomicDataset(Dataset):
    """Normalised gene-expression dataset with dynamic multi-head labels.

    Item: (x, targets)
        x       – float32 tensor (n_genes,)
        targets – dict[head_name -> scalar tensor]
                  binary/continuous heads → float32
                  multiclass heads        → int64
    """
    _FLOAT_HEADS = {"os_binary","vital_status","tmt","rt","additional_rt",
                    "additional_pharma","new_tumor_event","tumor_status",
                    "gender","er_status","pr_status","her2_status",
                    "kras_mutation","egfr_mutation",
                    "hpv_status","extrathyroid_extension","age"}

    def __init__(self, X: np.ndarray, Y: pd.DataFrame, heads: List[str]):
        self.X     = torch.from_numpy(X)
        self.heads = heads
        self._t: Dict[str, torch.Tensor] = {}
        for h in heads:
            col = Y[h].values
            if h in self._FLOAT_HEADS:
                self._t[h] = torch.from_numpy(col.astype(np.float32))
            else:
                self._t[h] = torch.from_numpy(col.astype(np.int64))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {h: self._t[h][idx] for h in self.heads}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class LoaderMeta:
    gene_names:     List[str]
    n_genes:        int
    heads:          List[str]
    head_dtypes:    Dict[str, str]         # "binary" | "multiclass" | "continuous"
    head_n_classes: Dict[str, int]         # number of classes (2 for binary)
    scaler:         Scaler
    splits:         Dict[str, np.ndarray]
    sample_ids:     Dict[str, List[str]]
    class_weights:  Dict[str, np.ndarray]
    cohorts:        List[str]
    n_samples:      int                    # after dropping NaN rows for chosen heads


def _class_weights(Y_train: pd.DataFrame, heads: List[str]) -> Dict[str, np.ndarray]:
    out = {}
    for h in heads:
        col  = Y_train[h].values
        spec = _PHENOTYPE_REGISTRY[h]
        if spec.dtype in ("binary", "continuous"):
            if spec.dtype == "binary":
                pos   = col.sum()
                neg   = len(col) - pos
                out[h] = np.clip([neg / max(pos, 1.0)], 0.1, 20.0).astype(np.float32)
            else:
                out[h] = np.array([1.0], dtype=np.float32)  # regression — no weighting
        else:
            classes, counts = np.unique(col, return_counts=True)
            freq    = counts / counts.sum()
            w       = 1.0 / np.maximum(freq, 1e-6)
            w       = w / w.sum() * len(classes)
            full_w  = np.ones(int(col.max()) + 1, dtype=np.float32)
            for c, wi in zip(classes, w):
                full_w[int(c)] = float(wi)
            out[h] = full_w
    return out


def build_loaders(
    heads: Optional[List[str]] = None,
    cohorts: Optional[List[str]] = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    data_root: Path = _ROOT,
    drop_na: bool = True,
) -> Tuple[Dict[str, DataLoader], LoaderMeta]:
    """Build train/val/test DataLoaders over raw gene expression.

    Auto-downloads any missing cohort from UCSC Xena on first call.

    Parameters
    ----------
    heads       : phenotype heads (any subset of ALL_HEADS)
    cohorts     : cohort keys; None = all five
    val_frac    : fraction of data for validation
    test_frac   : fraction of data for test
    batch_size  : mini-batch size
    seed        : RNG seed (controls split + DataLoader shuffle)
    num_workers : DataLoader worker count
    data_root   : path to data cache directory
    drop_na     : drop samples missing any requested head label (default True)

    Returns
    -------
    loaders : {"train", "val", "test"} → DataLoader
    meta    : LoaderMeta with gene names, scaler, splits, class weights, …
    """
    if heads is None:
        heads = list(BIO5_HEADS)
    if not heads:
        raise ValueError("heads must be non-empty.")
    bad = set(heads) - set(ALL_HEADS)
    if bad:
        raise ValueError(
            f"Unknown heads: {bad}.\n"
            f"Available: {sorted(ALL_HEADS)}\n"
            f"Use describe_heads() for descriptions."
        )
    if cohorts is None:
        cohorts = sorted(_COHORT_XENA)

    # Validate cohort-restricted heads
    for h in heads:
        spec = _PHENOTYPE_REGISTRY[h]
        if spec.cohorts and not set(spec.cohorts) & set(cohorts):
            raise ValueError(
                f"Head '{h}' is only available for cohorts {spec.cohorts}, "
                f"but requested cohorts are {cohorts}."
            )

    expr_df, labels_df = load_raw(cohorts, data_root)

    # Drop rows with NaN in any requested head
    if drop_na:
        mask = labels_df[heads].notna().all(axis=1)
        expr_df   = expr_df[mask]
        labels_df = labels_df[mask]
        if verbose := (len(expr_df) < len(mask)):
            dropped = int((~mask).sum())
            print(f"[build] Dropped {dropped} samples with NaN in requested heads; "
                  f"{len(expr_df)} remaining.", flush=True)

    gene_cols = list(expr_df.columns)
    Y_df      = labels_df[heads].copy()

    # Stratify on cancer_type × (first multiclass head or os_binary)
    strat_col = labels_df.get("cancer_type", pd.Series(0, index=labels_df.index))
    strat_labels = pd.factorize(strat_col.astype(str))[0]

    train_idx, val_idx, test_idx = _stratified_split(
        len(expr_df), strat_labels, val_frac, test_frac, seed
    )

    Xv     = expr_df.values.astype(np.float32)
    scaler = fit_scaler(Xv[train_idx])
    Xn     = scaler.transform(Xv)

    ids_all = expr_df.index.tolist()

    def _ds(idx):
        return GenomicDataset(Xn[idx], Y_df.iloc[idx], heads)

    g = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(_ds(train_idx), batch_size=batch_size, shuffle=True,
                            generator=g, num_workers=num_workers, pin_memory=True),
        "val":   DataLoader(_ds(val_idx),   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(_ds(test_idx),  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
    }

    # Count unique classes per multiclass head (after NaN-drop)
    head_n_classes: Dict[str, int] = {}
    for h in heads:
        spec = _PHENOTYPE_REGISTRY[h]
        if spec.dtype == "multiclass":
            head_n_classes[h] = int(Y_df[h].nunique())
        else:
            head_n_classes[h] = 2

    meta = LoaderMeta(
        gene_names     = gene_cols,
        n_genes        = len(gene_cols),
        heads          = heads,
        head_dtypes    = {h: _PHENOTYPE_REGISTRY[h].dtype for h in heads},
        head_n_classes = head_n_classes,
        scaler         = scaler,
        splits         = {"train": train_idx, "val": val_idx, "test": test_idx},
        sample_ids     = {
            "train": [ids_all[i] for i in train_idx],
            "val":   [ids_all[i] for i in val_idx],
            "test":  [ids_all[i] for i in test_idx],
        },
        class_weights  = _class_weights(Y_df.iloc[train_idx], heads),
        cohorts        = cohorts,
        n_samples      = len(expr_df),
    )
    return loaders, meta
