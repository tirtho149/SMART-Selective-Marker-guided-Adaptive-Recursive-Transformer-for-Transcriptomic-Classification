"""Measure REAL forward-pass FLOPs (torch FlopCounterMode) for the full Table-2 ladder on
ALL 14 datasets -- 8 single-cell + 6 multi-omics -- to answer "does bioMoR win on compute?".

Each model is built at its EXACT CV5-launcher config and traced on one real batch:
  * single-cell  (slurm/run_cv5_sc + bio_both ladder): d_model=96, d_ff=192, M=128 marker
    tokens, marker_mode=router.
  * multi-omics  (slurm/run_cv5_mo + biomorboth_ladder_mo): d_model=128, d_ff=256, M=256
    Reactome pathway tokens, marker_mode=pathway; bioMoR adds the learned pathway graph
    (pathway_learned_graph/fuse + zero-init graph-conv router), baseline uses the Reactome
    token-centrality prior. Channels per cohort: mut_cnv (Pro/BL/ST/PM), expr (PC),
    mut_cnv_expr (3M).

Keys "<fam>|<mode>|<K>" match scripts/build_cv5_tex.py SPECS. Writes per-dataset absolute
per-sample FLOPs to results/cv5/curves/all_flops.json; relative-to-Vanilla is derived in
the reporting step. Run on A100 (slurm) to avoid OOM on the login node; FLOP counts are
device-independent.
"""
import json, sys, traceback
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from recursive_marker_transformer.singlecell import _load_dataset, HEAD, _DTYPES
from recursive_marker_transformer.config import RMTConfig
from recursive_marker_transformer.model import RecursiveMarkerTransformer
from recursive_marker_transformer.bio_learned_genomap import _cfg as bio_cfg
from recursive_marker_transformer.pathway_data import load_cohort, load_pan_meta
from recursive_marker_transformer.pathway_tasks import PANMETA

SEED = 42
# Use each launcher's REAL batch size: bioMoR's learned-graph construction is a per-FORWARD
# fixed cost (independent of batch), so per-sample FLOPs are only faithful at the batch the
# model actually runs -- SC=128 (run_cv5_sc), MO=32 (run_cv5_mo / bio_both ladder).
SC_BATCH = 128
MO_BATCH = 32
DEV = "cuda" if torch.cuda.is_available() else "cpu"

SC = ["segerstolpe", "lung", "oesophagus", "baron", "muraro", "tcell", "spleen", "xin"]
MO = [("Pro", "prostate", "mut_cnv"), ("BL", "blca", "mut_cnv"), ("ST", "stad", "mut_cnv"),
      ("PM", "pan_meta_pri", "mut_cnv"), ("PC", "panmeta_response", "expr"),
      ("3M", "pan_meta_pri_3modal", "mut_cnv_expr")]

# (key, recursion_mode, share_weights, depth) -- mirrors run_cv5_{sc,mo}.sbatch
STD = [("std|independent|4", "expert", False, 4),
       ("std|fixed|2", "fixed", True, 2), ("std|fixed|3", "fixed", True, 3), ("std|fixed|4", "fixed", True, 4),
       ("std|expert|2", "expert", True, 2), ("std|expert|3", "expert", True, 3), ("std|expert|4", "expert", True, 4),
       ("std|token|2", "token", True, 2), ("std|token|3", "token", True, 3), ("std|token|4", "token", True, 4)]
# (key, recursion_mode, depth) -- mirrors bio_both ladder
BIO = [("biomor|expert|2", "expert", 2), ("biomor|expert|3", "expert", 3), ("biomor|expert|4", "expert", 4),
       ("biomor|token|2", "token", 2), ("biomor|token|3", "token", 3), ("biomor|token|4", "token", 4)]


def count(model, xb):
    model.eval()
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        model(xb)
    return int(fc.get_total_flops())


# ---------------- single-cell ----------------
def sc_std_cfg(mode, share, K, M):
    return RMTConfig(heads=(HEAD,), n_hvg=None, batch_size=SC_BATCH, d_model=96, d_ff=192,
                     n_markers=M, marker_mode="router", recursion_mode=mode, recursion_depth=K,
                     share_weights=share, seed=SEED, epochs=1, device=DEV)


def measure_sc(ds):
    X, y, _ = _load_dataset(ROOT / "data" / "singlecell" / ds)
    X = X.astype(np.float32); F = X.shape[1]; C = int(y.max() + 1); M = min(128, F)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    b = min(SC_BATCH, Xs.shape[0])
    xb = torch.from_numpy(Xs[:b].astype(np.float32)).to(DEV)
    gv = torch.from_numpy(Xs.var(0).astype(np.float32)).to(DEV)
    res = {}
    for key, mode, share, K in STD:
        torch.manual_seed(SEED)
        m = RecursiveMarkerTransformer(sc_std_cfg(mode, share, K, M), F, {HEAD: C}, _DTYPES).to(DEV)
        m.set_gene_variance(gv); res[key] = count(m, xb) / b
    for key, mode, K in BIO:
        torch.manual_seed(SEED)
        cfg = bio_cfg("bio_both", K, SEED, 1, n_markers=M); cfg.recursion_mode = mode; cfg.device = DEV
        m = RecursiveMarkerTransformer(cfg, F, {HEAD: C}, _DTYPES).to(DEV)
        m.set_gene_variance(gv); res[key] = count(m, xb) / b
    return {"kind": "single-cell", "n_genes": F, "n_classes": C, "M": M, "flops": res}


# ---------------- multi-omics ----------------
def load_mo(task, ch):
    if task in PANMETA:
        cd, lab = PANMETA[task]
        return load_pan_meta(label=lab, cohort=cd, min_genes=5)
    return load_cohort(task, channels=ch, min_genes=5)


def mo_cfg(bio, mode, share, K):
    d = dict(heads=("t",), n_hvg=None, batch_size=MO_BATCH, d_model=128, d_ff=256, n_markers=256,
             marker_mode="pathway", recursion_mode=mode, recursion_depth=K, share_weights=share,
             seed=SEED, epochs=1, device=DEV, pathway_pool="sum" if bio else "mean")
    if bio:
        d.update(gene_interaction="none", pathway_learned_graph=True,
                 pathway_learned_fuse=True, bio_graph_router=True)
    else:
        d.update(gene_interaction="reactome")
    return RMTConfig(**d)


def measure_mo(tag, task, ch):
    coh = load_mo(task, ch)
    X = coh.X.astype(np.float32); y = coh.y
    G = X.shape[1]; K = int(y.max() + 1); Cc = 1 if X.ndim == 2 else X.shape[2]
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-6; Xs = (X - mu) / sd
    b = min(MO_BATCH, Xs.shape[0])
    xb = torch.from_numpy(Xs[:b]).float().to(DEV)
    var0 = (Xs[:, :, 0] if Xs.ndim == 3 else Xs).var(0).astype(np.float32)
    gv = torch.from_numpy(var0).to(DEV)
    Pt = torch.from_numpy(coh.P); adj = torch.from_numpy(coh.adjacency); cen = torch.from_numpy(coh.centrality)
    dtypes = {"t": "multiclass"}

    def build(cfg):
        cfg = replace(cfg, heads=("t",), n_channels=Cc)
        m = RecursiveMarkerTransformer(cfg, G, {"t": K}, dtypes, pathway=Pt).to(DEV)
        m.set_gene_variance(gv)
        if cfg.gene_interaction == "reactome":
            m.set_token_prior(cen)
        if getattr(cfg, "pathway_learned_graph", False):
            m.set_pathway_graph(adj)
        return m

    res = {}
    for key, mode, share, K2 in STD:
        torch.manual_seed(SEED)
        res[key] = count(build(mo_cfg(False, mode, share, K2)), xb) / b
    for key, mode, K2 in BIO:
        torch.manual_seed(SEED)
        res[key] = count(build(mo_cfg(True, mode, True, K2)), xb) / b
    return {"kind": "multi-omics", "n_genes": G, "n_classes": K, "n_channels": Cc,
            "M": len(coh.pathways), "flops": res}


def report(key_data, label):
    van = key_data["flops"]["std|independent|4"]
    best_mor = min(  # cheapest MoR-general point is not the story; report the token/expert K4
        key_data["flops"]["std|token|4"], key_data["flops"]["std|expert|4"])
    bio2 = key_data["flops"]["biomor|token|2"]  # headline bioMoR (K2 token = accuracy leader)
    bio4 = key_data["flops"]["biomor|expert|4"]  # canonical bioMoR
    print(f"[{label:12s}] Vanilla=1.00x  MoR-K4={best_mor/van:.2f}x  "
          f"bioMoR(tokK2)={bio2/van:.2f}x  bioMoR(expK4)={bio4/van:.2f}x", flush=True)


def main():
    out = {"_meta": {"device": DEV, "sc_batch": SC_BATCH, "mo_batch": MO_BATCH,
                     "counter": "FlopCounterMode", "seed": SEED}}
    print(f"[flops-all] device={DEV}", flush=True)
    for ds in SC:
        try:
            out[ds] = measure_sc(ds); report(out[ds], ds)
        except Exception:
            print(f"[flops-all] SC {ds} FAILED:\n{traceback.format_exc()}", flush=True)
    for tag, task, ch in MO:
        try:
            out[tag] = measure_mo(tag, task, ch); report(out[tag], tag)
        except Exception:
            print(f"[flops-all] MO {tag} FAILED:\n{traceback.format_exc()}", flush=True)
    d = ROOT / "results" / "cv5" / "curves"; d.mkdir(parents=True, exist_ok=True)
    (d / "all_flops.json").write_text(json.dumps(out, indent=1))
    print(f"[flops-all] wrote {d/'all_flops.json'}", flush=True)


if __name__ == "__main__":
    main()
