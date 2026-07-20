"""Measure REAL forward-pass FLOPs (not the analytical flops_rel proxy) for every
architecture x depth point in Figure 5, on the Muraro single-cell cohort.

Each model is built at the EXACT Table-2 / Figure-5 config (d_model=96, d_ff=192,
M=128 marker tokens, n_genes = Muraro's 2000) with the SAME recursion_mode / share /
depth flags used by the CV5 launchers (slurm/run_cv5_sc.sbatch and the bioMoR bio_both
ladder), then a single real forward pass over a Muraro batch is traced with
torch.utils.flop_counter.FlopCounterMode. This captures the true, data-dependent active-
token routing of expert/token MoR -- exactly what the proxy only approximated.

Keys match scripts/build_cv5_tex.py SPECS: "<family>|<mode>|<K>" where family in
{std, biomor} and mode in {independent, fixed, expert, token}. Writes
results/cv5/curves/muraro_flops.json (absolute FLOPs for the batch + per-sample); the
Pareto figure normalises to Vanilla to get the relative x-axis.

Usage:  python scripts/measure_flops_muraro.py            # full, on A100 (or CPU: same count)
        python scripts/measure_flops_muraro.py --smoke    # just the first std + first bio point
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from recursive_marker_transformer.singlecell import _load_dataset, HEAD, _DTYPES
from recursive_marker_transformer.config import RMTConfig
from recursive_marker_transformer.model import RecursiveMarkerTransformer
from recursive_marker_transformer.bio_learned_genomap import _cfg as bio_cfg

DATASET = "muraro"
SEED = 42
BATCH = 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# std ladder -> (key, recursion_mode, share_weights, K). Mirrors slurm/run_cv5_sc.sbatch:
#   independent (Vanilla) = expert routing, UNSHARED weights, depth 4
#   fixed_k{2,3},fixed    = fixed depth (all tokens all K), shared
#   expert_k{2,3},shared  = MoR expert-choice, shared
#   token_k{2,3},token    = MoR token-choice, shared
STD = [
    ("std|independent|4", "expert", False, 4),
    ("std|fixed|2",       "fixed",  True,  2),
    ("std|fixed|3",       "fixed",  True,  3),
    ("std|fixed|4",       "fixed",  True,  4),
    ("std|expert|2",      "expert", True,  2),
    ("std|expert|3",      "expert", True,  3),
    ("std|expert|4",      "expert", True,  4),
    ("std|token|2",       "token",  True,  2),
    ("std|token|3",       "token",  True,  3),
    ("std|token|4",       "token",  True,  4),
]
# bioMoR bio_both ladder -> (key, recursion_mode, K). Mirrors run_biomorboth_ladder_sc /
# run_canonical_biomor_sc (bio_both; expert K2/K3/K4-canonical + token K2/K3/K4).
BIO = [
    ("biomor|expert|2", "expert", 2),
    ("biomor|expert|3", "expert", 3),
    ("biomor|expert|4", "expert", 4),
    ("biomor|token|2",  "token",  2),
    ("biomor|token|3",  "token",  3),
    ("biomor|token|4",  "token",  4),
]


def std_cfg(mode, share, K, M):
    return RMTConfig(heads=(HEAD,), n_hvg=None, batch_size=BATCH, d_model=96, d_ff=192,
                     n_markers=M, marker_mode="router", recursion_mode=mode,
                     recursion_depth=K, share_weights=share, seed=SEED, epochs=1,
                     lr=1e-3, weight_decay=1e-5, device=DEV)


def count_flops(model, xb, gene_var):
    model.eval()
    if gene_var is not None and hasattr(model, "set_gene_variance"):
        model.set_gene_variance(gene_var)
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        model(xb)
    return int(fc.get_total_flops())


def main(smoke=False):
    X, y, _ = _load_dataset(ROOT / "data" / "singlecell" / DATASET)
    X = X.astype(np.float32)
    F = X.shape[1]; C = int(y.max() + 1)
    M = min(128, F)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    xb = torch.from_numpy(Xs[:BATCH].astype(np.float32)).to(DEV)
    gene_var = torch.from_numpy(Xs.var(0).astype(np.float32)).to(DEV)
    print(f"[flops] dataset={DATASET} n_genes={F} n_classes={C} M={M} batch={xb.shape[0]} "
          f"device={DEV}", flush=True)

    std = STD[:1] if smoke else STD
    bio = BIO[:1] if smoke else BIO
    out = {"_meta": {"dataset": DATASET, "n_genes": F, "n_classes": C, "n_markers": M,
                     "batch": int(xb.shape[0]), "device": DEV, "counter": "FlopCounterMode"}}

    for key, mode, share, K in std:
        torch.manual_seed(SEED); np.random.seed(SEED)
        cfg = std_cfg(mode, share, K, M)
        model = RecursiveMarkerTransformer(cfg, F, {HEAD: C}, _DTYPES).to(DEV)
        fl = count_flops(model, xb, gene_var)
        out[key] = {"flops_total": fl, "flops_per_sample": fl / xb.shape[0]}
        print(f"[flops] {key:22s} K={K} mode={mode:10s} share={share!s:5s} "
              f"-> {fl/1e6:9.2f} MFLOPs/batch  ({fl/xb.shape[0]/1e3:8.1f} kFLOPs/sample)",
              flush=True)

    for key, mode, K in bio:
        torch.manual_seed(SEED); np.random.seed(SEED)
        cfg = bio_cfg("bio_both", K, SEED, 1, n_markers=M)
        cfg.recursion_mode = mode; cfg.device = DEV
        model = RecursiveMarkerTransformer(cfg, F, {HEAD: C}, _DTYPES).to(DEV)
        fl = count_flops(model, xb, gene_var)
        out[key] = {"flops_total": fl, "flops_per_sample": fl / xb.shape[0]}
        print(f"[flops] {key:22s} K={K} mode={mode:10s} (bio_both)      "
              f"-> {fl/1e6:9.2f} MFLOPs/batch  ({fl/xb.shape[0]/1e3:8.1f} kFLOPs/sample)",
              flush=True)

    d = ROOT / "results" / "cv5" / "curves"; d.mkdir(parents=True, exist_ok=True)
    outp = d / f"{DATASET}_flops.json"
    outp.write_text(json.dumps(out, indent=1))
    print(f"[flops] wrote {outp}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(smoke=ap.parse_args().smoke)
