#!/usr/bin/env python3
"""Convert the genomap capsule (6967747) datasets into readable, analysis-ready CSVs.

The capsule ships MATLAB ``.mat`` matrices, a 1.1 GB headerless ``TMdata.csv`` and
genomap image stacks. This script reads every dataset *directly from the zip*
(no multi-GB manual extraction), auto-detects its expression matrix and label
vector, and writes a tidy, pandas-friendly layout::

    <out>/<dataset>/expression.csv[.gz]   # rows = cells, cols = cell_id + feat_0001..
    <out>/<dataset>/labels.csv            # cell_id, label, class_name
    <out>/<dataset>/split.csv             # cell_id, split   (only if a split is provided)
    <out>/manifest.csv                    # one row per dataset: shapes / classes / sizes

The run is fully deterministic: no randomness, no network, fixed cell ordering.
Re-running on the same zip reproduces byte-identical CSVs.

Usage::

    python tools/convert_capsule_to_csv.py --zip capsule-6967747-data.zip --out capsule_csv
    python tools/convert_capsule_to_csv.py --datasets common_class pancreas --no-gzip
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

# --------------------------------------------------------------------------- #
# Dataset registry. Each entry declares, declaratively, where the feature
# matrix (x) and the integer labels (y) live inside the zip. Adding a new
# dataset is a matter of appending one spec here -- nothing else changes.
#   x.kind: "csv_matrix" | "mat_matrix" | "genomaps"
#   y:      {"member","key"}  or  {"member","keys":[...]} (concatenated in order)
# --------------------------------------------------------------------------- #
SPECS: dict[str, dict] = {
    "tabula_muris": {
        "x":     {"kind": "csv_matrix", "member": "TMdata.csv"},
        "y":     {"member": "GT_TM.mat", "key": "GT"},
        "names": "dataClasseNames.csv",
        "split": {"member": "index_TM.mat", "train": "indxTrain", "test": "indxTest"},
        "note":  "Tabula Muris; 1089-gene panel arranged by genomap.",
    },
    "common_class": {
        "x":    {"kind": "mat_matrix", "member": "data_comClass.mat", "key": "X"},
        "y":    {"member": "GT_comClass.mat", "key": "GT"},
        "note": "Common-class cross-tissue benchmark.",
    },
    "prototype": {
        "x":    {"kind": "mat_matrix", "member": "data_proto.mat", "key": "X"},
        "y":    {"member": "GT_proto.mat", "key": "GT"},
        "note": "Prototype benchmark (752-gene panel).",
    },
    "pancreas": {
        "x":    {"kind": "genomaps", "member": "panc_Surat_Geno.mat",
                 "keys": ["genomapsTrain", "genomapsTest"]},
        "y":    {"member": "panc_Surat_Geno.mat", "keys": ["GTTrain", "GTTest"]},
        "split_from_keys": ["genomapsTrain", "genomapsTest"],  # implicit train/test
        "note": "Human pancreas; features are flattened 44x44 genomaps.",
    },
}


# --------------------------------------------------------------------------- #
# Zip readers
# --------------------------------------------------------------------------- #
def _read_mat(zf: zipfile.ZipFile, member: str) -> dict:
    """Load a v5 .mat member straight from the zip (no extraction)."""
    return sio.loadmat(io.BytesIO(zf.read(member)))


def _labels(zf: zipfile.ZipFile, spec: dict) -> np.ndarray:
    """Return a 1-D int label vector, concatenating train/test keys if needed."""
    m = _read_mat(zf, spec["member"])
    if "keys" in spec:
        parts = [np.asarray(m[k]).ravel() for k in spec["keys"]]
        y = np.concatenate(parts)
    else:
        y = np.asarray(m[spec["key"]]).ravel()
    return y.astype(np.int64)


def _class_names(zf: zipfile.ZipFile, member: str | None) -> list[str] | None:
    if not member:
        return None
    text = zf.read(member).decode("utf-8", "replace")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _matrix(zf: zipfile.ZipFile, xspec: dict):
    """Return (features ndarray | None, n_features). For csv_matrix we stream
    later, so it returns (None, n_features) and signals streaming."""
    kind = xspec["kind"]
    if kind == "mat_matrix":
        X = np.asarray(_read_mat(zf, xspec["member"])[xspec["key"]], dtype=np.float64)
        return X, X.shape[1]
    if kind == "genomaps":
        m = _read_mat(zf, xspec["member"])
        stacks = []
        for k in xspec["keys"]:
            a = np.asarray(m[k], dtype=np.float64)          # (H, W, 1, N)
            a = a.reshape(a.shape[0] * a.shape[1] * a.shape[2], a.shape[3]).T  # (N, H*W)
            stacks.append(a)
        X = np.concatenate(stacks, axis=0)
        return X, X.shape[1]
    if kind == "csv_matrix":
        # Peek one line for the column count; the body is streamed at write time.
        with zf.open(xspec["member"]) as fh:
            first = fh.readline().decode("utf-8")
        return None, first.count(",") + 1
    raise ValueError(f"unknown x kind: {kind}")


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def _open_w(path: Path, gz: bool):
    return gzip.open(path, "wt", newline="") if gz else open(path, "w", newline="")


def _cell_ids(n: int) -> list[str]:
    w = max(5, len(str(n)))
    return [f"cell_{i:0{w}d}" for i in range(1, n + 1)]


def _write_expression_array(path: Path, X: np.ndarray, gz: bool) -> None:
    n, d = X.shape
    header = ["cell_id"] + [f"feat_{j:04d}" for j in range(1, d + 1)]
    ids = _cell_ids(n)
    with _open_w(path, gz) as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(n):
            w.writerow([ids[i]] + [format(v, ".6g") for v in X[i]])


def _write_expression_streamed(path: Path, zf: zipfile.ZipFile, member: str,
                               n_features: int, gz: bool) -> int:
    """Stream a headerless CSV member, prepending a header row and cell ids."""
    header = ["cell_id"] + [f"feat_{j:04d}" for j in range(1, n_features + 1)]
    n = 0
    with _open_w(path, gz) as out:
        w = csv.writer(out)
        w.writerow(header)
        with zf.open(member) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8")
            for i, line in enumerate(text, start=1):
                line = line.rstrip("\n").rstrip("\r")
                if not line:
                    continue
                out.write(f"cell_{i:05d},{line}\n")
                n += 1
    return n


def _write_labels(path: Path, y: np.ndarray, names: list[str] | None) -> None:
    ids = _cell_ids(len(y))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell_id", "label", "class_name"])
        for i, lab in enumerate(y):
            cname = ""
            if names is not None and 1 <= int(lab) <= len(names):
                cname = names[int(lab) - 1]          # labels are 1-based
            w.writerow([ids[i], int(lab), cname])


def _write_split(path: Path, n: int, train_idx=None, test_idx=None,
                 n_train=None) -> None:
    """Write cell_id,split using either explicit 1-based indices or a head/tail
    train_test cut (n_train)."""
    split = np.array(["unassigned"] * n, dtype=object)
    if n_train is not None:
        split[:n_train] = "train"
        split[n_train:] = "test"
    else:
        split[np.asarray(train_idx).ravel() - 1] = "train"
        split[np.asarray(test_idx).ravel() - 1] = "test"
    ids = _cell_ids(n)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell_id", "split"])
        for i in range(n):
            w.writerow([ids[i], split[i]])


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def convert(zip_path: Path, out_dir: Path, datasets: list[str], gz: bool) -> list[dict]:
    manifest = []
    with zipfile.ZipFile(zip_path) as zf:
        present = set(zf.namelist())
        for name in datasets:
            spec = SPECS[name]
            # Skip gracefully if a required member is missing from this zip.
            need = {spec["x"]["member"], spec["y"]["member"]}
            if not need.issubset(present):
                print(f"[skip] {name}: missing {sorted(need - present)}")
                continue

            print(f"[{name}] reading labels + features ...")
            y = _labels(zf, spec["y"])
            names = _class_names(zf, spec.get("names"))
            X, n_feat = _matrix(zf, spec["x"])

            dset_dir = out_dir / name
            dset_dir.mkdir(parents=True, exist_ok=True)
            ext = ".csv.gz" if gz else ".csv"
            expr_path = dset_dir / f"expression{ext}"

            if spec["x"]["kind"] == "csv_matrix":
                n = _write_expression_streamed(expr_path, zf, spec["x"]["member"], n_feat, gz)
            else:
                n = X.shape[0]
                _write_expression_array(expr_path, X, gz)

            if len(y) != n:
                print(f"  [warn] {name}: {n} feature rows vs {len(y)} labels "
                      f"-- truncating to min")
                n = min(n, len(y))
                y = y[:n]

            _write_labels(dset_dir / "labels.csv", y, names)

            # Split, if the capsule provides one.
            split_info = "none"
            if "split" in spec:
                s = _read_mat(zf, spec["split"]["member"])
                _write_split(dset_dir / "split.csv", n,
                             train_idx=s[spec["split"]["train"]],
                             test_idx=s[spec["split"]["test"]])
                split_info = "index file"
            elif "split_from_keys" in spec:
                ytr = np.asarray(_read_mat(zf, spec["y"]["member"])
                                 [spec["y"]["keys"][0]]).ravel()
                _write_split(dset_dir / "split.csv", n, n_train=len(ytr))
                split_info = "train/test stacks"

            row = {
                "dataset": name, "n_samples": int(n), "n_features": int(n_feat),
                "n_classes": int(len(np.unique(y))),
                "label_min": int(y.min()), "label_max": int(y.max()),
                "has_class_names": names is not None, "split": split_info,
                "expression_file": str(expr_path.relative_to(out_dir.parent)),
                "note": spec.get("note", ""),
            }
            manifest.append(row)
            print(f"  -> {row['n_samples']}x{row['n_features']}, "
                  f"{row['n_classes']} classes, split={split_info}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default="capsule-6967747-data.zip", type=Path)
    ap.add_argument("--out", default="capsule_csv", type=Path)
    ap.add_argument("--datasets", nargs="*", default=list(SPECS),
                    choices=list(SPECS), help="subset to convert (default: all)")
    ap.add_argument("--no-gzip", dest="gzip", action="store_false",
                    help="write plain .csv instead of .csv.gz")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = convert(args.zip, args.out, args.datasets, args.gzip)

    # Manifest as both CSV (human) and JSON (machine).
    if manifest:
        cols = list(manifest[0].keys())
        with open(args.out / "manifest.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(manifest)
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\n[done] {len(manifest)} dataset(s) -> {args.out}/  (see manifest.csv)")
    else:
        print("\n[done] nothing converted")


if __name__ == "__main__":
    main()
