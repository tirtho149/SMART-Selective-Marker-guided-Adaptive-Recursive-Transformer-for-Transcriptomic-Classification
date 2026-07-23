"""bioMOR CV orchestrator for Pathformer.

1. Builds Pathformer inputs from a bioMOR cohort (pathformer_biomor_prep.py),
   using bc.cv_folds for the sample split.
2. Runs the UNMODIFIED upstream per-fold trainer (Pathformer_train_2mod.py for
   2 modalities, Pathformer_train_3mod.py for 3) for each fold.
3. Reads each fold's fold<k>_result.json (test_metrics.f1_macro / acc) and emits
   a common scores CSV via bc.write_scores.

Usage:
    python scripts/pathformer_cv.py --cohort prostate
    python scripts/pathformer_cv.py --cohort pan_meta_pri_3modal \
        --modalities mutation cnv expression
    SMOKE=1 python scripts/pathformer_cv.py --cohort prostate   # 1 fold / few epochs
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
REPO = os.path.dirname(BASE)
sys.path.insert(0, REPO)
import biomor_common as bc  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"


def run(cmd):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--modalities", nargs="+", default=["mutation", "cnv"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    n_mod = len(args.modalities)
    trainer = "Pathformer_train_3mod.py" if n_mod == 3 else "Pathformer_train_2mod.py"
    data_dir = os.path.join(BASE, "work_dirs", args.cohort, "pf_data")
    save_dir = os.path.join(BASE, "work_dirs", args.cohort, "pf_runs")
    os.makedirs(save_dir, exist_ok=True)

    # 1) preprocess (writes sample_cross.tsv from bc folds)
    run([args.python, "-u", os.path.join(HERE, "pathformer_biomor_prep.py"),
         "--cohort", args.cohort, "--modalities", *args.modalities,
         "--output_dir", data_dir])

    # need y just for fold count / n_test
    _, y, _ = bc.load_omics(args.cohort, modalities=tuple(args.modalities))
    folds = bc.cv_folds(y)
    fold_ids = [1] if SMOKE else list(range(1, len(folds) + 1))
    epochs = 3 if SMOKE else args.epochs
    min_epochs = 1 if SMOKE else 50

    # 2) per-fold training (upstream trainer, untouched)
    for k in fold_ids:
        run([args.python, "-u", os.path.join(BASE, trainer),
             "--data_dir", data_dir, "--save_dir", save_dir,
             "--fold", str(k), "--seed", "42",
             "--epochs", str(epochs), "--min_epochs", str(min_epochs)])

    # 3) aggregate -> bc.write_scores
    f1s, accs, ns = [], [], []
    for k in fold_ids:
        p = os.path.join(save_dir, f"fold{k}_result.json")
        if not os.path.exists(p):
            print(f"WARNING: missing {p}")
            continue
        r = json.load(open(p))
        tm = r["test_metrics"]
        f1s.append(100.0 * tm["f1_macro"])
        accs.append(100.0 * tm["acc"])
        ns.append(int(len(folds[k - 1][2])))

    if not f1s:
        sys.exit("no fold results produced")
    wd = os.path.join(BASE, "work_dirs", args.cohort)
    out = bc.write_scores(wd, "Pathformer", args.cohort, f1s, accs, ns,
                          suffix="smoke" if SMOKE else "")
    print(f"macro_f1 mean={np.mean(f1s):.2f}  ->  {out}")


if __name__ == "__main__":
    main()
