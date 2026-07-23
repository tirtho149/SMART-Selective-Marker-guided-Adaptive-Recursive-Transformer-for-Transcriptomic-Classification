"""Build Pathformer-format inputs for a bioMOR cohort using the shared backbone.

Writes the exact files Pathformer_train_{2,3}mod.py expect into --output_dir,
but the sample-level CV split (sample_cross.tsv columns dataset_<k>_new) is taken
DIRECTLY from bc.cv_folds(y) so folds are byte-identical to bioMoR.

Pathway structure:
  * Uses data/<cohort>/filtered_pathways.csv + adjacency_matrix.csv if present
    (auto-detects a name column and a gene-list column / or a gene x pathway
    membership matrix). Otherwise falls back to a generic contiguous gene
    grouping with an identity-ish crosstalk (documented in NOTES.md).

Outputs (match pathformer_preprocess.py schema):
  gene_all.txt, gene_select.txt, modal_type_all.txt,
  pathway_gene_w.npy (G_select x P), pathway_crosstalk_network.npy (P x P),
  data_all.npy (N x G_all x n_omics), sample_cross.tsv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
REPO = os.path.dirname(BASE)
sys.path.insert(0, REPO)
import biomor_common as bc  # noqa: E402


def load_pathways(cohort, genes):
    """Return (pathway_names list, pathway_dict name->[genes], crosstalk PxP df or None)."""
    gene_set = set(genes)
    pw_path = os.path.join(bc.DATA, cohort, "filtered_pathways.csv")
    adj_path = os.path.join(bc.DATA, cohort, "adjacency_matrix.csv")
    pathway_dict = {}
    cross = None
    if os.path.exists(pw_path):
        df = pd.read_csv(pw_path)
        cols_lower = {c.lower(): c for c in df.columns}
        gene_col = next((cols_lower[c] for c in
                         ("genes", "gene", "gene_list", "members") if c in cols_lower), None)
        name_col = next((cols_lower[c] for c in
                         ("pathway_id", "pathway_name", "pathway", "name") if c in cols_lower),
                        df.columns[0])
        if gene_col is not None:
            for _, row in df.iterrows():
                raw = str(row[gene_col])
                sep = "|" if "|" in raw else ","
                members = [g.strip() for g in raw.split(sep) if g.strip() in gene_set]
                if len(members) >= 5:
                    pathway_dict[str(row[name_col])] = members
        else:
            # membership matrix: first col = gene, remaining cols = pathways (0/1)
            first = df.columns[0]
            df = df.set_index(first)
            for pw in df.columns:
                members = [g for g in df.index[df[pw] > 0] if g in gene_set]
                if len(members) >= 5:
                    pathway_dict[str(pw)] = members
    if os.path.exists(adj_path):
        try:
            cross = pd.read_csv(adj_path, index_col=0)
        except Exception:
            cross = None
    return pathway_dict, cross


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--modalities", nargs="+", default=["mutation", "cnv"])
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    X, y, meta = bc.load_omics(args.cohort, modalities=tuple(args.modalities))
    dims = [meta["modality_dims"][m] for m in args.modalities]
    assert len(set(dims)) == 1, f"Pathformer expects aligned gene sets, got {dims}"
    G = dims[0]
    n_omics = len(args.modalities)
    genes_all = [n.split(":", 1)[1] for n in meta["feature_names"][:G]]
    # data_all: (N, G, n_omics)
    data = np.stack([X[:, i * G:(i + 1) * G] for i in range(n_omics)], axis=-1).astype(np.float32)
    print(f"cohort={args.cohort} data_all={data.shape} genes={G} omics={n_omics}")

    pathway_dict, cross_df = load_pathways(args.cohort, genes_all)
    if not pathway_dict:
        print("no usable pathway file -> generic contiguous 50-gene grouping")
        block = 50
        pathway_dict = {}
        for i in range(0, G, block):
            members = genes_all[i:i + block]
            if len(members) >= 5:
                pathway_dict[f"block_{i//block}"] = members
    pathway_names = list(pathway_dict)
    P = len(pathway_names)
    print(f"pathways={P}")

    gene_select = sorted({g for v in pathway_dict.values() for g in v},
                         key=lambda g: genes_all.index(g))
    gidx = {g: i for i, g in enumerate(gene_select)}
    mask = np.zeros((len(gene_select), P), dtype=np.float32)
    for j, pw in enumerate(pathway_names):
        for g in pathway_dict[pw]:
            mask[gidx[g], j] = 1.0

    # crosstalk P x P
    if cross_df is not None and set(pathway_names).issubset(set(map(str, cross_df.index))):
        cross_df.index = cross_df.index.astype(str)
        cross_df.columns = cross_df.columns.astype(str)
        cross = cross_df.loc[pathway_names, pathway_names].to_numpy(np.float32)
    else:
        # Jaccard gene-overlap crosstalk as a generic default.
        sets = [set(pathway_dict[p]) for p in pathway_names]
        cross = np.eye(P, dtype=np.float32)
        for a in range(P):
            for b in range(a + 1, P):
                inter = len(sets[a] & sets[b])
                if inter:
                    j = inter / len(sets[a] | sets[b])
                    cross[a, b] = cross[b, a] = j
    cross[np.isnan(cross)] = 0.0

    # sample_cross.tsv from bc folds (byte-identical to bioMoR CV5)
    folds = bc.cv_folds(y)
    ids = meta["patient_ids"]
    label_tsv = pd.DataFrame({"id": ids, "y": y.astype(int)})
    for k, (tr, va, te) in enumerate(folds, start=1):
        split = np.array(["train"] * len(y), dtype=object)
        split[va] = "validation"
        split[te] = "test"
        label_tsv[f"dataset_{k}_new"] = split

    out = args.output_dir
    with open(os.path.join(out, "gene_all.txt"), "w") as f:
        f.write("\n".join(genes_all) + "\n")
    with open(os.path.join(out, "gene_select.txt"), "w") as f:
        f.write("\n".join(gene_select) + "\n")
    with open(os.path.join(out, "modal_type_all.txt"), "w") as f:
        f.write("\n".join(args.modalities) + "\n")
    np.save(os.path.join(out, "pathway_gene_w.npy"), mask)
    np.save(os.path.join(out, "pathway_crosstalk_network.npy"), cross)
    np.save(os.path.join(out, "data_all.npy"), data)
    label_tsv.to_csv(os.path.join(out, "sample_cross.tsv"), sep="\t", index=True)
    print("wrote Pathformer inputs to", out)


if __name__ == "__main__":
    main()
