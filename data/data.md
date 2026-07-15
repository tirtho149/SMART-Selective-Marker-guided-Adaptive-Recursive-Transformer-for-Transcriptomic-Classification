# Datasets

Cancer-genomics datasets used to train and benchmark PATH (a pathway-based graph
transformer). Each dataset is one cancer cohort. Patients are described by two
genomic modalities (somatic mutation and copy-number variation) over a shared set
of genes, and genes are grouped into biological pathways that form the nodes of a
pathway–pathway interaction graph.

All cohorts share the **same pathway graph**: 1,268 Reactome pathways and an
identical 1,268 × 1,268 adjacency matrix. They differ in the patient cohort, the
gene feature space, and the prediction task.

## Files in each dataset

Every dataset directory contains the following five files.

| File | Shape | Contents |
|---|---|---|
| `mutation_data.csv` | patients × genes | Somatic mutation matrix. Rows = patients, columns = genes. Binary {0,1} = mutated / wild-type for most cohorts (prostate holds mutation **counts**, 0–12). Row index = patient ID, header row = gene symbols. |
| `cnv_data.csv` | patients × genes | Copy-number variation matrix. Same patients × genes layout as `mutation_data.csv`. GISTIC thresholded values {−2,−1,0,1,2} = deep loss / loss / neutral / gain / amplification. |
| `labels.csv` | patients × 2 (or 4) | Patient labels. Column 1 = patient ID, column 2 = `response` (the classification target). `pan_meta_pri` additionally has `sample_type` and `primary_disease`. |
| `pathways.csv` | 1,268 × 3 | Pathway definitions. Columns: `Pathway_ID` (Reactome ID, e.g. `R-HSA-109581`), `Pathway_Name` (e.g. `Apoptosis`), `Genes` (comma-separated member gene symbols). These 1,268 pathways are the graph nodes. |
| `adjacency_matrix.csv` | 1,268 × 1,268 | Pathway–pathway interaction graph. Square matrix; entry (i,j) = interaction strength between pathway i and pathway j. Row/column order matches `pathways.csv`. Same file across all cohorts. |

The `mutation_data.csv` and `cnv_data.csv` files always cover the **same** patient
set and the **same** gene columns within a cohort (align by patient ID — row order
between the two files is not guaranteed identical).

## Datasets and tasks

| Dataset | Patients | Genes | Classes | Task / label meaning | Class distribution |
|---|---|---|---|---|---|
| `blca` | 404 | 23,384 | 2 | Bladder urothelial carcinoma, binary clinical outcome (TCGA primary tumors; `response` undocumented in file, likely early-vs-late stage) | 1: 273 (68%) / 0: 131 (32%) |
| `brca_5_class` | 526 omics / 518 labeled | 40,543 | 5 | Breast carcinoma, 5-class label (`response` 0–4, likely molecular subtype) | 2:262, 3:112, 0:95, 1:35, 4:14 |
| `stad` | 414 | 23,384 | 2 | Stomach adenocarcinoma, binary clinical outcome (TCGA primary tumors; likely early-vs-late stage) | 1: 228 (55%) / 0: 186 (45%) |
| `prostate` | 1,011 | 8,434 | 2 | Prostate cancer, binary label. Non-TCGA cohort (IDs `AAPC-STID…-Tumor-SM-…`); mutation matrix holds counts, not binary | 0: 678 (67%) / 1: 333 (33%) |
| `pan_meta_pri` | 8,893 | 23,384 | 2 | Pan-cancer **metastatic vs primary** classification across 32 cancer types. `response` 1 = Metastatic (361), 0 = Primary Tumor (8,532). Extra columns `sample_type`, `primary_disease` | 0: 8,532 (96%) / 1: 361 (4%) — highly imbalanced |

### Notes

- **Only `pan_meta_pri` has documented label semantics** (`sample_type` column
  confirms 1 = Metastatic, 0 = Primary Tumor). For `blca`, `stad`, and
  `brca_5_class` the `response` meaning is not stored in the files; sample IDs are
  TCGA primary-tumor barcodes (`-01`), so these are not metastasis tasks.
- **`brca_5_class`** has 526 patients in the omics files but only 518 in
  `labels.csv` — 8 patients are unlabeled. Join on patient ID (inner join) when
  loading. It also has the largest gene space (40,543).
- **`prostate`** is the most distinct cohort: an external (non-TCGA) dataset, its
  mutation matrix stores mutation counts rather than binary flags, and mutation/CNV
  row order differs (align by ID). `prostate/test.csv` is an empty placeholder.
- **`pan_meta_pri`** is heavily class-imbalanced (~4% metastatic); use class
  weighting or focal loss.
- `blca` labels are stored as floats (`1.0`/`0.0`); cast to int before training.
