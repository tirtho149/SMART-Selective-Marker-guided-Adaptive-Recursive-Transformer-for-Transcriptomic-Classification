# genomap_data/

Organized, per-dataset view of every dataset used in the genomapFormer study.
Files are the real data (stored via **Git-LFS** for the large `.npy`/`.mat`/`.csv`
arrays); structure is `Category / Dataset / files`.

Structure: `Category / Dataset / files`

| Category | Dataset | genomap fig | Task | Source of originals |
|---|---|---|---|---|
| **Ischaemic** | Lung | Fig-4c | Supervised cell-type CV | `fig4c_ischaemic/data/` |
| | Oesophagus | Fig-4c | Supervised cell-type CV | `fig4c_ischaemic/data/` |
| | Spleen | Fig-4c | Supervised cell-type CV | `fig4c_ischaemic/data/` |
| **Tcell** | Elyahu2019_SCP490 | Fig-5b | Supervised CV (7 CD4 states) | `tcell_fig5b/data/` |
| **Pancreas** | Baron | Fig-7c | Leave-one-dataset-out label transfer | `genomap/data/drive_data/` |
| | Muraro | Fig-7c | " | " |
| | Segerstolpe | Fig-7c | " | " |
| | Wang | Fig-7c | " | " |
| | Xin | Fig-7c | " | " |
| **Trajectory** | Ciona_proto | Fig-8c | Unsupervised trajectory (DEMaP) | `genomap_codeocean/capsule/data/` |
| **Retinal** | comClass | Fig-9c | Unsupervised clustering | `genomap_codeocean/capsule/data/` |

## File conventions

- **Ischaemic** (`{organ}`): `{organ}_data.npy` (cells × genes), `GT_{organ}.mat`
  (labels), `{organ}_hvg_idx.npy` (1089 HVG indices), `{organ}_labelmap.json`.
- **Tcell**: `tcell_data.npy`, `GT_tcell.mat`, `tcell_gene_names.npy`,
  `tcell_hvg_idx.npy`, `tcell_labelmap.json`, plus `SCP490_raw/` (original
  Single-Cell Portal download: expression / metadata / cluster).
- **Pancreas** (`{Dataset}`): `data{Dataset}X.mat` (Segerstolpe = `dataScapleX.mat`)
  plus the shared `classLabel.mat` / `batchLabel.mat` (cell order
  Baron → Muraro → Segerstolpe → Wang → Xin).
- **Trajectory / Retinal**: Code-Ocean capsule `.mat` pairs — `data_*.mat`
  (cells × genes) and `GT_*.mat` (labels / developmental stages).

## Notes

- Large arrays (`.npy`, `.mat`, and the raw `RawData1.csv`) are stored with
  Git-LFS — run `git lfs pull` after cloning to materialise them.
- `Tcell/Elyahu2019_SCP490/SCP490_raw/` is the original Single-Cell-Portal
  download; `tcell_data.npy` is the processed matrix the experiments actually use.
