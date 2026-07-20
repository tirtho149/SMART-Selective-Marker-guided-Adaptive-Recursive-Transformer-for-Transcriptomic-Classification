# Table data bundle (standalone reproducibility)

This folder makes the paper's tables reproducible **from data**, independent of the HPC
results tree. It contains every result JSON the table-building scripts read, plus the
scripts themselves.

## Contents
- `results/` — the exact **475** result JSON files (5-fold CV outputs) that the tables
  consume, with their original relative paths preserved (`results/cv5/...`,
  `results/repro/...`). Listed in `table_json_manifest.txt`.
- `scripts/` — the four table generators:
  - `build_cv5_tex.py`   → `cv5_main_table.tex` (Table 2), `cv5_baselines_table.tex`,
    `cv5_scaling_table.tex`
  - `build_injection_table.py` → `cv5_injection_table.tex`
  - `build_posf1_table.py` → `cv5_posf1_table.tex` (supplementary)
  - `build_pm_depth_tables.py` → `pm_depth_tables.tex` (supplementary)
- `table_json_manifest.txt` — the authoritative list of required JSONs (one path per line).

## Regenerate the tables
From this `tabledata/` directory (the scripts resolve paths relative to their parent):

```bash
mkdir -p paper                 # scripts write the .tex fragments into ./paper/
python scripts/build_cv5_tex.py
python scripts/build_injection_table.py
python scripts/build_posf1_table.py
python scripts/build_pm_depth_tables.py
```

Each `*.tex` written under `./paper/` is byte-identical to the fragment `\input` by the
paper (`../cv5_main_table.tex`, etc.). Only `numpy` is required.

## Notes
- Each JSON stores `cv_macro_f1` (mean/std over 5 folds), `config`, and metadata; the
  scripts read only the aggregate fields.
- Regenerating the JSONs themselves (training) is out of scope here — see the top-level
  repository `slurm/` and `recursive_marker_transformer/` for the training entry points.
