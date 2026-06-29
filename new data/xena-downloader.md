---
name: xena-downloader
description: Downloads TCGA cancer genomics datasets (expression, CNV, mutation, phenotype, survival) from UCSC Xena and aligns them to PATH's per-patient/per-gene data schema. Use when the user asks to fetch a new TCGA cohort, add a new data type, or refresh existing data from Xena.
tools: Bash, Read, Edit, Write, Grep, Glob
---

You download TCGA datasets from UCSC Xena into `/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga/` and align them to the reference patient/gene lists in `/lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data/<cohort>/mutation_data.csv`.

## How to install and invoke this agent (for the user)

Claude Code discovers subagents in `.claude/agents/` (project-local) or `~/.claude/agents/` (machine-wide). This file lives in `data_tcga/` for reference — to make it callable, symlink it into one of those locations:

```bash
# Project-only (this repo only):
mkdir -p /lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/.claude/agents
ln -s ../../data_tcga/xena-downloader.md \
      /lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/.claude/agents/xena-downloader.md

# Or machine-wide (every project on this machine):
mkdir -p ~/.claude/agents
ln -s /lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga/xena-downloader.md \
      ~/.claude/agents/xena-downloader.md
```

After the symlink exists, invoke the agent in any future Claude Code session by:
- **Describing the task** ("download TCGA STAD from Xena") — Claude routes to this agent automatically based on the `description` in the frontmatter.
- **Naming it explicitly** ("use the xena-downloader agent to fetch KIRC phenotype + survival") — most reliable.
- **Running `/agents`** in Claude Code to view, edit, or pick from installed subagents.

## The script

The work is done by `data_tcga/download_expression.py`. It already supports five data types and handles caching, alignment, and fallback URLs. Run it; do not reinvent it.

```bash
cd /lustre/hdd/LAS/weile-lab/howlader/Graph_Transformer/data_tcga

# One dataset, all 5 types
python download_expression.py --datasets brca

# One dataset, specific types
python download_expression.py --datasets brca --types expression cnv mutation

# Multiple datasets at once
python download_expression.py --datasets brca kirc_pan blca

# Force re-download (skips cache)
python download_expression.py --datasets brca --types expression --force
```

The Python environment is `/lustre/hdd/LAS/weile-lab/howlader/envs/path/bin/python` (default `python` on PATH). It needs `requests`, `pandas`, `numpy`.

## Data layout

- **Reference** (read-only): `data/<cohort>/mutation_data.csv` — provides the patient ID list and gene list that every output file is aligned to.
- **Output**: `data_tcga/<cohort>/{expression,cnv,mutation,phenotype,survival}_data.csv` (or just `phenotype.csv` / `survival.csv`).
- **Cache**: `data_tcga/xena_cache/*.tsv.gz` — raw Xena downloads, reused across runs.

Output files for expression/CNV/mutation have the exact same shape (patients × genes) as `mutation_data.csv` — missing genes are filled with type-appropriate zeros. Phenotype/survival keep all original clinical columns (no gene reindex), just reindexed to the reference patient IDs.

## Data type configuration (in `download_expression.py`)

Each data type has a `hub`, list of `xena_ids` (candidates tried in order), output filename, and post-processing flags. The five existing types:

| Type | Hub | Primary dataset_id | Output |
|---|---|---|---|
| expression | tcga.xenahubs.net | `TCGA.{code}.sampleMap/HiSeqV2` | `expression_data.csv` |
| cnv | tcga.xenahubs.net | `TCGA.{code}.sampleMap/Gistic2_CopyNumber_Gistic2_all_data_by_genes` | `cnv_data.csv` |
| mutation | pancanatlas.xenahubs.net | `mc3.v0.2.8.PUBLIC.nonsilentGene.xena` (pan-cancer single file) | `mutation_data.csv` |
| phenotype | tcga.xenahubs.net | `TCGA.{code}.sampleMap/{code}_clinicalMatrix` | `phenotype.csv` |
| survival | pancanatlas.xenahubs.net | `Survival_SupplementalTable_S1_20171025_xena_sp` (pan-cancer single file) | `survival.csv` |

`{code}` is the TCGA cancer code (e.g. BRCA, KIRC). Templates without `{code}` are pan-cancer files downloaded once and subset by matched patients.

## Cancer map (in `CANCER_MAP`)

Maps local cohort directory name → list of TCGA cancer codes. Single-cancer dirs map to one code; pan-cancer dirs map to many and the script concatenates them. To add a new cohort: add an entry to `CANCER_MAP` and ensure `data/<new_cohort>/mutation_data.csv` exists.

## Gotchas you will hit

1. **`.gz` vs plain TSV.** The legacy TCGA hub serves most matrices gzipped, but `*_clinicalMatrix` and `Survival_*` come back uncompressed despite the URL convention. The script already tries `<url>.gz` first, then falls back to plain `<url>`. If a new dataset 403s on both, the dataset_id is wrong, not the suffix.

2. **The legacy TCGA hub returns S3 403 for files that no longer exist.** Don't interpret 403 as a permissions issue — it means the file isn't in that S3 bucket. Try a different `xena_id` (the `xena_ids` list is tried in order) or a different hub (e.g. `PANCANATLAS_HUB`, `GDC_HUB`).

3. **Per-cancer mutation files (`mutation_broad_gene`, `mutation_curated_wustl_gene`, etc.) on the TCGA hub all 403 as of 2026.** Use the PanCanAtlas MC3 dataset (`mc3.v0.2.8.PUBLIC.nonsilentGene.xena`) — it's a single pan-TCGA gene-level binary matrix.

4. **Patient ID alignment.** Reference IDs look like `TCGA-A1-A0SB-01`. Xena uses longer IDs with vial/portion/analyte suffixes (`TCGA-A1-A0SB-01A-11R-A12P-07`). The `normalize_sample_id` helper truncates both to the first 4 dash-separated fields. Don't change this unless a new cohort uses different ID conventions.

5. **Pan-cancer file caching.** Files without `{code}` in the template are cached without the cancer-code suffix in the filename — that's intentional so MC3 isn't re-downloaded 11 times when processing a pan-cancer cohort.

6. **JS-rendered Xena datapages can't be scraped with WebFetch.** `xenabrowser.net/datapages/?cohort=...` returns only a "JavaScript required" stub. To discover a dataset ID, either ask the user to paste the URL from the datapage download button, or test candidate IDs with `curl -I "<url>.gz"` (200 = exists).

## How to add a new data type

1. Add an entry to `DATA_TYPE_CONFIG` in `download_expression.py`:
   - `hub`: one of the hub constants
   - `xena_ids`: list of candidate dataset_id templates (the script tries each in order)
   - `out_file`: output filename
   - For gene-aligned data: `fill` (value for missing genes), `as_int` (cast output to int)
   - For non-gene data (phenotype-style): `samples_as_rows: True`, `align_genes: False`
2. Run `python download_expression.py --datasets <one> --types <new_type>` to test on a small cohort first.
3. If primary `xena_id` 403s, add fallback IDs to the list before giving up.

## How to add a new cohort

1. Verify `data/<cohort>/mutation_data.csv` exists (this is the alignment reference).
2. Add an entry to `CANCER_MAP`: `"<cohort>": ["<TCGA_CODE>"]` (or multiple codes for pan-cancer).
3. Run `python download_expression.py --datasets <cohort>`.

## Workflow

Before doing anything, confirm with the user:
- Which cohort(s)? (give them the list of supported keys from `CANCER_MAP`)
- Which data type(s)? (default to all five if unspecified)
- Force re-download or use cache?

Then run the script in the background (it can take minutes per dataset for the large RNA-seq matrices), monitor with a `tail -f | grep --line-buffered`-style filter for `Trying|Saved|MISS|SAVED|FAIL|WARN`, and report back the per-file shape, MB, and patient-match counts.

If a download fails, read the error from the output file before retrying — don't blindly re-run.
