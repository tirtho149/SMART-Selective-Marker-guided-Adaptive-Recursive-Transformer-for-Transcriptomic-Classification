# SMART: Selective Marker-guided Adaptive Recursive Transformer

Parameter-efficient transformer for transcriptomic classification. SMART learns
which genes are *markers* worth dedicated computation (a cross-attention marker
router), compresses everything else into marker tokens (O(N²)→O(M²) attention),
and applies a **single** transformer block recursively K times (weight sharing)
with a Mixture-of-Recursions router that gives each gene an *adaptive* recursion
depth — an intrinsic importance signal. The whole pipeline (experiments → paper →
PDF) is reproducible from scripts.

> Formerly "GenomicRecursiveFormer". Renamed to **SMART** throughout.

## Repository layout

```
recursive_marker_transformer/   # the model + training + paper generator
  config.py        RMTConfig dataclass (all knobs)
  data.py          genomic_dataloader wrapper (TCGA bulk); raw-variance HVG; label remap
  embedding.py     gene-identity + value embedding
  marker.py        SlotRouter (headline), Concrete, variance/random selectors, refine gate
  recursion.py     SharedTransformerBlock + RecursiveStack (shared/independent, MoR routing)
  router.py        expert-choice / token-choice Mixture-of-Recursions routers
  model.py         RecursiveMarkerTransformer (SMART) + param counters
  losses.py        task + marker + diversity + compression + router losses
  train.py         run(cfg) on TCGA; per-class report; persists class names
  experiments.py   ablation SUITE + exact param-efficiency table -> results/*.json
  singlecell.py    run SMART on the single-cell CSV datasets -> results_singlecell/*.json
  make_paper.py    reads results/ + results_singlecell/ -> paper/*.tex (+ TikZ figures)
  bio_enrichment.py  Reactome enrichment of learned markers (in progress)
genomic_dataloader/  # TCGA loader (downloads/caches UCSC Xena RNA-seq)
tools/
  convert_capsule_to_csv.py   # genomap capsule .mat/.csv -> readable CSVs (dynamic, reproducible)
data/                # SINGLE home for all data
  tcga/              # TCGA bulk RNA-seq CSVs (HiSeqV2 from UCSC Xena), 4 cohorts
  singlecell/        # converted single-cell datasets (see data/singlecell/README.md)
results/             # TCGA experiment JSONs (feed the paper)
results_singlecell/  # single-cell run JSONs (created by singlecell.py; feeds the paper)
paper/               # generated .tex/.bib + compiled PDF
aaai_template/       # AAAI style files
requirements.txt     # pinned deps
run_all.sh           # TCGA experiments -> paper -> PDF, one command
```

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # on a GPU box, install the +cuXXX torch wheel
```

## Data

- **TCGA bulk RNA-seq** (`data/tcga/`): Illumina HiSeqV2 gene expression
  (`log2(norm_count+1)`, ~20.5k genes) for four cohorts — breast (BRCA),
  head-neck (HNSC), lung (LUNG), thyroid (THCA) — pulled from the UCSC Xena hub by
  `genomic_dataloader`. The 4-way `cancer_type` cohort label is the primary task.
- **Single-cell** (`data/singlecell/`): four datasets from the genomap capsule
  (CodeOcean 6967747) converted to readable CSVs by
  `tools/convert_capsule_to_csv.py` — Tabula Muris (55 classes), common_class (19),
  prototype (10), pancreas (15). See `data/singlecell/README.md`.

To (re)generate the single-cell CSVs from the original capsule zip:
```bash
python tools/convert_capsule_to_csv.py --zip capsule-6967747-data.zip --out data/singlecell
```

## Run

**TCGA experiments + paper (one command):**
```bash
./run_all.sh                  # experiments -> results/ -> paper/ -> PDF
```

**Single-cell generalization (fills the paper's single-cell table):**
```bash
python -m recursive_marker_transformer.singlecell \
    --epochs 15 --batch_size 1024 --lr 1e-3        # device=auto (cuda>mps>cpu)
```
Writes `results_singlecell/<dataset>.json`. The paper's
"Generalization to Single-Cell Datasets" table reads these automatically.

**Build the paper after any run:**
```bash
python -m recursive_marker_transformer.make_paper --results results --outdir paper
cd paper && pdflatex -interaction=nonstopmode genomicrecursiveformer.tex \
  && bibtex genomicrecursiveformer \
  && pdflatex -interaction=nonstopmode genomicrecursiveformer.tex \
  && pdflatex -interaction=nonstopmode genomicrecursiveformer.tex
```

## GPU note (important)

This project was developed on an **Intel Mac with no usable GPU** (the discrete
NVIDIA GT 755M is unsupported by PyTorch; `mps` needs Apple Silicon), so it falls
back to CPU. The TCGA runs are tractable, but the single-cell run is slow on CPU
(~2.5–3 h). `singlecell.py` already auto-selects `cuda` when available
(`resolve_device`), so on a CUDA machine the same command finishes in
**~5–15 min** — just install the CUDA torch wheel and (optionally) raise
`--batch_size`. Transfer everything **except `.venv/`** (recreate it from
`requirements.txt`).

## Reproducibility

- Deterministic: fixed seed; single-cell splits use each dataset's own `split.csv`
  when present, else a seeded stratified split.
- Every number in the paper is injected from `results/` and `results_singlecell/`
  via `@@TOKEN@@` placeholders — no hand-typed metrics (build prints
  "unresolved tokens: 0").
- The converter reads the capsule zip directly and is byte-reproducible.
