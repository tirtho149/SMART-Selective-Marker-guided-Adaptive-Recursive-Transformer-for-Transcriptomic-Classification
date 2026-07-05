# ============================================================================
# SMART: Selective Marker-guided Adaptive Recursive Transformer
#        for Transcriptomic Classification
#
# Authors:
#   Koushik Howlader   - Iowa State University
#   Tirtho Roy         - Iowa State University
#   Md Tauhidul Islam  - Stanford University
#   Wei Le             - Iowa State University
#
# Copyright (c) 2026 The SMART Authors. All Rights Reserved.
#
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, without the express written
# permission of the authors is STRICTLY PROHIBITED and will be prosecuted to
# the fullest extent permitted by law. See the LICENSE file for full terms.
# ============================================================================

"""Aggregated biological gene-gene network prior for SMART.

Builds a single ``(G, G)`` symmetric weighted adjacency over a caller-supplied
list of gene *symbols* by taking the confidence-weighted UNION of three curated
public sources:

* **STRING v12.0** -- physical/functional protein-protein association network.
  High-confidence edges (``combined_score >= 700``) only; the STRING score is
  normalised to ``[0, 1]`` as ``combined_score / 1000``. ENSP protein ids are
  mapped to HGNC symbols via STRING's own ``protein.info`` file. Mouse (10090)
  is supported by swapping the species prefix.

* **KEGG** -- pathway co-membership. Two genes that co-occur in at least one
  KEGG pathway get an edge. The weight for a single co-membership in a pathway
  of size ``s`` is ``1 / log2(s)`` (small dense pathways -> stronger edges),
  clipped to ``[0, 1]`` and accumulated as the max across shared pathways.
  hsa gene ids are mapped to symbols from KEGG ``list/hsa`` (primary symbol).

* **Reactome** -- pathway co-membership, reusing the repo's existing
  ``interaction._reactome_membership`` over ``data/brca/filtered_pathways.csv``.
  Two genes co-occurring in a Reactome pathway get an edge; the weight is
  ``1 / log2(pathway_size)`` accumulated as the max, exactly as for KEGG.

The final adjacency is the elementwise MAX across the three source adjacencies
(a confidence-weighted union: an edge supported by any source at weight ``w``
appears at ``>= w``; sources reinforce rather than dilute each other). The
matrix is symmetric with a zero diagonal and entries in ``[0, 1]``; genes with
no edges from any source are all-zero rows/cols (fine -- isolated nodes).

Caching:
* Raw downloads (STRING links/info, KEGG dumps) live under ``bio_networks/raw/``
  (gitignored). The STRING links file is streamed and pre-filtered to
  ``combined_score >= 700`` so the cache is a few MB, not ~800 MB uncompressed.
* Built adjacencies for a specific gene cohort are cached as scipy-sparse
  ``.npz`` under ``bio_networks/`` keyed by a hash of the gene list + species,
  so re-runs are instant and commit-friendly (small).

Public API::

    load_aggregated_adjacency(gene_symbols, species="human", cache_dir=None,
                              return_sources=False) -> np.ndarray | (np.ndarray, dict)

Returns a dense ``(G, G)`` float32 adjacency aligned to ``gene_symbols``. With
``return_sources=True`` also returns a dict of the per-source dense adjacencies.
"""

from __future__ import annotations

import gzip
import hashlib
import time
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE = _REPO_ROOT / "bio_networks"
_REACTOME_CSV = _REPO_ROOT / "data" / "brca" / "filtered_pathways.csv"

STRING_MIN_SCORE = 700               # high-confidence cutoff (combined_score)
STRING_VERSION = "v12.0"
_SPECIES = {"human": "9606", "hsa": "9606", "9606": "9606",
            "mouse": "10090", "mmu": "10090", "10090": "10090"}
_KEGG_ORG = {"9606": "hsa", "10090": "mmu"}

_STRING_BASE = "https://stringdb-downloads.org/download"
_KEGG_BASE = "https://rest.kegg.jp"

_HTTP_RETRIES = 4
_HTTP_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Networking helpers (retry + timeout; graceful failure)
# ---------------------------------------------------------------------------
class SourceUnavailable(RuntimeError):
    """Raised when a source download fails after all retries."""


def _requests():
    import requests  # local import so the module imports without network libs
    return requests


def _get(url: str, *, stream: bool = False):
    """GET with retry/backoff. Returns the response or raises SourceUnavailable."""
    requests = _requests()
    last = None
    for attempt in range(_HTTP_RETRIES):
        try:
            r = requests.get(url, timeout=_HTTP_TIMEOUT, stream=stream)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 - want to retry on anything transient
            last = e
            wait = 2 ** attempt
            print(f"[bio_network] GET {url} failed (attempt {attempt + 1}/"
                  f"{_HTTP_RETRIES}): {e}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise SourceUnavailable(f"{url}: {last}")


# ---------------------------------------------------------------------------
# STRING
# ---------------------------------------------------------------------------
def _string_info_path(cache_raw: Path, taxid: str) -> Path:
    return cache_raw / f"{taxid}.protein.info.{STRING_VERSION}.txt.gz"


def _string_links_filtered_path(cache_raw: Path, taxid: str) -> Path:
    # pre-filtered (score >= cutoff) to keep the cache small.
    return cache_raw / f"{taxid}.protein.links.{STRING_VERSION}.ge{STRING_MIN_SCORE}.tsv.gz"


def _download_string_info(cache_raw: Path, taxid: str) -> Path:
    dst = _string_info_path(cache_raw, taxid)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    url = f"{_STRING_BASE}/protein.info.{STRING_VERSION}/{taxid}.protein.info.{STRING_VERSION}.txt.gz"
    print(f"[bio_network] downloading STRING info: {url}", flush=True)
    r = _get(url)
    dst.write_bytes(r.content)
    return dst


def _download_string_links_filtered(cache_raw: Path, taxid: str) -> Path:
    """Stream the (large) STRING links file, keep only ``combined_score >= cutoff``,
    and cache the filtered subset gzip'd. The full file is never materialised."""
    dst = _string_links_filtered_path(cache_raw, taxid)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    url = f"{_STRING_BASE}/protein.links.{STRING_VERSION}/{taxid}.protein.links.{STRING_VERSION}.txt.gz"
    print(f"[bio_network] downloading+filtering STRING links (>= {STRING_MIN_SCORE}): {url}",
          flush=True)
    # Download the compressed links file to a temp path (streamed to disk to bound
    # memory), then decompress+filter locally. Reading the gzip directly off the
    # HTTP raw stream is fragile (the socket can close mid-decompress), so we stage
    # it; the ~80 MB .gz temp is deleted once the small filtered subset is written.
    r = _get(url, stream=True)
    gz_tmp = dst.with_suffix(".full.gz.tmp")
    with open(gz_tmp, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
    kept = 0
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with gzip.open(gz_tmp, mode="rt") as fin, gzip.open(tmp, mode="wt") as fout:
        fin.readline()  # header: 'protein1 protein2 combined_score'
        for line in fin:
            p = line.rstrip("\n").split(" ")
            if len(p) < 3:
                continue
            try:
                score = int(p[2])
            except ValueError:
                continue
            if score >= STRING_MIN_SCORE:
                fout.write(f"{p[0]}\t{p[1]}\t{score}\n")
                kept += 1
    tmp.replace(dst)
    gz_tmp.unlink(missing_ok=True)
    print(f"[bio_network] STRING: kept {kept} high-confidence edges -> {dst.name}",
          flush=True)
    return dst


def _string_ensp_to_symbol(info_gz: Path) -> dict:
    """Map ``taxid.ENSPxxxx`` -> preferred HGNC symbol from the info file."""
    m = {}
    with gzip.open(info_gz, mode="rt") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                m[parts[0]] = parts[1]
    return m


def _string_adjacency(gene_symbols: list, taxid: str, cache_raw: Path) -> sp.csr_matrix:
    """Build the (G, G) STRING adjacency (score/1000) aligned to ``gene_symbols``."""
    G = len(gene_symbols)
    sym_idx = {s: i for i, s in enumerate(gene_symbols)}
    info = _download_string_info(cache_raw, taxid)
    links = _download_string_links_filtered(cache_raw, taxid)
    ensp2sym = _string_ensp_to_symbol(info)

    rows, cols, vals = [], [], []
    with gzip.open(links, mode="rt") as f:
        for line in f:
            a, b, s = line.rstrip("\n").split("\t")
            ga = ensp2sym.get(a)
            gb = ensp2sym.get(b)
            if ga is None or gb is None:
                continue
            ia = sym_idx.get(ga)
            ib = sym_idx.get(gb)
            if ia is None or ib is None or ia == ib:
                continue
            w = int(s) / 1000.0
            rows.append(ia); cols.append(ib); vals.append(w)
    return _coo_to_sym_csr(rows, cols, vals, G)


# ---------------------------------------------------------------------------
# KEGG
# ---------------------------------------------------------------------------
def _kegg_list_path(cache_raw: Path, org: str) -> Path:
    return cache_raw / f"kegg.list.{org}.tsv"


def _kegg_pathway_link_path(cache_raw: Path, org: str) -> Path:
    return cache_raw / f"kegg.link.{org}.pathway.tsv"


def _download_kegg(cache_raw: Path, org: str) -> tuple[Path, Path]:
    lst = _kegg_list_path(cache_raw, org)
    lnk = _kegg_pathway_link_path(cache_raw, org)
    if not (lst.exists() and lst.stat().st_size > 0):
        print(f"[bio_network] downloading KEGG list/{org}", flush=True)
        lst.write_text(_get(f"{_KEGG_BASE}/list/{org}").text)
    if not (lnk.exists() and lnk.stat().st_size > 0):
        print(f"[bio_network] downloading KEGG link/{org}/pathway", flush=True)
        lnk.write_text(_get(f"{_KEGG_BASE}/link/{org}/pathway").text)
    return lst, lnk


def _kegg_geneid_to_symbol(list_path: Path) -> dict:
    """``org:ID`` -> primary HGNC symbol from KEGG ``list/<org>``.

    Line format: ``hsa:246181\\tCDS\\t1\\tAKR7L, AFAR3, ...; description``.
    The names field is the 4th tab-column; the primary symbol is the first
    comma-separated token before the ``;`` description."""
    m = {}
    for line in list_path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        gene_id = parts[0]
        names = parts[3].split(";", 1)[0]
        first = names.split(",")[0].strip()
        if first:
            m[gene_id] = first
    return m


def _kegg_adjacency(gene_symbols: list, org: str, cache_raw: Path) -> sp.csr_matrix:
    """(G, G) KEGG co-membership adjacency; edge weight per shared pathway of
    size s is 1/log2(s) (clipped to [0,1]), accumulated as the max."""
    G = len(gene_symbols)
    sym_idx = {s: i for i, s in enumerate(gene_symbols)}
    list_path, link_path = _download_kegg(cache_raw, org)
    id2sym = _kegg_geneid_to_symbol(list_path)

    # pathway -> set of local gene indices present in the cohort
    pw_members: dict[str, set] = {}
    for line in link_path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        pw, gene_id = parts[0], parts[1]
        sym = id2sym.get(gene_id)
        if sym is None:
            continue
        idx = sym_idx.get(sym)
        if idx is None:
            continue
        pw_members.setdefault(pw, set()).add(idx)

    return _pathway_membership_adjacency(pw_members, G)


# ---------------------------------------------------------------------------
# Reactome (reuse interaction._reactome_membership)
# ---------------------------------------------------------------------------
def _reactome_adjacency(gene_symbols: list, reactome_csv: Path) -> sp.csr_matrix:
    """(G, G) Reactome co-membership adjacency using the repo's existing
    membership reader. Same 1/log2(size) co-membership weighting as KEGG."""
    from .interaction import _reactome_membership
    G = len(gene_symbols)
    P = _reactome_membership(list(gene_symbols), reactome_csv)   # (G, M) binary
    pw_members: dict[str, set] = {}
    for j in range(P.shape[1]):
        members = set(np.nonzero(P[:, j])[0].tolist())
        if members:
            pw_members[str(j)] = members
    return _pathway_membership_adjacency(pw_members, G)


# ---------------------------------------------------------------------------
# Shared graph builders
# ---------------------------------------------------------------------------
def _pathway_membership_adjacency(pw_members: dict, G: int) -> sp.csr_matrix:
    """Co-membership graph: for each pathway, connect all member pairs with weight
    ``1/log2(size)`` (clipped to (0,1]); accumulate the MAX weight over pathways.
    Uses a dense accumulator when small, else a dict-of-edges to stay memory-lean.

    Pathways with >= 2 members contribute edges. A pathway of size s contributes
    O(s^2) pairs, so very large pathways are down-weighted (small 1/log2(s)) but
    still O(s^2) work -- fine for KEGG/Reactome pathway sizes (<~2000)."""
    edges: dict[tuple, float] = {}
    for members in pw_members.values():
        s = len(members)
        if s < 2:
            continue
        w = 1.0 / np.log2(max(s, 3))          # size>=3 -> in (0,1]; size 2 -> 1.0 via clip
        w = float(min(1.0, w if s > 2 else 1.0))
        idx = sorted(members)
        for a in range(len(idx)):
            ia = idx[a]
            for b in range(a + 1, len(idx)):
                ib = idx[b]
                key = (ia, ib)
                if w > edges.get(key, 0.0):
                    edges[key] = w
    if not edges:
        return sp.csr_matrix((G, G), dtype=np.float32)
    rows = np.fromiter((k[0] for k in edges), dtype=np.int64, count=len(edges))
    cols = np.fromiter((k[1] for k in edges), dtype=np.int64, count=len(edges))
    vals = np.fromiter(edges.values(), dtype=np.float32, count=len(edges))
    return _coo_to_sym_csr(rows, cols, vals, G)


def _coo_to_sym_csr(rows, cols, vals, G: int) -> sp.csr_matrix:
    """Build a symmetric CSR (0 diagonal) taking the MAX over duplicate/mirror
    entries, so both (i,j) and (j,i) hold the same weight."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float32)
    if len(rows) == 0:
        return sp.csr_matrix((G, G), dtype=np.float32)
    # symmetrise by stacking the transpose, then reduce duplicates with max.
    r = np.concatenate([rows, cols])
    c = np.concatenate([cols, rows])
    v = np.concatenate([vals, vals])
    # COO sums duplicates; to get MAX we use maximum.reduceat on a sorted flat index.
    flat = r.astype(np.int64) * G + c.astype(np.int64)
    order = np.argsort(flat, kind="stable")
    flat_s = flat[order]; v_s = v[order]
    uniq, start = np.unique(flat_s, return_index=True)
    vmax = np.maximum.reduceat(v_s, start)
    rr = (uniq // G).astype(np.int64)
    cc = (uniq % G).astype(np.int64)
    M = sp.csr_matrix((vmax, (rr, cc)), shape=(G, G), dtype=np.float32)
    M.setdiag(0.0)
    M.eliminate_zeros()
    return M


# ---------------------------------------------------------------------------
# Caching key
# ---------------------------------------------------------------------------
def _cohort_key(gene_symbols: list, taxid: str) -> str:
    h = hashlib.sha1()
    h.update(taxid.encode())
    h.update(b"\x00")
    h.update("\n".join(gene_symbols).encode())
    return f"agg_{taxid}_G{len(gene_symbols)}_{h.hexdigest()[:12]}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_aggregated_adjacency(gene_symbols: list,
                              species: str = "human",
                              cache_dir: Optional[Path] = None,
                              return_sources: bool = False,
                              rebuild: bool = False):
    """Return a dense ``(G, G)`` float32 symmetric adjacency (0 diagonal, weights
    in ``[0, 1]``) aligned to ``gene_symbols``, aggregating STRING + KEGG +
    Reactome as a confidence-weighted (elementwise-max) union.

    Parameters
    ----------
    gene_symbols : list[str]
        HGNC symbols (human) or MGI symbols (mouse). Defines row/col order.
    species : str
        ``"human"``/``"9606"`` (default) or ``"mouse"``/``"10090"``. Reactome
        membership uses the repo's human ``filtered_pathways.csv`` and is only
        applied for human.
    cache_dir : Path, optional
        Base cache dir (default ``<repo>/bio_networks``). Raw downloads go to
        ``<cache_dir>/raw``; built adjacencies to ``<cache_dir>/*.npz``.
    return_sources : bool
        If True, also return ``{"string":A, "kegg":A, "reactome":A}`` dense
        per-source adjacencies (same shape/order).
    rebuild : bool
        Ignore any cached built adjacency and rebuild from raw.

    Returns
    -------
    np.ndarray  (or tuple with the sources dict if ``return_sources``)
    """
    taxid = _SPECIES.get(str(species).lower())
    if taxid is None:
        raise ValueError(f"Unsupported species {species!r}; use human or mouse")
    gene_symbols = list(gene_symbols)
    G = len(gene_symbols)

    cache = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    cache_raw = cache / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    cache_raw.mkdir(parents=True, exist_ok=True)

    key = _cohort_key(gene_symbols, taxid)
    npz_path = cache / f"{key}.npz"

    if npz_path.exists() and not rebuild and not return_sources:
        M = sp.load_npz(npz_path)
        return _to_dense(M, G)

    # ---- build each source (failures are reported, not silently dropped) ----
    sources: dict[str, sp.csr_matrix] = {}
    unavailable: dict[str, str] = {}

    try:
        sources["string"] = _string_adjacency(gene_symbols, taxid, cache_raw)
    except SourceUnavailable as e:
        unavailable["string"] = str(e)
        sources["string"] = sp.csr_matrix((G, G), dtype=np.float32)

    org = _KEGG_ORG.get(taxid)
    if org is None:
        unavailable["kegg"] = f"no KEGG organism mapping for taxid {taxid}"
        sources["kegg"] = sp.csr_matrix((G, G), dtype=np.float32)
    else:
        try:
            sources["kegg"] = _kegg_adjacency(gene_symbols, org, cache_raw)
        except SourceUnavailable as e:
            unavailable["kegg"] = str(e)
            sources["kegg"] = sp.csr_matrix((G, G), dtype=np.float32)

    if taxid == "9606":
        if _REACTOME_CSV.exists():
            sources["reactome"] = _reactome_adjacency(gene_symbols, _REACTOME_CSV)
        else:
            unavailable["reactome"] = f"missing {_REACTOME_CSV}"
            sources["reactome"] = sp.csr_matrix((G, G), dtype=np.float32)
    else:
        unavailable["reactome"] = ("Reactome membership CSV is human; skipped for "
                                   f"taxid {taxid}")
        sources["reactome"] = sp.csr_matrix((G, G), dtype=np.float32)

    # ---- aggregate = elementwise MAX (confidence-weighted union) ----
    agg = sources["string"].copy()
    for name in ("kegg", "reactome"):
        agg = agg.maximum(sources[name])
    agg = agg.tocsr()
    agg.setdiag(0.0)
    agg.eliminate_zeros()

    if unavailable:
        for k, v in unavailable.items():
            print(f"[bio_network] WARNING source {k!r} contributed NOTHING: {v}",
                  flush=True)

    # ---- cache the built aggregate (sparse, small) ----
    sp.save_npz(npz_path, agg)

    dense = _to_dense(agg, G)
    if return_sources:
        src_dense = {k: _to_dense(v, G) for k, v in sources.items()}
        return dense, src_dense
    return dense


def _to_dense(M: sp.spmatrix, G: int) -> np.ndarray:
    A = np.asarray(M.todense(), dtype=np.float32)
    np.fill_diagonal(A, 0.0)
    return A


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------
def coverage_stats(gene_symbols: list, sources: dict, agg: np.ndarray) -> dict:
    """Fraction of genes with >= 1 edge, per source and for the union."""
    G = len(gene_symbols)
    out = {"n_genes": G}
    for name, A in sources.items():
        deg = (A > 0).sum(axis=1)
        out[name] = {
            "genes_with_edge": int((deg > 0).sum()),
            "coverage_frac": float((deg > 0).mean()),
            "n_edges": int((A > 0).sum() // 2),
            "mean_degree": float(deg.mean()),
        }
    deg = (agg > 0).sum(axis=1)
    out["union"] = {
        "genes_with_edge": int((deg > 0).sum()),
        "coverage_frac": float((deg > 0).mean()),
        "n_edges": int((agg > 0).sum() // 2),
        "mean_degree": float(deg.mean()),
    }
    return out


def _print_coverage(name: str, gene_symbols: list, cov: dict) -> None:
    print(f"\n=== coverage: {name}  (G={cov['n_genes']}) ===")
    for src in ("string", "kegg", "reactome", "union"):
        if src not in cov:
            continue
        s = cov[src]
        print(f"  {src:9s}  genes_with_edge={s['genes_with_edge']:6d}  "
              f"cov={s['coverage_frac']:.3f}  edges={s['n_edges']:8d}  "
              f"mean_deg={s['mean_degree']:.2f}")


# ---------------------------------------------------------------------------
# CLI: build + cache adjacencies for the P-NET cohorts, print coverage.
# ---------------------------------------------------------------------------
def _cli(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohorts", nargs="+",
                    default=["prostate", "blca", "stad"],
                    help="P-NET cohort names to build (channels=mut_cnv genes).")
    ap.add_argument("--species", default="human")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit-genes", type=int, default=0,
                    help="debug: cap the gene list length (0 = all).")
    args = ap.parse_args(argv)

    from .pathway_data import load_cohort
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    for c in args.cohorts:
        coh = load_cohort(c, channels="mut_cnv")
        genes = list(coh.genes)
        if args.limit_genes:
            genes = genes[: args.limit_genes]
        print(f"\n[bio_network] building cohort={c} G={len(genes)} species={args.species}")
        agg, srcs = load_aggregated_adjacency(
            genes, species=args.species, cache_dir=cache_dir,
            return_sources=True, rebuild=args.rebuild)
        cov = coverage_stats(genes, srcs, agg)
        _print_coverage(c, genes, cov)
        key = _cohort_key(genes, _SPECIES[args.species.lower()])
        base = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
        print(f"  cached: {base / (key + '.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
