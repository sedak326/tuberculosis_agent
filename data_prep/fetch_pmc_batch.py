#!/usr/bin/env python3
"""
fetch_pmc_batch.py — Run multiple PMC queries in sequence and download full-text XMLs.

Deduplicates across all queries by:
  1. PMC ID   — file already exists in out-dir (PMC<id>.xml)
  2. Title    — normalised title already seen this session or in pre-existing files

Usage:
    python fetch_pmc_batch.py --out-dir mtubercolosis/ --api-key YOUR_KEY
    python fetch_pmc_batch.py --out-dir mtubercolosis/ --dry-run

Get a free NCBI API key at: https://www.ncbi.nlm.nih.gov/account/
Without an API key you are limited to 3 requests/sec (slower but works fine).
"""

import argparse
import re
import time
from pathlib import Path

import requests

INTER_QUERY_DELAY = 5.0  # seconds between queries to avoid burst rate-limiting


def get_with_retry(url: str, params: dict, timeout: int = 30, max_retries: int = 5) -> requests.Response:
    """GET with exponential backoff on 429/5xx."""
    for attempt in range(max_retries):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 429 or r.status_code >= 500:
            wait = 2 ** attempt * 2
            print(f"  [{r.status_code}] rate-limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r

# ---------------------------------------------------------------------------
# Query list — each tuple is (label, query_string, max_results)
# All queries target open-access full-text available in PMC.
# ---------------------------------------------------------------------------
QUERIES = [
    # NOTE: proteomics papers already fetched to mtubercolosis/ — not included here.
    # Merge corpus.jsonl files at the corpus level after extraction.

    # Transcriptomics: gene expression studies, RNA-seq, microarray
    (
        "transcriptomics",
        '"Mycobacterium tuberculosis"[MeSH] AND transcriptomics[tiab]',
        9999,
    ),
    # Gene expression: broader than transcriptomics tag, catches older literature
    (
        "gene expression",
        '"Mycobacterium tuberculosis"[MeSH] AND "gene expression"[MeSH]',
        9999,
    ),
    # Genomics: whole-genome sequencing, comparative genomics, SNP studies
    (
        "genomics",
        '"Mycobacterium tuberculosis"[MeSH] AND genomics[MeSH]',
        9999,
    ),
    # Metabolomics: metabolite profiling, flux analysis
    (
        "metabolomics",
        '"Mycobacterium tuberculosis"[MeSH] AND metabolomics[tiab]',
        9999,
    ),
    # Lipidomics: TB has an unusually lipid-rich cell wall, key to pathogenesis
    (
        "lipidomics",
        '"Mycobacterium tuberculosis"[MeSH] AND lipidomics[tiab]',
        9999,
    ),
    # Protein structure: crystallography, cryo-EM, structural biology of TB proteins
    (
        "protein structure",
        '"Mycobacterium tuberculosis"[MeSH] AND "protein structure"[MeSH]',
        9999,
    ),
    # Drug resistance: AMR mechanisms, resistance mutations, efflux pumps
    (
        "drug resistance",
        '"Mycobacterium tuberculosis"[MeSH] AND "drug resistance"[MeSH]',
        9999,
    ),
    # Host-pathogen interactions: macrophage responses, immune evasion
    (
        "host-pathogen interactions",
        '"Mycobacterium tuberculosis"[MeSH] AND "host-pathogen interactions"[MeSH]',
        9999,
    ),
    # Virulence: pathogenicity factors, secretion systems (ESX), toxin-antitoxin
    (
        "virulence",
        '"Mycobacterium tuberculosis"[MeSH] AND virulence[MeSH]',
        9999,
    ),
    # Signal transduction: two-component systems, kinases, phosphorylation
    (
        "signal transduction",
        '"Mycobacterium tuberculosis"[MeSH] AND "signal transduction"[MeSH]',
        9999,
    ),
    # Molecular biology: broad catch-all for mechanistic studies
    (
        "molecular biology",
        '"tuberculosis"[MeSH] AND "molecular biology"[MeSH]',
        9999,
    ),
    # Epigenomics / methylation
    (
        "epigenomics",
        '"Mycobacterium tuberculosis"[MeSH] AND epigenomics[tiab]',
        9999,
    ),
    # Drug targets / essential genes
    (
        "drug targets",
        '"Mycobacterium tuberculosis"[MeSH] AND "drug design"[MeSH]',
        9999,
    ),
    # Latency / dormancy: non-replicating persistence, hypoxia response
    (
        "latency and dormancy",
        '"Mycobacterium tuberculosis"[MeSH] AND dormancy[tiab]',
        9999,
    ),
    # Biofilm
    (
        "biofilm",
        '"Mycobacterium tuberculosis"[MeSH] AND biofilm[tiab]',
        9999,
    ),
]

# ---------------------------------------------------------------------------
# E-utilities endpoints
# ---------------------------------------------------------------------------
ESEARCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

TITLE_PATTERNS = [
    re.compile(rb"<dc:title[^>]*>(.*?)</dc:title>", re.DOTALL),
    re.compile(rb"<article-title[^>]*>(.*?)</article-title>", re.DOTALL),
]
TAG_RE = re.compile(rb"<[^>]+>")


def normalise(title: str | bytes) -> str:
    if isinstance(title, bytes):
        title = TAG_RE.sub(b"", title).decode("utf-8", errors="ignore")
    return re.sub(r"[^a-z0-9]", "", title.lower())


def load_existing_titles(out_dir: Path) -> set[str]:
    titles: set[str] = set()
    for f in out_dir.iterdir():
        if f.suffix != ".xml":
            continue
        try:
            content = f.read_bytes()
        except OSError:
            continue
        for pattern in TITLE_PATTERNS:
            m = pattern.search(content)
            if m:
                norm = normalise(m.group(1))
                if norm:
                    titles.add(norm)
                break
    return titles


def search_pmc(query: str, max_results: int, api_key: str | None, delay: float) -> list[str]:
    ids: list[str] = []
    retstart = 0
    batch = min(max_results, 10000)

    while retstart < max_results:
        params = {
            "db": "pmc",
            "term": query,
            "retmax": min(batch, max_results - retstart),
            "retstart": retstart,
            "retmode": "json",
        }
        if api_key:
            params["api_key"] = api_key

        r = get_with_retry(ESEARCH_URL, params, timeout=30)
        result = r.json()["esearchresult"]
        batch_ids = result["idlist"]
        ids.extend(batch_ids)

        total_available = int(result.get("count", 0))
        retstart += len(batch_ids)

        if not batch_ids or retstart >= total_available:
            break

        time.sleep(delay)

    return ids, total_available


def fetch_titles_batch(pmc_ids: list[str], api_key: str | None, delay: float) -> dict[str, str]:
    result: dict[str, str] = {}
    for i in range(0, len(pmc_ids), 200):
        chunk = pmc_ids[i : i + 200]
        params = {"db": "pmc", "id": ",".join(chunk), "retmode": "json"}
        if api_key:
            params["api_key"] = api_key
        r = get_with_retry(ESUMMARY_URL, params, timeout=30)
        for uid, rec in r.json().get("result", {}).items():
            if uid == "uids":
                continue
            title = rec.get("title", "")
            if title:
                result[uid] = normalise(title)
        time.sleep(delay)
    return result


def fetch_xml(pmc_id: str, api_key: str | None) -> bytes | None:
    params = {"db": "pmc", "id": pmc_id, "rettype": "xml", "retmode": "xml"}
    if api_key:
        params["api_key"] = api_key
    try:
        r = get_with_retry(EFETCH_URL, params, timeout=60)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    if b"<article" not in r.content and b"<pmc-articleset" not in r.content:
        return None
    return r.content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",  default="mtubercolosis")
    parser.add_argument("--api-key",  default=None)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Search only — print counts, do not download anything.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    delay = 0.11 if args.api_key else 0.34

    # Load all titles already on disk once, shared across all queries
    print("Scanning existing files...")
    known_titles = load_existing_titles(out_dir)
    existing_files = {f.stem for f in out_dir.glob("PMC*.xml")}
    print(f"  {len(existing_files)} XML files already on disk ({len(known_titles)} titles indexed)\n")

    session_downloaded = 0
    session_skipped    = 0
    session_failed     = 0

    for label, query, max_results in QUERIES:
        print(f"{'=' * 60}")
        print(f"Query: {label}")
        print(f"  {query}")

        ids, total_available = search_pmc(query, max_results, args.api_key, delay)
        print(f"  PMC reports {total_available} total results, retrieved {len(ids)} IDs")

        # Filter IDs already on disk by filename
        new_ids = [i for i in ids if f"PMC{i}" not in existing_files]
        print(f"  {len(ids) - len(new_ids)} already on disk by ID, {len(new_ids)} to check")

        if not new_ids or args.dry_run:
            if args.dry_run:
                print(f"  [dry-run] would fetch up to {len(new_ids)} new papers")
            print()
            time.sleep(INTER_QUERY_DELAY)
            continue

        # Fetch titles for new IDs to catch title-level duplicates
        print(f"  Fetching titles for {len(new_ids)} candidates...")
        title_map = fetch_titles_batch(new_ids, args.api_key, delay)

        downloaded = failed = skipped_title = 0

        for i, pmc_id in enumerate(new_ids, 1):
            norm_title = title_map.get(pmc_id, "")
            if norm_title and norm_title in known_titles:
                skipped_title += 1
                continue

            label_str = title_map.get(pmc_id, f"PMC{pmc_id}")[:60]
            print(f"  [{i}/{len(new_ids)}] {label_str!r}...", end=" ", flush=True)

            xml = fetch_xml(pmc_id, args.api_key)
            if xml:
                out_path = out_dir / f"PMC{pmc_id}.xml"
                out_path.write_bytes(xml)
                existing_files.add(f"PMC{pmc_id}")
                if norm_title:
                    known_titles.add(norm_title)
                print("ok")
                downloaded += 1
            else:
                print("no full text")
                failed += 1

            time.sleep(delay)

        print(f"  Downloaded: {downloaded}  |  Skipped (dup title): {skipped_title}  |  No full text: {failed}")
        print()

        time.sleep(INTER_QUERY_DELAY)

        session_downloaded += downloaded
        session_skipped    += skipped_title
        session_failed     += failed

    print("=" * 60)
    print("Session total")
    print(f"  Downloaded : {session_downloaded}")
    print(f"  Skipped    : {session_skipped}")
    print(f"  Failed     : {session_failed}")
    total_on_disk = len(list(out_dir.glob("PMC*.xml")))
    print(f"  XMLs on disk now: {total_on_disk}")


if __name__ == "__main__":
    main()
