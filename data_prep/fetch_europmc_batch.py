#!/usr/bin/env python3
"""
fetch_europmc_batch.py — Fetch TB papers from Europe PMC that are not already
in the existing corpora (mtubercolosis/ and tb_corpus_v3/).

Covers papers from UK/European funders, WHO, and preprints (bioRxiv/medRxiv)
not indexed in US PubMed Central.

Usage:
    python fetch_europmc_batch.py --out-dir /uss/skavlak/tb_corpus_v3_extra/
"""

import argparse
import re
import time
from pathlib import Path

import requests

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Restricting the TB term to TITLE/ABSTRACT (rather than unrestricted full-text search) is
# critical for precision: EuropePMC's default full-text match means a paper that merely name-drops
# "Mycobacterium tuberculosis" once in a citation (e.g. a review of unrelated pathogens) counts as
# a "match" otherwise. A sampled audit found only 47% of full-text-matched papers even mentioned TB
# in title/abstract, and 27% mentioned it <=2 times in the whole document (tangential citations).
TB_FILTER = '(TITLE:"Mycobacterium tuberculosis" OR ABSTRACT:"Mycobacterium tuberculosis")'

# Exclude retracted publications so we don't train on since-withdrawn findings.
QUALITY_FILTER = 'NOT PUB_TYPE:"Retracted Publication"'

TOPICS = [
    ("drug resistance",         '"drug resistance"'),
    ("host-pathogen",           '"host-pathogen"'),
    ("virulence",               'virulence'),
    ("genomics",                'genomics'),
    ("transcriptomics",         'transcriptomics'),
    ("metabolomics",            'metabolomics'),
    ("latency dormancy",        'dormancy'),
    ("drug targets",            '"drug target"'),
    ("signal transduction",     '"signal transduction"'),
    ("epigenomics",             'epigenomics'),
    ("biofilm",                 'biofilm'),
    ("protein structure",       '"protein structure"'),
    ("clinical trials",         '"clinical trial"'),
    ("vaccine",                 'vaccine'),
    ("diagnosis",               'diagnosis'),
    ("proteomics",              'proteomics'),
    ("protein-protein interaction", '("protein-protein interaction" OR interactome)'),
    ("secretion systems",       '("ESX" OR "secretion system")'),
    ("cell wall biosynthesis",  '"cell wall"'),
    ("gene regulation",         '("gene regulation" OR regulon)'),
    ("metabolic pathway",       '"metabolic pathway"'),
    ("enzyme catalysis",        '(enzyme OR catalysis)'),
    ("crystal structure",       '"crystal structure"'),
]

QUERIES = [
    (name, f'{TB_FILTER} AND {topic}')
    for name, topic in TOPICS
]

PAGE_SIZE = 200
MAX_PER_QUERY = 9999
REQUEST_DELAY = 0.4


def collect_existing_ids(dirs: list[str]) -> set[str]:
    """Collect PMC IDs already on disk across all corpus directories."""
    ids = set()
    for d in dirs:
        path = Path(d)
        if not path.exists():
            continue
        for f in path.iterdir():
            stem = f.stem  # e.g. "PMC1234567" or "1234567"
            stem = stem.upper()
            if not stem.startswith("PMC"):
                stem = "PMC" + stem
            ids.add(stem)
    print(f"Existing IDs loaded: {len(ids)} across {[d for d in dirs]}")
    return ids


def iter_europmc(query: str, page_size: int = PAGE_SIZE, max_retries: int = 5):
    """Yield articles one page at a time so we can process without loading all into memory."""
    cursor = "*"
    total_seen = 0

    while True:
        params = {
            "query": query + f" AND OPEN_ACCESS:y NOT SRC:PPR AND {QUALITY_FILTER}",
            "format": "json",
            "pageSize": page_size,
            "cursorMark": cursor,
            "resultType": "idlist",  # lightweight: only IDs + pmcid, no full metadata
        }

        data = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(SEARCH_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                wait = 2 ** attempt  # 1, 2, 4, 8, 16s
                print(f"  [search error, retry {attempt+1}/{max_retries} in {wait}s] {e}")
                time.sleep(wait)

        if data is None:
            print(f"  [search failed after {max_retries} retries — skipping rest of this query]")
            break

        articles = data.get("resultList", {}).get("result", [])
        total_seen += len(articles)
        yield from articles

        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor or len(articles) < page_size:
            print(f"  Search complete: {total_seen} articles scanned")
            break
        cursor = next_cursor
        time.sleep(REQUEST_DELAY)


def fetch_fulltext(numeric_pmcid: str) -> tuple[bytes | None, str]:
    """Fetch full text XML from NCBI E-utilities using numeric PMC ID."""
    try:
        resp = requests.get(NCBI_EFETCH_URL, params={
            "db": "pmc",
            "id": numeric_pmcid,
            "rettype": "full",
            "retmode": "xml",
        }, timeout=15)
        if resp.status_code == 200 and b"<article" in resp.content:
            return resp.content, "ok"
        return None, f"http_{resp.status_code}"
    except requests.exceptions.Timeout:
        return None, "timeout"
    except Exception as e:
        return None, str(e)[:60]


def safe_filename(title: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]", "", title.lower())
    return slug[:max_len]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/uss/skavlak/tb_corpus_v3_extra/")
    parser.add_argument("--existing-dirs", nargs="+", default=[
        "/home/skavlak/finetuning/mtubercolosis",
        "/uss/skavlak/tb_corpus_v3",
    ])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect IDs already on disk so we don't re-download
    existing_ids = collect_existing_ids(args.existing_dirs + [str(out_dir)])

    session_downloaded = 0
    session_skipped = 0
    session_failed = 0

    for query_name, query in QUERIES:
        print(f"\nQuery: {query_name}")

        new_this_query = 0
        for article in iter_europmc(query):
            if session_downloaded >= MAX_PER_QUERY * len(QUERIES):
                break

            # Only include peer-reviewed papers with a PMCID — skip preprints
            source = article.get("source", "")
            pmcid = article.get("pmcid", "")

            if not pmcid or source == "PPR":
                session_skipped += 1
                continue

            dedup_key = pmcid.upper()
            if not dedup_key.startswith("PMC"):
                dedup_key = "PMC" + dedup_key
            # NCBI E-utilities needs numeric ID only (strip "PMC" prefix)
            fetch_id = dedup_key.replace("PMC", "", 1)

            if dedup_key in existing_ids:
                session_skipped += 1
                continue

            # Build output filename
            title = article.get("title", dedup_key)
            fname = out_dir / f"{safe_filename(title)}.xml"
            if fname.exists():
                existing_ids.add(dedup_key)
                session_skipped += 1
                continue

            xml, reason = fetch_fulltext(fetch_id)
            if xml is None:
                session_failed += 1
                if session_failed <= 5 or session_failed % 50 == 0:
                    print(f"  [fail #{session_failed}] {fetch_id}: {reason}")
                continue

            fname.write_bytes(xml)
            existing_ids.add(dedup_key)
            session_downloaded += 1
            new_this_query += 1

            if new_this_query % 25 == 0 or new_this_query == 1:
                print(f"  [{new_this_query} new this query | {session_downloaded} total]")

            time.sleep(REQUEST_DELAY)

        print(f"  New papers this query: {new_this_query}")

    print(f"\nSession total")
    print(f"  Downloaded : {session_downloaded}")
    print(f"  Skipped    : {session_skipped}")
    print(f"  Failed     : {session_failed}")
    print(f"  XMLs on disk now: {sum(1 for _ in out_dir.glob('*.xml'))}")


if __name__ == "__main__":
    main()
