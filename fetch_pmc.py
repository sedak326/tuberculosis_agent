#!/usr/bin/env python3
"""
fetch_pmc.py — Download full-text XMLs from PubMed Central via E-utilities.

Deduplicates against papers already in out-dir by comparing normalised titles.

Usage:
    python fetch_pmc.py --query '"Mycobacterium tuberculosis"[MeSH] AND proteomics[MeSH]'
    python fetch_pmc.py --query '...' --out-dir mtubercolosis/ --max 500 --api-key KEY

NCBI rate limit: 3 req/sec without API key, 10/sec with one.
Get a free API key at: https://www.ncbi.nlm.nih.gov/account/
"""

import argparse
import re
import time
from pathlib import Path

import requests

ESEARCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Title patterns for existing files
TITLE_PATTERNS = [
    re.compile(rb"<dc:title[^>]*>(.*?)</dc:title>", re.DOTALL),       # Elsevier
    re.compile(rb"<article-title[^>]*>(.*?)</article-title>", re.DOTALL),  # PMC
]
TAG_RE = re.compile(rb"<[^>]+>")


def normalise(title: str | bytes) -> str:
    """Lowercase, strip tags and non-alphanumeric chars for fuzzy comparison."""
    if isinstance(title, bytes):
        title = TAG_RE.sub(b"", title).decode("utf-8", errors="ignore")
    return re.sub(r"[^a-z0-9]", "", title.lower())


def existing_titles(out_dir: Path) -> set[str]:
    titles: set[str] = set()
    for f in out_dir.iterdir():
        if f.suffix not in (".xml",):
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
    print(f"  {len(titles)} existing titles found in {out_dir}.")
    return titles


def search_pmc(query: str, max_results: int, api_key: str | None) -> list[str]:
    params = {"db": "pmc", "term": query, "retmax": max_results, "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    r = requests.get(ESEARCH_URL, params=params, timeout=30)
    r.raise_for_status()
    ids = r.json()["esearchresult"]["idlist"]
    print(f"Found {len(ids)} articles in search.")
    return ids


def fetch_titles(pmc_ids: list[str], api_key: str | None, delay: float) -> dict[str, str]:
    """Return {pmc_id: normalised_title} for all IDs via esummary (batched)."""
    result: dict[str, str] = {}
    batch_size = 200
    for i in range(0, len(pmc_ids), batch_size):
        batch = pmc_ids[i:i + batch_size]
        params = {"db": "pmc", "id": ",".join(batch), "retmode": "json"}
        if api_key:
            params["api_key"] = api_key
        r = requests.get(ESUMMARY_URL, params=params, timeout=30)
        r.raise_for_status()
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
    r = requests.get(EFETCH_URL, params=params, timeout=60)
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}, skipping.")
        return None
    if b"<article" not in r.content and b"<pmc-articleset" not in r.content:
        print(f"  No full-text XML available, skipping.")
        return None
    return r.content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",   required=True)
    parser.add_argument("--out-dir", default="mtubercolosis")
    parser.add_argument("--max",     type=int, default=500)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    delay = 0.11 if args.api_key else 0.34

    pmc_ids = search_pmc(args.query, args.max, args.api_key)

    print("Fetching titles for search results...")
    search_titles = fetch_titles(pmc_ids, args.api_key, delay)

    print("Scanning existing files for titles...")
    known_titles = existing_titles(out_dir)

    skipped_dup = skipped_done = downloaded = failed = 0

    for i, pmc_id in enumerate(pmc_ids, 1):
        out_path = out_dir / f"PMC{pmc_id}.xml"

        if out_path.exists():
            skipped_done += 1
            continue

        norm_title = search_titles.get(pmc_id, "")
        if norm_title and norm_title in known_titles:
            print(f"[{i}/{len(pmc_ids)}] Already have: {search_titles[pmc_id][:80]!r}")
            skipped_dup += 1
            continue

        label = search_titles.get(pmc_id, f"PMC{pmc_id}")[:60]
        print(f"[{i}/{len(pmc_ids)}] {label!r}...", end=" ", flush=True)
        xml_bytes = fetch_xml(pmc_id, args.api_key)

        if xml_bytes:
            out_path.write_bytes(xml_bytes)
            if norm_title:
                known_titles.add(norm_title)
            print("done")
            downloaded += 1
        else:
            failed += 1

        time.sleep(delay)

    print(f"\nDone.")
    print(f"  Downloaded      : {downloaded}")
    print(f"  Skipped (dup)   : {skipped_dup}")
    print(f"  Skipped (exist) : {skipped_done}")
    print(f"  Failed (no XML) : {failed}")


if __name__ == "__main__":
    main()
