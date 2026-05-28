"""
Fetch title + authors for all papers in mtubercolosis/ by querying PubMed.
Filenames are PMIDs. Outputs papers_metadata.tsv (pmid, title, authors, source).
"""

import os
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

PAPERS_DIR = Path("/home/skavlak/finetuning/mtubercolosis")
OUT_FILE = Path("/home/skavlak/finetuning/papers_metadata.tsv")
BATCH_SIZE = 200
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def get_pmids():
    seen = set()
    pmids = []
    for f in sorted(PAPERS_DIR.iterdir()):
        pmid = f.stem
        if pmid.isdigit() and pmid not in seen:
            seen.add(pmid)
            pmids.append(pmid)
    return pmids


def fetch_batch(pmids):
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
        "rettype": "docsum",
    })
    url = f"{ESUMMARY_URL}?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def parse_batch(data):
    results = {}
    for pmid, doc in data.get("result", {}).items():
        if pmid == "uids":
            continue
        title = doc.get("title", "").strip().rstrip(".")
        authors = "; ".join(a.get("name", "") for a in doc.get("authors", []))
        results[pmid] = (title, authors)
    return results


def main():
    pmids = get_pmids()
    print(f"Found {len(pmids)} unique PMIDs")

    already_done = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            next(f)  # header
            for line in f:
                parts = line.split("\t")
                if parts:
                    already_done.add(parts[0].strip())
        print(f"Resuming — {len(already_done)} already fetched")

    pending = [p for p in pmids if p not in already_done]
    print(f"Fetching {len(pending)} remaining")

    mode = "a" if already_done else "w"
    with open(OUT_FILE, mode) as out:
        if not already_done:
            out.write("pmid\ttitle\tauthors\n")

        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i:i + BATCH_SIZE]
            print(f"  batch {i // BATCH_SIZE + 1} / {(len(pending) + BATCH_SIZE - 1) // BATCH_SIZE}  ({batch[0]}..)")
            try:
                data = fetch_batch(batch)
                parsed = parse_batch(data)
                for pmid in batch:
                    title, authors = parsed.get(pmid, ("NOT FOUND", ""))
                    out.write(f"{pmid}\t{title}\t{authors}\n")
                out.flush()
            except Exception as e:
                print(f"  ERROR on batch starting {batch[0]}: {e}")
            time.sleep(0.34)  # NCBI rate limit: ~3 req/s without API key

    print(f"Done. Written to {OUT_FILE}")


if __name__ == "__main__":
    main()
