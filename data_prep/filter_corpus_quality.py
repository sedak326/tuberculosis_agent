#!/usr/bin/env python3
"""
filter_corpus_quality.py — Remove non-English and off-topic documents from an
extracted corpus, checking FULL document text rather than just the title
(title-only checks under-count legitimately on-topic papers that don't restate
"tuberculosis" and over-trust off-topic papers that happen to mention it once).

Groups chunks by document_name, decides per-document (not per-chunk) so a
document is either fully kept or fully dropped, then writes the filtered
corpus plus a report of what was excluded and why.

Usage:
    python filter_corpus_quality.py --corpus corpus_all.jsonl --out corpus_all_filtered.jsonl \
        --excluded-report excluded_postfilter.csv
"""
import argparse
import csv
import json
import re
from collections import defaultdict

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 42  # deterministic

HEADER_RE = re.compile(r"^\[Document:.*?\]\n\n", re.DOTALL)
TB_RE = re.compile(r"tubercul|mycobact", re.IGNORECASE)

MIN_TB_MENTIONS = 3  # matches the project's own earlier finding that <=2 mentions
                      # in a full document usually means a tangential/false-positive match


def strip_header(text: str) -> str:
    return HEADER_RE.sub("", text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--excluded-report", required=True)
    args = parser.parse_args()

    docs = defaultdict(list)
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("content_type") != "text":
                continue
            docs[obj["document_name"]].append(obj)

    print(f"Loaded {len(docs)} documents from {args.corpus}")

    kept_chunks = []
    excluded_rows = []
    n_nonenglish = 0
    n_offtopic = 0

    for doc_name, chunks in docs.items():
        full_text = " ".join(strip_header(c.get("text", "")) for c in chunks)

        tb_mentions = len(TB_RE.findall(full_text))

        lang = None
        try:
            # First 3000 chars is plenty for reliable language detection and
            # keeps this fast across thousands of documents.
            lang = detect(full_text[:3000]) if len(full_text.strip()) > 20 else "unknown"
        except LangDetectException:
            lang = "unknown"

        if lang != "en":
            n_nonenglish += 1
            excluded_rows.append({
                "document_name": doc_name, "reason": "NON_ENGLISH",
                "detected_lang": lang, "tb_mentions": tb_mentions, "n_chunks": len(chunks),
            })
            continue

        if tb_mentions < MIN_TB_MENTIONS:
            n_offtopic += 1
            excluded_rows.append({
                "document_name": doc_name, "reason": "OFF_TOPIC_LOW_TB_MENTIONS",
                "detected_lang": lang, "tb_mentions": tb_mentions, "n_chunks": len(chunks),
            })
            continue

        kept_chunks.extend(chunks)

    with open(args.out, "w", encoding="utf-8") as f:
        for c in kept_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(args.excluded_report, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["document_name", "reason", "detected_lang", "tb_mentions", "n_chunks"])
        writer.writeheader()
        writer.writerows(excluded_rows)

    kept_docs = len(docs) - n_nonenglish - n_offtopic
    print(f"\nKept:     {kept_docs} documents, {len(kept_chunks)} chunks -> {args.out}")
    print(f"Excluded: {n_nonenglish} non-English, {n_offtopic} off-topic (<{MIN_TB_MENTIONS} TB mentions)")
    print(f"Excluded report: {args.excluded_report}")


if __name__ == "__main__":
    main()
