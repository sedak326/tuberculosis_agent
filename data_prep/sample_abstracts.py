"""
Pick a random sample of papers and write clean abstracts to abstracts_sample.md.
"""

import random
import re
import subprocess
from pathlib import Path

METADATA = Path("papers_metadata.tsv")
PAPERS_DIR = Path("mtubercolosis")
OUT = Path("abstracts_sample.md")
N = 15
SEED = 42

TAG_RE = re.compile(r"<[^>]+>")


def clean(text):
    text = TAG_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def abstract_from_xml(path):
    content = path.read_text(errors="ignore")
    m = re.search(r"<dc:description>(.*?)</dc:description>", content, re.DOTALL)
    if m:
        return clean(m.group(1))
    # fallback: <abstract> tag (PMC format)
    m = re.search(r"<abstract[^>]*>(.*?)</abstract>", content, re.DOTALL)
    if m:
        return clean(m.group(1))
    return None


def abstract_from_pdf(path):
    try:
        text = subprocess.check_output(
            ["pdftotext", str(path), "-"], stderr=subprocess.DEVNULL
        ).decode("utf-8", errors="ignore")
    except Exception:
        return None
    # find Abstract or Summary heading, grab following paragraph
    m = re.search(
        r"(?:^|\n)(Abstract|Summary)\s*\n(.*?)(?=\n[A-Z][a-z]|\nIntroduction|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(2)).strip()
    return None


def get_abstract(pmid):
    # prefer XML over PDF
    for suffix in (".xml", ".pdf"):
        p = PAPERS_DIR / f"{pmid}{suffix}"
        if p.exists():
            fn = abstract_from_xml(p) if suffix == ".xml" else abstract_from_pdf(p)
            if fn and len(fn) > 80:
                return fn
    return None


def main():
    rows = []
    with open(METADATA) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))

    random.seed(SEED)
    sample = random.sample(rows, min(N * 3, len(rows)))  # oversample to account for misses

    results = []
    for pmid, title, authors in sample:
        if len(results) >= N:
            break
        abstract = get_abstract(pmid)
        if abstract:
            results.append((pmid, title, authors, abstract))
        else:
            print(f"  no abstract: {pmid}")

    with open(OUT, "w") as f:
        f.write(f"# Paper Abstracts — Random Sample ({len(results)} papers)\n\n")
        for i, (pmid, title, authors, abstract) in enumerate(results, 1):
            f.write(f"---\n\n")
            f.write(f"## {i}. {title}\n\n")
            f.write(f"**Authors:** {authors}  \n")
            f.write(f"**PMID:** {pmid}\n\n")
            f.write(f"{abstract}\n\n")

    print(f"Written {len(results)} abstracts to {OUT}")


if __name__ == "__main__":
    main()
