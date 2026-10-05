#!/usr/bin/env python3
"""
fetch_postcutoff_papers.py — Bulk-download TB / M. tuberculosis research papers
whose earliest known public availability is on or after 2024-01-01, i.e. after
Llama 3.1/3.3's documented pretraining cutoff (December 2023). Standalone task:
discovers, verifies, and downloads papers only — no corpus/dataset/fine-tuning
processing here.

INFRASTRUCTURE NOTE (checked live, 2026-09): NCBI's PMC Open Access Web Service
(oa.fcgi), the standard bulk-PDF-by-PMCID API, was retired in August 2026 and
now 404s unconditionally. There is a new AWS-based PMC Article Datasets system,
but its access requirements (AWS account? cost?) weren't verified, so it's not
used here. Two things were confirmed still working:
  - Unpaywall API -> direct publisher PDF URLs for ~40% of candidates tested
    (Nature, Frontiers succeed; MDPI, Wiley 403 scripted requests even for
    their own open-access content — not a paywall, just bot-blocking, and we
    do not attempt to spoof/evade that, per the no-scraping instruction).
  - JATS full-text XML remains 100% reliable via two independent channels
    (EuropePMC's own REST API, and NCBI's core E-utilities efetch), for any
    paper that's inEPMC/open access.
So the PDF strategy is a documented hybrid: try the real publisher PDF via
Unpaywall first (source_used_for_pdf="publisher_direct"); if that's blocked
or unavailable, fetch the JATS XML (guaranteed available) and render it
locally into a simple, readable PDF (source_used_for_pdf="rendered_from_jats_xml").
Every record's metadata says exactly which happened — nothing is silently
downgraded.

Usage:
    # Test run — stop after 20 accepted+downloaded papers
    python fetch_postcutoff_papers.py --out-dir /uss/skavlak/version_3 --target-accepted 20

    # Full run — unlimited, exhausts all queries
    python fetch_postcutoff_papers.py --out-dir /uss/skavlak/version_3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html as html_module
import io
import json
import logging
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUTOFF = date(2023, 12, 31)  # earliest acceptable public date is 2024-01-01

EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_FULLTEXT_URL_TMPL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{id}/fullTextXML"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CROSSREF_URL_TMPL = "https://api.crossref.org/works/{doi}"
UNPAYWALL_URL_TMPL = "https://api.unpaywall.org/v2/{doi}"
S2_URL_TMPL = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
OPENALEX_URL = "https://api.openalex.org/works"

# OpenAlex indexes a broader swath of publishers/venues than EuropePMC (which
# leans PMC-centric); empirically, ~60% of its post-cutoff TB candidates were
# not found by any of the 29 EuropePMC topic queries. Its title_and_abstract
# filter gives the same precision restriction that mattered for EuropePMC
# (avoiding papers that just tangentially mention TB), so one broad query
# covers the whole date range without needing per-topic splitting the way
# EuropePMC did.
OPENALEX_TB_FILTER = 'title_and_abstract.search:Mycobacterium tuberculosis|M. tuberculosis'

CONTACT_EMAIL = "sedak98@googlemail.com"
USER_AGENT = f"tb-postcutoff-corpus-research (mailto:{CONTACT_EMAIL})"

TB_FILTER = '(TITLE:"Mycobacterium tuberculosis" OR ABSTRACT:"Mycobacterium tuberculosis" OR TITLE:"M. tuberculosis" OR ABSTRACT:"M. tuberculosis")'
QUALITY_FILTER = 'NOT PUB_TYPE:"Retracted Publication"'

# Prioritized bacterium/molecular-biology topics first, disease-level topics after.
TOPICS = [
    # --- bacterial biology / molecular biology (priority) ---
    ("genetics genomics",        '(genetics OR genomics OR "whole genome")'),
    ("mutations gene function",  '(mutation OR "gene function" OR "essential gene")'),
    ("genome evolution phylo",   '("genome evolution" OR phylogenetics OR phylogenomics)'),
    ("transcriptomics",          'transcriptomics'),
    ("proteomics",               'proteomics'),
    ("metabolomics metabolism",  '(metabolomics OR metabolism OR "metabolic pathway")'),
    ("bacterial physiology",     '("bacterial physiology" OR physiology)'),
    ("cell wall envelope",       '("cell wall" OR "cell envelope" OR "mycolic acid")'),
    ("virulence",                'virulence'),
    ("secretion systems ESX",    '("secretion system" OR "ESX" OR "type VII secretion")'),
    ("PE PPE proteins",          '("PE protein" OR "PPE protein" OR "PE/PPE")'),
    ("dormancy persistence",     '(dormancy OR persistence OR persister)'),
    ("stress response",          '("stress response" OR "oxidative stress")'),
    ("antibiotic resistance",    '("antibiotic resistance" OR "antimicrobial resistance" OR "drug resistance")'),
    ("resistance mutations",     '("resistance mutation" OR "resistance-conferring")'),
    ("drug mechanisms targets",  '("drug mechanism" OR "drug target" OR "mechanism of action")'),
    ("drug susceptibility",      '("drug susceptibility" OR susceptibility)'),
    ("rifampicin isoniazid",     '(rifampicin OR rifampin OR isoniazid)'),
    ("bedaquiline linezolid",    '(bedaquiline OR linezolid)'),
    ("fluoroquinolone MDR XDR",  '(fluoroquinolone OR "MDR-TB" OR "XDR-TB" OR "multidrug-resistant" OR "extensively drug-resistant")'),
    ("host pathogen macrophage", '("host-pathogen" OR macrophage OR "immune evasion")'),
    ("structural biology",       '("crystal structure" OR "protein structure" OR "structural biology")'),
    ("enzyme function",          '(enzyme OR catalysis OR "enzyme function")'),
    ("experimental evolution",   '("experimental evolution")'),
    ("molecular microbiology",   '("molecular microbiology" OR "molecular biology")'),
    # --- disease-level (acceptable, lower priority) ---
    ("treatment",                '(treatment OR therapy OR regimen)'),
    ("diagnostics",               'diagnos*'),
    ("vaccine immunology",       '(vaccine OR immunology OR immunogenicity)'),
    ("epidemiology transmission", '(epidemiology OR transmission)'),
    ("clinical research",        '("clinical trial" OR "clinical study")'),
]

QUERIES = [(name, f"{TB_FILTER} AND {topic}") for name, topic in TOPICS]

PAGE_SIZE = 100
REQUEST_DELAY = 0.4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[],
)
log = logging.getLogger("fetch_postcutoff")


def setup_file_logging(log_path: Path):
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(sh)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get(url, params=None, timeout=25, max_retries=4, headers=None, treat_403_as_final=False):
    """GET with exponential backoff. Returns Response or None. A 403 is treated
    as a definitive block (no retry, no header-spoofing escalation) when
    treat_403_as_final=True — used for direct publisher PDF fetches, where
    retrying/evading would cross into scraping behavior we're avoiding."""
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=hdrs)
            if resp.status_code == 403 and treat_403_as_final:
                return resp
            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, 30)
                time.sleep(wait)
                continue
            return resp
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            log.warning(f"  [http retry {attempt+1}/{max_retries} in {wait}s] {url}: {e}")
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# SQLite checkpoint store
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    key TEXT PRIMARY KEY,          -- dedup key: DOI, else PMID, else PMCID, else norm-title
    doi TEXT,
    pmid TEXT,
    pmcid TEXT,
    norm_title TEXT,
    status TEXT NOT NULL,          -- accepted_downloaded | rejected | failed
    reason TEXT,
    record_json TEXT NOT NULL,
    sha256 TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
CREATE INDEX IF NOT EXISTS idx_papers_pmid ON papers(pmid);
CREATE INDEX IF NOT EXISTS idx_papers_pmcid ON papers(pmcid);
CREATE INDEX IF NOT EXISTS idx_papers_normtitle ON papers(norm_title);
CREATE INDEX IF NOT EXISTS idx_papers_sha256 ON papers(sha256);
"""


class Store:
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def seen_key(self, key: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM papers WHERE key = ?", (key,)).fetchone()
        return row is not None

    def seen_any(self, doi=None, pmid=None, pmcid=None, norm_title=None) -> bool:
        clauses, params = [], []
        if doi:
            clauses.append("doi = ?"); params.append(doi)
        if pmid:
            clauses.append("pmid = ?"); params.append(pmid)
        if pmcid:
            clauses.append("pmcid = ?"); params.append(pmcid)
        if norm_title:
            clauses.append("norm_title = ?"); params.append(norm_title)
        if not clauses:
            return False
        q = f"SELECT 1 FROM papers WHERE {' OR '.join(clauses)} LIMIT 1"
        return self.conn.execute(q, params).fetchone() is not None

    def hash_exists(self, sha256: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM papers WHERE sha256 = ?", (sha256,)).fetchone()
        return row is not None

    def upsert(self, key, doi, pmid, pmcid, norm_title, status, reason, record: dict, sha256=None):
        self.conn.execute(
            """INSERT INTO papers (key, doi, pmid, pmcid, norm_title, status, reason, record_json, sha256, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(key) DO UPDATE SET
                 status=excluded.status, reason=excluded.reason, record_json=excluded.record_json,
                 sha256=excluded.sha256, updated_at=excluded.updated_at""",
            (key, doi, pmid, pmcid, norm_title, status, reason, json.dumps(record, ensure_ascii=False), sha256),
        )
        self.conn.commit()

    def count_accepted(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM papers WHERE status = 'accepted_downloaded'").fetchone()
        return row[0]

    def all_rows(self, status=None):
        if status:
            return self.conn.execute("SELECT record_json FROM papers WHERE status = ?", (status,)).fetchall()
        return self.conn.execute("SELECT record_json, status FROM papers").fetchall()


# ---------------------------------------------------------------------------
# EuropePMC search
# ---------------------------------------------------------------------------

def openalex_to_article(work: dict) -> dict:
    """Normalize an OpenAlex work into the same shape verify_cutoff() and the
    rest of the pipeline expect from an EuropePMC article dict."""
    ids = work.get("ids") or {}
    doi = (ids.get("doi") or "").replace("https://doi.org/", "").lower() or None
    pmid_url = ids.get("pmid") or ""
    pmid = pmid_url.rstrip("/").split("/")[-1] if pmid_url else None
    pmcid_url = ids.get("pmcid") or ""
    pmcid = None
    if pmcid_url:
        # e.g. "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567" -> PMC1234567
        tail = pmcid_url.rstrip("/").split("/")[-1].upper()
        pmcid = tail if tail.startswith("PMC") else f"PMC{tail}"

    authors = work.get("authorships") or []
    author_string = ", ".join(a.get("author", {}).get("display_name", "") for a in authors if a.get("author"))
    journal = ((work.get("primary_location") or {}).get("source") or {}).get("display_name", "")

    return {
        "doi": doi,
        "pmid": pmid,
        "pmcid": pmcid,
        "title": work.get("title") or "",
        "authorString": author_string,
        "journalInfo": {"journal": {"title": journal}},
        "pubYear": work.get("publication_year"),
        "firstPublicationDate": work.get("publication_date"),
        "license": ((work.get("primary_location") or {}).get("license")),
        "_discovery_source": "openalex",
    }


def iter_openalex(page_size: int = 200):
    cursor = "*"
    while True:
        params = {
            "filter": f"{OPENALEX_TB_FILTER},from_publication_date:2024-01-01,to_publication_date:2026-12-31,"
                      f"type:article,open_access.is_oa:true",
            "per-page": page_size,
            "cursor": cursor,
            "mailto": CONTACT_EMAIL,
        }
        resp = http_get(OPENALEX_URL, params=params, timeout=30)
        if resp is None or resp.status_code != 200:
            log.warning(f"  [openalex search failed] status={resp.status_code if resp else None}")
            break
        data = resp.json()
        results = data.get("results", [])
        for w in results:
            yield openalex_to_article(w)
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor or not results:
            break
        time.sleep(0.15)


def iter_europmc(query: str, page_size: int = PAGE_SIZE):
    cursor = "*"
    while True:
        params = {
            "query": f"{query} AND PUB_YEAR:[2024 TO 2026] AND OPEN_ACCESS:y AND LANG:eng NOT SRC:PPR AND {QUALITY_FILTER}",
            "format": "json",
            "pageSize": page_size,
            "cursorMark": cursor,
            "resultType": "core",
        }
        resp = http_get(EUROPEPMC_SEARCH_URL, params=params, timeout=30)
        if resp is None or resp.status_code != 200:
            log.warning(f"  [search failed, stopping this query] status={resp.status_code if resp else None}")
            break
        data = resp.json()
        articles = data.get("resultList", {}).get("result", [])
        yield from articles
        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor or len(articles) < page_size:
            break
        cursor = next_cursor
        time.sleep(REQUEST_DELAY)


# ---------------------------------------------------------------------------
# Cutoff verification
# ---------------------------------------------------------------------------

def parse_date_parts(date_parts) -> Optional[date]:
    if not date_parts or not date_parts.get("date-parts"):
        return None
    parts = date_parts["date-parts"][0]
    if not parts:
        return None
    y = parts[0]
    m = parts[1] if len(parts) > 1 else 1
    d = parts[2] if len(parts) > 2 else 1
    try:
        return date(y, m, d)
    except ValueError:
        return date(y, 1, 1)


def crossref_lookup(doi: str) -> Optional[dict]:
    resp = http_get(CROSSREF_URL_TMPL.format(doi=doi), params={"mailto": CONTACT_EMAIL}, timeout=20)
    if resp is None or resp.status_code != 200:
        return None
    try:
        return resp.json().get("message")
    except Exception:
        return None


def unpaywall_lookup(doi: str) -> Optional[dict]:
    resp = http_get(UNPAYWALL_URL_TMPL.format(doi=doi), params={"email": CONTACT_EMAIL}, timeout=20)
    if resp is None or resp.status_code != 200:
        return None
    try:
        return resp.json()
    except Exception:
        return None


def s2_lookup(doi: str) -> Optional[dict]:
    resp = http_get(S2_URL_TMPL.format(doi=doi),
                     params={"fields": "publicationDate,year,externalIds"}, timeout=20)
    if resp is None or resp.status_code != 200:
        return None
    try:
        return resp.json()
    except Exception:
        return None


@dataclass
class CutoffResult:
    decision: str  # INCLUDE | EXCLUDE | AMBIGUOUS
    reason: str
    earliest_public_date: Optional[str]
    evidence: list = field(default_factory=list)
    preprint_doi: Optional[str] = None
    preprint_date: Optional[str] = None
    checked_preprint: bool = False


def verify_cutoff(article: dict) -> CutoffResult:
    doi = article.get("doi")
    evidence = []
    dates = []  # list of (date, source_label)

    # 1. EuropePMC's own firstPublicationDate / pubYear
    fpd = article.get("firstPublicationDate")
    if fpd:
        try:
            y, m, d = [int(x) for x in fpd.split("-")]
            dates.append((date(y, m, d), "europepmc.firstPublicationDate"))
            evidence.append(f"EuropePMC firstPublicationDate={fpd}")
        except Exception:
            pass
    elif article.get("pubYear"):
        try:
            dates.append((date(int(article["pubYear"]), 1, 1), "europepmc.pubYear(day-imprecise)"))
            evidence.append(f"EuropePMC pubYear={article['pubYear']} (no exact date)")
        except Exception:
            pass

    crossref_checked = False
    preprint_doi = None
    preprint_date_val = None
    if doi:
        cr = crossref_lookup(doi)
        crossref_checked = cr is not None
        if cr:
            created = parse_date_parts(cr.get("created"))
            published = parse_date_parts(cr.get("published")) or parse_date_parts(cr.get("published-online")) \
                or parse_date_parts(cr.get("published-print"))
            if created:
                dates.append((created, "crossref.created"))
                evidence.append(f"Crossref created={created}")
            if published:
                dates.append((published, "crossref.published"))
                evidence.append(f"Crossref published={published}")

            relation = cr.get("relation") or {}
            preprint_links = relation.get("has-preprint") or []
            for link in preprint_links:
                pdoi = link.get("id")
                if not pdoi:
                    continue
                preprint_doi = pdoi
                pcr = crossref_lookup(pdoi)
                time.sleep(0.2)
                if pcr:
                    p_created = parse_date_parts(pcr.get("created"))
                    p_published = parse_date_parts(pcr.get("published"))
                    ppd = p_created or p_published
                    if ppd:
                        preprint_date_val = ppd
                        dates.append((ppd, f"crossref.preprint({pdoi})"))
                        evidence.append(f"Linked preprint {pdoi} date={ppd}")
        time.sleep(0.2)

        s2 = s2_lookup(doi)
        if s2 and s2.get("publicationDate"):
            try:
                y, m, d = [int(x) for x in s2["publicationDate"].split("-")]
                dates.append((date(y, m, d), "semanticscholar.publicationDate"))
                evidence.append(f"Semantic Scholar publicationDate={s2['publicationDate']}")
            except Exception:
                pass
        time.sleep(0.2)

    if not dates:
        return CutoffResult(
            decision="AMBIGUOUS", reason="AMBIGUOUS_DATE",
            earliest_public_date=None, evidence=["No date evidence from any source"],
            checked_preprint=crossref_checked,
        )

    earliest_date, earliest_source = min(dates, key=lambda t: t[0])

    if earliest_date < date(2024, 1, 1):
        reason = "PRE_CUTOFF_PREPRINT" if "preprint" in earliest_source else "PRE_CUTOFF_EPUBLICATION"
        return CutoffResult(
            decision="EXCLUDE", reason=reason,
            earliest_public_date=str(earliest_date), evidence=evidence,
            preprint_doi=preprint_doi,
            preprint_date=str(preprint_date_val) if preprint_date_val else None,
            checked_preprint=crossref_checked,
        )

    return CutoffResult(
        decision="INCLUDE", reason="POST_CUTOFF_CONFIRMED",
        earliest_public_date=str(earliest_date), evidence=evidence,
        preprint_doi=preprint_doi,
        preprint_date=str(preprint_date_val) if preprint_date_val else None,
        checked_preprint=crossref_checked,
    )


# ---------------------------------------------------------------------------
# Full text retrieval
# ---------------------------------------------------------------------------

def try_publisher_pdf(doi: str) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
    """Returns (pdf_bytes, url_used, license) or (None, None, None)."""
    up = unpaywall_lookup(doi)
    if not up:
        return None, None, None
    loc = up.get("best_oa_location") or {}
    pdf_url = loc.get("url_for_pdf")
    if not pdf_url:
        return None, None, None
    resp = http_get(pdf_url, timeout=30, max_retries=2, treat_403_as_final=True)
    if resp is None or resp.status_code != 200:
        return None, None, None
    content = resp.content
    if not content.startswith(b"%PDF"):
        return None, None, None
    return content, pdf_url, loc.get("license")


def recover_pmcid_via_doi(doi: str) -> Optional[str]:
    """OpenAlex's own pmcid field is unreliable — empirically, ~88% of OpenAlex
    candidates missing a pmcid actually have one in EuropePMC. Without this,
    those candidates would lose the JATS-XML full-text fallback entirely and
    depend solely on the ~40%-success Unpaywall path."""
    resp = http_get(EUROPEPMC_SEARCH_URL, params={
        "query": f'DOI:"{doi}"', "format": "json", "pageSize": 1, "resultType": "idlist",
    }, timeout=15)
    if resp is None or resp.status_code != 200:
        return None
    results = resp.json().get("resultList", {}).get("result", [])
    if results and results[0].get("pmcid"):
        return results[0]["pmcid"].upper()
    return None


def fetch_jats_xml(pmcid: str) -> Optional[bytes]:
    resp = http_get(EUROPEPMC_FULLTEXT_URL_TMPL.format(id=pmcid), timeout=30)
    if resp is not None and resp.status_code == 200 and b"<article" in resp.content:
        return resp.content
    numeric = pmcid.upper().replace("PMC", "")
    resp2 = http_get(NCBI_EFETCH_URL, params={"db": "pmc", "id": numeric, "rettype": "full", "retmode": "xml"}, timeout=30)
    if resp2 is not None and resp2.status_code == 200 and b"<article" in resp2.content:
        return resp2.content
    return None


def jats_text_sections(xml_bytes: bytes) -> tuple[str, list[tuple[str, str]]]:
    """Very small JATS parser: returns (abstract_text, [(section_title, paragraph_text), ...])."""
    root = ET.fromstring(xml_bytes)

    def strip_ns(tag):
        return tag.split("}")[-1]

    def text_of(el) -> str:
        return "".join(el.itertext()).strip() if el is not None else ""

    abstract_text = ""
    for el in root.iter():
        if strip_ns(el.tag) == "abstract":
            abstract_text = text_of(el)
            # JATS abstracts often start with a redundant "ABSTRACT" label run
            # together with no separating whitespace (e.g. "ABSTRACTMethionine...");
            # strip it since we already add our own "Abstract" heading above it.
            abstract_text = re.sub(r"^\s*ABSTRACT\s*", "", abstract_text, count=1, flags=re.IGNORECASE)
            break

    sections = []
    body = None
    for el in root.iter():
        if strip_ns(el.tag) == "body":
            body = el
            break
    if body is not None:
        for sec in body.iter():
            if strip_ns(sec.tag) != "sec":
                continue
            title_el = None
            for child in sec:
                if strip_ns(child.tag) == "title":
                    title_el = child
                    break
            title = text_of(title_el) if title_el is not None else ""
            paras = []
            for p in sec:
                if strip_ns(p.tag) == "p":
                    paras.append(text_of(p))
            if paras:
                sections.append((title, "\n\n".join(paras)))
    return abstract_text, sections


def render_pdf_from_jats(xml_bytes: bytes, title: str, authors: str, journal: str, doi: str) -> Optional[bytes]:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from xml.sax.saxutils import escape

    try:
        abstract, sections = jats_text_sections(xml_bytes)
    except ET.ParseError:
        return None

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TBTitle", parent=styles["Title"], fontSize=15)
    meta_style = ParagraphStyle("TBMeta", parent=styles["Normal"], fontSize=9, textColor="#555555")
    heading_style = ParagraphStyle("TBHeading", parent=styles["Heading2"], spaceBefore=12)
    body_style = ParagraphStyle("TBBody", parent=styles["BodyText"], spaceAfter=8, leading=14)
    note_style = ParagraphStyle("TBNote", parent=styles["Normal"], fontSize=8, textColor="#888888", spaceBefore=20)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    flow = [
        Paragraph(escape(title), title_style),
        Spacer(1, 6),
        Paragraph(escape(f"{authors} — {journal} — DOI: {doi}"), meta_style),
        Spacer(1, 14),
    ]
    if abstract:
        flow.append(Paragraph("Abstract", heading_style))
        flow.append(Paragraph(escape(abstract), body_style))
    for sec_title, sec_text in sections:
        if sec_title:
            flow.append(Paragraph(escape(sec_title), heading_style))
        for para in sec_text.split("\n\n"):
            if para.strip():
                flow.append(Paragraph(escape(para), body_style))
    flow.append(Paragraph(
        "Rendered locally from JATS full-text XML (source_used_for_pdf=rendered_from_jats_xml) — "
        "not the publisher's original PDF layout. See papers.jsonl for provenance.",
        note_style,
    ))
    try:
        doc.build(flow)
    except Exception as e:
        log.warning(f"  [pdf render failed] {e}")
        return None
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def clean_markup(s: str) -> str:
    """EuropePMC titles/journal names often carry inline italics markup, either
    literal (<i>Mycobacterium</i>) or HTML-entity-escaped (&lt;i&gt;Mycobacterium&lt;/i&gt;).
    Strip both so it never leaks into filenames, PDF headers, or CSV/JSONL fields."""
    if not s:
        return s
    s = re.sub(r"<[^>]+>", "", s)
    s = html_module.unescape(s)
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip()


def normalize_title(title: str) -> str:
    t = re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()
    return re.sub(r"\s+", " ", t)


def sanitize_filename_part(s: str, max_len: int) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s or "").strip("_")
    return s[:max_len] or "x"


def build_filename(year: str, first_author: str, title: str, identifier: str) -> str:
    author_part = sanitize_filename_part(first_author, 25)
    title_part = sanitize_filename_part(title, 50)
    id_part = sanitize_filename_part(identifier, 20)
    return f"{year}_{author_part}_{title_part}_{id_part}.pdf"


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/uss/skavlak/version_3")
    parser.add_argument("--target-accepted", type=int, default=0, help="Stop after N accepted+downloaded (0 = unlimited)")
    parser.add_argument("--max-candidates-per-query", type=int, default=0, help="Safety cap (0 = unlimited)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    papers_dir = out_dir / "papers"
    out_dir.mkdir(parents=True, exist_ok=True)
    papers_dir.mkdir(parents=True, exist_ok=True)

    setup_file_logging(out_dir / "download.log")
    store = Store(out_dir / "checkpoint.db")

    papers_csv_path = out_dir / "papers.csv"
    papers_jsonl_path = out_dir / "papers.jsonl"
    excluded_csv_path = out_dir / "excluded.csv"

    csv_fields = [
        "local_filename", "title", "authors", "pub_year", "journal", "doi", "pmid", "pmcid",
        "pub_date", "earliest_public_date", "earliest_public_date_evidence", "checked_for_preprint",
        "preprint_doi", "preprint_date", "source_used_for_pdf", "pdf_url_or_source", "discovery_source", "license",
        "download_timestamp", "sha256", "file_size_bytes",
    ]
    excluded_fields = [
        "title", "doi", "pmid", "journal_pub_date", "earliest_public_date_found",
        "preprint_doi", "preprint_date", "exclusion_reason", "evidence",
    ]

    write_papers_header = not papers_csv_path.exists()
    write_excl_header = not excluded_csv_path.exists()
    papers_csv_f = open(papers_csv_path, "a", newline="", encoding="utf-8")
    excluded_csv_f = open(excluded_csv_path, "a", newline="", encoding="utf-8")
    papers_writer = csv.DictWriter(papers_csv_f, fieldnames=csv_fields)
    excluded_writer = csv.DictWriter(excluded_csv_f, fieldnames=excluded_fields)
    if write_papers_header:
        papers_writer.writeheader()
    if write_excl_header:
        excluded_writer.writeheader()
    papers_jsonl_f = open(papers_jsonl_path, "a", encoding="utf-8")

    stats = {
        "candidates_discovered": 0, "accepted": 0, "rejected_pre_cutoff": 0,
        "rejected_ambiguous": 0, "rejected_no_fulltext": 0, "failed_download": 0,
        "skipped_duplicate": 0,
    }
    example_decisions = []

    already_accepted = store.count_accepted()
    log.info(f"Resuming with {already_accepted} already accepted+downloaded in {out_dir}")

    target = args.target_accepted
    stop = False

    # EuropePMC (29 topic-restricted queries) + OpenAlex (one broad query — its
    # title_and_abstract filter already covers the whole date range precisely,
    # no per-topic splitting needed). Empirically ~60% of OpenAlex's candidates
    # aren't found by EuropePMC at all, so both are real, non-redundant sources.
    sources = [(name, "europmc", q) for name, q in QUERIES] + [("openalex_broad", "openalex", None)]

    for query_name, source_label, query in sources:
        if stop:
            break
        log.info(f"=== Query: {query_name} ({source_label}) ===")
        n_this_query = 0
        article_iter = iter_europmc(query) if source_label == "europmc" else iter_openalex()
        for article in article_iter:
            if target and store.count_accepted() >= target:
                log.info(f"Target of {target} accepted papers reached. Stopping.")
                stop = True
                break
            if args.max_candidates_per_query and n_this_query >= args.max_candidates_per_query:
                break
            n_this_query += 1
            stats["candidates_discovered"] += 1
            discovery_source = article.get("_discovery_source", "europmc")

            doi = (article.get("doi") or "").lower() or None
            pmid = article.get("pmid") or None
            pmcid = article.get("pmcid") or None
            title = clean_markup(article.get("title", "").rstrip("."))
            norm_title = normalize_title(title)
            key = doi or pmid or pmcid or norm_title
            if not key:
                continue

            if store.seen_key(key) or store.seen_any(doi=doi, pmid=pmid, pmcid=pmcid, norm_title=norm_title):
                stats["skipped_duplicate"] += 1
                continue

            author_string = clean_markup(article.get("authorString", ""))
            first_author = author_string.split(",")[0].strip() if author_string else "unknown"
            journal = clean_markup((article.get("journalInfo") or {}).get("journal", {}).get("title", ""))
            pub_year = article.get("pubYear", "")
            pub_date = article.get("firstPublicationDate", "")

            cutoff = verify_cutoff(article)

            if len(example_decisions) < 8:
                example_decisions.append({
                    "title": title, "doi": doi, "decision": cutoff.decision,
                    "reason": cutoff.reason, "earliest_public_date": cutoff.earliest_public_date,
                    "evidence": cutoff.evidence,
                })

            if cutoff.decision == "EXCLUDE":
                stats["rejected_pre_cutoff"] += 1
                excluded_writer.writerow({
                    "title": title, "doi": doi, "pmid": pmid, "journal_pub_date": pub_date,
                    "earliest_public_date_found": cutoff.earliest_public_date,
                    "preprint_doi": cutoff.preprint_doi, "preprint_date": cutoff.preprint_date,
                    "exclusion_reason": cutoff.reason, "evidence": " | ".join(cutoff.evidence),
                })
                excluded_csv_f.flush()
                store.upsert(key, doi, pmid, pmcid, norm_title, "rejected", cutoff.reason,
                             {"title": title, "doi": doi, "cutoff": asdict(cutoff)})
                continue

            if cutoff.decision == "AMBIGUOUS":
                stats["rejected_ambiguous"] += 1
                excluded_writer.writerow({
                    "title": title, "doi": doi, "pmid": pmid, "journal_pub_date": pub_date,
                    "earliest_public_date_found": cutoff.earliest_public_date,
                    "preprint_doi": cutoff.preprint_doi, "preprint_date": cutoff.preprint_date,
                    "exclusion_reason": "AMBIGUOUS_DATE", "evidence": " | ".join(cutoff.evidence),
                })
                excluded_csv_f.flush()
                store.upsert(key, doi, pmid, pmcid, norm_title, "rejected", "AMBIGUOUS_DATE",
                             {"title": title, "doi": doi, "cutoff": asdict(cutoff)})
                continue

            # INCLUDE -> try to get a PDF
            if pmcid is None and doi:
                pmcid = recover_pmcid_via_doi(doi)

            pdf_bytes, source_used, license_used = (None, None, None)
            pdf_url_or_source = None
            if doi:
                pdf_bytes, pdf_url, license_used = try_publisher_pdf(doi)
                if pdf_bytes:
                    source_used = "publisher_direct"
                    pdf_url_or_source = pdf_url

            if pdf_bytes is None and pmcid:
                xml_bytes = fetch_jats_xml(pmcid)
                if xml_bytes:
                    pdf_bytes = render_pdf_from_jats(xml_bytes, title, author_string, journal, doi or "")
                    if pdf_bytes:
                        source_used = "rendered_from_jats_xml"
                        pdf_url_or_source = f"europepmc_fulltext:{pmcid}"

            if pdf_bytes is None:
                stats["rejected_no_fulltext"] += 1
                excluded_writer.writerow({
                    "title": title, "doi": doi, "pmid": pmid, "journal_pub_date": pub_date,
                    "earliest_public_date_found": cutoff.earliest_public_date,
                    "preprint_doi": cutoff.preprint_doi, "preprint_date": cutoff.preprint_date,
                    "exclusion_reason": "NO_LEGAL_FULLTEXT", "evidence": " | ".join(cutoff.evidence),
                })
                excluded_csv_f.flush()
                store.upsert(key, doi, pmid, pmcid, norm_title, "rejected", "NO_LEGAL_FULLTEXT",
                             {"title": title, "doi": doi, "cutoff": asdict(cutoff)})
                continue

            sha256 = sha256_of(pdf_bytes)
            if store.hash_exists(sha256):
                stats["skipped_duplicate"] += 1
                store.upsert(key, doi, pmid, pmcid, norm_title, "rejected", "DUPLICATE",
                             {"title": title, "doi": doi, "sha256": sha256})
                continue

            year_str = str(pub_year) if pub_year else cutoff.earliest_public_date[:4]
            year_dir = papers_dir / year_str
            year_dir.mkdir(parents=True, exist_ok=True)
            identifier = (pmcid or pmid or (doi.split("/")[-1] if doi else key))[:20]
            fname = build_filename(year_str, first_author, title, identifier)
            fpath = year_dir / fname
            n = 1
            while fpath.exists():
                fpath = year_dir / f"{fname[:-4]}_{n}.pdf"
                n += 1
            fpath.write_bytes(pdf_bytes)

            record = {
                "local_filename": str(fpath.relative_to(out_dir)),
                "title": title, "authors": author_string, "pub_year": pub_year, "journal": journal,
                "doi": doi, "pmid": pmid, "pmcid": pmcid, "pub_date": pub_date,
                "earliest_public_date": cutoff.earliest_public_date,
                "earliest_public_date_evidence": " | ".join(cutoff.evidence),
                "checked_for_preprint": cutoff.checked_preprint,
                "preprint_doi": cutoff.preprint_doi, "preprint_date": cutoff.preprint_date,
                "source_used_for_pdf": source_used, "pdf_url_or_source": pdf_url_or_source,
                "discovery_source": discovery_source,
                "license": license_used or article.get("license"),
                "download_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sha256": sha256, "file_size_bytes": len(pdf_bytes),
            }
            papers_writer.writerow(record)
            papers_csv_f.flush()
            papers_jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            papers_jsonl_f.flush()
            store.upsert(key, doi, pmid, pmcid, norm_title, "accepted_downloaded", "OK", record, sha256=sha256)

            stats["accepted"] += 1
            if stats["accepted"] % 5 == 0 or stats["accepted"] == 1:
                log.info(f"  [{stats['accepted']} accepted] latest: {title[:70]} ({source_used})")

            time.sleep(REQUEST_DELAY)

        log.info(f"  Query '{query_name}' done: {n_this_query} candidates scanned this run")

    papers_csv_f.close()
    excluded_csv_f.close()
    papers_jsonl_f.close()

    log.info("=" * 70)
    log.info("SUMMARY")
    for k, v in stats.items():
        log.info(f"  {k}: {v}")
    log.info(f"  Total accepted so far (all runs): {store.count_accepted()}")
    log.info("=" * 70)
    log.info("Example cutoff decisions:")
    for ex in example_decisions:
        log.info(f"  [{ex['decision']}/{ex['reason']}] {ex['title'][:70]}")
        log.info(f"    earliest_public_date={ex['earliest_public_date']}")
        for e in ex["evidence"]:
            log.info(f"    evidence: {e}")


if __name__ == "__main__":
    main()
