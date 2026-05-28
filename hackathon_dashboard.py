#!/usr/bin/env python3
"""
TB-LLM Seed Writing Hackathon Dashboard
Run: python3 hackathon_dashboard.py
Open: http://localhost:5050
"""

import json
import os
import glob
import random
import fcntl
from collections import Counter
from datetime import datetime
from xml.etree import ElementTree as ET
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SEEDS_FILE = os.path.join(BASE_DIR, "seeds.json")
CORPUS_DIR = os.path.join(BASE_DIR, "mtubercolosis")

NS = {
    "dc":    "http://purl.org/dc/elements/1.1/",
    "prism": "http://prismstandard.org/namespaces/basic/2.0/",
}

CATEGORIES = [
    ("data_interpretation",       "📊 Data Interpretation",       "Researcher pastes protein lists, fold-change tables, or abundance data and asks what it means biologically — differential abundance reasoning, unexpected hits, canonical infection-stage signatures."),
    ("fractionation_localization","🧫 Fractionation & Localization","ESX substrate identification, cell envelope fraction interpretation, lipoprotein vs. secreted vs. surface-exposed disambiguation. Requires knowing the unique Mtb envelope and secretion machinery."),
    ("methods_protocol",          "⚗️ Methods & BSL-3 Protocols",  "Sample prep constraints under BSL-3, inactivation tradeoffs, enrichment strategy advice (phosphoproteomics, secretome pulldowns) and where standard workflows break down specifically for Mtb."),
    ("database_integration",      "🗄️ Database & Cross-Experiment","Reconciling proteomics hits with Tn-seq essentiality, annotation quality caveats in MtbBase/UniProt/PATRIC, common co-IP sticky proteins (e.g. GroEL2), interpreting conflicting datasets."),
    ("host_pathogen",             "🦠 Host–Pathogen Interface",    "Dual proteomics (separating Mtb signal from macrophage signal), virulence factor abundance as infection-stage proxies, and proteins conserved between human and Mtb that are hard to disambiguate."),
    ("mechanism_hypothesis",      "💡 Mechanism & Hypothesis",     "Given an experimental observation, propose the most plausible molecular mechanisms and a distinguishing experiment. Open-ended reasoning grounded in TB biology."),
    ("literature_search",         "📚 Literature Search Strategy", "Scaffold targeted PubMed search queries for a specific TB proteomics research question, with a brief note on what each query is designed to surface."),
]

CATEGORY_TARGETS = {
    "data_interpretation":        16,
    "fractionation_localization": 12,
    "methods_protocol":           12,
    "database_integration":       12,
    "host_pathogen":              10,
    "mechanism_hypothesis":       10,
    "literature_search":           8,
}
TOTAL_TARGET = 80

SUBMIT_MSGS = [
    "That would stump GPT-4. Nice work. 🧬",
    "Training signal acquired. 💪",
    "The future TB-LLM thanks you. 🦠",
    "Exactly the kind of question that makes a model actually useful. 🔬",
    "BSL-3 knowledge, encoded. ⚗️",
    "Your expertise just became training data. 🧠",
    "One more question Claude can't answer. You're fixing that. ✨",
    "Solid output. The model will learn from that. 📚",
    "That's a high-MIWV seed for sure. 🎯",
]

EXAMPLE_SEEDS = [
    {
        "category": "data_interpretation",
        "ambiguous": False,
        "instruction": "Hey, I just got back my macrophage infection proteome results and I want to sanity-check my top hits before I start following anything up. Can you go through this list and tell me what's real biology, what's likely artifact, and what's interesting but unclear?",
        "input": "Top hits enriched >2-fold, FDR<0.05 vs. axenic culture: EsxA, EsxB, GroEL2, Rv3303c (LpdA), DnaK, Tuf, SigA, Rv0340, HspX (Rv2031c)",
        "output": "EsxA and EsxB are real biology — they're well-validated ESX-1 substrates and their enrichment during macrophage infection is a meaningful proxy for active secretion. HspX (Rv2031c, alpha-crystallin) is also expected: it's a canonical DosR-regulon protein and a textbook hypoxia/latency marker, so seeing it enriched during macrophage infection is consistent with the bacterium sensing the intracellular environment and entering a stress response. GroEL2 and DnaK should be treated with skepticism — they're notorious sticky proteins and some of the most common non-specific co-IP contaminants in Mtb proteomics. Their appearance at the top of a pulldown list is a red flag, not a hit. Tuf and SigA are both very highly abundant housekeeping proteins; their enrichment is almost certainly bleed-through from total protein rather than regulated changes. LpdA (Rv3303c) is genuinely interesting: it's a flavoprotein that was annotated as lipoamide dehydrogenase but structural work showed it lacks the catalytic residues for that activity and prefers NADPH; its role during infection is not well defined, so this is worth following up. Rv0340 is poorly annotated — I'd park it and come back only if you see it again in a second experiment.",
    },
    {
        "category": "methods_protocol",
        "ambiguous": False,
        "instruction": "I want to do phosphoproteomics on live Mtb. I know the basics of phosphoproteomics but I have no idea what's specific to working with mycobacteria under BSL-3. What do I need to know before I start?",
        "input": "Planning to use TiO2 enrichment, standard FASP workflow, inactivation by heat (80°C, 1h) before lysis.",
        "output": "Your planned heat inactivation step is the first thing to revisit. Heat inactivation at 80°C for 1 hour causes rapid and widespread dephosphorylation — phosphatases remain active during the temperature ramp before the cells die, and you'll systematically lose low-stoichiometry phosphosites before you've even touched the cells. The field standard for phosphoproteomics is to kill and denature as fast as possible: immediate lysis in boiling SDS (e.g. drop cells directly into 95°C SDS buffer) quenches phosphatase activity fastest. This is harder to do safely at BSL-3 and your containment setup will constrain your options, but it's worth the effort. Second: Mtb's kinome is dominated by PASTA-domain Ser/Thr kinases (PknA through PknL) rather than the His kinases that dominate other bacteria, so your TiO2 enrichment targeting pSer/pThr is the right choice — don't use Tyr-specific enrichment. Third: phosphosite stoichiometry is low in Mtb relative to eukaryotes, so budget for at least two rounds of TiO2 enrichment, or switch to Fe-IMAC which some labs find gives better coverage. Fourth: the cell wall makes lysis inefficient with standard sonication — bead-beating in lysis buffer with phosphatase inhibitors is standard, but you're in a race against the phosphatases the whole time. I'd point you to Prisic et al. 2010 (PNAS) and Ge et al. 2010 as the foundational Mtb phosphoproteomics papers, and to more recent work from the Bhatt lab for protocol updates.",
    },
    {
        "category": "database_integration",
        "ambiguous": True,
        "instruction": "I have a protein that shows up as strongly essential in multiple Tn-seq screens but I can't detect it at all in my whole-cell proteome — not even a single peptide. Is this a technical problem on my end, or is this a real thing that happens?",
        "input": "Protein: Rv2671, essential in Griffin 2011 and Sassetti 2003 Tn-seq datasets. Undetectable in my shotgun whole-cell proteome of H37Rv grown in 7H9 to mid-log. Coverage elsewhere in proteome is good (>2000 proteins).",
        "output": "This is a real phenomenon, not necessarily a technical failure on your part, and it happens for several reasons. First, essentiality by Tn-seq reflects the inability of the bacterium to tolerate loss of the gene — but that says nothing about how abundant the protein is. Some essential proteins are regulatory factors or enzymes that function at very low copy number; if the protein is present at fewer than ~100-500 copies per cell, shotgun proteomics will routinely miss it regardless of sample prep quality. Second, Rv2671 specifically: it's annotated as a conserved hypothetical with no well-characterized function, and low-abundance hypotheticals are among the most commonly missed proteins in Mtb whole-cell proteomes. Third, there are cases in the literature where Tn-seq essentiality calls are condition-specific or reflect indirect effects — a transposon in one gene can affect the fitness of neighboring genes through polar effects, so the essentiality call may not perfectly reflect the gene you're looking at. That said, I'd be cautious about concluding the protein is simply low-abundance without trying targeted approaches: parallel reaction monitoring (PRM) or a complementary enrichment step (e.g. immunoprecipitation if you have an antibody) would give you a more definitive answer. Whether the literature consensus on Rv2671's essentiality is fully settled is also worth checking — the Griffin and Sassetti datasets used different conditions and strain backgrounds, and there is some variability in essentiality calls across screens.",
    },
]


def load_abstracts(n=150, seed=42):
    rng = random.Random(seed)
    paths = glob.glob(os.path.join(CORPUS_DIR, "*.xml"))
    rng.shuffle(paths)
    abstracts = []
    for path in paths:
        if len(abstracts) >= n:
            break
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            title_el    = root.find(".//dc:title", NS)
            desc_el     = root.find(".//dc:description", NS)
            creators    = root.findall(".//dc:creator", NS)
            if desc_el is None or not desc_el.text or len(desc_el.text.strip()) < 80:
                continue
            pmid    = os.path.basename(path).replace(".xml", "")
            title   = (title_el.text or "").strip() if title_el is not None else "Unknown"
            authors = ", ".join(c.text.strip() for c in creators if c.text)
            abstracts.append({"pmid": pmid, "title": title, "authors": authors, "abstract": desc_el.text.strip()})
        except Exception:
            continue
    return abstracts


ABSTRACTS = load_abstracts()


def read_seeds():
    if not os.path.exists(SEEDS_FILE):
        return []
    with open(SEEDS_FILE) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def write_seed(seed):
    lock_path = SEEDS_FILE + ".lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        seeds = read_seeds()
        seeds.append(seed)
        with open(SEEDS_FILE, "w") as f:
            json.dump(seeds, f, indent=2, ensure_ascii=False)
        fcntl.flock(lf, fcntl.LOCK_UN)


@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/abstracts")
def api_abstracts():
    return jsonify(ABSTRACTS)

@app.route("/api/seeds", methods=["GET"])
def api_seeds_get():
    seeds = read_seeds()
    return jsonify({"count": len(seeds), "seeds": seeds})

@app.route("/api/seeds", methods=["POST"])
def api_seeds_post():
    data = request.get_json(force=True)
    for field in ("submitter", "category", "instruction", "input", "output"):
        if not data.get(field, "").strip():
            return jsonify({"error": f"'{field}' is required"}), 400
    seed = {
        "id":           f"hack-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
        "submitted_at": datetime.now().isoformat(),
        "submitter":    data["submitter"].strip(),
        "category":     data["category"].strip(),
        "ambiguous":    bool(data.get("ambiguous", False)),
        "instruction":  data["instruction"].strip(),
        "input":        data["input"].strip(),
        "output":       data["output"].strip(),
        "notes":        data.get("notes", "").strip(),
    }
    write_seed(seed)
    total = len(read_seeds())
    return jsonify({"ok": True, "id": seed["id"], "total": total})

@app.route("/api/stats")
def api_stats():
    seeds = read_seeds()
    counts = Counter(s.get("category", "") for s in seeds)
    categories = [
        {"id": cid, "name": name, "count": counts.get(cid, 0), "target": CATEGORY_TARGETS.get(cid, 10)}
        for cid, name, _ in CATEGORIES
    ]
    return jsonify({"total": len(seeds), "target": TOTAL_TARGET, "categories": categories})

@app.route("/api/leaderboard")
def api_leaderboard():
    seeds = read_seeds()
    scores = {}
    for s in seeds:
        name = (s.get("submitter") or "Anonymous").strip() or "Anonymous"
        pts  = 2 if s.get("ambiguous") else 1
        if name not in scores:
            scores[name] = {"name": name, "regular": 0, "ambiguous": 0, "points": 0}
        if s.get("ambiguous"):
            scores[name]["ambiguous"] += 1
        else:
            scores[name]["regular"] += 1
        scores[name]["points"] += pts
    board = sorted(scores.values(), key=lambda x: (-x["points"], x["name"]))
    return jsonify(board)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TB-LLM Seed Hackathon</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:       #12141e;
  --s1:       #1a1d2e;
  --s2:       #21253a;
  --s3:       #272b42;
  --border:   #2e3352;
  --blue:     #7b9fff;
  --blue-dim: #2d3f6a;
  --green:    #4ade80;
  --green-dim:#1a4730;
  --amber:    #fbbf24;
  --amber-dim:#44370a;
  --red:      #f87171;
  --purple:   #c084fc;
  --text:     #dde1f0;
  --muted:    #7a82a0;
  --muted2:   #3e4460;
  --r:        10px;
  --sidebar:  310px;
  --topbar:   54px;
}
body { font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; font-size: 14px; line-height: 1.5; }

/* ── TOP BAR ─────────────────────────────── */
.topbar {
  position: fixed; top: 0; left: 0; right: 0; height: var(--topbar);
  background: var(--s1); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 20px; padding: 0 20px; z-index: 50;
}
.topbar-title { font-size: 15px; font-weight: 700; color: var(--blue); white-space: nowrap; }
.topbar-progress { flex: 1; min-width: 0; }
.topbar-progress-label { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-bottom: 4px; }
.topbar-progress-label strong { color: var(--text); font-weight: 700; }
.progress-track {
  height: 8px; background: var(--s3); border-radius: 4px; position: relative; overflow: visible;
}
.progress-fill {
  height: 100%; background: linear-gradient(90deg, var(--blue), var(--green));
  border-radius: 4px; transition: width 0.6s ease; min-width: 0;
}
.milestone-pip {
  position: absolute; top: -3px; width: 2px; height: 14px;
  background: var(--border); border-radius: 1px;
}
.milestone-pip span {
  position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
  font-size: 9px; color: var(--muted2); white-space: nowrap;
}
.topbar-leader { font-size: 12px; color: var(--muted); white-space: nowrap; }
.topbar-leader strong { color: var(--amber); }
.milestone-flash {
  display: none; position: fixed; top: 66px; left: 50%; transform: translateX(-50%);
  background: var(--green); color: #0a1a10; font-weight: 700; font-size: 14px;
  padding: 8px 22px; border-radius: 20px; z-index: 100; white-space: nowrap;
  animation: fadeSlide 3s ease forwards;
}
@keyframes fadeSlide {
  0%   { opacity:0; transform:translateX(-50%) translateY(-6px); }
  15%  { opacity:1; transform:translateX(-50%) translateY(0); }
  75%  { opacity:1; }
  100% { opacity:0; }
}

/* ── LAYOUT ──────────────────────────────── */
.layout { display: flex; height: 100vh; padding-top: var(--topbar); overflow: hidden; }
.sidebar {
  width: var(--sidebar); min-width: var(--sidebar);
  height: 100%; overflow-y: auto; background: var(--s1);
  border-right: 1px solid var(--border); display: flex; flex-direction: column;
}
.main { flex: 1; overflow-y: auto; padding: 20px 24px 32px; }

/* ── SIDEBAR ─────────────────────────────── */
.sb-section { border-top: 1px solid var(--border); }
.sb-toggle {
  width: 100%; background: none; border: none; color: var(--text);
  padding: 11px 18px; display: flex; align-items: center; justify-content: space-between;
  cursor: pointer; font-size: 11px; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase; text-align: left; font-family: inherit;
}
.sb-toggle:hover { background: var(--s2); }
.sb-toggle .arr { color: var(--muted); font-size: 9px; transition: transform .2s; }
.sb-toggle.open .arr { transform: rotate(180deg); }
.sb-body { padding: 0 18px 14px; font-size: 13px; line-height: 1.65; color: #b0b8d0; display: none; }
.sb-body.open { display: block; }
.sb-body p { margin-bottom: 8px; }
.sb-body strong { color: var(--text); }

.cat-card { background: var(--s2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; margin-bottom: 7px; }
.cat-card-name { font-size: 12px; font-weight: 700; color: var(--blue); margin-bottom: 3px; }
.cat-card-desc { font-size: 11.5px; color: var(--muted); line-height: 1.5; }

.tip-row { display: flex; gap: 8px; margin-bottom: 7px; font-size: 12.5px; }
.tip-row .dot { color: var(--blue); flex-shrink: 0; font-weight: 700; margin-top: 1px; }

/* ── EXAMPLES ────────────────────────────── */
.ex-card { background: var(--s2); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 10px; font-size: 12px; }
.ex-badge { display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: 10px; font-weight: 700; letter-spacing: .04em; margin-bottom: 6px; }
.ex-badge.regular { background: var(--blue-dim); color: var(--blue); }
.ex-badge.ambiguous { background: var(--amber-dim); color: var(--amber); }
.ex-field { margin-bottom: 6px; }
.ex-field-lbl { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 2px; }
.ex-field-val { color: var(--text); line-height: 1.55; max-height: 75px; overflow-y: auto; }

/* ── MAIN AREA ───────────────────────────── */
.section-label {
  font-size: 10.5px; font-weight: 700; color: var(--muted2);
  text-transform: uppercase; letter-spacing: .07em; margin-bottom: 10px;
}

/* ── ABSTRACT CARD ───────────────────────── */
.abs-card {
  background: var(--s1); border: 1px solid var(--border); border-radius: var(--r);
  padding: 18px 22px; margin-bottom: 18px;
}
.abs-nav { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.nav-btn {
  background: var(--s2); border: 1px solid var(--border); color: var(--text);
  padding: 5px 13px; border-radius: 6px; cursor: pointer; font-size: 12px;
  font-family: inherit; transition: background .15s;
}
.nav-btn:hover:not(:disabled) { background: var(--s3); }
.nav-btn:disabled { opacity: .3; cursor: default; }
.abs-idx { font-size: 12px; color: var(--muted); flex: 1; text-align: center; }
.use-btn {
  background: none; border: 1px solid var(--blue); color: var(--blue);
  padding: 5px 13px; border-radius: 6px; cursor: pointer; font-size: 12px;
  font-family: inherit; transition: all .15s;
}
.use-btn:hover { background: var(--blue); color: #fff; }
.abs-title { font-size: 15px; font-weight: 700; line-height: 1.4; margin-bottom: 5px; }
.abs-authors { font-size: 11.5px; color: var(--muted); margin-bottom: 12px; }
.abs-pmid { font-size: 10px; color: var(--muted2); }
.abs-body {
  font-size: 13px; line-height: 1.75; color: #c0c8d8;
  max-height: 180px; overflow-y: auto;
  border-top: 1px solid var(--border); padding-top: 12px; margin-top: 10px;
}
.abs-body::-webkit-scrollbar { width: 3px; }
.abs-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
.kbd-note { font-size: 10px; color: var(--muted2); margin-top: 8px; }
.kbd { background: var(--s2); border: 1px solid var(--border); border-radius: 3px; padding: 1px 5px; font-family: monospace; font-size: 10px; }

/* ── BOTTOM ROW ──────────────────────────── */
.bottom-row { display: grid; grid-template-columns: 1fr 360px; gap: 16px; align-items: start; }

/* ── SEED FORM ───────────────────────────── */
.seed-form { background: var(--s1); border: 1px solid var(--border); border-radius: var(--r); padding: 20px 22px; }
.form-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.fg { margin-bottom: 13px; }
.fg label {
  display: block; font-size: 10.5px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: .05em; margin-bottom: 5px;
}
.fg label .req { color: var(--blue); }
.fg label .hint { font-weight: 400; text-transform: none; letter-spacing: 0; font-size: 10px; color: var(--muted2); margin-left: 4px; }
input[type=text], select, textarea {
  width: 100%; background: var(--s2); border: 1px solid var(--border); border-radius: 7px;
  color: var(--text); padding: 8px 11px; font-size: 13px; font-family: inherit;
  transition: border-color .15s; resize: vertical;
}
input[type=text]:focus, select:focus, textarea:focus { outline: none; border-color: var(--blue); }
select option { background: var(--s2); }
textarea.short { min-height: 70px; }
textarea.tall  { min-height: 110px; }
textarea.xtall { min-height: 140px; }

/* ── AMBIGUOUS TOGGLE ────────────────────── */
.amb-toggle {
  display: flex; align-items: flex-start; gap: 12px;
  background: var(--s2); border: 1.5px solid var(--border); border-radius: 8px;
  padding: 12px 14px; cursor: pointer; transition: border-color .15s, background .15s;
  margin-bottom: 14px; user-select: none;
}
.amb-toggle:hover { border-color: var(--amber); }
.amb-toggle.active { border-color: var(--amber); background: #2a2010; }
.amb-check {
  width: 22px; height: 22px; border: 1.5px solid var(--border); border-radius: 5px;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  font-size: 13px; transition: all .15s; background: var(--s3);
}
.amb-toggle.active .amb-check { border-color: var(--amber); background: var(--amber-dim); }
.amb-text { flex: 1; }
.amb-title { font-size: 13px; font-weight: 600; color: var(--text); }
.amb-toggle.active .amb-title { color: var(--amber); }
.amb-desc { font-size: 11.5px; color: var(--muted); margin-top: 2px; line-height: 1.4; }

.submit-row { display: flex; align-items: center; gap: 14px; }
.submit-btn {
  background: var(--blue); color: #fff; border: none; border-radius: 7px;
  padding: 9px 26px; font-size: 14px; font-weight: 600; cursor: pointer;
  font-family: inherit; transition: opacity .15s;
}
.submit-btn:hover { opacity: .85; }
.submit-btn:disabled { opacity: .4; cursor: default; }
.form-msg { font-size: 12.5px; }
.form-msg.ok  { color: var(--green); }
.form-msg.err { color: var(--red); }

/* ── RIGHT PANEL ─────────────────────────── */
.right-panel { display: flex; flex-direction: column; gap: 14px; }

/* ── LEADERBOARD ─────────────────────────── */
.leaderboard { background: var(--s1); border: 1px solid var(--border); border-radius: var(--r); padding: 16px 18px; }
.lb-note { font-size: 11px; color: var(--muted2); margin-bottom: 10px; }
.lb-row {
  display: flex; align-items: center; gap: 10px;
  padding: 7px 10px; border-radius: 7px; margin-bottom: 4px;
}
.lb-row:nth-child(1) { background: #1e1a08; }
.lb-row:nth-child(2) { background: #16181f; }
.lb-row:nth-child(3) { background: #14161c; }
.lb-rank { font-size: 15px; width: 24px; text-align: center; flex-shrink: 0; }
.lb-name { flex: 1; font-size: 13px; font-weight: 600; }
.lb-detail { font-size: 11px; color: var(--muted); }
.lb-pts { font-size: 15px; font-weight: 800; color: var(--amber); min-width: 28px; text-align: right; }
.lb-empty { font-size: 12px; color: var(--muted2); text-align: center; padding: 12px 0; }

/* ── CATEGORY STATS ──────────────────────── */
.cat-stats { background: var(--s1); border: 1px solid var(--border); border-radius: var(--r); padding: 16px 18px; }
.cs-row { margin-bottom: 9px; }
.cs-meta { display: flex; justify-content: space-between; font-size: 11.5px; margin-bottom: 3px; }
.cs-name { color: var(--text); font-weight: 500; }
.cs-count { color: var(--muted); }
.cs-count.done { color: var(--green); }
.cs-track { height: 5px; background: var(--s3); border-radius: 3px; overflow: hidden; }
.cs-fill { height: 100%; border-radius: 3px; transition: width .5s ease; background: var(--blue); }
.cs-fill.done { background: var(--green); }

/* ── TOAST ───────────────────────────────── */
#toast {
  position: fixed; bottom: 24px; right: 24px;
  background: var(--green); color: #061a0d; padding: 10px 18px;
  border-radius: 9px; font-size: 13px; font-weight: 600;
  opacity: 0; transform: translateY(6px); transition: all .25s;
  pointer-events: none; z-index: 200; max-width: 320px;
}
#toast.show { opacity: 1; transform: translateY(0); }
#toast.err  { background: var(--red); color: #fff; }

/* ── SCROLLBARS ──────────────────────────── */
.sidebar::-webkit-scrollbar, .main::-webkit-scrollbar { width: 4px; }
.sidebar::-webkit-scrollbar-thumb, .main::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── TABS ────────────────────────────────── */
.tab-bar { display: flex; gap: 2px; margin-bottom: 20px; border-bottom: 1px solid var(--border); }
.tab-btn {
  background: none; border: none; border-bottom: 2px solid transparent;
  color: var(--muted); padding: 8px 18px; font-size: 13px; font-weight: 600;
  cursor: pointer; font-family: inherit; transition: all .15s; margin-bottom: -1px;
}
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--blue); border-bottom-color: var(--blue); }

/* ── SEEDS VIEW ──────────────────────────── */
.sv-filters { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
.sv-filters select, .sv-filters input[type=text] { flex: 1; min-width: 140px; }
.sv-meta { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.sv-count { font-size: 12px; color: var(--muted); }
.sv-sort { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
.sv-sort select { padding: 4px 8px; font-size: 12px; flex: none; width: auto; }
.seed-card { background: var(--s1); border: 1px solid var(--border); border-radius: var(--r); margin-bottom: 8px; overflow: hidden; }
.sc-header { display: flex; align-items: center; gap: 10px; padding: 11px 14px; cursor: pointer; transition: background .12s; }
.sc-header:hover { background: var(--s2); }
.sc-badges { display: flex; gap: 6px; align-items: center; flex-shrink: 0; }
.sc-cat-badge { font-size: 10.5px; font-weight: 700; color: var(--blue); background: var(--blue-dim); padding: 2px 7px; border-radius: 4px; white-space: nowrap; }
.sc-amb-badge { font-size: 11px; color: var(--amber); }
.sc-submitter { font-size: 12px; font-weight: 600; color: var(--muted); white-space: nowrap; flex-shrink: 0; }
.sc-instr { flex: 1; font-size: 13px; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sc-time { font-size: 11px; color: var(--muted2); white-space: nowrap; flex-shrink: 0; }
.sc-chevron { color: var(--muted2); font-size: 10px; flex-shrink: 0; transition: transform .2s; }
.sc-chevron.open { transform: rotate(180deg); }
.sc-body { display: none; border-top: 1px solid var(--border); padding: 14px 16px; }
.sc-body.open { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.sc-body.open.no-input { grid-template-columns: 1fr; }
.sc-field-lbl { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px; font-weight: 600; }
.sc-field-val { font-size: 12.5px; line-height: 1.65; color: #c0c8d8; max-height: 200px; overflow-y: auto; }
.sc-field-val::-webkit-scrollbar { width: 3px; }
.sc-field-val::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
.sc-full { grid-column: 1 / -1; }
.sc-notes { grid-column: 1 / -1; margin-top: 4px; }
.sc-notes .sc-field-val { color: var(--amber); font-size: 12px; }
.sv-empty { text-align: center; color: var(--muted2); padding: 50px 0; font-size: 13px; }
</style>
</head>
<body>

<!-- TOP BAR -->
<header class="topbar">
  <div class="topbar-title">🔬 TB-LLM Seed Hackathon</div>
  <div class="topbar-progress">
    <div class="topbar-progress-label">
      <span><strong id="hdr-count">—</strong> / 80 seeds</span>
      <span id="hdr-pct">0%</span>
    </div>
    <div class="progress-track">
      <div class="progress-fill" id="hdr-fill" style="width:0%"></div>
      <div class="milestone-pip" style="left:31.25%"><span>25</span></div>
      <div class="milestone-pip" style="left:62.5%"><span>50</span></div>
      <div class="milestone-pip" style="left:93.75%"><span>75</span></div>
    </div>
  </div>
  <div class="topbar-leader" id="hdr-leader"></div>
</header>
<div id="milestone-flash" class="milestone-flash"></div>

<div class="layout">

  <!-- SIDEBAR -->
  <aside class="sidebar">

    <div class="sb-section" style="border-top:none;">
      <button class="sb-toggle open" onclick="toggleSb(this)">What we're building <span class="arr">▼</span></button>
      <div class="sb-body open">
        <p>We're fine-tuning a language model to be a <strong>research assistant for M. tuberculosis proteomics</strong>. The goal is a model that answers the questions a TB researcher asks daily — questions GPT-4 or Claude get wrong because they lack deep domain knowledge.</p>
        <p>The pipeline: continued pre-training on ~1,400 TB papers, then supervised fine-tuning on curated instruction-response pairs. <strong>Your job today is to write those pairs.</strong></p>
      </div>
    </div>

    <div class="sb-section">
      <button class="sb-toggle open" onclick="toggleSb(this)">Why your seeds matter <span class="arr">▼</span></button>
      <div class="sb-body open">
        <p>We generate thousands of training examples using a large LLM — but we seed it with hand-written examples first. The diversity and quality of your seeds directly controls what gets generated.</p>
        <p>We use a method called <strong>MIWV</strong> to filter generated data: it keeps only examples that genuinely challenge the base model. Seeds that encode hard, domain-specific knowledge produce high-value training data. Generic questions produce noise that gets filtered out.</p>
        <p><strong>Write the questions GPT-4 would get wrong.</strong></p>
      </div>
    </div>

    <div class="sb-section">
      <button class="sb-toggle open" onclick="toggleSb(this)">Categories <span class="arr">▼</span></button>
      <div class="sb-body open" id="cat-list-sb"></div>
    </div>

    <div class="sb-section">
      <button class="sb-toggle" onclick="toggleSb(this)">Tips for good seeds <span class="arr">▼</span></button>
      <div class="sb-body">
        <div class="tip-row"><span class="dot">→</span><span>Use a <strong>naturalistic researcher voice</strong>. "Hey, I just got these results and I'm trying to make sense of them…" beats "Explain the mechanism of…"</span></div>
        <div class="tip-row"><span class="dot">→</span><span>The <strong>output is the training signal</strong>. Write it like an expert TB proteomics researcher would actually respond — specific gene names, mechanisms, caveats, what to do next.</span></div>
        <div class="tip-row"><span class="dot">→</span><span>The <strong>input grounds the seed</strong>. Paste a protein list, fold-change data, an experimental context, or the relevant part of an abstract. Don't leave it vague.</span></div>
        <div class="tip-row"><span class="dot">→</span><span>Vary your instruction phrasing across seeds — if you've written two starting with "I got a hit for…", change the register.</span></div>
        <div class="tip-row"><span class="dot">→</span><span>Include <strong>uncertainty where it's real</strong>. "This is genuinely contested in the literature — here's why" is a valuable output. Don't fake confidence you don't have.</span></div>
        <div class="tip-row"><span class="dot">→</span><span>Use the <strong>⭐ Ambiguous</strong> toggle for questions where the answer is genuinely unclear or the literature is inconclusive — these are worth double points and produce the most valuable training data.</span></div>
      </div>
    </div>

    <div class="sb-section">
      <button class="sb-toggle" onclick="toggleSb(this)">Example seeds <span class="arr">▼</span></button>
      <div class="sb-body" id="ex-container"></div>
    </div>

  </aside>

  <!-- MAIN -->
  <main class="main">

    <!-- TAB BAR -->
    <div class="tab-bar">
      <button class="tab-btn active" id="tab-submit-btn" onclick="switchTab('submit')">Submit a Seed</button>
      <button class="tab-btn" id="tab-seeds-btn" onclick="switchTab('seeds')">View All Seeds</button>
    </div>

    <!-- TAB: SUBMIT -->
    <div id="tab-submit">

    <!-- ABSTRACT BROWSER -->
    <div class="section-label">Abstract Browser — click next for a random abstract, or paste one directly into the Input field</div>
    <div class="abs-card">
      <div class="abs-nav">
        <button class="nav-btn" id="prev-btn" onclick="navigate(-1)" disabled>← Prev</button>
        <span class="abs-idx" id="abs-idx">Loading…</span>
        <button class="nav-btn" id="next-btn" onclick="navigate(1)">Next →</button>
        <button class="use-btn" onclick="useAbstract()">Use this abstract ↓</button>
      </div>
      <div class="abs-title" id="abs-title">Loading abstracts…</div>
      <div class="abs-authors" id="abs-authors"></div>
      <div class="abs-pmid" id="abs-pmid"></div>
      <div class="abs-body" id="abs-body"></div>
      <div class="kbd-note">Keyboard: <span class="kbd">←</span> <span class="kbd">→</span> arrows to navigate</div>
    </div>

    <!-- BOTTOM ROW: form + right panel -->
    <div class="bottom-row">

      <!-- SEED FORM -->
      <div>
        <div class="section-label">Submit a Seed</div>
        <div class="seed-form">
          <div class="form-2col">
            <div class="fg">
              <label>Your name <span class="req">*</span></label>
              <input type="text" id="f-submitter" placeholder="e.g. Seda">
            </div>
            <div class="fg">
              <label>Category <span class="req">*</span></label>
              <select id="f-category"><option value="">— select —</option></select>
            </div>
          </div>

          <div class="fg">
            <label>Instruction <span class="req">*</span> <span class="hint">— the question, written as a researcher would actually say it</span></label>
            <textarea class="tall" id="f-instruction" placeholder="e.g. Hey, I keep seeing GroEL2 as a top hit in my co-IP pulldown — is there any reason to follow this up or is it just a sticky protein?"></textarea>
          </div>

          <div class="fg">
            <label>Input <span class="req">*</span> <span class="hint">— paste protein list, fold-change data, abstract excerpt, or experimental context</span></label>
            <textarea class="short" id="f-input" placeholder="Paste your protein list, experimental data, or the relevant passage here. Be specific — this grounds the seed."></textarea>
          </div>

          <div class="fg">
            <label>Output <span class="req">*</span> <span class="hint">— the expert answer; this is the training signal, write it carefully</span></label>
            <textarea class="xtall" id="f-output" placeholder="The answer a TB proteomics expert would give. Include gene names, mechanisms, caveats, what to do next. If the answer is genuinely uncertain, say so explicitly."></textarea>
          </div>

          <div class="fg">
            <label>Notes <span class="hint">— things to verify, sources to check, known uncertainty</span></label>
            <textarea class="short" id="f-notes" placeholder="e.g. Need to verify GroEL2 sticky-protein claim — check Rao et al. and Levin et al."></textarea>
          </div>

          <!-- AMBIGUOUS TOGGLE -->
          <div class="amb-toggle" id="amb-toggle" onclick="toggleAmb()">
            <div class="amb-check" id="amb-check"></div>
            <div class="amb-text">
              <div class="amb-title">⭐ Ambiguous / literature inconclusive — 2 points!</div>
              <div class="amb-desc">The answer is genuinely contested, unclear, or where the literature gives conflicting results. The output must explicitly acknowledge this uncertainty. Bonus points because these are the hardest and most valuable seeds.</div>
            </div>
          </div>

          <div class="submit-row">
            <button class="submit-btn" id="submit-btn" onclick="submitSeed()">Submit seed</button>
            <span class="form-msg" id="form-msg"></span>
          </div>
        </div>
      </div>

      <!-- RIGHT PANEL -->
      <div class="right-panel">

        <!-- LEADERBOARD -->
        <div>
          <div class="section-label">Leaderboard</div>
          <div class="leaderboard">
            <div class="lb-note">Regular seed = 1pt &nbsp;·&nbsp; ⭐ Ambiguous = 2pts</div>
            <div id="lb-rows"><div class="lb-empty">No seeds yet — be first!</div></div>
          </div>
        </div>

        <!-- CATEGORY PROGRESS -->
        <div>
          <div class="section-label">Category Quota</div>
          <div class="cat-stats" id="cat-stats"></div>
        </div>

      </div>
    </div>

    </div><!-- /tab-submit -->

    <!-- TAB: VIEW ALL SEEDS -->
    <div id="tab-seeds" style="display:none">
      <div class="sv-filters">
        <select id="sv-cat" onchange="renderSeeds()"><option value="">All categories</option></select>
        <input type="text" id="sv-name" placeholder="Filter by name…" oninput="renderSeeds()">
        <input type="text" id="sv-search" placeholder="Search instructions…" oninput="renderSeeds()">
      </div>
      <div class="sv-meta">
        <span class="sv-count" id="sv-count"></span>
        <span class="sv-sort">
          Sort:
          <select id="sv-sort" onchange="renderSeeds()">
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
            <option value="ambiguous">Ambiguous first</option>
            <option value="cat">By category</option>
          </select>
        </span>
      </div>
      <div id="sv-list"><div class="sv-empty">Loading seeds…</div></div>
    </div><!-- /tab-seeds -->

  </main>
</div>

<div id="toast"></div>

<script>
const CATS = """ + json.dumps(CATEGORIES) + r""";
const EXAMPLES = """ + json.dumps(EXAMPLE_SEEDS) + r""";
const SUBMIT_MSGS = """ + json.dumps(SUBMIT_MSGS) + r""";
const MILESTONES = {10:"Double digits! 🔥", 25:"Quarter way there! 🎉", 50:"HALFWAY. Incredible. 🚀", 75:"Almost there — final push! 💪", 80:"GOAL REACHED. 🏆"};

let abstracts = [], curIdx = 0, ambiguous = false, lastTotal = -1;
let seen = new Set(), allSeeds = [];

// ── INIT ──────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  buildSidebarCats();
  buildExamples();
  populateCatSelect();
  populateSvCatSelect();
  const saved = localStorage.getItem('tb_submitter');
  if (saved) document.getElementById('f-submitter').value = saved;
  fetchAbstracts();
  refreshAll();
  setInterval(refreshAll, 20000);
  document.addEventListener('keydown', e => {
    if (['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) return;
    if (e.key === 'ArrowLeft')  navigate(-1);
    if (e.key === 'ArrowRight') navigate(1);
  });
});

// ── SIDEBAR ───────────────────────────────
function toggleSb(btn) {
  btn.classList.toggle('open');
  btn.nextElementSibling.classList.toggle('open');
}

function buildSidebarCats() {
  document.getElementById('cat-list-sb').innerHTML = CATS.map(([id, name, desc]) =>
    `<div class="cat-card"><div class="cat-card-name">${name}</div><div class="cat-card-desc">${desc}</div></div>`
  ).join('');
}

function buildExamples() {
  document.getElementById('ex-container').innerHTML = EXAMPLES.map((ex, i) => {
    const catName = CATS.find(c => c[0] === ex.category)?.[1] || ex.category;
    const isAmb = ex.ambiguous;
    return `
      <div class="ex-card">
        <div class="ex-badge ${isAmb ? 'ambiguous' : 'regular'}">${isAmb ? '⭐ AMBIGUOUS · 2pts' : 'REGULAR · 1pt'}</div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:6px;">${catName}</div>
        <div class="ex-field"><div class="ex-field-lbl">Instruction</div><div class="ex-field-val">${ex.instruction}</div></div>
        <div class="ex-field"><div class="ex-field-lbl">Input</div><div class="ex-field-val">${ex.input || '(closed-book)'}</div></div>
        <div class="ex-field"><div class="ex-field-lbl">Output</div><div class="ex-field-val">${ex.output}</div></div>
      </div>`;
  }).join('');
}

function populateCatSelect() {
  const sel = document.getElementById('f-category');
  CATS.forEach(([id, name]) => {
    const o = document.createElement('option');
    o.value = id; o.textContent = name;
    sel.appendChild(o);
  });
}

// ── ABSTRACTS ─────────────────────────────
async function fetchAbstracts() {
  try {
    const r = await fetch('/api/abstracts');
    abstracts = await r.json();
    renderAbstract();
  } catch(e) { document.getElementById('abs-title').textContent = 'Could not load abstracts.'; }
}

function renderAbstract() {
  if (!abstracts.length) return;
  const a = abstracts[curIdx];
  document.getElementById('abs-title').textContent   = a.title || 'Untitled';
  document.getElementById('abs-authors').textContent = a.authors || '';
  document.getElementById('abs-pmid').textContent    = `PMID: ${a.pmid}`;
  document.getElementById('abs-body').textContent    = a.abstract;
  document.getElementById('abs-idx').textContent     = `Paper ${curIdx+1} of ${abstracts.length}`;
  document.getElementById('prev-btn').disabled = curIdx === 0;
  document.getElementById('next-btn').disabled = curIdx === abstracts.length - 1;
}

function navigate(d) {
  if (!abstracts.length) return;
  seen.add(curIdx);
  if (d > 0) {
    // pick a random unseen abstract; reset if all seen
    const unseen = abstracts.map((_, i) => i).filter(i => !seen.has(i));
    if (!unseen.length) { seen.clear(); seen.add(curIdx); }
    const pool = unseen.length ? unseen : abstracts.map((_, i) => i).filter(i => i !== curIdx);
    curIdx = pool[Math.floor(Math.random() * pool.length)];
  } else {
    curIdx = Math.max(0, curIdx - 1);
  }
  renderAbstract();
}

function useAbstract() {
  if (!abstracts.length) return;
  const el = document.getElementById('f-input');
  el.value = abstracts[curIdx].abstract;
  el.scrollIntoView({behavior:'smooth', block:'center'});
  el.focus();
  showToast('Abstract pasted into Input field', false);
}

// ── AMBIGUOUS TOGGLE ─────────────────────
function toggleAmb() {
  ambiguous = !ambiguous;
  const tog = document.getElementById('amb-toggle');
  const chk = document.getElementById('amb-check');
  tog.classList.toggle('active', ambiguous);
  chk.textContent = ambiguous ? '✓' : '';
}

// ── SUBMIT ────────────────────────────────
async function submitSeed() {
  const btn = document.getElementById('submit-btn');
  const msg = document.getElementById('form-msg');
  msg.textContent = ''; msg.className = 'form-msg';

  const payload = {
    submitter:   document.getElementById('f-submitter').value,
    category:    document.getElementById('f-category').value,
    instruction: document.getElementById('f-instruction').value,
    input:       document.getElementById('f-input').value,
    output:      document.getElementById('f-output').value,
    notes:       document.getElementById('f-notes').value,
    ambiguous:   ambiguous,
  };

  if (!payload.submitter.trim())   { setMsg('Enter your name.', true); return; }
  localStorage.setItem('tb_submitter', payload.submitter.trim());
  if (!payload.category)            { setMsg('Select a category.', true); return; }
  if (!payload.instruction.trim()) { setMsg('Instruction is required.', true); return; }
  if (!payload.input.trim())        { setMsg('Input is required — paste your data or experimental context.', true); return; }
  if (!payload.output.trim())       { setMsg('Output is required.', true); return; }

  btn.disabled = true; btn.textContent = 'Submitting…';
  try {
    const r = await fetch('/api/seeds', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const d = await r.json();
    if (d.ok) {
      const funMsg = SUBMIT_MSGS[Math.floor(Math.random() * SUBMIT_MSGS.length)];
      showToast(funMsg, false);
      setMsg(`Submitted! Team total: ${d.total} seed${d.total===1?'':'s'}`, false);
      clearForm();
      refreshAll();
    } else {
      setMsg(d.error || 'Something went wrong.', true);
    }
  } catch(e) { setMsg('Network error — try again.', true); }
  finally { btn.disabled = false; btn.textContent = 'Submit seed'; }
}

function setMsg(text, isErr) {
  const el = document.getElementById('form-msg');
  el.textContent = text;
  el.className = 'form-msg ' + (isErr ? 'err' : 'ok');
}

function clearForm() {
  ['f-instruction','f-input','f-output','f-notes'].forEach(id => document.getElementById(id).value = '');
  if (ambiguous) toggleAmb();
}

// ── REFRESH ───────────────────────────────
async function refreshAll() {
  try {
    const [statsR, lbR] = await Promise.all([fetch('/api/stats'), fetch('/api/leaderboard')]);
    const stats = await statsR.json();
    const lb    = await lbR.json();
    updateHeader(stats.total, stats.target);
    updateLeaderboard(lb);
    updateCatStats(stats.categories);
    checkMilestone(stats.total);
  } catch(e) {}
}

function updateHeader(total, target) {
  document.getElementById('hdr-count').textContent = total;
  const pct = Math.min(100, Math.round(total / target * 100));
  document.getElementById('hdr-pct').textContent = pct + '%';
  document.getElementById('hdr-fill').style.width = pct + '%';
}

function updateLeaderboard(lb) {
  const MEDALS = ['👑','🥈','🥉'];
  const el = document.getElementById('lb-rows');
  if (!lb.length) { el.innerHTML = '<div class="lb-empty">No seeds yet — be first!</div>'; return; }

  // Update leader in top bar
  if (lb[0]) {
    document.getElementById('hdr-leader').innerHTML = `Leader: <strong>${lb[0].name}</strong> (${lb[0].points}pts)`;
  }

  el.innerHTML = lb.map((row, i) => {
    const medal   = MEDALS[i] || `${i+1}.`;
    const ambBadge = row.ambiguous > 0 ? ` <span style="color:var(--amber);font-size:10px;">⭐×${row.ambiguous}</span>` : '';
    return `
      <div class="lb-row">
        <span class="lb-rank">${medal}</span>
        <div class="lb-name">${row.name}${ambBadge}</div>
        <div class="lb-detail">${row.regular + row.ambiguous} seed${row.regular+row.ambiguous!==1?'s':''}</div>
        <div class="lb-pts">${row.points}</div>
      </div>`;
  }).join('');
}

function updateCatStats(categories) {
  const el = document.getElementById('cat-stats');
  el.innerHTML = categories.map(c => {
    const pct  = Math.min(100, Math.round(c.count / c.target * 100));
    const done = c.count >= c.target;
    return `
      <div class="cs-row">
        <div class="cs-meta">
          <span class="cs-name">${c.name}</span>
          <span class="cs-count ${done?'done':''}">${c.count}/${c.target}${done?' ✓':''}</span>
        </div>
        <div class="cs-track"><div class="cs-fill ${done?'done':''}" style="width:${pct}%"></div></div>
      </div>`;
  }).join('');
}

function checkMilestone(total) {
  if (total === lastTotal) return;
  const prev = lastTotal;
  lastTotal  = total;
  if (prev < 0) return;
  const msg = MILESTONES[total];
  if (msg) showMilestone(msg);
}

function showMilestone(msg) {
  const el = document.getElementById('milestone-flash');
  el.textContent = msg;
  el.style.display = 'block';
  el.style.animation = 'none';
  void el.offsetWidth;
  el.style.animation = 'fadeSlide 3.5s ease forwards';
  setTimeout(() => el.style.display = 'none', 3600);
}

// ── TABS ──────────────────────────────────
function switchTab(name) {
  document.getElementById('tab-submit').style.display = name === 'submit' ? '' : 'none';
  document.getElementById('tab-seeds').style.display  = name === 'seeds'  ? '' : 'none';
  document.getElementById('tab-submit-btn').classList.toggle('active', name === 'submit');
  document.getElementById('tab-seeds-btn').classList.toggle('active',  name === 'seeds');
  if (name === 'seeds') loadSeeds();
}

// ── SEEDS VIEW ────────────────────────────
function populateSvCatSelect() {
  const sel = document.getElementById('sv-cat');
  CATS.forEach(([id, name]) => {
    const o = document.createElement('option');
    o.value = id; o.textContent = name;
    sel.appendChild(o);
  });
}

async function loadSeeds() {
  try {
    const r = await fetch('/api/seeds');
    const d = await r.json();
    allSeeds = d.seeds;
    renderSeeds();
  } catch(e) { document.getElementById('sv-list').innerHTML = '<div class="sv-empty">Could not load seeds.</div>'; }
}

function renderSeeds() {
  const catF    = document.getElementById('sv-cat').value;
  const nameF   = document.getElementById('sv-name').value.toLowerCase();
  const searchF = document.getElementById('sv-search').value.toLowerCase();
  const sortF   = document.getElementById('sv-sort').value;

  let filtered = allSeeds.filter(s => {
    if (catF   && s.category !== catF) return false;
    if (nameF  && !(s.submitter||'').toLowerCase().includes(nameF)) return false;
    if (searchF && !(s.instruction||'').toLowerCase().includes(searchF) &&
                   !(s.output||'').toLowerCase().includes(searchF)) return false;
    return true;
  });

  filtered = [...filtered].sort((a, b) => {
    if (sortF === 'oldest')    return (a.submitted_at||'').localeCompare(b.submitted_at||'');
    if (sortF === 'ambiguous') return (b.ambiguous ? 1 : 0) - (a.ambiguous ? 1 : 0);
    if (sortF === 'cat')       return (a.category||'').localeCompare(b.category||'');
    return (b.submitted_at||'').localeCompare(a.submitted_at||''); // newest
  });

  document.getElementById('sv-count').textContent = `${filtered.length} of ${allSeeds.length} seeds`;

  const el = document.getElementById('sv-list');
  if (!filtered.length) {
    el.innerHTML = '<div class="sv-empty">No seeds match the current filters.</div>';
    return;
  }

  el.innerHTML = filtered.map((s, i) => {
    const catName = CATS.find(c => c[0] === s.category)?.[1] || s.category || '—';
    const dt = s.submitted_at ? new Date(s.submitted_at) : null;
    const timeStr = dt ? dt.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) : '';
    const hasInput = s.input && s.input.trim();
    const uid = `sc-${i}`;
    return `
      <div class="seed-card">
        <div class="sc-header" onclick="toggleCard('${uid}')">
          <div class="sc-badges">
            <span class="sc-cat-badge">${catName}</span>
            ${s.ambiguous ? '<span class="sc-amb-badge">⭐</span>' : ''}
          </div>
          <span class="sc-submitter">${s.submitter || 'Anonymous'}</span>
          <span class="sc-instr">${(s.instruction||'').slice(0, 130)}${(s.instruction||'').length > 130 ? '…' : ''}</span>
          <span class="sc-time">${timeStr}</span>
          <span class="sc-chevron" id="${uid}-chev">▼</span>
        </div>
        <div class="sc-body ${hasInput ? '' : 'no-input'}" id="${uid}-body">
          <div class="sc-full">
            <div class="sc-field-lbl">Instruction</div>
            <div class="sc-field-val">${s.instruction || ''}</div>
          </div>
          ${hasInput ? `
          <div>
            <div class="sc-field-lbl">Input</div>
            <div class="sc-field-val">${s.input}</div>
          </div>` : ''}
          <div ${hasInput ? '' : 'class="sc-full"'}>
            <div class="sc-field-lbl">Output</div>
            <div class="sc-field-val">${s.output || ''}</div>
          </div>
          ${s.notes ? `<div class="sc-notes"><div class="sc-field-lbl">Notes</div><div class="sc-field-val">${s.notes}</div></div>` : ''}
        </div>
      </div>`;
  }).join('');
}

function toggleCard(uid) {
  const body = document.getElementById(uid + '-body');
  const chev = document.getElementById(uid + '-chev');
  const open = body.classList.toggle('open');
  chev.classList.toggle('open', open);
}

// ── TOAST ─────────────────────────────────
function showToast(msg, isErr) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className   = 'show' + (isErr ? ' err' : '');
  setTimeout(() => t.className = '', 3200);
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"Loaded {len(ABSTRACTS)} abstracts from corpus.")
    print(f"Seeds file: {SEEDS_FILE}")
    print("Starting on http://0.0.0.0:5050")
    app.run(host="0.0.0.0", port=8080, debug=False)
