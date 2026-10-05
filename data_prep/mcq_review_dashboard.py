#!/usr/bin/env python3
"""
mcq_review_dashboard.py — Interactively review the 800 eval MCQs and see how each model did on them.

Shows, per question: the choices + gold key + gold explanation, the best-matching source passage
from the held-out corpus, and every evaluated condition's answer (right/wrong, which choice it
picked, GPT-4o judge scores, raw output). Reviewers tag each question (good / ambiguous / wrong key
/ ...) and the tags are saved; the Stats tab re-scores every condition on the questions reviewers
did not flag.

Usage (on the cluster; tunnel the port with `ssh -L 8890:localhost:8890 <host>`):
    python mcq_review_dashboard.py \
        --mcq         /uss/skavlak/version_3/mcq_eval.jsonl \
        --corpus      /uss/skavlak/version_3/corpus_v3_eval.jsonl \
        --results-dir /uss/skavlak/version_3/eval_results \
        --port 8890            # add --host 0.0.0.0 to let labmates connect directly

Conditions are every `<name>_scored.jsonl` in --results-dir; the matching raw model output is read
from `<name>.jsonl` if present. If a run used --randomize-answers, the shuffled choice order is
replayed from --seed (default 42) and verified per question against the recorded correct letter;
questions that can't be verified show letters only.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict

from flask import Flask, Response, jsonify, request

app = Flask(__name__)
STATE: dict = {}

VERDICTS = [
    ("good", "Good"),
    ("ambiguous", "Ambiguous"),
    ("wrong_key", "Wrong key"),
    ("bad_distractor", "Bad distractor"),
    ("passage_specific", "Needs the passage"),
    ("other", "Other"),
]
LETTERS = ["A", "B", "C", "D"]
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9-]{3,}")


# --------------------------------------------------------------------------- loading

def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def strip_header(text: str) -> str:
    return text[text.index("\n\n") + 2:] if "\n\n" in text else text


def shuffle_choices(mcq: dict, rng: random.Random) -> tuple[dict, str]:
    """Mirror of evaluate_emcqa.py's shuffle_choices."""
    key_text = mcq["choices"][mcq["correct"]]
    texts = [mcq["choices"][l] for l in LETTERS]
    rng.shuffle(texts)
    shuffled = dict(zip(LETTERS, texts))
    return shuffled, next(l for l, t in shuffled.items() if t == key_text)


def load_chunks(corpus_path: str, docs: set[str]) -> dict[str, list[dict]]:
    by_doc = defaultdict(list)
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c.get("document_name") in docs and c.get("content_type", "text") == "text":
                body = strip_header(c.get("text", ""))
                by_doc[c["document_name"]].append({
                    "id": c.get("id", ""), "section": c.get("section_title", ""),
                    "body": body, "tok": tokens(body),
                })
    return by_doc


def find_passages(mcq: dict, by_doc: dict, k: int = 3) -> list[dict]:
    """MCQs only store document_name (generate_mcq.py's chunk-id fallback), so rank that paper's
    chunks by vocabulary overlap with the question, key and gold explanation."""
    cands = by_doc.get(mcq.get("document_name", ""), [])
    exact = [c for c in cands if c["id"] and c["id"] == mcq.get("source_chunk_id")]
    if exact:
        return [{"id": c["id"], "section": c["section"], "body": c["body"], "score": 1.0, "exact": True} for c in exact]
    q = tokens(mcq["question"] + " " + mcq["choices"][mcq["correct"]] + " " + mcq.get("explanation", ""))
    scored = []
    for c in cands:
        if c["tok"]:
            scored.append((len(q & c["tok"]) / math.sqrt(len(c["tok"])), c))
    scored.sort(key=lambda x: -x[0])
    return [{"id": c["id"], "section": c["section"], "body": c["body"], "score": round(s, 2), "exact": False}
            for s, c in scored[:k]]


def load_conditions(results_dir: str, mcqs: list[dict], seed: int) -> dict:
    by_id = {m["id"]: m for m in mcqs}
    conds = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_scored.jsonl"))):
        name = os.path.basename(path)[: -len("_scored.jsonl")]
        scored = {r["id"]: r for r in read_jsonl(path)}
        raw_path = os.path.join(results_dir, name + ".jsonl")
        raw = {r["id"]: r for r in read_jsonl(raw_path)} if os.path.exists(raw_path) else {}
        randomized = any(r.get("correct_presented") != by_id[i]["correct"] for i, r in raw.items() if i in by_id) \
            or any(r.get("correct_presented") != by_id[i]["correct"] for i, r in scored.items() if i in by_id)
        presented = {}
        if randomized:
            rng = random.Random(seed)
            for m in mcqs:
                choices, new_correct = shuffle_choices(m, rng)
                rec = scored.get(m["id"]) or raw.get(m["id"])
                if rec and rec.get("correct_presented") == new_correct:
                    presented[m["id"]] = choices   # verified
        else:
            presented = {m["id"]: m["choices"] for m in mcqs}
        conds[name] = {"scored": scored, "raw": raw, "randomized": randomized, "presented": presented}
        ok = sum(1 for i in scored if i in presented)
        print(f"  condition {name}: {len(scored)} scored, randomized={randomized}, choice order verified for {ok}")
    return conds


def build_state(args) -> None:
    mcqs = read_jsonl(args.mcq)
    print(f"Loaded {len(mcqs)} MCQs")
    by_doc = load_chunks(args.corpus, {m.get("document_name", "") for m in mcqs}) if os.path.exists(args.corpus) else {}
    print(f"Loaded chunks for {len(by_doc)} papers")
    conds = load_conditions(args.results_dir, mcqs, args.seed)
    STATE.update(mcqs=mcqs, by_id={m["id"]: m for m in mcqs}, by_doc=by_doc, conds=conds,
                 reviews_path=args.reviews, passage_cache={})


# --------------------------------------------------------------------------- per-question views

def condition_view(cname: str, mcq: dict) -> dict | None:
    c = STATE["conds"][cname]
    s = c["scored"].get(mcq["id"])
    if s is None:
        return None
    presented = c["presented"].get(mcq["id"])
    pred = s.get("predicted")
    pred_text = presented.get(pred) if (presented and pred) else None
    orig_letter = next((l for l, t in mcq["choices"].items() if t == pred_text), None) if pred_text else None
    if pred is None:
        orig_letter = None
    return {
        "name": cname, "predicted": pred, "correct_presented": s.get("correct_presented"),
        "is_correct": bool(s.get("is_correct")), "pred_text": pred_text, "pred_orig_letter": orig_letter,
        "order_verified": presented is not None,
        "factual_accuracy": s.get("factual_accuracy"), "coherence": s.get("coherence"),
        "naturalness": s.get("naturalness"), "completeness": s.get("completeness"),
        "raw_output": (c["raw"].get(mcq["id"]) or {}).get("raw_output"),
        "choices": presented,
    }


def summary_row(mcq: dict) -> dict:
    views = [v for v in (condition_view(n, mcq) for n in STATE["conds"]) if v]
    n_correct = sum(v["is_correct"] for v in views)
    picks = Counter(v["pred_orig_letter"] for v in views if v["pred_orig_letter"])
    top, top_n = (picks.most_common(1)[0] if picks else (None, 0))
    key_suspect = bool(top and top != mcq["correct"] and top_n >= max(3, 0.6 * sum(picks.values())))
    return {
        "id": mcq["id"], "level": mcq["cognitive_level"], "topic": mcq["topic"], "doc": mcq.get("document_name", ""),
        "question": mcq["question"], "n_cond": len(views), "n_correct": n_correct,
        "right": [v["name"] for v in views if v["is_correct"]],
        "wrong": [v["name"] for v in views if not v["is_correct"]],
        "key_suspect": key_suspect, "suspect_letter": top if key_suspect else None,
    }


# --------------------------------------------------------------------------- reviews

def load_reviews() -> dict:
    p = STATE["reviews_path"]
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def locked_update(fn) -> dict:
    p = STATE["reviews_path"]
    with open(p + ".lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            data = load_reviews()
            fn(data)
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, p)
            return data
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def review_status(revs: list[dict]) -> str:
    if not revs:
        return "unreviewed"
    return "good" if all(r["verdict"] == "good" for r in revs) else "flagged"


# --------------------------------------------------------------------------- routes

@app.route("/")
def index():
    return Response(PAGE.replace("__VERDICTS__", json.dumps(VERDICTS)), mimetype="text/html")


@app.route("/api/list")
def api_list():
    reviews = load_reviews()
    rows = []
    for m in STATE["mcqs"]:
        r = summary_row(m)
        r["review"] = review_status(list(reviews.get(m["id"], {}).values()))
        rows.append(r)
    return jsonify({
        "rows": rows, "conditions": list(STATE["conds"]),
        "levels": sorted({m["cognitive_level"] for m in STATE["mcqs"]}),
        "topics": sorted({m["topic"] for m in STATE["mcqs"]}),
    })


@app.route("/api/question/<qid>")
def api_question(qid):
    m = STATE["by_id"].get(qid)
    if m is None:
        return jsonify({"error": "not found"}), 404
    if qid not in STATE["passage_cache"]:
        STATE["passage_cache"][qid] = find_passages(m, STATE["by_doc"])
    views = [v for v in (condition_view(n, m) for n in STATE["conds"]) if v]
    return jsonify({
        "mcq": {k: m[k] for k in ("id", "question", "choices", "correct", "explanation", "cognitive_level", "topic")}
                | {"doc": m.get("document_name", "")},
        "passages": STATE["passage_cache"][qid], "conditions": views,
        "reviews": load_reviews().get(qid, {}),
    })


@app.route("/api/review", methods=["POST"])
def api_review():
    b = request.get_json(force=True)
    qid, reviewer, verdict = b.get("id"), (b.get("reviewer") or "").strip(), b.get("verdict")
    if qid not in STATE["by_id"] or not reviewer or verdict not in dict(VERDICTS):
        return jsonify({"error": "need valid id, reviewer and verdict"}), 400
    entry = {"verdict": verdict, "notes": (b.get("notes") or "")[:2000], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    locked_update(lambda d: d.setdefault(qid, {}).__setitem__(reviewer, entry))
    return jsonify({"ok": True})


@app.route("/api/stats")
def api_stats():
    reviews = load_reviews()
    flagged = {i for i, rs in reviews.items() if review_status(list(rs.values())) == "flagged"}
    reviewed = {i for i, rs in reviews.items() if rs}
    out = []
    for name, c in STATE["conds"].items():
        def acc(pred):
            ids = [i for i in c["scored"] if i in STATE["by_id"] and pred(STATE["by_id"][i])]
            return (sum(1 for i in ids if c["scored"][i].get("is_correct")) / len(ids), len(ids)) if ids else (None, 0)
        row = {"name": name, "overall": acc(lambda m: True), "excluding_flagged": acc(lambda m: m["id"] not in flagged)}
        for lv in sorted({m["cognitive_level"] for m in STATE["mcqs"]}):
            row[f"level:{lv}"] = acc(lambda m, lv=lv: m["cognitive_level"] == lv)
        no_ans = sum(1 for r in c["scored"].values() if r.get("predicted") is None)
        row["no_answer"] = no_ans
        scores = {k: [r[k] for r in c["scored"].values() if isinstance(r.get(k), (int, float))]
                  for k in ("factual_accuracy", "coherence", "naturalness", "completeness")}
        row["judge"] = {k: (sum(v) / len(v) if v else None) for k, v in scores.items()}
        out.append(row)
    verdicts = Counter(r["verdict"] for rs in reviews.values() for r in rs.values())
    return jsonify({"conditions": out, "n_reviewed": len(reviewed), "n_flagged": len(flagged),
                    "verdicts": dict(verdicts), "n_total": len(STATE["mcqs"])})


@app.route("/api/export")
def api_export():
    return Response(json.dumps(load_reviews(), indent=1, ensure_ascii=False), mimetype="application/json",
                    headers={"Content-Disposition": "attachment; filename=mcq_reviews.json"})


# --------------------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MCQ Review</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e2dc;--ok:#1a7f4b;--bad:#b3261e;--warn:#b26a00;--acc:#2456c6}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecece8;--muted:#9a9a93;--line:#34342f;--ok:#4cc38a;--bad:#ff7b72;--warn:#e3a53c;--acc:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:5}
header h1{font-size:16px;margin:0}header nav button{margin-right:4px}
button,select,input,textarea{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
#app{display:grid;grid-template-columns:340px 1fr;height:calc(100vh - 53px)}
#side{border-right:1px solid var(--line);overflow:auto;background:var(--card)}
#filters{padding:8px;display:grid;grid-template-columns:1fr 1fr;gap:6px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--card)}
#filters .full{grid-column:1/3}
.row{padding:7px 10px;border-bottom:1px solid var(--line);cursor:pointer}.row:hover{background:var(--bg)}.row.sel{background:var(--bg);box-shadow:inset 3px 0 var(--acc)}
.row .q{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.meta{color:var(--muted);font-size:12px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
#main{overflow:auto;padding:16px 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:12px}
.choice{padding:5px 8px;border-radius:6px;margin:3px 0;border:1px solid transparent}
.choice.key{border-color:var(--ok);background:color-mix(in srgb,var(--ok) 12%,transparent)}
.pill{display:inline-block;padding:0 7px;border-radius:10px;font-size:12px;border:1px solid var(--line);margin-right:4px}
.pill.ok{color:var(--ok);border-color:var(--ok)}.pill.bad{color:var(--bad);border-color:var(--bad)}.pill.warn{color:var(--warn);border-color:var(--warn)}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
td.ok{color:var(--ok);font-weight:600}td.bad{color:var(--bad);font-weight:600}
pre{white-space:pre-wrap;margin:6px 0 0;font:13px/1.45 ui-monospace,monospace;background:var(--bg);padding:8px;border-radius:6px}
.passage{border-left:3px solid var(--line);padding-left:10px;white-space:pre-wrap;max-height:260px;overflow:auto}
#stats{display:none;padding:16px 20px;overflow:auto}
kbd{border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:11px}
@media(max-width:800px){#app{grid-template-columns:1fr;height:auto}#side{max-height:40vh}}
</style></head><body>
<header><h1>MCQ review</h1>
<nav><button id="tabQ" class="on">Questions</button><button id="tabS">Stats</button></nav>
<span class="meta" id="count"></span>
<span style="margin-left:auto" class="meta">Reviewer <input id="reviewer" size="10" placeholder="your name"></span>
<a class="meta" href="/api/export">export reviews</a></header>
<div id="app">
 <div id="side"><div id="filters">
  <select id="fLevel" class="full"><option value="">all cognitive levels</option></select>
  <select id="fTopic" class="full"><option value="">all topics</option></select>
  <select id="fResult" class="full">
   <option value="">any model result</option><option value="allright">all models right</option>
   <option value="allwrong">all models wrong</option><option value="mixed">models disagree</option>
   <option value="suspect">key looks suspect (models agree on a non-key answer)</option></select>
  <select id="fWrong"><option value="">wrong in…</option></select>
  <select id="fRight"><option value="">right in…</option></select>
  <select id="fRev" class="full"><option value="">any review status</option><option value="unreviewed">unreviewed</option>
   <option value="flagged">flagged</option><option value="good">marked good</option></select>
  <input id="fText" class="full" placeholder="search question text / paper">
  <select id="fSort" class="full"><option value="id">sort: id</option><option value="hard">sort: hardest first</option>
   <option value="easy">sort: easiest first</option></select>
 </div><div id="list"></div></div>
 <div id="main"><p class="meta">Select a question. <kbd>j</kbd>/<kbd>k</kbd> next/previous, <kbd>1</kbd>–<kbd>6</kbd> set verdict.</p></div>
 <div id="stats"></div>
</div>
<script>
const VERDICTS = __VERDICTS__;
let ROWS=[], VIEW=[], CUR=null, DATA={};
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const store={get(k){try{return localStorage.getItem(k)}catch(e){return null}},set(k,v){try{localStorage.setItem(k,v)}catch(e){}}};
$('reviewer').value=store.get('reviewer')||'';$('reviewer').onchange=e=>store.set('reviewer',e.target.value.trim());

async function init(){
  DATA=await (await fetch('/api/list')).json(); ROWS=DATA.rows;
  const fill=(id,vals)=>vals.forEach(v=>$(id).add(new Option(v,v)));
  fill('fLevel',DATA.levels);fill('fTopic',DATA.topics);fill('fWrong',DATA.conditions);fill('fRight',DATA.conditions);
  ['fLevel','fTopic','fResult','fWrong','fRight','fRev','fText','fSort'].forEach(id=>$(id).oninput=render);
  render();
}
function render(){
  const f=id=>$(id).value, t=f('fText').toLowerCase();
  VIEW=ROWS.filter(r=>(!f('fLevel')||r.level==f('fLevel'))&&(!f('fTopic')||r.topic==f('fTopic'))
   &&(!f('fWrong')||r.wrong.includes(f('fWrong')))&&(!f('fRight')||r.right.includes(f('fRight')))
   &&(!f('fRev')||r.review==f('fRev'))&&(!t||(r.question+' '+r.doc).toLowerCase().includes(t))
   &&(!f('fResult')||(f('fResult')=='allright'&&r.n_cond&&r.n_correct==r.n_cond)||(f('fResult')=='allwrong'&&r.n_cond&&r.n_correct==0)
     ||(f('fResult')=='mixed'&&r.n_correct>0&&r.n_correct<r.n_cond)||(f('fResult')=='suspect'&&r.key_suspect)));
  const s=f('fSort'), frac=r=>r.n_cond?r.n_correct/r.n_cond:0;
  if(s=='hard')VIEW.sort((a,b)=>frac(a)-frac(b));else if(s=='easy')VIEW.sort((a,b)=>frac(b)-frac(a));
  $('count').textContent=VIEW.length+' / '+ROWS.length+' questions';
  $('list').innerHTML=VIEW.map(r=>{
    const col=r.review=='good'?'var(--ok)':r.review=='flagged'?'var(--warn)':'var(--line)';
    return `<div class="row ${r.id==CUR?'sel':''}" data-id="${esc(r.id)}"><div class="q">${esc(r.question)}</div>
     <div class="meta"><span class="dot" style="background:${col}"></span>${esc(r.level)} · ${esc(r.topic)} ·
     ${r.n_correct}/${r.n_cond} models right${r.key_suspect?' · <b style="color:var(--bad)">key?</b>':''}</div></div>`}).join('');
  document.querySelectorAll('.row').forEach(el=>el.onclick=()=>open(el.dataset.id));
}
async function open(id){
  CUR=id; render();
  const d=await (await fetch('/api/question/'+encodeURIComponent(id))).json(); const m=d.mcq;
  const choices=['A','B','C','D'].map(l=>`<div class="choice ${l==m.correct?'key':''}"><b>${l}</b> ${esc(m.choices[l])}${l==m.correct?' <span class="pill ok">key</span>':''}</div>`).join('');
  const passages=d.passages.length?d.passages.map((p,i)=>`<details ${i==0?'open':''}><summary class="meta">${p.exact?'source chunk':'candidate passage (match score '+p.score+')'} · ${esc(p.section)} · ${esc(p.id)}</summary><div class="passage">${esc(p.body)}</div></details>`).join('')
     :'<span class="meta">No passage found (corpus file missing, or paper not in corpus).</span>';
  const rows=d.conditions.map((c,i)=>{
    const pick=c.predicted?`${esc(c.predicted)}${c.pred_text?') '+esc(c.pred_text):''}`:'<i>no answer</i>';
    return `<tr><td>${esc(c.name)}</td><td class="${c.is_correct?'ok':'bad'}">${c.is_correct?'right':'wrong'}</td><td>${pick}${c.order_verified?'':' <span class="pill warn" title="shuffled choice order could not be verified">letter only</span>'}</td>
     <td class="meta">F${c.factual_accuracy??'–'} C${c.coherence??'–'} N${c.naturalness??'–'} Cm${c.completeness??'–'}</td>
     <td>${c.raw_output?`<details><summary class="meta">raw output</summary><pre>${esc(c.raw_output)}</pre></details>`:''}</td></tr>`}).join('');
  const mine=(d.reviews[$('reviewer').value.trim()]||{});
  const others=Object.entries(d.reviews).filter(([k])=>k!=$('reviewer').value.trim()).map(([k,v])=>`<div class="meta"><b>${esc(k)}</b>: ${esc(v.verdict)} ${esc(v.notes)}</div>`).join('');
  $('main').innerHTML=`
   <div class="card"><div class="meta">${esc(m.id)} · ${esc(m.cognitive_level)} · ${esc(m.topic)} · ${esc(m.doc)}</div>
    <p style="font-size:15px"><b>${esc(m.question)}</b></p>${choices}</div>
   <div class="card"><b>Gold explanation</b><p>${esc(m.explanation)}</p></div>
   <div class="card"><b>Source passage</b>${passages}</div>
   <div class="card"><b>Model results</b>${rows?`<table><tr><th>condition</th><th></th><th>picked</th><th>judge</th><th></th></tr>${rows}</table>`:'<p class="meta">No results loaded.</p>'}</div>
   <div class="card"><b>Your review</b><div style="margin:8px 0" id="vbtns">${VERDICTS.map((v,i)=>`<button data-v="${v[0]}" class="${mine.verdict==v[0]?'on':''}">${i+1} ${v[1]}</button>`).join(' ')}</div>
    <textarea id="notes" rows="2" style="width:100%" placeholder="notes (what's wrong / which passage sentence)">${esc(mine.notes||'')}</textarea>
    <div style="margin-top:6px"><button id="save">Save &amp; next</button> <span id="saved" class="meta"></span></div>${others}</div>`;
  document.querySelectorAll('#vbtns button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#vbtns button').forEach(x=>x.classList.remove('on'));b.classList.add('on')});
  $('save').onclick=save;
}
async function save(){
  const reviewer=$('reviewer').value.trim(), btn=document.querySelector('#vbtns button.on');
  if(!reviewer){$('saved').textContent='enter your name (top right)';return}
  if(!btn){$('saved').textContent='pick a verdict';return}
  const r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:CUR,reviewer,verdict:btn.dataset.v,notes:$('notes').value})});
  if(!r.ok){$('saved').textContent='save failed';return}
  const row=ROWS.find(x=>x.id==CUR); row.review=btn.dataset.v=='good'?'good':'flagged'; render(); step(1);
}
function step(d){const i=VIEW.findIndex(r=>r.id==CUR);const n=VIEW[Math.min(VIEW.length-1,Math.max(0,i+d))];if(n)open(n.id)}
document.addEventListener('keydown',e=>{
  if(['TEXTAREA','INPUT','SELECT'].includes(e.target.tagName)||e.metaKey||e.ctrlKey)return;
  if(e.key=='j')step(1);else if(e.key=='k')step(-1);
  else if(/^[1-6]$/.test(e.key)){const b=document.querySelectorAll('#vbtns button')[+e.key-1];if(b)b.click()}
});
async function stats(){
  const d=await (await fetch('/api/stats')).json(), pct=x=>x[0]==null?'–':(100*x[0]).toFixed(1)+'% <span class="meta">('+x[1]+')</span>';
  const levels=Object.keys(d.conditions[0]||{}).filter(k=>k.startsWith('level:'));
  $('stats').innerHTML=`<p>${d.n_reviewed} / ${d.n_total} reviewed · ${d.n_flagged} flagged · ${Object.entries(d.verdicts).map(([k,v])=>k+': '+v).join(', ')||'no reviews yet'}</p>
   <table><tr><th>condition</th><th>accuracy</th><th>excluding flagged</th>${levels.map(l=>`<th>${esc(l.slice(6))}</th>`).join('')}<th>no answer</th><th>judge F/C/N/Cm</th></tr>
   ${d.conditions.map(c=>`<tr><td>${esc(c.name)}</td><td>${pct(c.overall)}</td><td>${pct(c.excluding_flagged)}</td>${levels.map(l=>`<td>${pct(c[l])}</td>`).join('')}
    <td>${c.no_answer}</td><td class="meta">${['factual_accuracy','coherence','naturalness','completeness'].map(k=>c.judge[k]==null?'–':c.judge[k].toFixed(2)).join(' / ')}</td></tr>`).join('')}</table>
   <p class="meta">Accuracy is over each condition's scored questions (count in brackets). "Excluding flagged" drops every question any reviewer marked as not good.</p>`;
}
function tab(s){$('app').children[0].style.display=$('app').children[1].style.display=s?'none':'';$('stats').style.display=s?'block':'none';
  $('tabQ').classList.toggle('on',!s);$('tabS').classList.toggle('on',s);if(s)stats()}
$('tabQ').onclick=()=>tab(false);$('tabS').onclick=()=>tab(true);
init();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcq", default="/uss/skavlak/version_3/mcq_eval.jsonl")
    ap.add_argument("--corpus", default="/uss/skavlak/version_3/corpus_v3_eval.jsonl")
    ap.add_argument("--results-dir", default="/uss/skavlak/version_3/eval_results")
    ap.add_argument("--reviews", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcq_reviews.json"))
    ap.add_argument("--seed", type=int, default=42, help="--seed used by evaluate_emcqa.py for randomized runs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8890)
    args = ap.parse_args()
    build_state(args)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
