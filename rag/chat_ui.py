#!/usr/bin/env python3
"""
chat_ui.py — Simple web chat UI for the TB RAG agent.

Single-file Flask app (same convention as data_prep/mcq_review_dashboard.py):
inline HTML/JS, no separate template files, no new frontend framework.

Usage (on the cluster; tunnel the port with `ssh -L 8891:localhost:8891 <host>`):
    python chat_ui.py
    python chat_ui.py --host 0.0.0.0   # let labmates connect directly

Defaults to the same model as chat_rag.py (v5 SFT-Instruct-100k, the best
standalone condition so far). Generation is serialized behind a lock since
one 8B model instance can't safely serve two concurrent .generate() calls.
"""

import argparse
import threading

from flask import Flask, Response, jsonify, request

from rag_pipeline import load_generator, load_vectorstore, query_rag
from chat_rag import DEFAULT_MODEL, DEFAULT_ADAPTER

app = Flask(__name__)
GEN_LOCK = threading.Lock()
ARGS = None

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>TB Research Assistant</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{
    --bg:#0b1220; --panel:#111a2e; --panel2:#0e1626; --border:#1f2c47; --border-soft:#1a2440;
    --text:#e6ecf5; --text-dim:#8b9ac2; --text-faint:#5b6b8f;
    --accent:#2dd4bf; --accent-dim:#134e4a; --accent-glow:rgba(45,212,191,.18);
    --user:#1b2a4a; --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --sans:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
  }
  *{box-sizing:border-box}
  body{font-family:var(--sans);max-width:820px;margin:0 auto;padding:0 20px 120px;
       background:
         radial-gradient(circle at 15% 0%, rgba(45,212,191,.06), transparent 40%),
         radial-gradient(circle at 85% 10%, rgba(45,212,191,.04), transparent 35%),
         var(--bg);
       color:var(--text);min-height:100vh}
  header{padding:32px 0 20px;border-bottom:1px solid var(--border-soft);margin-bottom:24px}
  .tag{display:inline-flex;align-items:center;gap:6px;font:11px var(--mono);letter-spacing:.14em;
       text-transform:uppercase;color:var(--accent);background:var(--accent-glow);
       border:1px solid var(--accent-dim);border-radius:20px;padding:4px 10px;margin-bottom:12px}
  .tag::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--accent);
               box-shadow:0 0 8px var(--accent)}
  h1{font-size:21px;margin:0 0 6px;font-weight:600;letter-spacing:-.01em}
  .sub{color:var(--text-dim);font-size:13px;font-family:var(--mono)}
  .sub b{color:var(--text)}
  #chat{display:flex;flex-direction:column;gap:18px}
  .row{display:flex}
  .row.user{justify-content:flex-end}
  .row.bot{justify-content:flex-start}
  .msg{padding:14px 16px;border-radius:12px;max-width:85%;white-space:pre-wrap;line-height:1.55;font-size:14.5px}
  .user .msg{background:var(--user);border:1px solid #28406e;color:#dbe6ff}
  .bot .msg{background:var(--panel);border:1px solid var(--border);color:var(--text);
            box-shadow:0 1px 0 rgba(0,0,0,.2)}
  .label{font:10px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--text-faint);
         margin-bottom:6px}
  .bot .msg p{margin:0 0 10px}
  .bot .msg p:last-child{margin-bottom:0}
  .bot .msg h4,.bot .msg h5{margin:14px 0 6px;color:var(--accent);font-weight:600}
  .bot .msg h4:first-child,.bot .msg h5:first-child{margin-top:0}
  .bot .msg h4{font-size:15px}
  .bot .msg h5{font-size:13.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--text-dim)}
  .bot .msg ul{margin:0 0 10px;padding-left:20px}
  .bot .msg ul:last-child{margin-bottom:0}
  .bot .msg li{margin:3px 0}
  .bot .msg strong{color:var(--accent);font-weight:600}
  .sources{margin-top:12px;border-top:1px dashed var(--border);padding-top:10px}
  .sources summary{cursor:pointer;color:var(--accent);font:12px var(--mono);letter-spacing:.03em;
                    list-style:none;outline:none}
  .sources summary::before{content:"▸ ";display:inline-block}
  .sources[open] summary::before{content:"▾ "}
  .src{margin:8px 0 0;padding:9px 11px;background:var(--panel2);border:1px solid var(--border-soft);
       border-left:2px solid var(--accent-dim);border-radius:6px;font-size:12.5px;color:var(--text-dim)}
  .src b{color:var(--accent);font-family:var(--mono);font-weight:500;letter-spacing:.02em}
  .src .pg{color:var(--text-faint);font-family:var(--mono);font-size:11px}
  #bar{position:fixed;bottom:0;left:0;right:0;background:linear-gradient(to top, var(--bg) 60%, transparent);
       padding:28px 20px 22px}
  #barInner{max-width:820px;margin:0 auto;display:flex;gap:10px;background:var(--panel);
            border:1px solid var(--border);border-radius:12px;padding:6px 6px 6px 16px;
            box-shadow:0 8px 30px rgba(0,0,0,.45)}
  #barInner:focus-within{border-color:var(--accent-dim);box-shadow:0 0 0 3px var(--accent-glow)}
  #q{flex:1;padding:12px 4px;border:none;background:transparent;color:var(--text);font-size:14.5px;outline:none;
     font-family:var(--sans)}
  #q::placeholder{color:var(--text-faint)}
  #send{padding:10px 20px;border:none;border-radius:8px;background:var(--accent);color:#06201c;
        cursor:pointer;font-size:13.5px;font-weight:600;letter-spacing:.02em}
  #send:disabled{background:var(--border);color:var(--text-faint);cursor:default}
  #send:not(:disabled):hover{background:#5eead4}
  .dots{display:inline-flex;gap:4px;align-items:center}
  .dots span{width:5px;height:5px;border-radius:50%;background:var(--text-faint);
             animation:pulse 1.2s infinite ease-in-out}
  .dots span:nth-child(2){animation-delay:.15s}
  .dots span:nth-child(3){animation-delay:.3s}
  @keyframes pulse{0%,80%,100%{opacity:.25;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
  .err{color:#f87171}
  #chat:empty::after{content:"Ask about TB pathogenesis, drug resistance, diagnostics, or anything in the indexed literature.";
    color:var(--text-faint);font-size:13.5px;font-family:var(--mono);display:block;padding:40px 0;text-align:center}
</style></head>
<body>
  <header>
    <div class="tag">retrieval-augmented · online</div>
    <h1>TB Research Assistant</h1>
    <div class="sub">corpus: <b>__N_CHUNKS__</b> indexed passages &nbsp;·&nbsp; model: <b>__MODEL_LABEL__</b></div>
  </header>
  <div id="chat"></div>
  <div id="bar"><div id="barInner">
    <input id="q" placeholder="Ask a TB research question..." autocomplete="off">
    <button id="send">Ask</button>
  </div></div>
<script>
const chat = document.getElementById('chat'), q = document.getElementById('q'), send = document.getElementById('send');
function addRow(role, innerHTML){
  const row = document.createElement('div'); row.className = 'row ' + role;
  const bubble = document.createElement('div'); bubble.className = 'msg';
  bubble.innerHTML = innerHTML;
  row.appendChild(bubble); chat.appendChild(row);
  window.scrollTo(0, document.body.scrollHeight);
  return bubble;
}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
// Minimal Markdown-subset renderer for model answers: escapes first, then
// turns the handful of constructs the system prompt asks for (##/### headers,
// - bullets, **bold**, blank-line paragraphs) into real HTML. Source-panel
// snippets stay plain-escaped (esc) - they're raw corpus text, not model output.
function renderMd(raw){
  let s = esc(raw).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  const lines = s.split('\n');
  let html = '', para = [], list = [];
  const flushPara = () => { if(para.length){ html += '<p>' + para.join(' ') + '</p>'; para = []; } };
  const flushList = () => { if(list.length){ html += '<ul>' + list.map(li => '<li>' + li + '</li>').join('') + '</ul>'; list = []; } };
  for(const raw2 of lines){
    const line = raw2.trim();
    if(!line){ flushPara(); flushList(); continue; }
    const h = line.match(/^(#{2,3})\s+(.*)$/);
    if(h){ flushPara(); flushList(); const tag = h[1].length === 2 ? 'h4' : 'h5'; html += '<' + tag + '>' + h[2] + '</' + tag + '>'; continue; }
    const li = line.match(/^[-*]\s+(.*)$/);
    if(li){ flushPara(); list.push(li[1]); continue; }
    flushList();
    para.push(line);
  }
  flushPara(); flushList();
  return html || s;
}
async function ask(){
  const question = q.value.trim();
  if(!question) return;
  q.value = '';
  send.disabled = true;
  addRow('user', esc(question));
  const loading = addRow('bot', '<span class="label">assistant</span><div class="dots"><span></span><span></span><span></span></div>');
  try{
    const r = await fetch('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'},
                                        body: JSON.stringify({question})});
    const data = await r.json();
    if(!r.ok){ loading.innerHTML = '<span class="label">assistant</span><span class="err">Error: ' + esc(data.error || r.statusText) + '</span>'; return; }
    let html = '<span class="label">assistant</span>' + renderMd(data.answer);
    if(data.sources && data.sources.length){
      html += '<details class="sources"><summary>' + data.sources.length + ' retrieved passages</summary>' +
        data.sources.map(s => '<div class="src"><b>' + esc(s.document_name) + '</b> <span class="pg">p.' + esc(String(s.page)) +
          '</span><br>' + esc(s.snippet) + '…</div>').join('') + '</details>';
    }
    loading.innerHTML = html;
  }catch(e){
    loading.innerHTML = '<span class="label">assistant</span><span class="err">Error: ' + esc(String(e)) + '</span>';
  }finally{
    send.disabled = false;
    q.focus();
  }
}
send.onclick = ask;
q.addEventListener('keydown', e => { if(e.key === 'Enter') ask(); });
q.focus();
</script>
</body></html>"""


def _model_label() -> str:
    base = ARGS.model.split("/")[-1]
    if not ARGS.adapter:
        return base
    adapter_name = ARGS.adapter.rstrip("/").split("/")[-2]  # .../<checkpoint-dir>/final -> <checkpoint-dir>
    return f"{base} + {adapter_name} (LoRA)"


@app.route("/")
def index():
    vs = load_vectorstore()
    n_chunks = vs._collection.count()
    page = PAGE.replace("__N_CHUNKS__", f"{n_chunks:,}").replace("__MODEL_LABEL__", _model_label())
    return Response(page, mimetype="text/html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "empty question"}), 400

    with GEN_LOCK:
        answer, retrieved = query_rag(
            question,
            model_path=ARGS.model,
            tokenizer_path=ARGS.tokenizer,
            adapter_path=ARGS.adapter,
            top_k=ARGS.top_k,
            max_new_tokens=ARGS.max_new_tokens,
        )

    sources = [
        {
            "document_name": d.metadata.get("document_name", "?"),
            "page": d.metadata.get("page_range") or d.metadata.get("page_number") or "?",
            "snippet": d.page_content[:200].replace("\n", " "),
        }
        for d in retrieved
    ]
    return jsonify({"answer": answer, "sources": sources})


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
    ARGS = parser.parse_args()

    print("Loading vectorstore and generator (this happens once at startup)...")
    load_vectorstore()
    load_generator(ARGS.model, ARGS.tokenizer, ARGS.adapter)
    if ARGS.adapter:
        print(f"Adapter loaded on top of {ARGS.model}: {ARGS.adapter}")
    else:
        print(f"No adapter - running base {ARGS.model} with no fine-tuning.")
    print(f"Ready. Visit http://{ARGS.host}:{ARGS.port} (tunnel with: ssh -L {ARGS.port}:localhost:{ARGS.port} <host>)")

    app.run(host=ARGS.host, port=ARGS.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
