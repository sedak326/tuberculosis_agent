#!/usr/bin/env python3
"""One-off audit of seeds.json for the specific issue classes flagged:
facts leaking into instruction, instruction/output mismatch, output copying
input verbatim, suspiciously short outputs. Not the full build/ pipeline —
just enough to get real counts across all 59 seeds."""
import json
import re

seeds = json.load(open("/home/skavlak/finetuning/data_prep/seeds.json"))

GENE_RE = re.compile(r"\bRv\d{4}[A-Za-z]?\b")
MEASURE_RE = re.compile(r"\d+(\.\d+)?\s*(mM|uM|µM|nM|ug/mL|µg/mL|mg/mL|°C|nm|rpm|%)")
FIRST_PERSON_RE = re.compile(r"\b(we performed|we screened|they confirmed|was purified|we have identified|we propose)\b", re.I)

MISMATCH_TRIGGER_RE = re.compile(r"interpret|critique|evaluate|flag|artifact|follow[- ]up|think through|propose|mechanis|steer|correct", re.I)
ALT_RE = re.compile(r"could|might|may|one possibility|alternatively|another explanation|two mechanisms|rather than", re.I)
ACTION_RE = re.compile(r"I'd|I would|next step|to test|to distinguish|suggest|recommend|it would help to know", re.I)


def lcs_len(a, b):
    # cheap approximate check: look for the longest shared substring via a sliding window
    # rather than full DP (fine for a one-off audit on 59 short records)
    best = 0
    a, b = a.lower(), b.lower()
    for i in range(0, len(a), 20):
        chunk = a[i:i+40]
        if len(chunk) < 40:
            continue
        if chunk in b:
            best = max(best, 40)
    return best


flags = {"instruction_contains_facts": [], "instruction_output_mismatch": [],
         "output_copies_input": [], "output_too_short": []}

for s in seeds:
    sid = s.get("id", "?")
    instr = s.get("instruction", "")
    inp = s.get("input", "")
    out = s.get("output", "")
    cat = s.get("category", "")

    if GENE_RE.search(instr) or MEASURE_RE.search(instr) or "Passage:" in instr or FIRST_PERSON_RE.search(instr):
        flags["instruction_contains_facts"].append(sid)

    if MISMATCH_TRIGGER_RE.search(instr):
        wc = len(out.split())
        has_alt = bool(ALT_RE.search(out))
        has_action = bool(ACTION_RE.search(out))
        if not (wc >= 60 and has_alt and has_action):
            flags["instruction_output_mismatch"].append(sid)

    if lcs_len(inp, out) >= 40:
        flags["output_copies_input"].append(sid)

    if cat != "structured_extraction" and len(out.split()) < 25:
        flags["output_too_short"].append(sid)

print(f"Total seeds: {len(seeds)}\n")
all_flagged = set()
for check, ids in flags.items():
    print(f"{check}: {len(ids)} flagged")
    for i in ids:
        print(f"   {i}")
    all_flagged.update(ids)
    print()

print(f"UNIQUE seeds flagged by at least one check: {len(all_flagged)} / {len(seeds)}")
