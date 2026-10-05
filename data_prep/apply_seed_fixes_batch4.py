#!/usr/bin/env python3
"""Fourth fix pass on seeds.json. Addresses what survived batch 3: a
grammatically broken clause (adrian-rc-007), a mechanism/strategy
contradiction (adrian-rc-004), a near-duplicate correction-preamble opener
(gen-rc-007), residual editorial voice (gen-hyp-004), an
instruction/output mismatch where no misconception exists to correct
(hack-...115641073), a dense three-clause sentence (gen-rc-003), and a
content-diversity duplication flagged three times over and never resolved
(adrian-rc-006, cyclopropane synthases already covered by adrian-rc-005).
The gen-rc-009 PMID (28096490) was checked against PubMed directly and
confirmed correct - no change needed there.
"""
import json

PATH = "/home/skavlak/finetuning/data_prep/seeds.json"
seeds = json.load(open(PATH))
by_id = {s["id"]: s for s in seeds}

REQUIRED_IDS = [
    "adrian-rc-007", "adrian-rc-004", "gen-rc-007", "gen-hyp-004",
    "hack-20260602115641073", "gen-rc-003", "adrian-rc-006",
]
missing = [i for i in REQUIRED_IDS if i not in by_id]
if missing:
    raise SystemExit(f"Missing IDs, aborting: {missing}")

# 1. adrian-rc-007 — fix the ungrammatical clause.
old_out = by_id["adrian-rc-007"]["output"]
old_frag = (
    "The structure showed LpdA is missing one of the pair of catalytic "
    "cysteines, plus the His-Glu acid/base pair — three residues gone, "
    "not a full second cysteine that real lipoamide dehydrogenases need, "
    "so it physically can't run that reaction."
)
assert old_frag in old_out, "adrian-rc-007 fragment not found"
new_frag = (
    "The structure showed LpdA is missing one of the two catalytic "
    "cysteines and the His-Glu acid/base pair that real lipoamide "
    "dehydrogenases need, so it physically can't run that reaction."
)
by_id["adrian-rc-007"]["output"] = old_out.replace(old_frag, new_frag)
by_id["adrian-rc-007"]["notes"] = (
    by_id["adrian-rc-007"]["notes"]
    + " Cleaned up an ungrammatical clause in the grounding-fix sentence."
)

# 2. adrian-rc-004 — justify why locking the enzyme off is the coherent
# drug strategy (Mtb encounters acidified phagosomes; the whole reason
# adenylyl cyclases are of interest as TB targets is their link to cAMP
# signaling during infection).
old_out = by_id["adrian-rc-004"]["output"]
old_frag = (
    "Conceptually, instead of blocking the conserved active site, you "
    "might design compounds that engage that N-terminal domain to lock in "
    "the autoinhibited state — preventing the pH-triggered "
    "de-repression rather than trying to out-compete substrate at the "
    "catalytic site."
)
assert old_frag in old_out, "adrian-rc-004 fragment not found"
new_frag = (
    "Mtb's own adenylyl cyclases are of interest as targets because their "
    "cAMP output has been implicated in manipulating host macrophage "
    "signaling during infection, and Mtb spends much of that infection "
    "sitting in acidified phagosomes — exactly the condition that "
    "switches Rv1264 on. So conceptually, instead of blocking the "
    "conserved active site, you might design compounds that engage that "
    "N-terminal domain to lock in the autoinhibited state, dampening the "
    "acid-triggered cAMP burst rather than trying to out-compete substrate "
    "at the catalytic site (that connection between Rv1264 specifically "
    "and virulence would still need confirming before betting a program "
    "on it)."
)
by_id["adrian-rc-004"]["output"] = old_out.replace(old_frag, new_frag)

# 3. gen-rc-007 — vary the opener so it doesn't near-duplicate
# hack-...113457046's "Worth flagging before going further".
old_out = by_id["gen-rc-007"]["output"]
old_frag = "Worth a correction before going further: RelZ is"
assert old_frag in old_out, "gen-rc-007 opener not found"
new_frag = "Quick correction on the protein itself: RelZ is"
by_id["gen-rc-007"]["output"] = old_out.replace(old_frag, new_frag)

# 4. gen-hyp-004 — drop the remaining editorial-voice opener; state the
# fact directly instead of narrating that it's being flagged.
old_out = by_id["gen-hyp-004"]["output"]
old_frag = (
    "Worth flagging directly, since it changes the framing above: more "
    "recent biochemical work reclassifies Rv2131c as CysQ"
)
assert old_frag in old_out, "gen-hyp-004 fragment not found"
new_frag = "More recent biochemical work reclassifies Rv2131c as CysQ"
by_id["gen-hyp-004"]["output"] = old_out.replace(old_frag, new_frag)

# 5. hack-...115641073 (MhuD) — the input contains no misconception, so an
# instruction promising to "correct any misconceptions" is a mismatch with
# what the output actually (correctly) does: answer the question straight.
by_id["hack-20260602115641073"]["instruction"] = (
    "The user wants to understand a concept or gene in M. tuberculosis. "
    "Use your knowledge of the literature to give a clear, accurate "
    "explanation."
)
by_id["hack-20260602115641073"]["notes"] = (
    "Instruction swapped from a 'correct any misconceptions' framing to a "
    "plain closed_book_qa instruction: the input asks a neutral factual "
    "question with no embedded misconception, so the original instruction "
    "promised a correction the output never had reason to make."
)

# 6. gen-rc-003 — split the dense three-clause sentence into two.
old_out = by_id["gen-rc-003"]["output"]
old_frag = (
    "Mtb's non-replicating persistence is one of the most clinically "
    "significant problems in the field — it's the main reason "
    "drug-susceptible TB treatment has traditionally taken 6 months — "
    "though a newer 4-month rifapentine-moxifloxacin-based regimen (HPMZ) "
    "is now a guideline-endorsed alternative per 2025 ATS/CDC/IDSA/ERS "
    "recommendations, not yet universal in practice."
)
assert old_frag in old_out, "gen-rc-003 fragment not found"
new_frag = (
    "Mtb's non-replicating persistence is one of the most clinically "
    "significant problems in the field — it's the main reason "
    "drug-susceptible TB treatment has traditionally taken 6 months. A "
    "newer 4-month rifapentine-moxifloxacin-based regimen (HPMZ) is now "
    "available too: the 2025 ATS/CDC/IDSA/ERS guidelines endorse it as an "
    "alternative, though it's not yet universal in practice."
)
by_id["gen-rc-003"]["output"] = old_out.replace(old_frag, new_frag)

# 7. adrian-rc-006 — swap the example off cyclopropane synthases (already
# covered by adrian-rc-005) onto Erm(37), a real, well-documented, and
# enzymologically distinct Mtb SAM-methyltransferase. Flagged in this
# seed's own original notes and raised again in two later audit rounds.
by_id["adrian-rc-006"]["output"] = (
    "This is a great angle. There's a long tradition of enzymologists who, "
    "without any machine learning, have studied these structures for "
    "decades and worked out the conserved features within an enzyme class "
    "that dictate substrate specificity. A good example for your "
    "SAM-methyltransferase focus is Erm(37), Mtb's intrinsic "
    "macrolide-resistance methyltransferase — it's mechanistically "
    "unusual among Erm enzymes: rather than the typical single "
    "methylation at 23S rRNA position A2058, it can go on to methylate the "
    "neighboring A2057 and A2059 as well, and the degree of methylation "
    "determines which macrolides it confers resistance to. That's a clean "
    "case where the enzyme's exact chemistry — how many times it "
    "methylates, and where — directly determines a phenotype, which "
    "is exactly the kind of conserved active-site logic worth encoding in "
    "a substrate-specificity model. I'd start with the structural and "
    "biochemical literature on Erm(37)'s atypical multi-site methylation "
    "mechanism and its comparison to canonical single-site Erm enzymes."
)
by_id["adrian-rc-006"]["notes"] = (
    "Verified (PMID 16174779, 'Methyltransferase Erm(37) slips on rRNA to "
    "confer atypical resistance in Mycobacterium tuberculosis' — "
    "atypical multi-site 23S rRNA methylation mechanism confirmed via "
    "independent search). Swapped the example from cyclopropane mycolic "
    "acid synthases (already covered in adrian-rc-005) to Erm(37) for "
    "category diversity, per this seed's own original note flagging the "
    "duplication."
)

with open(PATH, "w") as f:
    json.dump(seeds, f, indent=2)

print(f"Batch 4 applied. {len(REQUIRED_IDS)} seeds touched. Total active seeds: {len(seeds)}")
