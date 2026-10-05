#!/usr/bin/env python3
"""Second batch of seed fixes — the remaining 15 (+1 missed) flagged items."""
import json

SEEDS_PATH = "/home/skavlak/finetuning/data_prep/seeds.json"
QUARANTINE_PATH = "/home/skavlak/finetuning/data_prep/seeds_quarantine.json"

seeds = json.load(open(SEEDS_PATH))
by_id = {s["id"]: s for s in seeds if "id" in s}
quarantined = json.load(open(QUARANTINE_PATH))


def quarantine(sid, reason):
    s = by_id.pop(sid, None)
    if s is None:
        return
    s["notes"] = (s.get("notes", "") + f" | QUARANTINED: {reason}").strip(" |")
    quarantined.append(s)


# --- Quarantine: SSB redundancy cluster (keep the other 2, this one is generic) ---
quarantine("hack-20260602113348616", "part of MtuSSB redundancy cluster (4 near-duplicate seeds "
           "from 1-2 papers) — generic textbook definition, no added domain value over the other "
           "kept seeds in the cluster")

# --- Mechanical fixes, no external verification needed ---

# adrian-rc-007: passage says "one of two" cysteines, output said "a pair of cysteines" (both).
s = by_id["adrian-rc-007"]
s["output"] = s["output"].replace(
    "missing two key catalytic pieces (a pair of cysteines, plus an acid/base residue pair)",
    "missing one of the pair of catalytic cysteines, plus the His-Glu acid/base pair — three "
    "residues gone, not a full second cysteine"
)
s["notes"] = "Fixed grounding error: output said 'a pair of cysteines' (both) but the passage says 'one of two' cysteine residues plus the His-Glu pair — output now matches the passage."

# gen-hyp-002: granuloma core temperature is ~37C (human core body temp), not 33C.
s = by_id["gen-hyp-002"]
s["output"] = s["output"].replace(
    "granuloma-mimicking conditions — low oxygen, acidic pH, 33°C",
    "granuloma-mimicking conditions — low oxygen, acidic pH, 37°C (human core body temperature, "
    "not the 33°C sometimes used for skin/extremity infections)"
)
s["notes"] = "Fixed fabricated figure: granuloma core temperature is ~37C (human core body temp), not 33C."

# hack-...112835589: mismatch — question asks what mycolic acids ARE and WHY they matter;
# original output only answered the first half.
s = by_id["hack-20260602112835589"]
s["output"] = (
    s["output"] + " They matter for several reasons. Structurally, they form a highly hydrophobic "
    "outer membrane that acts as a formidable permeability barrier — this is a major reason Mtb is "
    "intrinsically resistant to many antibiotics that easily cross other bacterial membranes. "
    "Diagnostically, this same lipid-rich cell wall is the basis of acid-fast staining "
    "(Ziehl-Neelsen), the classic way TB is identified microscopically. Mycolic acids also "
    "contribute to immune evasion and intracellular survival in macrophages. And clinically, "
    "mycolic acid biosynthesis is a validated drug target: isoniazid, a first-line TB drug, works "
    "by inhibiting InhA, an enzyme in the mycolic acid biosynthesis pathway."
)
s["notes"] = "Fixed instruction/output mismatch: original answer only defined mycolic acids, never addressed 'why they matter' (permeability barrier/drug resistance, acid-fast staining basis, immune evasion, isoniazid's mechanism)."

# gen-rc-008: internal contradiction — whole passage describes a CATABOLIC pathway
# (breaking cholesterol down for carbon/energy) but calls it "biosynthetic".
s = by_id["gen-rc-008"]
s["output"] = s["output"].replace(
    "one of the most thoroughly worked-out biosynthetic pathways",
    "one of the most thoroughly worked-out catabolic pathways"
)
s["notes"] = "Fixed wording error: called a catabolic pathway 'biosynthetic', contradicting the rest of the same answer (which correctly describes breaking cholesterol down for carbon/energy)."

# hack-...120959646: LaTeX artifacts + add alternatives per audit (Pro-Q Diamond specificity,
# destabilization vs catalytic-dead alternative explanation).
s = by_id["hack-20260602120959646"]
s["input"] = s["input"].replace(
    "$\\Delta\\text{PAS}\\Delta\\text{PP2C}$", "ΔPASΔPP2C"
)
s["output"] = (
    s["output"]
    .replace("$\\text{D328A}$", "D328A")
    .replace("$\\text{E444K/D328A}$", "E444K/D328A")
    .replace("$\\text{[}\\gamma\\text{-}^{32}\\text{P]ATP}$", "[γ-32P]ATP")
)
s["output"] = s["output"].replace(
    "Your data demonstrates a dynamic, intramolecular kinetic antagonism",
    "The leading explanation is a dynamic, intramolecular kinetic antagonism"
)
s["output"] += (
    " Two things worth ruling out before treating this as settled, though. First, D328A could be "
    "destabilizing the phosphatase domain's fold rather than cleanly killing only its catalytic "
    "activity — a limited proteolysis or thermal-stability comparison between D328A and wild-type "
    "would help rule that out. Second, if you're reading this out via Pro-Q Diamond staining, it has "
    "known specificity limitations (it can also react with some non-phosphoprotein species) — "
    "confirming the signal with a second method, like autoradiography or phospho-specific "
    "antibodies, would strengthen the interpretation."
)
s["notes"] = "Fixed LaTeX artifacts (plain text now). Softened the confident single-explanation framing and added the two alternatives the audit flagged (destabilization vs. catalytic-only effect; Pro-Q Diamond specificity limitations)."

# --- Science-verified corrections (each independently re-confirmed via web search) ---

# sigG upregulation confirmed strongly (even more so than the seed hedged); also add the
# induction-vs-fitness-signal distinction the audit flagged, and drop the AI-persona framing.
s = by_id["adrian-mech-002"]
s["input"] = s["input"].replace(
    " As an AI agent specialized in TB, what can you tell me about this gene?",
    " What can you tell me about this gene?"
)
s["output"] = (
    "sigG is a strong hit to see here, and it's well-supported: of all 13 sigma factors in M. "
    "tuberculosis, sigG is the most highly induced following both DNA damage and macrophage "
    "infection, so its appearance in your expression screen fits an established pattern. Worth "
    "being precise about what each screen is actually telling you, though — your expression screen "
    "reports induction (sigG's transcript goes up under infection-like conditions), while your "
    "Tn-seq screen reports fitness cost (disrupting sigG measurably affects survival/growth). A gene "
    "being a top hit in both means it's both turned on during infection AND functionally needed once "
    "it is — that's a stronger signal than either screen alone, and worth stating explicitly rather "
    "than treating both hits as the same kind of evidence. I'd point you to the macrophage-infection "
    "transcriptomics literature on sigG to see which downstream genes it's been linked to."
)
s["notes"] = "Verified (multiple sources confirm sigG is the most highly induced of the 13 Mtb sigma factors following macrophage infection/DNA damage — stronger support than the original hedge). Added the induction-vs-fitness distinction the audit flagged. Removed 'As an AI agent specialized in TB' persona framing from the input."

# Rv1264: fix vague/potentially-backward directionality — confirmed activated (40x) at
# acidic pH (pH 6) vs neutral/basic (pH 8), via the N-terminal regulatory domain.
s = by_id["adrian-rc-004"]
s["output"] = s["output"].replace(
    "constrains cyclase activity under specific conditions",
    "is directly responsible for pH-dependent activation: Rv1264 shows a roughly 40-fold increase "
    "in adenylyl cyclase activity at pH 6 versus pH 8, so acidic conditions activate the enzyme "
    "rather than constrain it"
) if "constrains cyclase activity under specific conditions" in s["output"] else s["output"]
s["notes"] = s.get("notes", "") + " | Verified (Tews et al. 2005, Science 308:1020-1023, confirmed via independent search): fixed the directionality — acidic pH activates the enzyme (40-fold), doesn't constrain it."

# FAS-I: bimodal (two distinct peaks), not a continuous C16-C26 range.
s = by_id["gen-rc-004"]
s["output"] = s["output"].replace(
    "a single multifunctional polypeptide that makes C16-C26 fatty acids",
    "a single multifunctional polypeptide that makes fatty acids in a bimodal distribution — "
    "C16-C18 and C24-C26, not a continuous range in between"
)
s["notes"] = "Verified and corrected (confirmed bimodal C16-C18/C24-C26 FAS-I product distribution via independent search): fixed 'C16-C26 fatty acids' framing, which implied a continuous range rather than two distinct peaks."

# TB treatment: acknowledge the newer 4-month regimen alongside the traditional 6-month one.
s = by_id["gen-rc-003"]
s["output"] = s["output"].replace(
    "it's the main reason TB treatment takes 6 months",
    "it's the main reason drug-susceptible TB treatment has traditionally taken 6 months — though "
    "a newer 4-month rifapentine-moxifloxacin-based regimen (HPMZ) is now a guideline-endorsed "
    "alternative per 2025 ATS/CDC/IDSA/ERS recommendations, not yet universal in practice"
)
s["notes"] = "Verified and updated (2025 ATS/CDC/IDSA/ERS guidelines, confirmed via independent search): acknowledged the newer 4-month regimen alongside the traditional 6-month standard."

# Tn-seq: name the actual mechanism (TA-dinucleotide site density) instead of just
# gesturing at "GC-richness effects".
s = by_id["gen-rc-009"]
s["output"] = s["output"].replace(
    "There are also GC-richness effects on transposon insertion bias to account for.",
    "There's also a mechanism worth understanding rather than just flagging: Himar1 only inserts "
    "at TA dinucleotides, so AT-rich regions have denser TA site availability and give finer-"
    "resolution essentiality calls, while GC-rich regions have sparser TA sites (coarser, noisier "
    "signal) — plus local sequence context immediately around a TA site further modulates insertion "
    "probability. A gene with few TA sites can look artificially 'hard to call' just from low "
    "sampling density, not necessarily from biology."
)
s["notes"] = "Verified (confirmed via independent search): replaced the symptom-only 'GC-richness effects' mention with the actual TA-dinucleotide-density mechanism."

# MIC90 vs MIC99: correct the convention for a single-strain endpoint.
s = by_id["claude-methcrit-002"]
s["output"] = s["output"].replace(
    "run a serial dilution and get an MIC (MIC90 is the usual TB convention) so the result is comparable to the literature",
    "run a serial dilution and get an MIC — for a single strain like this, MIC99 (or a defined "
    "visual no-growth endpoint) is the usual convention, not MIC90; MIC90 specifically means the "
    "concentration inhibiting 90% of isolates *in a population*, which isn't what you're measuring "
    "here with one strain"
)
s["notes"] = "Verified and corrected (confirmed via independent search): MIC90 is a population-level convention, not the single-strain endpoint this scenario calls for (MIC99/ECV99)."

# gen-rc-012: TAP appears to facilitate, not interfere with, Mtb antigen loading onto MHC-I —
# correct the speculative mechanism.
s = by_id["gen-rc-012"]
s["output"] = s["output"].replace(
    "A complication is that Mtb appears to actively suppress MHC class I presentation in infected "
    "macrophages — there's evidence that infection reduces surface MHC class I levels, possibly via "
    "interference with TAP-dependent peptide loading.",
    "One correction worth making here: TAP itself appears to be functional and to actively "
    "facilitate presentation of Mtb-derived peptides on MHC-I, not interfere with it — that's fairly "
    "well established. If overall surface MHC-I levels are reduced during infection, the mechanism "
    "is more likely something other than TAP interference specifically; I wouldn't lead with that "
    "explanation without checking the more recent literature on it."
)
s["notes"] = "Verified and corrected (confirmed via independent search — TAP facilitates rather than interferes with Mtb peptide loading onto MHC-I): removed the speculative 'TAP interference' mechanism."

# gen-hyp-004: add the newer primary-function understanding (sulfur metabolism/CysQ) as a
# third hypothesis, without discarding the existing ones (which remain plausible).
s = by_id["gen-hyp-004"]
s["output"] += (
    "\n\nOne more angle worth adding: more recent work reclassifies Rv2131c as CysQ, a "
    "3'-phosphoadenosine-5'-phosphatase whose primary physiological role may actually be regulating "
    "the sulfate assimilation pathway (by controlling PAP/PAPS pools), not myo-inositol/PIM "
    "biosynthesis. That doesn't rule out the coordinated-regulation hypothesis above, but it does "
    "mean the PIM-biosynthesis connection might be a secondary or vestigial activity rather than "
    "the primary selective pressure for keeping both activities in one enzyme — worth testing "
    "sulfur-pathway phenotypes (PAPS/PAP levels, sulfation status) alongside the PIM/FBPase readouts "
    "in the separation-of-function experiment."
)
s["notes"] = s.get("notes", "") + " | Verified (confirmed via independent search): added the newer CysQ/sulfur-metabolism primary-role hypothesis as a third option, since it changes which function might be under selection."

with open(SEEDS_PATH, "w", encoding="utf-8") as f:
    json.dump(list(by_id.values()), f, indent=2, ensure_ascii=False)
with open(QUARANTINE_PATH, "w", encoding="utf-8") as f:
    json.dump(quarantined, f, indent=2, ensure_ascii=False)

print(f"Active seeds: {len(by_id)}")
print(f"Quarantined total: {len(quarantined)}")
