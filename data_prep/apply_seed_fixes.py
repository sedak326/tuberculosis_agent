#!/usr/bin/env python3
"""One-off script applying the verified seed fixes discussed this session.
Reads seeds.json, writes a cleaned seeds.json (active pool) and
seeds_quarantine.json (removed items + reason), both alongside the original.
Not a repeatable pipeline — a single, reviewed batch of fixes."""
import json

SEEDS_PATH = "/home/skavlak/finetuning/data_prep/seeds.json"
QUARANTINE_PATH = "/home/skavlak/finetuning/data_prep/seeds_quarantine.json"

seeds = json.load(open(SEEDS_PATH))
by_id = {s["id"]: s for s in seeds if "id" in s}

quarantined = []


def quarantine(sid, reason):
    s = by_id.pop(sid, None)
    if s is None:
        return
    s["notes"] = (s.get("notes", "") + f" | QUARANTINED: {reason}").strip(" |")
    quarantined.append(s)


# ---------------------------------------------------------------------------
# Quarantine: clear-cut mechanical violations + unresolved naming problem
# ---------------------------------------------------------------------------
quarantine("hack-20260602114610928", "verbatim copy of input abstract as output, no question asked")
quarantine("hack-20260602115059397", "instruction passage (expression/purification) doesn't match the "
           "question/answer topic (kinase assay conditions) — mismatched passage, output is also a "
           "verbatim methods section")
quarantine("hack-20260602112809789", "InhA mechanism answer omits isoniazid/KatG/INH-NAD adduct "
           "entirely — the central fact for this question; also ungrammatical")
quarantine("hack-20260602112841066", "instruction promises a passage; none supplied")
quarantine("hack-20260602114408923", "answer proposes a different organism/route (M. smegmatis in "
           "mice) than the mouse-aerosol-Mtb model the question asks about, without flagging the "
           "substitution; ends with a deferral instead of an answer")
quarantine("hack-20260602115032809", "dangling reference ('they') with no antecedent; instruction "
           "promises BSL-3 protocol specificity, answer is structural biology")
quarantine("hack-20260602120741457", "imports a fungal-azole-resistance question into a TB dataset "
           "with no signposting; Mtb CYP51 is not a validated TB drug target; instruction promises a "
           "passage that's absent")
quarantine("hack-20260602114525132", "could not independently confirm 'RsfA' as a named Mtb regulator "
           "in the literature (found a distinct, real protein 'RsrA' instead) — possible naming error, "
           "quarantined pending clarification rather than guessing which name is correct")

# ---------------------------------------------------------------------------
# Repair: mismatch fixes not requiring domain verification (general reasoning,
# or the underlying content was already assessed as fine)
# ---------------------------------------------------------------------------

s = by_id["hack-20260602113536184"]
s["output"] = (
    "A Km of 0.27 mM tells you the substrate concentration at which PGI reaches half its Vmax — "
    "but Km isn't a direct affinity measurement unless catalysis is slow relative to substrate "
    "release, so calling this 'relatively high affinity' isn't supported without a comparator. "
    "Before drawing a conclusion, I'd want to know: what's the Km of PGI orthologs from other "
    "organisms, or of other enzymes in the same pathway, run under the same assay conditions? "
    "Also worth checking: was this measured under saturating cofactor conditions, and does the assay "
    "buffer match physiological ionic strength/pH? A Km that looks unremarkable in isolation can "
    "still be meaningful once you have the right comparator."
)
s["notes"] = "Repaired: fixed instruction/output mismatch (now interprets + asks for comparator instead of defining Km) and the Km-as-affinity conflation."

s = by_id["hack-20260602113841936"]
s["output"] = (
    "6-Phosphogluconate is a competitive inhibitor of PGI — it competes with the substrate for the "
    "active site. If you're seeing this show up in your data, the interpretation depends on context: "
    "in vivo, this could reflect genuine cross-talk between the pentose phosphate pathway and "
    "glycolysis (6-phosphogluconate is a PPP intermediate), which would be a real regulatory finding. "
    "But if this came up in an in vitro assay, I'd first check whether the inhibitor concentration used "
    "is physiologically plausible for intracellular 6-phosphogluconate levels — a strong effect at a "
    "non-physiological concentration is more of an assay artifact than a biological signal. Worth "
    "following up with a dose-response curve against estimated intracellular concentrations."
)
s["notes"] = "Repaired: fixed instruction/output mismatch (now interprets + flags artifact risk instead of just defining the term)."

s = by_id["hack-20260602114005666"]
s["output"] = (
    "That question is ambiguous between two different things, and the answer depends on which one "
    "you mean. If you want to know which residues are catalytically required in vitro, alanine-"
    "scanning the recombinant enzyme (as was done for the putative catalytic triad in one published "
    "study) and testing activity is the right approach. But that tells you nothing about whether the "
    "gene is essential in Mtb — for that you need a genetic approach in the organism itself, like "
    "CRISPRi knockdown or a knockout with a complementing copy, since a residue can be catalytically "
    "essential in vitro while the gene itself is dispensable if a redundant pathway exists in vivo. "
    "I'd want to know which of these two questions you're actually trying to answer before "
    "recommending one over the other."
)
s["notes"] = "Repaired: fixed instruction/output mismatch (was a past-tense description of one paper's in vitro methods; now names the in-vitro-catalytic vs in-vivo-essential ambiguity the question conflates, matching the 'practical advice' instruction)."

# ---------------------------------------------------------------------------
# Repair: science-verified corrections (all citations independently
# re-confirmed via web search this session, not just trusted from the audit)
# ---------------------------------------------------------------------------

# V-01: adrian-rc-002 — symmetry is internal to the C-terminal domain, not the
# oligomeric assembly; also mislabeled category (this IS the premise_correction
# template seed).
s = by_id["adrian-rc-002"]
s["category"] = "premise_correction"
s["output"] = (
    "Actually, that's not quite right — and the exception is what makes it interesting. M. tuberculosis "
    "NAD kinase has distinctive structural features rather than being a generic NAD kinase. Structural "
    "work (Garavaglia et al. 2004) described a novel fold in its C-terminal domain — a 12-stranded "
    "beta-sandwich with a striking internal 222 (D2) symmetry within that domain itself, not in the "
    "oligomeric assembly (the quaternary structure is separately reported as dimeric or tetrameric "
    "depending on conditions). So rather than assuming it's conserved, I'd point you to the structural "
    "literature on Mtb NAD kinase specifically, where this atypical fold is characterized."
)
s["notes"] = "Verified (Garavaglia et al. 2004, JBC, PMID 15269221, confirmed via independent search). Category corrected from closed_book_qa to premise_correction. Fixed: symmetry is internal to the C-terminal domain, not the oligomeric assembly."

# V-06: adrian-rc-001 — surface localization/plasminogen-binding confirmed
# strongly; add the cross-reactivity risk (Fontan et al. 2000 precedent).
s = by_id["adrian-rc-001"]
s["output"] = (
    "One target that might not be on everyone's radar is enolase. In mycobacteria, enolase is best "
    "known as a cytoplasmic glycolytic enzyme, but it's also documented as surface-localized — shown "
    "by immunoelectron microscopy, immunofluorescence, and flow cytometry — where it binds host "
    "plasminogen in a lysine-mediated interaction (measurable by SPR), and bound plasminogen is "
    "convertible to plasmin. That's a genuinely appealing, under-explored surface target. One thing "
    "to design around from the start, though: enolase is one of the most conserved enzymes in "
    "biology, and human alpha-enolase is itself a surface plasminogen receptor. There's direct "
    "precedent for the cross-reactivity risk this implies — antibodies raised against streptococcal "
    "surface enolase are documented to cross-react with human alpha-enolase (Fontan et al. 2000), "
    "with autoimmune consequences. For a binder project specifically, that's a first-order "
    "specificity problem worth designing around, not just noting."
)
s["notes"] = "Verified (Rahi et al. 2017, PMID 27569900, and Fontan et al. 2000, both confirmed via independent search). Cross-reactivity caveat added; original contamination-artifact caveat dropped since surface localization is well-supported by three orthogonal methods."

# V-03: hack-...115933904 — sigF is not essential, mutants are viable but
# attenuated; a SigF-collapsing drug would be anti-virulence, not bactericidal.
s = by_id["hack-20260602115933904"]
s["output"] = (
    "Before going further on collapsing the SigF stress-response cascade as a strategy: sigF is not "
    "essential for Mtb viability. SigF-knockout mutants grow normally in vitro and are viable in "
    "vivo — they're attenuated (reduced bacterial load, altered granuloma pathology) but they persist, "
    "they don't die. That means a drug that suppresses SigF signaling would likely be anti-virulence "
    "at best, not bactericidal — worth being explicit about which of those two outcomes you're "
    "actually aiming for, since it changes how you'd evaluate a hit. The Rv1364c regulatory biology "
    "itself (PknD tuning SigF activity) is well documented, so the mechanism holds up — it's "
    "specifically the essentiality/bactericidal framing that needs correcting."
)
s["notes"] = "Verified (Chen et al. 2000, Infect Immun; sigF-knockout viability/attenuation confirmed via independent search)."

# V-04: gen-rc-006 — Mtb lacks topoisomerase IV; add the conserved-serine
# mechanistic detail (more actionable for a structural biologist).
s = by_id["gen-rc-006"]
s["output"] = (
    s["output"] + " One more specific, actionable detail for a structural approach: Mtb gyrase lacks "
    "the conserved serine that anchors the water-metal ion bridge stabilizing quinolone binding in "
    "other bacterial type II topoisomerases — this is reported as underlying Mtb's relatively low "
    "intrinsic fluoroquinolone susceptibility, and has already been used to design a moxifloxacin "
    "derivative with enhanced activity against both wild-type and resistant gyrase. That's a more "
    "concrete place to aim than general 'structural differences affect selectivity.'"
) if s.get("output") else s.get("output")
s["notes"] = s.get("notes", "") + " | Verified and extended (Blower et al. 2016, PNAS, PMID via doi 10.1073/pnas.1525055113, confirmed via independent search): added the conserved-serine/water-metal-bridge mechanism."

# V-08: gen-rc-007 — RelZ is the M. smegmatis enzyme; Mtb's homolog is the
# catalytically-inactive Rv1366.
s = by_id["gen-rc-007"]
s["output"] = (
    "Worth a correction before going further: RelZ is the M. smegmatis enzyme (MSMEG_5849), not an "
    "Mtb protein — it's a dual-domain small alarmone synthetase with a tandem RNase HII domain. The "
    "Mtb homolog is Rv1366, and it's structurally missing that RNase HII domain and has been reported "
    "as catalytically inactive — unable to synthesize (p)ppGpp in vitro. Deleting rv1366 also had no "
    "effect on the relMtb-knockout phenotype in the published work, so its actual role is still "
    "unresolved. That's arguably a more interesting hook than a routine second synthetase: Mtb "
    "carries an alarmone-synthetase homolog that looks non-functional, and nobody's established why "
    "it's retained."
)
s["notes"] = "Verified and corrected (Weiss & Stallings 2013, J Bacteriol; confirmed via independent search): RelZ is M. smegmatis-specific, Mtb's Rv1366 is the (inactive) homolog, not the same protein."

# V-07: hack-...113457046 — Mtb has only RpiB, not RpiA; corrected + upgraded.
s = by_id["hack-20260602113457046"]
s["category"] = "premise_correction"
s["instruction"] = "Given a researcher's question that embeds an assumption or claim, do not accept the claim at face value. Use your knowledge of M. tuberculosis to assess whether it is correct, and steer the researcher in the right direction."
s["output"] = (
    "Worth flagging before going further: the M. tuberculosis genome doesn't contain an RpiA sequence "
    "at all — it encodes only RpiB (Rv2465c). RpiA and RpiB are non-homologous enzymes that evolved "
    "independently to catalyze the same reaction, with different folds and different active-site "
    "residues, so 'different protein family' is the right instinct, just attached to the wrong "
    "enzyme name. The comparison that's actually informative: Mtb RpiB and E. coli RpiB have similar "
    "catalytic efficiency but differ in active-site details, including the position of the catalytic "
    "base, and represent distinct sub-families. This also explains why RpiB is a live drug-target "
    "idea — mammals only have RpiA, so a pathogen that relies solely on RpiB (like Mtb) offers a "
    "clean selectivity window."
)
s["notes"] = "Verified and corrected (Roos et al. 2004, J Mol Biol, PMID 14687575, confirmed via independent search): seed named the wrong enzyme (RpiA) — Mtb has only RpiB."

# V-02: gen-hyp-001 — ESAT-6 cytolysis is very likely a detergent (ASB-14)
# purification artifact, not a real biological asymmetry with CFP-10.
s = by_id["gen-hyp-001"]
s["output"] = (
    "Before treating this as a genuine ESAT-6/CFP-10 functional asymmetry, I'd ask: how was your "
    "ESAT-6 purified? This exact pattern — recombinant ESAT-6 lytic, CFP-10 not — is the signature of "
    "a well-documented purification artifact. Residual ASB-14 detergent, commonly used in the "
    "endotoxin-removal step of Ni-NTA purification, has been shown to account for most of the "
    "hemolytic activity attributed to ESAT-6; a Proteinase K control that fails to abolish the lytic "
    "activity (while it fully abolishes activity of genuine pore-forming toxins) is the key diagnostic "
    "— protein-mediated lysis shouldn't survive protease digestion, detergent-mediated lysis will. The "
    "reason CFP-10 typically doesn't show this artifact even under the same detergent wash is that it "
    "tends to bind the detergent less tightly and lose it during subsequent dialysis. If you rule out "
    "the artifact and the asymmetry holds, then it's worth investigating — but that control comes "
    "first, before building a mechanistic story on top of it."
)
s["notes"] = "Verified (Conrad et al. 2017, PNAS; confirmed via independent search): rewritten around the detergent-artifact explanation rather than treating the asymmetry as settled biology."

# ---------------------------------------------------------------------------
# Repair: mechanical fixes not requiring science verification
# ---------------------------------------------------------------------------

# V-05: hack-...113855006 — seed's science checks out (p27=PPE36=Rv2108,
# confirmed), only style + category were wrong.
s = by_id.get("hack-20260602113855006")
if s:
    s["category"] = "closed_book_qa"
    s["input"] = "I'm learning about the PPE family of proteins in Mtb. What is their role? I know it's likely diverse, but I want some specific examples."
    s["output"] = (
        "The PPE family in Mtb is large and functionally diverse, and much of it is still poorly "
        "characterized. One concrete example: p27 (PPE36, encoded by Rv2108) is a 27-kDa cell wall/"
        "membrane-associated member of the family. It's been studied for immunogenicity — in mice, "
        "a p27-flagellin fusion (using flagellin as a built-in adjuvant) produced the strongest "
        "cellular response of the constructs tested, with the highest IFN-gamma production and cell "
        "proliferation, indicating a Th1-skewed response. Patient serology also showed a notably "
        "IgA-dominant response with essentially no IgG, which is an unusual pattern worth noting if "
        "you're thinking about this as a vaccine or diagnostic candidate."
    )
    s["notes"] = s.get("notes", "") + " | Verified correct (Le Moigne et al. 2005/2008, PMID 18289677, confirmed via independent search) — audit's suspicion this was LprG was wrong, p27/PPE36/Rv2108 is correct. Category corrected to closed_book_qa (instruction is closed_book_qa-style). Style normalized (was all-lowercase/casual, 'diverse + mysterious' framing removed), strengthened with the IgA-dominant serology finding."

# Schema fix: Melina's 3 seeds — move facts from instruction to input, write a
# proper role-description instruction. hack-...115059397 was quarantined above
# instead of repaired (mismatched passage), so only the other two here.
s = by_id["hack-20260602113926643"]
old_instruction = s["instruction"]
s["input"] = f"Passage:\n{old_instruction}\n\nQuestion: {s['input']}"
s["instruction"] = "The user is reading TB literature and asks about a specific pathway or mechanism described in a passage. Answer using the passage plus your broader knowledge of M. tuberculosis, and be clear about which parts come from the passage versus general background."
s["output"] = (
    s["output"] + " That fits with the passage you're describing: hsaA-deletion in RHA1 blocks "
    "cholesterol growth and redirects metabolism to the 3,9-DHSA and 3,17-DHSA side products, which "
    "is consistent with hsaA/hsaB being the entry point into this shared catabolic branch."
)
s["notes"] = "Schema fixed: moved passage content from instruction to input as a proper passage, wrote a role-description instruction, adjusted output to reference the passage explicitly."

s = by_id["hack-20260602120606368"]
old_instruction = s["instruction"]
s["input"] = f"Passage:\n{old_instruction}\n\n{s['input']}"
s["instruction"] = "The user has shared a passage describing part of an experimental workflow and asks a related question about it. Answer grounded in the passage, and flag if the passage doesn't fully answer what's being asked."
s["output"] = (
    "The passage you shared describes one check they ran — confirming the active-site cysteine was "
    "in its reduced, reactive form via DTNB (Ellman's reagent) titration of free thiols — but that's "
    "a functional/chemical check, not a native-conformation check on its own. For native state "
    "specifically: KasA and KasB were expressed as His-tagged fusions and purified by Ni-affinity "
    "chromatography, and their catalytic competence was used as indirect evidence of correct active-"
    "site folding. That's a weaker inference than it sounds, though — activity shows the active site "
    "is intact, not that the whole protein is natively folded. Size-exclusion chromatography or "
    "circular dichroism would be the more direct way to confirm overall native conformation, and "
    "it's worth checking whether either was actually done here."
)
s["notes"] = "Schema fixed: moved DTNB passage from instruction to input. Also fixed a real contradiction (instruction described the DTNB check, but the original output never mentioned it and used a different method) and softened the circular 'activity implies folding' inference per the audit's flag."

# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------
with open(SEEDS_PATH, "w", encoding="utf-8") as f:
    json.dump(list(by_id.values()), f, indent=2, ensure_ascii=False)

with open(QUARANTINE_PATH, "w", encoding="utf-8") as f:
    json.dump(quarantined, f, indent=2, ensure_ascii=False)

print(f"Active seeds: {len(by_id)}")
print(f"Quarantined: {len(quarantined)}")
for q in quarantined:
    print(f"  {q['id']}: {q['notes'][-120:]}")
