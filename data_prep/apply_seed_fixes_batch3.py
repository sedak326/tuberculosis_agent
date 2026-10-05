#!/usr/bin/env python3
"""Third fix pass on seeds.json. Addresses issues introduced or missed in
batch 2: instruction/output mismatches (gen-hyp-001, hack-...115933904),
miscategorization (gen-rc-007), notes-field corruption (hack-...113855006),
an incomplete fact (gen-rc-006 topo IV), a residual LaTeX artifact
(hack-...120959646 input), an unanswered question (hack-...113926643),
inconsistent style normalization (hack-...114445270), editor-voice/scaffolding
language leaking into outputs (gen-rc-012, gen-hyp-004, gen-hyp-002), an
internally-inconsistent mechanism (adrian-rc-004), an incomplete fact
(hack-...112835589 KatG), and uncited "confirmed via independent search"
notes now backed with real citations found via WebSearch this session.
"""
import json

PATH = "/home/skavlak/finetuning/data_prep/seeds.json"
seeds = json.load(open(PATH))
by_id = {s["id"]: s for s in seeds}

REQUIRED_IDS = [
    "gen-hyp-001", "hack-20260602115933904", "gen-rc-007",
    "hack-20260602113855006", "gen-rc-006", "hack-20260602120959646",
    "hack-20260602113926643", "hack-20260602114445270", "gen-rc-012",
    "gen-hyp-004", "gen-hyp-002", "adrian-rc-004",
    "hack-20260602112835589", "adrian-mech-002", "gen-rc-004",
    "gen-rc-009", "claude-methcrit-002", "gen-rc-003",
]
missing = [i for i in REQUIRED_IDS if i not in by_id]
if missing:
    raise SystemExit(f"Missing IDs, aborting: {missing}")

# 1. gen-hyp-001 — add the mechanisms + distinguishing experiment the
# instruction asks for, on top of the (correct) artifact-check framing.
by_id["gen-hyp-001"]["output"] = (
    "Before treating this as a genuine ESAT-6/CFP-10 functional asymmetry, "
    "I'd ask: how was your ESAT-6 purified? This exact pattern — recombinant "
    "ESAT-6 lytic, CFP-10 not — is the signature of a well-documented "
    "purification artifact. Residual ASB-14 detergent, commonly used in the "
    "endotoxin-removal step of Ni-NTA purification, has been shown to "
    "account for most of the hemolytic activity attributed to ESAT-6; a "
    "Proteinase K control that fails to abolish the lytic activity (while it "
    "fully abolishes activity of genuine pore-forming toxins) is the key "
    "diagnostic — protein-mediated lysis shouldn't survive protease "
    "digestion, detergent-mediated lysis will. The reason CFP-10 typically "
    "doesn't show this artifact even under the same detergent wash is that "
    "it tends to bind the detergent less tightly and lose it during "
    "subsequent dialysis.\n\nIf you rule out the artifact and the asymmetry "
    "holds, a couple of genuine mechanisms are worth testing. One is "
    "conformational occlusion: ESAT-6's membrane-interacting surface may "
    "only be exposed or active in the unpaired state, with heterodimerization "
    "sterically blocking it — meaning the 1:1 complex is intrinsically less "
    "lytic than free ESAT-6, not that CFP-10 is simply inert. The other is "
    "that CFP-10 isn't lytic on its own because its role isn't lysis at all "
    "— it may function as a chaperone/delivery module that stabilizes ESAT-6 "
    "and positions it at the membrane, with the lytic activity remaining "
    "ESAT-6-intrinsic rather than something CFP-10 contributes to directly. "
    "To distinguish these, compare membrane permeabilization by the "
    "preformed 1:1 heterodimer against free ESAT-6 at matched "
    "concentrations: substantially reduced lysis from the complex supports "
    "the occlusion model, while lysis comparable to free ESAT-6 is more "
    "consistent with CFP-10 as a non-lytic delivery partner. Either way, "
    "that control comes first, before building a mechanistic story on top of it."
)

# 2. hack-...115933904 — keep the (correct) essentiality correction, but
# actually prioritize a target and propose an experiment as the instruction asks.
by_id["hack-20260602115933904"]["output"] = (
    "Before going further on collapsing the SigF stress-response cascade as "
    "a strategy: sigF is not essential for Mtb viability. SigF-knockout "
    "mutants grow normally in vitro and are viable in vivo — they're "
    "attenuated (reduced bacterial load, altered granuloma pathology) but "
    "they persist, they don't die. That means a drug that suppresses SigF "
    "signaling would likely be anti-virulence at best, not bactericidal — "
    "worth being explicit about which of those two outcomes you're actually "
    "aiming for, since it changes how you'd evaluate a hit.\n\nGiven that, "
    "the more prioritizable target in this pathway is probably PknD itself "
    "rather than SigF — as a Ser/Thr kinase, PknD is far more tractable for "
    "small-molecule inhibition (ATP-competitive or allosteric) than a sigma "
    "factor, which has no enzymatic pocket to drug directly. I'd prioritize "
    "the PknD kinase domain and build the assay around it: purify the "
    "kinase domain, run an ATP-competition biochemical screen to find hits, "
    "then validate cell-based activity with a SigF-dependent reporter "
    "construct (a SigF-regulated promoter driving luciferase or GFP) to "
    "confirm pathway inhibition, followed by a macrophage or mouse "
    "infection assay scoring bacterial burden and granuloma pathology. "
    "Since the expected readout is attenuation of virulence rather than "
    "loss of viability, a standalone CFU-kill assay would likely read as a "
    "false negative here."
)

# 3. gen-rc-007 — recategorize; output is already pure premise correction.
by_id["gen-rc-007"]["category"] = "premise_correction"
by_id["gen-rc-007"]["instruction"] = (
    "Given a researcher's description of their background or question that "
    "embeds an assumption, do not accept the claim at face value. Use your "
    "knowledge of M. tuberculosis to assess whether it is correct, and "
    "steer the researcher in the right direction."
)
by_id["gen-rc-007"]["notes"] = (
    "Verified and corrected (Weiss & Stallings 2013, J Bacteriol; confirmed "
    "via independent search): RelZ is M. smegmatis-specific, Mtb's Rv1366 "
    "is the (inactive) homolog, not the same protein. Category corrected "
    "from research_consult to premise_correction — output is pure premise "
    "correction, not general consult."
)

# 4. hack-...113855006 — strip corrupted leading " | " from notes.
old_notes = by_id["hack-20260602113855006"]["notes"]
assert old_notes.startswith(" | "), old_notes[:10]
by_id["hack-20260602113855006"]["notes"] = old_notes[len(" | "):]

# 5. gen-rc-006 — add the topo IV fact, remove the duplicated/tacked-on
# postscript, integrate the serine detail into the body instead.
by_id["gen-rc-006"]["output"] = (
    "Gyrase is actually one of the better-validated targets in Mtb — "
    "fluoroquinolones target the GyrA subunit and are used in second-line "
    "TB treatment, so there's clinical proof of concept. One structural "
    "fact worth knowing up front: unlike E. coli, Mtb has no topoisomerase "
    "IV — gyrase is its sole type II topoisomerase, so it's also the sole "
    "fluoroquinolone target, with no redundant enzyme to pick up the slack. "
    "That's part of why fluoroquinolone resistance mutations in gyrA/gyrB "
    "have such a direct, unbuffered effect on susceptibility. On the "
    "structural side, Mtb gyrase lacks the conserved serine that anchors "
    "the water-metal ion bridge stabilizing quinolone binding in other "
    "bacterial type II topoisomerases — this is reported as underlying "
    "Mtb's relatively low intrinsic fluoroquinolone susceptibility, and has "
    "already been used to design a moxifloxacin derivative with enhanced "
    "activity against both wild-type and resistant gyrase, which is a much "
    "more concrete place to aim than a generic 'structural differences "
    "affect selectivity.'\n\nThat said, fluoroquinolone resistance is a "
    "defining feature of XDR-TB and is growing, so novel binding sites are "
    "the real opportunity — the GyrB ATPase domain is less exploited than "
    "the GyrA quinolone resistance-determining region. If you want adjacent "
    "targets where structural work is still early: Mtb topoisomerase I "
    "(TopA, Rv3646c) has no close human homologue and has attracted "
    "interest, and Mtb's DNA repair machinery has some unusual gaps — it "
    "lacks several canonical repair enzymes found in E. coli — which has "
    "raised interest in its recombination machinery. I'd look at the "
    "structural comparison of Mtb vs. E. coli GyrB as a starting point, and "
    "the emerging literature on Mtb TopA."
)
by_id["gen-rc-006"]["notes"] = (
    "Fluoroquinolones targeting GyrA in Mtb and their role in XDR-TB is "
    "well established. GyrB as alternative site and TopA as target are in "
    "the literature. Verify Rv3646c as TopA gene assignment before "
    "bootstrapping. Verified and extended (Blower et al. 2016, PNAS, PMID "
    "via doi 10.1073/pnas.1525055113, confirmed via independent search): "
    "added the conserved-serine/water-metal-bridge mechanism. Added: Mtb "
    "lacks topoisomerase IV, confirmed via independent search (PMID "
    "16876125, 'First functional characterization of a singly expressed "
    "bacterial type II topoisomerase: the enzyme from Mycobacterium "
    "tuberculosis') — removed duplicated/tacked-on postscript, integrated "
    "the serine detail into the body instead of appending it after the "
    "closing sentence."
)

# 6. hack-...120959646 — fix residual LaTeX artifacts in the input field
# (the output was already cleaned in batch 2; the input was missed).
old_input = by_id["hack-20260602120959646"]["input"]
assert r"$\text{D328A}$" in old_input
by_id["hack-20260602120959646"]["input"] = old_input.replace(r"$\text{D328A}$", "D328A")

# 7. hack-...113926643 — answer the actual question (the shared pathway),
# not just the HsaAB step.
by_id["hack-20260602113926643"]["output"] = (
    "The shared pathway is the broader 9,10-seco steroid degradation route "
    "for cholesterol catabolism, first worked out in detail in RHA1 and "
    "found to be conserved gene-for-gene in Mtb. Broadly: cholesterol is "
    "taken up, its side chain is degraded through a beta-oxidation-like "
    "process, then the steroid ring system is opened — KshAB introduces a "
    "9alpha-hydroxyl that destabilizes the A-ring, triggering ring-opening "
    "— producing seco-steroid intermediates that are progressively "
    "degraded by a series of oxygenases and hydrolases, eventually "
    "funneling carbon into propionyl-CoA and pyruvate for central "
    "metabolism.\n\nWithin that larger pathway, hsaAB sits downstream of "
    "the ring-opening step: it encodes a flavin-dependent monooxygenase "
    "(hsaA = oxygenase, hsaB = reductase) that hydroxylates 3-HSA into a "
    "catechol — one step in that degradative sequence, not the pathway's "
    "starting point. That's why deleting hsaA in RHA1 blocks "
    "cholesterol-dependent growth and redirects metabolism into the "
    "3,9-DHSA and 3,17-DHSA side products described in the passage: hsaA/hsaB "
    "is a required downstream step in the shared route, and losing it "
    "backs up the pathway rather than stopping it at the very first step."
)
by_id["hack-20260602113926643"]["notes"] = (
    "Schema fixed: moved passage content from instruction to input as a "
    "proper passage, wrote a role-description instruction, adjusted output "
    "to reference the passage explicitly. Content fixed: output now "
    "answers the actual question asked (the shared cholesterol catabolic "
    "pathway), not just the HsaAB step within it."
)

# 8. hack-...114445270 — style normalize (was untouched in prior passes).
by_id["hack-20260602114445270"]["input"] = (
    "Let's talk about transporters and sulfur metabolism in Mycobacterium "
    "tuberculosis. In particular I want to understand details about sulfur "
    "transport in Mtb. What is known about the families/types of sulfur "
    "transporters in Mtb?"
)
by_id["hack-20260602114445270"]["output"] = (
    "The Mtb genome contains several types of sulfur transporters. There's "
    "the ABC sulfate permease (CysTWA), and separately three genes related "
    "to the SLC26 gene superfamily (a family of eukaryotic "
    "bicarbonate/anion transporters). For example, researchers found that "
    "overexpressing one of these genes, Rv1739c — specifically its "
    "transmembrane domain — significantly increased sulfate uptake."
)
by_id["hack-20260602114445270"]["notes"] = (
    "Style normalized (was all-lowercase/casual with a 'great question' "
    "opener, 'it's transmembrane domain' typo) to match the rest of the pool."
)

# 9. gen-rc-012 — remove the mid-answer "correction" framing (the user
# never claimed TAP interferes); state it as established background instead.
by_id["gen-rc-012"]["output"] = (
    "Mtb engages CD8 T cells through MHC class I, which requires antigen to "
    "reach the cytosol — and ESX-1 is central to this. ESAT-6 permeabilizes "
    "the phagosomal membrane, allowing Mtb proteins to leak into the "
    "cytosol where they can be proteasomally processed and loaded onto MHC "
    "class I. CD8 T cells specific for ESAT-6, Ag85B, and other Mtb "
    "antigens have been characterized in infected individuals and in mouse "
    "models. Worth knowing going in: TAP itself is functional in this "
    "setting and actively facilitates presentation of Mtb-derived peptides "
    "on MHC-I, including within the phagosome itself, rather than being a "
    "bottleneck Mtb needs to defeat. If overall surface MHC-I levels drop "
    "during infection, that's better explained by mechanisms other than "
    "TAP interference — I'd be cautious about leading with a TAP-blockade "
    "story without checking the more recent literature on it. On the "
    "antigen side, the PE/PPE protein family is thought to contain targets "
    "for both CD4 and CD8 responses, but antigenic variation across strains "
    "makes them difficult to study systematically. The distinction between "
    "cross-presentation (by uninfected DCs taking up Mtb antigens) and "
    "direct presentation (by infected macrophages) is also important in "
    "Mtb — cross-presentation is thought to be a major pathway for CD8 "
    "priming. I'd start with the ESX-1-dependent cytosolic access "
    "literature and the studies on Mtb-specific CD8 T cell responses in "
    "humans."
)
by_id["gen-rc-012"]["notes"] = (
    "Verified and corrected (PMID 24244525, PLOS ONE, 'TAP Mediates Import "
    "of Mycobacterium tuberculosis-Derived Peptides into Phagosomes and "
    "Facilitates Loading onto HLA-I' — TAP facilitates rather than "
    "interferes with Mtb peptide loading onto MHC-I): removed the "
    "speculative 'TAP interference' mechanism, and rewrote as established "
    "background rather than a mid-answer correction, since the researcher "
    "never claimed TAP interferes."
)

# 10. gen-hyp-004 — fold the CysQ point in as a direct refinement instead of
# "one more angle worth adding" scaffolding; add the real citation.
old_out = by_id["gen-hyp-004"]["output"]
old_tail = (
    "One more angle worth adding: more recent work reclassifies Rv2131c as "
    "CysQ, a 3'-phosphoadenosine-5'-phosphatase whose primary physiological "
    "role may actually be regulating the sulfate assimilation pathway (by "
    "controlling PAP/PAPS pools), not myo-inositol/PIM biosynthesis. That "
    "doesn't rule out the coordinated-regulation hypothesis above, but it "
    "does mean the PIM-biosynthesis connection might be a secondary or "
    "vestigial activity rather than the primary selective pressure for "
    "keeping both activities in one enzyme — worth testing sulfur-pathway "
    "phenotypes (PAPS/PAP levels, sulfation status) alongside the PIM/FBPase "
    "readouts in the separation-of-function experiment."
)
assert old_tail in old_out
new_tail = (
    "Worth flagging directly, since it changes the framing above: more "
    "recent biochemical work reclassifies Rv2131c as CysQ, a "
    "3'-phosphoadenosine-5'-phosphatase whose primary physiological role is "
    "regulating sulfate assimilation (by controlling PAP/PAPS pools) — its "
    "turnover rate for PAP is over three orders of magnitude higher than "
    "for either inositol monophosphate or fructose-1,6-bisphosphate, the "
    "substrates discussed above. That doesn't rule out the "
    "coordinated-regulation hypothesis, but it does mean the "
    "PIM-biosynthesis connection is likely a secondary or vestigial "
    "activity rather than the primary selective pressure for keeping both "
    "activities in one enzyme. It's worth testing sulfur-pathway phenotypes "
    "(PAPS/PAP levels, sulfation status) alongside the PIM/FBPase readouts "
    "in the separation-of-function experiment."
)
by_id["gen-hyp-004"]["output"] = old_out.replace(old_tail, new_tail)
old_notes = by_id["gen-hyp-004"]["notes"]
old_note_tail = (
    " | Verified (confirmed via independent search): added the newer "
    "CysQ/sulfur-metabolism primary-role hypothesis as a third option, "
    "since it changes which function might be under selection."
)
assert old_note_tail in old_notes
new_note_tail = (
    " | Verified (PMID 18454554, Biochemistry 2008, 'Rv2131c from "
    "Mycobacterium tuberculosis Is a CysQ 3'-Phosphoadenosine-5'-phosphatase' "
    "— PAP turnover rate >1000x higher than for IMP/FBP): added the newer "
    "CysQ/sulfur-metabolism primary-role hypothesis as a third option, and "
    "folded it into the output directly rather than as an appended aside, "
    "since it changes which function might be under selection."
)
by_id["gen-hyp-004"]["notes"] = old_notes.replace(old_note_tail, new_note_tail)

# 11. gen-hyp-002 — drop the "editor talking about the edit" parenthetical;
# just state the correct temperature.
old_out = by_id["gen-hyp-002"]["output"]
old_frag = (
    "37°C (human core body temperature, not the 33°C sometimes "
    "used for skin/extremity infections) — to map"
)
assert old_frag in old_out
by_id["gen-hyp-002"]["output"] = old_out.replace(old_frag, "37°C — to map")

# 12. adrian-rc-004 — fix the internally inconsistent autoinhibition framing.
by_id["adrian-rc-004"]["output"] = (
    "Funny you ask — there's a great example. The gene Rv1264 encodes a "
    "class III adenylyl cyclase whose C-terminal domain carries the "
    "conserved catalytic core shared with other class III cyclases, while a "
    "novel N-terminal domain acts as a pH-sensitive autoinhibitory module: "
    "at neutral-to-basic pH it represses catalytic activity, and acidic pH "
    "relieves that repression — Rv1264 shows a roughly 40-fold increase in "
    "activity at pH 6 versus pH 8. Because that N-terminal module is a "
    "distinctive regulatory element rather than the conserved catalytic "
    "core, it's an appealing place to aim for specificity. Conceptually, "
    "instead of blocking the conserved active site, you might design "
    "compounds that engage that N-terminal domain to lock in the "
    "autoinhibited state — preventing the pH-triggered de-repression rather "
    "than trying to out-compete substrate at the catalytic site. I'd point "
    "you to the structural work on the Rv1264 holoenzyme to map the domain "
    "interface you'd want to target."
)

# 13. hack-...112835589 — add the INH prodrug/KatG activation detail (the
# same gap that got the InhA seed quarantined).
old_out = by_id["hack-20260602112835589"]["output"]
old_frag = (
    "isoniazid, a first-line TB drug, works by inhibiting InhA, an enzyme "
    "in the mycolic acid biosynthesis pathway."
)
assert old_frag in old_out
new_frag = (
    "isoniazid, a first-line TB drug, is a prodrug that must first be "
    "activated by the mycobacterial catalase-peroxidase KatG before its "
    "active form can inhibit InhA, an enzyme in the mycolic acid "
    "biosynthesis pathway."
)
by_id["hack-20260602112835589"]["output"] = old_out.replace(old_frag, new_frag)

# 14. adrian-mech-002 — add real citation.
old_notes = by_id["adrian-mech-002"]["notes"]
old_frag = (
    "Verified (multiple sources confirm sigG is the most highly induced of "
    "the 13 Mtb sigma factors following macrophage infection/DNA damage "
    "— stronger support than the original hedge)."
)
assert old_frag in old_notes
new_frag = (
    "Verified (PMID 16483748, macrophage-infection expression profiling "
    "identifying sigG as induced ~9.7-fold and among the most highly "
    "induced M. tuberculosis genes/sigma factors, corroborated by PMC3776920, "
    "the SigG operon/regulon characterization — stronger support than "
    "the original hedge)."
)
by_id["adrian-mech-002"]["notes"] = old_notes.replace(old_frag, new_frag)

# 15. gen-rc-004 — add real citation.
old_notes = by_id["gen-rc-004"]["notes"]
old_frag = "confirmed bimodal C16-C18/C24-C26 FAS-I product distribution via independent search"
assert old_frag in old_notes
new_frag = (
    "bimodal C16-C18/C24-C26 FAS-I product distribution confirmed via "
    "independent search, corroborated by 'Structure of Type-I Mycobacterium "
    "tuberculosis fatty acid synthase at 3.3 Å resolution,' Nature "
    "Communications 2018"
)
by_id["gen-rc-004"]["notes"] = old_notes.replace(old_frag, new_frag)

# 16. gen-rc-009 — add real citation.
old_notes = by_id["gen-rc-009"]["notes"]
old_frag = "Verified (confirmed via independent search):"
assert old_frag in old_notes
new_frag = (
    "Verified (DeJesus et al. 2017, mBio, PMID 28096490, saturating Himar1 "
    "transposon mutagenesis; DeJesus & Ioerger, mSystems 2021, PMC8525568, "
    "modeling site-specific nucleotide biases in Himar1 TnSeq):"
)
by_id["gen-rc-009"]["notes"] = old_notes.replace(old_frag, new_frag)

# 17. claude-methcrit-002 — add real citation.
old_notes = by_id["claude-methcrit-002"]["notes"]
old_frag = "Verified and corrected (confirmed via independent search):"
assert old_frag in old_notes
new_frag = (
    "Verified and corrected (EUCAST broth microdilution reference method "
    "for Mtb MIC determination, Clin Microbiol Infect 2020 / AAC 2024 "
    "multicentre validation):"
)
by_id["claude-methcrit-002"]["notes"] = old_notes.replace(old_frag, new_frag)

# 18. gen-rc-003 — no content change (2025 ATS/CDC/ERS/IDSA guideline is
# real and distinct from the 2022 CDC MMWR interim guidance); add precise
# citation to make the note auditable.
old_notes = by_id["gen-rc-003"]["notes"]
old_frag = "Verified and updated (2025 ATS/CDC/IDSA/ERS guidelines, confirmed via independent search):"
assert old_frag in old_notes
new_frag = (
    "Verified and updated (ATS/CDC/ERS/IDSA Clinical Practice Guideline, "
    "Am J Respir Crit Care Med, Jan 2025, 'Updates on the Treatment of "
    "Drug-Susceptible and Drug-Resistant Tuberculosis' — distinct from "
    "and supersedes the CDC's 2022 MMWR interim guidance on the same "
    "regimen):"
)
by_id["gen-rc-003"]["notes"] = old_notes.replace(old_frag, new_frag)

with open(PATH, "w") as f:
    json.dump(seeds, f, indent=2)

print(f"Batch 3 applied. {len(REQUIRED_IDS)} seeds touched. Total active seeds: {len(seeds)}")
