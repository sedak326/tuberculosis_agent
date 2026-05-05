# TB Seed Tasks for Instruction Tuning
# 20 diverse task examples for generating synthetic TB instruction-response pairs
You are asked to come up with a set of 20 diverse task instructions specifically related to tuberculosis and tuberculosis research domain. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

---

## Task 1
**Instruction:** Rewrite the following tuberculosis research text in plain language suitable for a general audience without losing the core information.

**Input:** Mycobacterium tuberculosis persists within the phagosomal compartment of macrophages by inhibiting phagosome-lysosome fusion, thereby evading the host's primary intracellular killing mechanism.

**Output:** The bacteria that cause tuberculosis survive inside immune cells called macrophages by blocking the process the cell uses to destroy them, which allows the bacteria to avoid being killed by the body's defences.

---

## Task 2
**Instruction:** Extract all TB-related drugs mentioned in the following text. For each drug, include the drug class, mechanism of action, and any resistance information if stated.

**Input:** Isoniazid remains a cornerstone of first-line TB treatment, inhibiting mycolic acid synthesis via KatG-mediated activation. Rifampicin targets the beta subunit of RNA polymerase encoded by rpoB, and mutations in this gene account for over 95% of rifampicin-resistant clinical isolates. Pyrazinamide, a prodrug activated by PncA, disrupts membrane energy metabolism and is particularly effective against semi-dormant bacilli.

**Output:** - Drug: Isoniazid, class: first-line antitubercular, mechanism: inhibits mycolic acid synthesis after KatG activation, resistance: KatG mutations reduce activation. - Drug: Rifampicin, class: first-line antitubercular, mechanism: inhibits RNA polymerase via rpoB subunit binding, resistance: rpoB mutations in >95% of resistant isolates. - Drug: Pyrazinamide, class: first-line antitubercular, mechanism: prodrug activated by PncA, disrupts membrane energy metabolism, resistance: pncA mutations.

---

## Task 3
**Instruction:** Answer the following question about tuberculosis based on established biomedical knowledge.

**Input:** What is the difference between latent TB infection and active TB disease?

**Output:** In latent TB infection the bacteria are present in the body but contained by the immune system, causing no symptoms and no transmission. In active TB disease the bacteria multiply and cause symptoms such as persistent cough, fever, night sweats, and weight loss, and the person can transmit the infection to others. Latent infection can progress to active disease if the immune system is weakened, for example by HIV infection, malnutrition, or immunosuppressive therapy.

---

## Task 4
**Instruction:** Identify the main research conclusion from the following results excerpt.

**Input:** Deletion of the fadE28 gene in Mycobacterium tuberculosis H37Rv resulted in significantly attenuated growth in macrophages and reduced bacterial burden in mouse lungs at weeks 4 and 8 post-infection compared to the wild-type strain. Complementation with the intact fadE28 allele fully restored virulence, confirming the gene's role in pathogenesis.

**Output:** The fadE28 gene is required for full virulence of M. tuberculosis, as its deletion attenuates bacterial survival in macrophages and in vivo, and this effect is reversed by genetic complementation.

---

## Task 5
**Instruction:** Explain the function of the following M. tuberculosis gene and its relevance to drug resistance.

**Input:** katG

**Output:** katG encodes a bifunctional catalase-peroxidase enzyme in M. tuberculosis. Its primary relevance to drug resistance lies in its role in activating isoniazid: KatG oxidises the prodrug isoniazid into its active form, which then inhibits InhA, an enzyme required for mycolic acid biosynthesis. Mutations in katG, most commonly S315T, reduce or abolish this activation, rendering the bacterium resistant to isoniazid. This is the most common mechanism of isoniazid resistance in clinical isolates.

---

## Task 6
**Instruction:** Classify the following TB case as drug-susceptible, MDR-TB, or XDR-TB based on the resistance profile provided and justify your classification.

**Input:** A clinical isolate from a TB patient shows resistance to isoniazid and rifampicin. Susceptibility testing confirms the isolate remains sensitive to fluoroquinolones, bedaquiline, and all second-line injectable agents.

**Output:** This isolate is classified as MDR-TB (multidrug-resistant tuberculosis). MDR-TB is defined as resistance to at least isoniazid and rifampicin, which this isolate meets. It does not qualify as XDR-TB. According to current WHO definitions, XDR-TB is MDR/RR-TB with additional resistance to any fluoroquinolone and at least one Group A drug (such as bedaquiline or linezolid), which is not present here

---

## Task 7
**Instruction:** Summarise the experimental methods described in the following passage in two to three sentences.

**Input:** Bacterial strains were cultured in Middlebrook 7H9 broth supplemented with 10% OADC enrichment and 0.05% Tween-80 at 37°C with shaking. Gene deletion mutants were constructed by specialised transduction using a temperature-sensitive phage delivery system. Macrophage infection assays were performed using differentiated THP-1 cells at a multiplicity of infection of 1:10, and intracellular bacterial survival was quantified by colony-forming unit counts at 24, 48, and 72 hours post-infection.

**Output:** M. tuberculosis strains were grown in standard liquid culture, and gene knockouts were generated using phage-mediated recombination. Virulence was assessed by infecting THP-1 macrophages and counting viable bacteria at multiple time points over 72 hours.

---

## Task 8
**Instruction:** What does the following finding suggest about the role of the described protein in M. tuberculosis metabolism? Provide a mechanistic explanation.

**Input:** Disruption of the GlnD gene in M. tuberculosis resulted in a severe growth defect under nitrogen-limiting conditions, but growth was restored to near wild-type levels when excess ammonium was supplied.

**Output:** This finding suggests GlnD is essential for nitrogen sensing and assimilation under limiting conditions. GlnD is a uridylyltransferase that modifies the PII signal transduction proteins in response to intracellular glutamine levels, regulating downstream nitrogen assimilation pathways including the GS-GOGAT system. When nitrogen is scarce, functional GlnD is required to signal nitrogen starvation and upregulate assimilation. Excess ammonium bypasses this regulatory requirement by restoring glutamine pools directly, which explains why supplementation rescues the growth defect.

---

## Task 9
**Instruction:** Identify the limitations of the study as stated or implied by the following discussion excerpt.

**Input:** Although our in vitro data strongly support a role for Rv1538c in asparaginase activity, we acknowledge that experiments were conducted using a recombinant protein expressed in E. coli, which may not fully recapitulate the native folding or post-translational modifications present in M. tuberculosis. Furthermore, all infection experiments were performed using the H37Rv laboratory-adapted strain, and whether these findings extend to clinical isolates with diverse genetic backgrounds remains to be established.

**Output:** The study has several limitations: first, the recombinant protein was expressed in a non-native host (E. coli), which may not reflect its native structure or modifications in M. tuberculosis; second, the experiments relied solely on the H37Rv laboratory strain, limiting generalisability to diverse clinical isolates; and third, the absence of in vivo validation limits conclusions about physiological relevance during infection.

---

## Task 10
**Instruction:** Translate the following molecular biology finding into its potential clinical or therapeutic significance.

**Input:** The crystal structure of M. tuberculosis InhA revealed a hydrophobic binding pocket that accommodates the isoniazid-NAD adduct. Virtual screening against this pocket identified three candidate compounds that inhibited InhA activity with IC50 values below 1 µM and retained activity against KatG-mutant isoniazid-resistant strains in vitro.

**Output:** This finding is clinically significant because isoniazid resistance most commonly arises from mutations in KatG, the enzyme that activates isoniazid, rather than in InhA itself. Compounds that bind directly to InhA bypass the need for KatG activation and therefore retain activity against the most prevalent isoniazid-resistant strains. The three candidates identified could serve as starting points for developing new TB drugs effective against MDR strains.

---

## Task 11
**Instruction:** Answer the following epidemiology question about tuberculosis using current global health knowledge.

**Input:** Which populations are at the highest risk of progressing from latent TB infection to active TB disease?

**Output:** People living with HIV have the highest risk of progression from latent to active TB, particularly if untreated, with an annual risk of approximately 5–10%. This risk is substantially reduced with effective antiretroviral therapy. Other high-risk groups include individuals on immunosuppressive therapies (such as TNF-alpha inhibitors), people with diabetes, those who are severely malnourished, patients with silicosis, and individuals recently infected within the past two years. Children under five years of age are also at increased risk of progression and severe forms such as miliary TB and TB meningitis.

---

## Task 12
**Instruction:** Compare the two TB treatment drugs described below in terms of mechanism of action, spectrum of activity, and known resistance mechanisms.

**Input:** Drug A: Bedaquiline. Drug B: Linezolid.

**Output:** Bedaquiline inhibits mycobacterial ATP synthase, targeting the c subunit and disrupting energy production. It is active against both replicating and non-replicating M. tuberculosis and is a key component of MDR/RR-TB regimens. Resistance most commonly arises through mutations in regulatory genes such as Rv0678, leading to overexpression of the MmpL5 efflux pump, although atpE mutations can also occur.

Linezolid is an oxazolidinone that inhibits protein synthesis by binding the 50S ribosomal subunit and preventing formation of the initiation complex. It is active against drug-resistant TB strains. Resistance occurs via mutations in the rrl gene (23S rRNA) or rplC. Compared to bedaquiline, linezolid is associated with significant toxicity, including myelosuppression and peripheral neuropathy with prolonged use.

---

## Task 13
**Instruction:** Based on the following abstract, identify what the authors state remains unknown or unresolved after their study.

**Input:** Our findings demonstrate that PimB catalyses the second mannosylation step in phosphatidylinositol mannoside biosynthesis, a pathway essential for lipoarabinomannan production. However, the precise structural basis for PimB substrate specificity and whether additional mannosyltransferases act redundantly at this step in different Mycobacterium species remain to be determined. The contribution of individual PIM species to host immune evasion also warrants further investigation.

**Output:** Three questions remain unresolved: the structural mechanism underlying PimB's substrate specificity, whether other mannosyltransferases can substitute for PimB in related mycobacterial species, and the specific roles of individual PIM intermediates in helping M. tuberculosis evade the host immune response.

---

## Task 14
**Instruction:** Summarise the following paper abstract in one paragraph suitable for inclusion in a literature review.

**Input:** Title: Mycobacterial Cells Have Dual Nickel-Cobalt Sensors. Abstract: Mycobacterium tuberculosis must sense and respond to metal ion concentrations in the host environment. We identified two transcriptional regulators, KmtR and NmtR, that respond to cobalt and nickel respectively. Structural and biochemical analyses revealed distinct metal coordination geometries that confer selectivity. Deletion of either regulator resulted in metal sensitivity and attenuated survival in macrophages, suggesting that tight metal homeostasis is required for intracellular survival.

**Output:** This study characterised two transcriptional regulators in M. tuberculosis, KmtR and NmtR, that separately sense cobalt and nickel through distinct metal coordination mechanisms. Deletion of either regulator impaired metal homeostasis and reduced bacterial survival in macrophages, demonstrating that precise regulation of intracellular metal levels is important for the pathogen's ability to survive within the host.

---

## Task 15
**Instruction:** Given the following TB diagnostic test results, state which test would be most appropriate as a confirmatory follow-up and explain why.

**Input:** A patient presents with a three-week cough, night sweats, and weight loss. A tuberculin skin test (TST) returns positive at 14mm induration. Chest X-ray shows upper lobe infiltrates with early cavitation. Sputum smear microscopy is negative for acid-fast bacilli.

**Output:** Sputum nucleic acid amplification testing (NAAT), such as the Xpert MTB/RIF assay, is the most appropriate next step for rapid confirmation. Smear microscopy has limited sensitivity, so a negative result does not exclude active TB. NAAT can detect M. tuberculosis DNA even in smear-negative cases and simultaneously assess rifampicin resistance. Mycobacterial culture should also be performed, as it remains the gold standard for diagnosis and allows comprehensive drug susceptibility testing, although results take longer. Given the clinical and radiological findings, active TB remains highly likely.

---

## Task 16
**Instruction:** Formulate a testable research hypothesis based on the following observation.

**Input:** Transcriptomic analysis of M. tuberculosis isolated from the lungs of immunocompromised mice shows significant upregulation of genes involved in lipid catabolism and downregulation of genes involved in carbohydrate metabolism compared to bacteria grown in standard laboratory culture.

**Output:** Hypothesis: M. tuberculosis preferentially catabolises host-derived lipids rather than carbohydrates as its primary carbon source during infection in vivo, and disruption of key lipid catabolism genes will attenuate bacterial survival and virulence in a mouse infection model. This is testable by generating targeted deletion mutants of upregulated lipid catabolism genes and comparing their growth and bacterial burden in immunocompromised mice to the wild-type strain.

---

## Task 17
**Instruction:** Extract all quantitative results from the following results passage and present them in a structured list.

**Input:** Infection of THP-1 macrophages with the ΔfadE28 mutant resulted in a 3.2-fold reduction in intracellular bacterial counts at 72 hours compared to wild-type H37Rv (p < 0.001). In the murine aerosol infection model, lung CFU counts at week 4 were reduced by approximately 1.8 log10 in mice infected with the mutant strain (p = 0.003). Spleen CFU counts showed a smaller but significant reduction of 0.9 log10 at the same time point (p = 0.01). Complementation of the mutant with the wild-type fadE28 allele restored lung CFU to within 0.2 log10 of the parental strain at week 4 (p = 0.43, not significant).

**Output:** - Intracellular survival (THP-1 macrophages, 72h): 3.2-fold reduction in ΔfadE28 vs wild-type, p < 0.001. - Lung CFU (mouse aerosol model, week 4): ~1.8 log10 reduction in ΔfadE28 vs wild-type, p = 0.003. - Spleen CFU (mouse aerosol model, week 4): 0.9 log10 reduction in ΔfadE28 vs wild-type, p = 0.01. - Lung CFU after complementation (week 4): restored to within 0.2 log10 of parental strain, p = 0.43 (not significant).

---

## Task 18
**Instruction:** Explain why the following gene is considered essential for M. tuberculosis survival and what this means for its potential as a drug target.

**Input:** inhA, which encodes enoyl-ACP reductase, an enzyme involved in the final reductive step of fatty acid elongation in the mycobacterial type II fatty acid synthesis system.

**Output:** inhA is essential because mycobacteria rely on the type II fatty acid synthesis system to produce mycolic acids, which are long-chain fatty acids unique to mycobacteria and critical components of the mycobacterial cell wall. Without functional InhA, the cell cannot complete fatty acid elongation, mycolic acid biosynthesis fails, and the cell wall is disrupted, leading to bacterial death. Humans use a type I fatty acid synthesis system with a fundamentally different architecture, meaning InhA has no direct human counterpart. This makes it an attractive drug target: inhibitors can be designed to be selectively toxic to mycobacteria without affecting human metabolism. Isoniazid already exploits this vulnerability, and the structural characterisation of InhA has enabled the development of direct InhA inhibitors that bypass isoniazid resistance.

---

## Task 19
**Instruction:** The following two studies report conflicting results on the same topic. Describe the possible reasons for the discrepancy.

**Input:** Study A reports that deletion of the mmpL3 gene in M. tuberculosis is bactericidal under standard in vitro growth conditions. Study B reports that a conditional knockdown of mmpL3 expression results in growth arrest but not cell death under the same conditions, with viability restored upon re-expression of the gene.

**Output:** Several factors could explain the discrepancy. First, complete gene deletion in Study A eliminates all MmpL3 protein and any potential residual activity, whereas a conditional knockdown in Study B may not achieve full depletion, leaving enough residual protein to prevent outright cell death. Second, the two approaches differ in kinetics: deletion is instantaneous while knockdown depletion is gradual, and bacteria may adapt to low MmpL3 levels differently than to its complete absence. Third, the genetic backgrounds of the strains used, or differences in the growth media composition, could influence the outcome. Finally, bactericidal versus bacteriostatic outcomes can be sensitive to how cell viability is measured, and differences in CFU counting methodology or time points assessed between the two studies may contribute to the apparent conflict.

---

## Task 20
**Instruction:** Given the following description of a TB patient's treatment history and current bacterial isolate resistance profile, recommend an appropriate treatment regimen and justify each drug choice.

**Input:** A 34-year-old patient previously treated for TB with a standard first-line regimen of HRZE (isoniazid, rifampicin, pyrazinamide, ethambutol) relapsed after six months. Drug susceptibility testing of the current isolate shows resistance to isoniazid and rifampicin. The isolate is susceptible to fluoroquinolones, bedaquiline, linezolid, clofazimine, and pyrazinamide. The patient is HIV-negative with no significant comorbidities.

**Output:** This is a case of MDR/RR-TB (resistant to isoniazid and rifampicin). Based on current WHO recommendations, an appropriate all-oral longer regimen should include Group A drugs: bedaquiline, a fluoroquinolone (levofloxacin or moxifloxacin), and linezolid. These form the core of treatment due to their strong efficacy. Additional agents such as clofazimine (Group B) should be included to strengthen the regimen. Pyrazinamide may be added if susceptibility is confirmed, as it has activity against semi-dormant bacilli, although its contribution can vary. The regimen should include at least four effective drugs, tailored to susceptibility results. Monitoring is essential: bedaquiline and clofazimine can prolong QT interval, and linezolid is associated with myelosuppression and neuropathy. Treatment duration typically ranges from 6–18 months depending on regimen type and response.
