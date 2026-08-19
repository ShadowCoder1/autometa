# Evaluation of `runs/nine` against Cisneros 2024 — 2026-08-18

Code at HEAD `a9a5883`. Run: 9 PDFs → 6 eligible, 3 excluded; 20 extracted cells; 3 pooled; 34
questions; late adaptation k=2 (d = −0.74 [−2.47, 1.00]), aftereffect k=0. Spend on this run
$56.69 over three passes (pass 1 died at $12.93 on a PyMuPDF float-dpi bug; pass 2 $34.47; pass 3
$9.29). Gold: `validation/reference/cisneros2024/{late,aft}_gsheet.csv` + her two notebooks (raw
WebPlotDigitizer read-outs).

## 1. Every number, correctly paired with hers

Pairing is by the paper's own group labels and n (not list order). "Held" = the tool produced the
value but did not pool it; "no value" = it produced no effect size.

### Late adaptation

| dataset | Cisneros d (n young/old) | tool d (n) | status | Δ |
|---|---|---|---|---|
| Bock 2005, 60° | −1.676 (12/12) | −1.659 (12/12) | pooled | 0.017 |
| Cressman 2010, 30° | −0.224 (10/9) | −0.226 (10/9) | held (#5, #6 confirm) | 0.002 |
| Vachon 2020, non-instructed | +0.170 (16/18 analysed) | +0.112 (20/19 recruited) | pooled | 0.058 — mostly the young CI (tool 2.6 vs her 3.760, the same duplicated value as below) |
| Vachon 2020, instructed | +0.325 (21/17) | +0.831 (21/19) | held (#1, #2) | 0.51 — means agree (28.16/30.31 vs 28.01/30.20); her CI for instructed-old (3.760) is identical to her non-instructed-young CI (3.760), i.e. the gap is in the dispersion and possibly her copy-paste |
| Buch 2003, sudden 90° | −0.437 (sheet says 10/10; her seTE 0.642 is only reproducible with 5/5) | −0.489 (5/5) | held (#7, #8) | 0.052 |
| Buch 2003, gradual | not in her analysis | −0.099 (5/5) | held (#3, #4) | — |
| Heuer & Hegele 2008, Exp 1a | −0.624 (20/20, Fig 2a) | no d pooled — the RESOLVED value is the printed text pair 27.7°/18.9° (last practice block; no SD, so unconvertible), chosen by route precedence over the Fig 2a candidates −27.0 (SE 4.71) / −40.3 (SE 4.75), which are exactly her across-target means (27.11 / 40.33, SE 4.76 / 4.71) and give **d = −0.627** | held (#13, #14; picking "v2" in both reproduces her number) | 0.003 if the figure candidate were taken |
| Heuer & Hegele 2008, Exp 2 | not in her analysis | no d (text −6.4/−7.0 is the wrong quantity, verifier right) | held (#15, #16) | — |
| Langan & Seidler 2011, 30° | +0.654 (9/9; her reads old −15.71 / young −20.88) | no d: older −15.8 (right, panel B) / young −18.5 = the average of a correct panel-A read (−20.5) and a wrong-panel-B read (−16.5), because the mapper listed BOTH panels for each group and the resolver averaged the two locator ensembles; series identity (black vs grey = presentation order) unresolved | held (#9–#10) | — |
| Langan & Seidler 2011, 45° | −1.275 (9/9) | one group only | held (#11–#12) | — |
| Wolpe 2020 | −0.374 (109/108, Fig 1c age bands) | excluded by mapper: age continuous, bands "for illustration purposes only" | excluded | judgment call — reversible with an eligibility override |
| Anguera 2010 | −0.379 (18/18) | the PDF in the folder is a different Anguera 2010 (young-only fMRI) → correctly excluded | wrong PDF | — |
| Pan & Van Gemmert 2013 | +0.642, +0.913 (14/14) | the PDF in the folder is a different Pan & van Gemmert paper (young-only) → correctly excluded | wrong PDF | — |

### Aftereffect

| dataset | Cisneros d | tool | status | Δ |
|---|---|---|---|---|
| Cressman 2010 | −0.096 | −0.105 | accepted (alone → not pooled) | 0.009 |
| Bock 2005 | −0.537 | means −21.3 (SD 7.8) / −25.4 (SD 5.7) → −0.600 once "larger deviation = larger aftereffect" is applied | held (#25, #26 orientation) | 0.063 |
| Heuer & Hegele 2008, Exp 1a | +0.284 | −13.18 (SE 2.23) / −10.14 (SE 2.38) → +0.295 | held (#27, #28 orientation) | 0.011 |
| Vachon 2020, non-instructed | +0.118 (her reads 15.71 CI 3.11 / 15.06 CI 1.8) | one found read 15.89 CI 3.18 / 15.13 CI 1.85 → +0.13; the two model read-outs were nulled by the categorical-x-axis rule ("nothing says whether the x categories are conditions or the groups") and the found read was then dropped by overlay verification → score 0.00 | held (#21, #22) | 0.01 |
| Vachon 2020, instructed | +1.180 (19.8 CI 1.88 / 13.58 CI 2.62) | one found read 19.81 CI 2.03 / 13.55 CI 2.34 → +1.24, discarded the same way | held (#23, #24) | 0.06 |
| Buch 2003 (sudden, Fig 4) | +0.676 (her Fig 4 means 2.991 SE 0.424 / 1.886 SE 0.839; her notebook cell computes 0.743) | reads from two figures: old −1.8…−3.84, young −1.72…−4.56 (one sonnet read 0.0 ± 411); opus's Fig 3 post-exposure reads −3.0 / −1.83 match her means almost exactly → d ≈ +0.9 (gap = young SE 0.69 vs 0.839) | held (#31–#34) | genuinely hard bar chart, n = 5 |
| Langan & Seidler 2011, 30° | +0.58 | 12.4 (SE 3.5) / 6.0 (SE 1.1) → +0.82; the numbers match Fig 1B/1A at AE by eye, the locator text does not | held (#17–#18) | 0.24 |
| Langan & Seidler 2011, 45° | +0.972 | 12.0 / 6.8, dispersion type unknown | held (#19–#20) | — |
| Heuer & Hegele 2008, Exp 2 | not in her analysis | −1.72 / −2.42 (across-target average; text says ≈ −8° at the practised target) | held (#29, #30) | — |

**Summary.** On the seven correct papers, wherever the tool produced an effect size it is within
0.06 of hers (Bock, Cressman ×2, Vachon NI, Buch sudden) except Vachon-instructed (0.51, held; the
gap is in the dispersion, where her own notebook carries a duplicated CI value). Wherever it produced
means but no d (Bock aft, Heuer aft ×1 and late ×1 via its figure candidates, Vachon aft ×2, Langan
aft 30°) the implied d is within 0.07 of hers in six of seven cases. It never pooled anything with the
wrong sign. Caveats the adversarial check added: two of the agreements rest on a different raw quantity
than hers (Bock late = the paper's printed fit asymptotes, not her digitised markers; Buch sudden late =
fitted-curve endpoints); and Vachon's n are recruited, hers analysed (she is right). It also caught two
things she has differently: Buch's n (5/5 — her SE agrees, only her sheet's N column says 10/10) and
the duplicated Vachon CI.

Run-to-run stability: `ceiling-3` pooled Cressman late (−0.207) and held Bock late (−1.755); `nine`
pooled Bock (−1.659) and held Cressman (−0.226). Values stay within 0.1, but which cells clear the
gate is not stable between runs — the gate flickers on cache/route mix.

Independent check: every claim in this section was adversarially re-derived from the files by a second
agent (`scratchpad/eval/adversarial-report.md`): 17 confirmed, 0 refuted, 7 partly (corrected above),
2 judgement calls.

### What the forest plot would look like if the tool went with its best guess (its own numbers, no human)

Computed with the repo's REML pooling (`canopy.stats.meta.random_effects`), same estimator settings:

- **Late adaptation, k = 6** (Bock, Cressman, Vachon ×2, Buch ×2): **d = −0.21 [−0.91, +0.49]**, I² 74%.
  Cisneros on the same five datasets: −0.32 [−1.02, +0.39]. (Heuer and Langan cannot be built without dispersion / series identity.)
- **Aftereffect, k = 7** (Cressman, Bock, Heuer, Vachon ×2, Buch, Langan 30°): **d = +0.36 [−0.13, +0.84]**, I² 59%.
  Cisneros on the same seven: +0.31 [−0.13, +0.74].
- Her full analysis (50 / 40 datasets): late −0.5 [−0.7, −0.3]; aftereffect +0.4 [+0.2, +0.5].
  Her conclusion: "Older individuals exhibited a marked impairment in their ability to employ an
  explicit strategy … However, they exhibited enhanced implicit recalibration."

So on this seven-paper subset the tool's best guess points the same way she does on both outcomes,
with the same non-significance at k ≈ 7. The tool has no "best guess" mode and writes no
conclusion; both are missing features (see §3).

## 2. The 34 questions, honestly

| cluster | ids | what they are | verdict |
|---|---|---|---|
| orientation | #10 #11 #12 #19 #25 #26 #27 #28 (8) | "does a larger raw value mean more?" — asked once per GROUP although each says the answer settles every cell of that measure in the paper | **automatable**; at most 3–4 distinct decisions, decidable from the measure name + protocol definition by a rule or one adjudicator call |
| Langan verifier-refuted | #9 #17 #18 #20 (4) | verifier: "the locator says Fig 1A but the caption says A = YA" — one mapper defect (both panels listed for each group) surfaced four times, plus a resolver defect (averaging across the two locators); the aftereffect values themselves are right, the late young value is a right/wrong-panel average | **should be one automatic re-map (or one question)**, not four |
| Vachon aft confirms | #21 #22 #23 #24 (4) | reads that match her WebPlotDigitizer values within 0.2°, nulled by the categorical-x-axis rule and an overlay drop; the questions carry no dispersion and no reason | **pipeline defect** (the categorical-x rule fires on a figure whose x is condition and whose series are the groups) — automatable |
| Buch which-ladder | #3 #4 #7 #8 #33 #34 (6) | axis calibration disputed on a standardized-score plot; #3/#4/#7/#8 options differ by 0.06–0.25 z (impact 0.20 / 0.09); #33/#34 span two ladders (1.4–1.8 z) but that cell is unconvertible anyway | low value for #7/#8 (an impact threshold removes them); the rest should be one question per dataset with the figure |
| confirms | #5 #6 (Cressman late) | 31.4 vs 31.78; d within 0.002 of hers; ceiling-3 pooled it, nine held it | low value — pool with a note in a best-guess mode |
| Vachon-instructed late | #1 #2 | the mean is fine (30.2 vs her 30.31); the disputed quantity is the CI (1.5 vs 0.95; hers 3.760 = a duplicated value); impact 0.53 d; #1 was already answered "yes" (human_override) | **legitimate** — but the question should have been about the ribbon width, not the mean |
| Heuer Exp 1a late | #13 #14 (2) | resolved value (text 27.7/18.9) has no SD; the figure alternative "v2" (−27.0 SE 4.71 / −40.3 SE 4.75) IS her number (d −0.627 vs −0.624); the verifier's "single target" objection is wrong for it | **automatable**: an unconvertible resolved value must yield to a convertible candidate (with a precedence-override flag); as posed, a human picking v2 twice gets it right |
| Heuer Exp 2 | #15 #16 #29 #30 (4) | a dataset she did not use; the resolved text values −6.4/−7.0 are the wrong quantity (verifier right); #15/#16's v1 are Fig 6a across-direction averages; #29/#30 average Fig 6b across targets where the paper says the effect is local (≈ −8° at the practised target) | should have been ONE inclusion question ("Exp 2 is a single-target generalisation design — include?") first |
| Buch aftereffect | #31 #32 (2, plus #33 #34 above) | two figures read for one quantity, wide spread, n = 5 | **legitimately hard**; one question with the figure |

Net: 34 questions ≈ 13 distinct decisions (the adversarial checker's count; mine was ~10). Of the 13:
~7 automatable (orientation ×3–4, Langan panel ×1, Vachon aft ×1, Heuer 1a precedence ×1), ~4 real
(Buch sudden aftereffect, Vachon-instructed dispersion, Heuer Exp 2 inclusion, Langan series identity),
~2 low-value (Cressman late confirm, Buch sudden-late ladder). Three structural fixes drive most of the
reduction: ask per dataset/measure not per group; treat verifier objections about the MAP (wrong
panel, wrong quantity) as re-map actions; never let an unconvertible resolved value outrank a
convertible candidate. One question that should exist and does not: Wolpe's eligibility call was made
silently by the mapper ($1.63) with no reviewable question.

## 3. Missing features
- **Best-guess mode**: no CLI flag / UI toggle exists (`--resume`, `--budget-usd`, caps only). Needed: a
  second analysis line ("strict" = today's; "best guess" = strict + held rows at their resolved value,
  orientation from measure semantics, lone confident reads) shown side by side, every best-guess row
  marked, with the verifier-refuted rows following the verifier (Heuer's text value is unbuildable, not
  wrong; nothing here would have pooled a wrong-signed value).
- **Conclusion**: report has no interpretation section. Needed: direction + magnitude + CI + I² +
  prediction interval + how many rows are held and what they would change, in plain sentences, per outcome.

## 4. Why only nine papers
`~/Downloads/Systematic Review/papers` = 79 files = 9 unique PDFs (sha256); the zip they came from
(`CisnerosPapersStudy.zip`, 2024-08-07) already contains only those 9. Two of the 9 are the wrong
paper (Anguera; Pan & van Gemmert — same authors/year, young-only studies). Cisneros' corpus is
~42 published studies (44 author-year rows incl. duplicates and 2 unpublished). Already fetched
open-access: 10 in `validation/papers_oa/` (Bindra 2021, Binyamin-Netser 2023, Cornelis 2022,
Hegele & Heuer 2010, Hermans 2025, Kitchen & Miall 2021, Li 2021, Panouillères 2015, Uresti-Cabrera
2015, T. Wang 2022). Need manual retrieval (library/browser): 22 listed in
`validation/papers_oa/MANIFEST.csv`, of which Wang 2011, Hardwick 2014, Nemanich 2015, Bansal 2023
are open on PMC/Springer behind bot gates a browser passes.

## 5. Cost (from `scratchpad/eval/cost-audit.md`, a read-only code + records audit)

Spend so far on Canopy's API key: ≈ $39 (validation before 08-17) + $1 fixtures + $16.08 (ceiling-3)
+ $56.69 (nine) ≈ **$113**. Per eligible paper in `nine`: Bock $0.75, Cressman $1.35, Vachon $3.45,
Heuer $10.45, Buch $11.10, Langan ≈ $16.65 (three passes; $5.40 of it the map). Excluding a paper at
the map costs $0.34–1.63. Stage split of the big pass: digitize 65%, map 19%, verify 16%.

Why the hit rate is what it is (60–78% of input tokens read from cache in our passes): one breakpoint per
digitize call on the image; system + tools + image are the cached prefix and the per-cell task text
comes after it (right order); but every cell buys a sonnet-5 read-out as the mandatory second family
(27.6% of read-out calls) which can never hit the opus prefix, and the mapper deliberately opts out
(four calls, four schemas). The email's "17%" is Anthropic's generic estimate over the whole org's
direct traffic (which also carries the Wren proxy); Canopy's own runs are already at 60–78%.

Where the money actually goes / is lost:
1. **The per-paper cap.** In pass 2, three papers hit the $8 cap and errored with nothing pooled —
   $24.63 of $34.47 (71%). The calls are disk-cached so a resume recovers them for ~$0 (that is what
   pass 3 did), but a user who does not know to resume loses the papers. Fix: fail soft (finish with
   what exists, flag thin coverage) — the cheapest lever, ~20–30% of nominal spend turned into output.
2. **Third read-outs bought needlessly.** Early stopping exists (`digitizer.py:784/1981`) but requires
   agreement on mean AND error half-length within a tight tolerance; Heuer's −27.01/4.76 vs −27.13/4.93
   still bought a third read. Loosening the SE side of the test is a low-risk cut of maybe 10–15% of
   digitize.
3. **Batches API** (−50% list price) is unused; only the two mandatory up-front read-outs + coords are
   batchable without redesign (the zoom tool-loop and the adaptive third read are synchronous by
   construction) → ~10–15% of a run, at real engineering cost and hours of turnaround.
4. `coords()` bought unconditionally (~8–12%, but it is a corroborating witness — needs the panel);
   overlay-verify trigger/iterations (~3–7%); mapper prefix caching (~2%).
5. 28% of digitize candidates return no value ("ambiguous", mean None), concentrated in three papers
   (Langan 60%, Buch 47%, Vachon 46%) — figure legibility, and the categorical-x rule above.

Projection, 50 PDFs → ~30 eligible: **$170–360 per full run today** (of which ~$70–120 stalls at the
cap without a resume); with the levers **≈ $120–260**, and ~all of it turned into rows. Every prompt
change invalidates the cache, so a full re-run after a prompt change costs the same again; iterate on
the cached records (free), then run the corpus once.

The larger cost line is not the API: the development agents in this session are an order of magnitude
above the ~$113 of API spend.
