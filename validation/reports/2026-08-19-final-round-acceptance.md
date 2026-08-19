# Final round — acceptance on the regenerated `runs/nine` (2026-08-19)

Code: 21 commits a9a5883..HEAD (see `.superpowers/sdd/2026-08-15-canopy-implementation/final/DECISION.md`, PLAN.md, per-task reviews).
Resume: `canopy run … --resume --budget-usd 12 --max-usd-per-paper 6` → **$6.76**, 875 calls (815 cached), 805 s, exit 0, no paper in error (`runs/nine/run4.log`).

## Strict line (pooled without a human) — after the two whole-branch fix rounds (HEAD 2d435aa; resumes $6.76 + $4.95)
| dataset | Cisneros d | tool d | Δ |
|---|---|---|---|
| Bock 2005 late | −1.676 | −1.659 | 0.017 |
| Vachon 2020 non-instructed late | +0.170 | +0.112 | 0.059 |
| Bock 2005 aftereffect | −0.537 | −0.600 | 0.063 (orientation settled by the tiebreak ballot, majority on the same scale) |
| Cressman 2010 aftereffect | −0.096 | −0.105 | 0.009 |
| Heuer & Hegele 2008 aftereffect | +0.284 | +0.295 | 0.011 (tiebreak ballot) |
| Vachon 2020 non-instructed aftereffect | +0.118 | +0.134 | 0.016 (categorical point read) |
Late k = 2: −0.74 [−2.47, 1.00]; aftereffect k = 4: +0.01 [−0.35, 0.37]. 6/6 within 0.10, 4/6 within 0.05, all signs agree.

## Best-guess line (strict + held rows at the value the tool read; each row carries rule + reason)
Late k = 6: **−0.48** — adds Cressman late −0.226 (hers −0.224), Heuer 1a late **−0.627 via `precedence_override`** (hers −0.624; the printed 27.7°/18.9° pair has no SD so the figure pair it quotes is used, flagged), Buch sudden −0.489 (−0.437), Buch gradual −0.099 (not in hers).
Aftereffect: nothing added — Heuer Exp 2 aft is vetoed (`sign_mismatch`), Buch gradual aft has one group only after the disagreeing-vote rule.
Held in neither line (reason on the card): Vachon instructed late +0.831 / aft +1.27 (hers +0.325 / +1.18; `series_identity_conflict`), Langan ×4 (`locator_panel_mismatch` / series identity; aft 30° reads +0.809 vs her +0.58), Buch sudden aft (one group only), Heuer Exp 2 late (`axis_conflict`).

## Site / artefacts
- Forests drawn by **R `meta::forest.meta`** for both outcomes and both lines (`pooled.json["renderer"]`, cross-check ok, meta 8.2.1): Author/Year/Task/Perturbation/N targets/N (O/Y) · Cohen's d / 95% CI / Weight · "Reduced in Old"/"Enhanced in Old" · xlim ±4 (cisneros2024 profile), sorted by TE, prediction interval; the best-guess forest is a `subgroup` plot (Confirmed / Best guess).
- Conclusion per outcome + overall (`pooled.json["conclusion"]`, `conclusion.md`, report.html, SPA card); deterministic, no forbidden words.
- Questions: **26 cards** (was 34 per-cell questions): pair 5, verifier_refuted 7, include_paper 3 (Wolpe, Anguera, Pan — answering "include" maps the paper on the next resume), precedence_override 2, analysed_n 2 (Vachon), which_series 2, which_value 2, which_axis 2, orientation 1; ordered by impact.
- SPA: Strict | Best guess toggle (default strict), best-guess badges with the reason, conclusion card, consolidated cards, renderer caption. Server restarted on http://127.0.0.1:8000.

## Deviations from the acceptance list (recorded, not hidden)
- Strict aftereffect is k = 4 (Bock, Cressman, Heuer, Vachon NI), not "k ≤ 1": the tiebreak ballot and the categorical point read released rows the plan expected to stay held; every one is within 0.07 of the human's value.
- Bock aftereffect settled only after the majority-on-the-same-scale ruling (whole-branch review finding 17).
- Vachon instructed aftereffect (+1.27) is vetoed by `series_identity_conflict`; answering its pair card releases it.
- Whole-branch review: 2 blockers + 14 majors found and fixed (commits 733fb8d, 73680d7); re-review verdict recorded in `.superpowers/sdd/2026-08-15-canopy-implementation/final/WHOLE-BRANCH-REVIEW.md`.
