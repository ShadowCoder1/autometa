# Reference (gold-standard) data for the Cisneros et al. (2024) aging meta-analysis

- `late_gsheet.csv`, `aft_gsheet.csv`, `combined_gsheet.csv` — the "updated spreadsheets" referenced in
  `Aging_Meta_Analysis_v5.Rmd` (public Google Sheets export, fetched 2026-08-14). Per-dataset human
  extraction: Author, Year, Task, N, figure used, dependent measure, phase, TE (Cohen's d), CI, seTE.
- `Adaptation_Forest_Aging_LA.xlsx`, `Adaptation_Forest_Aging_AFT.xlsx`, `Aging_Meta_Analysis_v5.Rmd` —
  from OSF https://osf.io/s7h5e/ ("Meta-analysis" folder). Analysis: meta::metagen(sm="SMD",
  random=TRUE, method.tau="REML", hakn=FALSE), forest.meta sorted by TE with prediction interval.
- `Late_Adaptation_v2.ipynb`, `Aftereffects_v2.ipynb` — Elizabeth Cisneros' per-study Cohen's d
  computations (WebPlotDigitizer values → SD → pooled-SD Cohen's d; SE(d); 95% CI).

These files are used ONLY by validation scripts (validation/), never by the pipeline itself.
