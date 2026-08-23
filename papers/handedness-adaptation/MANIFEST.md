# Learning meta-analysis — included studies

The 23 studies pooled in the **adaptation** meta-analysis (dominant vs non-dominant limb).
Not the transfer meta-analysis, which is a separate set of 27 studies.

**Where this list comes from.** These studies are *not* in the preprint's reference list — they
appear only as labels inside the Figure 4 forest plots (4A late adaptation, 29 datasets; 4B
aftereffects, 19 datasets). The list below was read off those two panels and resolved to DOIs via
Crossref. 23 unique studies, which matches the count the paper states.

`probable` means the forest-plot label was ambiguous (a common surname plus a year) and the DOI is
a best match on author, year, task type and sample size rather than a certainty — check those
against the forest plot before trusting them.

---

## In this folder (5)

| Study | DOI | Confidence |
|---|---|---|
| Carroll et al. 2016 — *Enhanced crosslimb transfer of force-field learning…* | `10.1152/jn.00485.2015` | confirmed |
| Kumar et al. 2020 — *Mechanistic determinants of effector-independent motor memory encoding* (PNAS) | `10.1073/pnas.2001179117` | probable |
| Poh et al. 2016 — *Effect of coordinate frame compatibility on the transfer…* | `10.1152/jn.00410.2016` | confirmed |
| Yadav & Sainburg 2014 — *Limb Dominance Results from Asymmetries in Predictive and Impedance Control* | `10.1371/journal.pone.0093892` | confirmed |
| Yokoi et al. 2011 — *Gain Field Encoding of the Kinematics of Both Arms…* (J Neurosci) | `10.1523/JNEUROSCI.4634-11.2011` | probable |

---

## Still to fetch (18)

Free full text exists for the first eight — the download just could not be scripted (PMC, HAL and
Wiley all gate automated access now). Open the link and save into this folder.

| Study | DOI | Free copy |
|---|---|---|
| Sainburg 2002 — *Evidence for a dynamic-dominance hypothesis of handedness* | `10.1007/s00221-001-0913-8` | PMC10710695 |
| Sainburg & Wang 2002 — *Interlimb transfer of visuomotor rotations: independence of direction and final position* | `10.1007/s00221-002-1140-7` | PMC10704413 |
| Wang & Sainburg 2003 — *Mechanisms underlying interlimb transfer of visuomotor rotations* | `10.1007/s00221-003-1392-x` | PMC3697093 |
| Wang & Sainburg 2006 — *Interlimb transfer of visuomotor rotations depends on handedness* | `10.1007/s00221-006-0543-2` | PMC10705045 |
| Wang et al. 2011 — *Aging reduces asymmetries in interlimb transfer of visuomotor adaptation* | `10.1007/s00221-011-2631-1` | PMC3116897 |
| Duff & Sainburg 2007 — *Lateralization of motor adaptation reveals independence in control of trajectory and steady-state position* | `10.1007/s00221-006-0811-1` | PMC10681153 |
| Coudière et al. 2023 — *Right-left hand asymmetry in manual tracking…* | `10.1007/s00426-023-01858-0` | hal.science/hal-04198019 |
| Scarpina et al. 2015 — *Prism adaptation changes the subjective proprioceptive localization of the hands* | `10.1111/jnp.12032` | Wiley (free to read) — **probable** |

Institutional access needed (CMU library):

| Study | DOI |
|---|---|
| Addison & Van Gemmert 2023 — *Bilateral Transfer of a Visuomotor Task in Different Workspace Configurations* | `10.1080/00222895.2023.2293002` (green copy: repository.lsu.edu/kinesiology_pubs/441) |
| Bagesteiro et al. 2021 — *Interlimb differences in visuomotor and dynamic adaptation…* — **probable** | `10.1016/j.humov.2021.102788` |
| Balitsky Thompson & Henriques 2010 — *Visuomotor adaptation and intermanual transfer under different viewing conditions* | `10.1007/s00221-010-2155-0` |
| Carroll et al. 2014 — *New visuomotor maps are immediately available to the opposite limb* | `10.1152/jn.00042.2014` |
| Kirby et al. 2019 — *Brain functional differences in visuo-motor task adaptation…* | `10.1007/s00221-019-05653-5` (green copy: digitalcommons.lsu.edu/psychology_pubs/721) |
| Kumar & Mutha 2023 — *Spontaneous recovery in an untrained arm…* | `10.1037/xhp0001124` |
| Redding & Wallace 2009 — *Asymmetric Visual Prism Adaptation and Intermanual Transfer* | `10.1080/00222895.2009.10125920` |
| Schabowsky et al. 2007 — *Greater reliance on impedance control in the nondominant arm…* | `10.1007/s00221-007-1017-x` |
| Striemer & Morrill 2023 — *Direction of visual shift and hand congruency enhance spatial realignment…* | `10.1007/s00221-023-06697-4` (preprint: `10.31234/osf.io/trk23`) |
| Jones et al. 2020 — *Does hand-dominance matter in non-standard visuomotor transformations?* — **DOI unresolved**, PubMed `32933401` | — |

---

## One thing to check first

Several of these test **both limbs in the same participants**. In Figure 4 the `N` column is the
dominant and non-dominant group sizes combined, so a within-subject study appears as e.g. N = 48
for 24 people. Canopy computes an independent-groups standardized mean difference and will treat
those two arms as separate samples. That is the second limitation recorded in the protocol, and it
is worth confirming against a couple of these papers before comparing pooled numbers.
