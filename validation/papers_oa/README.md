# Open-access validation corpus (not distributed)

The 21 PDFs this directory held are not committed. "Open access" here meant *freely retrievable*
— from Europe PMC, PubMed Central, NASA NTRS, a repository or an author's page — which is not the
same as being licensed for redistribution, so the safe thing is to ship the recipe rather than the
files.

Everything needed to rebuild the corpus is here:

- **`MANIFEST.csv`** — one row per paper: author, year, title, DOI, whether an open copy was found,
  the file name it was saved under, the exact `source_url` it came from, and a note recording how
  that copy was verified as the right paper.
- **`fetch_oa_papers.py`** — re-fetches them from those sources into this directory.
- **`fetch_report_*.csv`, `run_log*.txt`** — what each fetching pass actually did, kept so the
  corpus is auditable rather than merely reproducible.

Papers marked `paywalled` in the manifest had no open copy at the time of the search; the notes
column records every index that was checked. Supply those yourself if you have institutional
access.
