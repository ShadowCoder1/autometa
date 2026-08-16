#!/usr/bin/env python3
"""
fetch_oa_papers.py

Build a validation corpus of open-access PDFs for the studies included in the
Cisneros et al. (2024) aging x sensorimotor-adaptation meta-analysis.

Pipeline per study:
  1. Union unique (Author, Title, Year) rows from late_gsheet.csv + aft_gsheet.csv,
     skipping "Unpublished" rows and rows with no title.
  2. Skip a fixed list of 9 studies we already have on disk.
  3. Look up the DOI via Crossref (bibliographic query on the title), verifying
     the returned title is actually a match (string-similarity + first-author
     surname check) before trusting it.
  4. Look up an open-access PDF location via Unpaywall, falling back to Europe
     PMC, for that DOI.
  5. Download the PDF with curl and verify it's really a PDF, of plausible
     size, whose extracted text mentions the title or first author.
  6. Save as <FirstAuthorLastName>_<Year>.pdf and record the outcome.

This script makes network calls to Crossref, Unpaywall, and Europe PMC (free,
public, no-auth APIs) and to whatever host serves the OA PDF. It never
attempts to bypass a paywall: if Unpaywall/EuropePMC/publisher-OA-page don't
offer a legal PDF, the study is recorded as "paywalled" and left alone.

Usage:
    python3 fetch_oa_papers.py --list          # just print the dedup'd, filtered study list
    python3 fetch_oa_papers.py --dry-run        # do DOI + OA lookups, no downloads
    python3 fetch_oa_papers.py                  # full run (resumable)
    python3 fetch_oa_papers.py --only "Bindra,2021"   # process a single study (debugging)
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import urllib.error
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.normpath(os.path.join(HERE, "..", "reference", "cisneros2024"))
LATE_CSV = os.path.join(REF_DIR, "late_gsheet.csv")
AFT_CSV = os.path.join(REF_DIR, "aft_gsheet.csv")
MANIFEST_PATH = os.path.join(HERE, "MANIFEST.csv")

# NOTE: Unpaywall hard-rejects placeholder addresses on the example.com domain
# (HTTP 422: "Please use your own email address in API calls"), which would
# silently zero out Unpaywall coverage and make every OA lookup fall through
# to Europe PMC only. Using a real, working contact address here, per
# Unpaywall's and Crossref's usage policies (https://unpaywall.org/products/api).
CONTACT_EMAIL = "sritej.paddy@gmail.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
CROSSREF_UA = f"canopy-validation-corpus/1.0 (mailto:{CONTACT_EMAIL})"

MAX_DOWNLOAD_ATTEMPTS = 4
MIN_PDF_BYTES = 50 * 1024
REQUEST_TIMEOUT = 60

# Studies we already have -- skip these. Matched on (normalized first-author
# token, year). Normalization strips " et al.", trailing/leading whitespace,
# and lowercases. See build_skip_key().
ALREADY_HAVE = {
    ("langan & seidler", 2011),
    ("wolpe", 2020),
    ("vachon", 2020),
    ("cressman", 2010),
    ("bock", 2005),
    ("heuer & hegele", 2008),
    ("anguera", 2010),
    ("buch", 2003),
    ("pan & van gemmert", 2013),
}

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "and", "to", "for", "with", "is",
    "are", "by", "as", "at", "from", "that", "this", "across", "during",
    "after", "between", "into", "their", "its", "vs", "via",
}


# --------------------------------------------------------------------------
# Text normalization helpers
# --------------------------------------------------------------------------

def ascii_fold(text):
    """Normalize unicode punctuation/diacritics to plain ASCII-ish text."""
    if text is None:
        return ""
    text = text.replace("‑", "-").replace("–", "-").replace("—", "-")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_for_compare(text):
    text = ascii_fold(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_similarity(a, b):
    na, nb = normalize_for_compare(a), normalize_for_compare(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def significant_words(text, min_len=4):
    words = normalize_for_compare(text).split()
    return [w for w in words if len(w) >= min_len and w not in STOPWORDS]


def first_author_surname(author_field):
    """Extract a best-guess surname token from a CSV Author field.

    Examples: 'Bock ' -> 'bock'; 'Vachon et al.' -> 'vachon';
    'Langan & Seidler' -> 'langan'; 'Heuer & Hegele' -> 'heuer'.
    """
    a = ascii_fold(author_field).strip()
    a = re.sub(r"\bet al\.?", "", a, flags=re.IGNORECASE).strip()
    a = re.split(r"[&,]", a)[0].strip()
    a = re.sub(r"\s+", " ", a)
    # last whitespace-separated token as surname (handles "Van Gemmert" ->
    # take last token 'gemmert' only if no '&'; for compound surnames we keep
    # the fallback of the whole first-chunk for filename use elsewhere).
    return a.lower()


def surname_last_token(author_field):
    s = first_author_surname(author_field)
    parts = s.split(" ")
    return parts[-1] if parts else s


def build_skip_key(author_field, year):
    a = ascii_fold(author_field).strip().lower()
    a = re.sub(r"\bet al\.?", "", a).strip()
    a = re.sub(r"\s+", " ", a)
    try:
        y = int(year)
    except (ValueError, TypeError):
        y = None
    return (a, y)


def filename_author(author_field):
    """First author's last name only, ASCII, for filenames."""
    a = ascii_fold(author_field).strip()
    a = re.sub(r"\bet al\.?", "", a, flags=re.IGNORECASE).strip()
    a = re.split(r"[&,]", a)[0].strip()
    tokens = [t for t in re.split(r"\s+", a) if t]
    if not tokens:
        return "Unknown"
    # Handle multi-token surnames like "Van Gemmert" by joining without
    # spaces only when the first token is a lowercase particle; otherwise
    # just take the last token as the surname.
    particles = {"van", "von", "de", "der", "den", "la", "le"}
    if len(tokens) > 1 and tokens[0].lower() in particles:
        surname = "".join(tokens)
    else:
        surname = tokens[-1]
    surname = re.sub(r"[^A-Za-z0-9]", "", surname)
    return surname or "Unknown"


# --------------------------------------------------------------------------
# CSV loading / union / dedup
# --------------------------------------------------------------------------

def load_studies():
    rows = []
    for path in (LATE_CSV, AFT_CSV):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                author = (row.get("Author") or "").strip()
                title = (row.get("Title") or "").strip()
                year_raw = (row.get("Year") or "").strip()
                if not title or title == "-":
                    continue
                if "unpublished" in title.lower() or "unpublished" in year_raw.lower():
                    continue
                m = re.match(r"^(\d{4})", year_raw)
                if not m:
                    continue
                year = int(m.group(1))
                rows.append((author, title, year))

    seen = {}
    for author, title, year in rows:
        key = (normalize_for_compare(author), normalize_for_compare(title), year)
        if key not in seen:
            seen[key] = (author.strip(), title.strip(), year)
    return list(seen.values())


def filter_studies(studies):
    keep = []
    skipped = []
    for author, title, year in studies:
        key = build_skip_key(author, year)
        if key in ALREADY_HAVE:
            skipped.append((author, title, year))
        else:
            keep.append((author, title, year))
    return keep, skipped


def merge_title_duplicates(studies):
    """Collapse rows sharing the same (author, title) that disagree on year
    across the two source spreadsheets (epub-ahead-of-print vs. issue-date
    drift between late_gsheet.csv and aft_gsheet.csv) into one study entry,
    so we don't fetch/download the same physical paper twice under two
    filenames. Rows with the same author+year but genuinely DIFFERENT titles
    (e.g. two separate 2015 Uresti-Cabrera papers) are left as separate
    studies.

    Returns (merged_list, notes_by_key) where notes_by_key is keyed by
    (normalized_author, normalized_title) and holds a human-readable note
    about the year disagreement for the manifest.
    """
    groups = {}
    order = []
    for author, title, year in studies:
        key = (normalize_for_compare(author), normalize_for_compare(title))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((author, title, year))

    merged = []
    notes_by_key = {}
    for key in order:
        rows = groups[key]
        years = sorted(set(y for _, _, y in rows))
        author, title = rows[0][0], rows[0][1]
        nominal_year = years[0]
        merged.append((author, title, nominal_year))
        if len(years) > 1:
            notes_by_key[key] = (
                "source spreadsheets disagree on year ("
                + "/".join(str(y) for y in years)
                + "); treated as one study, filename year taken from Crossref"
            )
    return merged, notes_by_key


# --------------------------------------------------------------------------
# Polite HTTP with per-host rate limiting
# --------------------------------------------------------------------------

_last_request_time = {}


def _throttle(host, min_interval=1.05):
    last = _last_request_time.get(host, 0)
    wait = min_interval - (time.time() - last)
    if wait > 0:
        time.sleep(wait)
    _last_request_time[host] = time.time()


def http_get_json(url, headers=None):
    host = urllib.parse.urlparse(url).netloc
    _throttle(host)
    req = urllib.request.Request(url, headers=headers or {"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = resp.read()
        return json.loads(data.decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"    [warn] HTTP {e.code} for {url}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"    [warn] request failed for {url}: {e}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# Crossref
# --------------------------------------------------------------------------

def crossref_lookup(title, author, year):
    q = urllib.parse.quote(title)
    url = f"https://api.crossref.org/works?query.bibliographic={q}&rows=3"
    data = http_get_json(url, headers={"User-Agent": CROSSREF_UA})
    if not data:
        return None, [], "crossref request failed"

    items = data.get("message", {}).get("items", [])
    if not items:
        return None, [], "crossref returned no items"

    my_surname = surname_last_token(author)
    candidates = []
    for it in items:
        ct_list = it.get("title") or []
        ct = ct_list[0] if ct_list else ""
        sim = title_similarity(title, ct)
        cr_authors = it.get("author") or []
        cr_first_surname = ""
        if cr_authors:
            cr_first_surname = normalize_for_compare(cr_authors[0].get("family", ""))
        author_match = bool(cr_first_surname) and (
            cr_first_surname == my_surname
            or cr_first_surname in my_surname
            or my_surname in cr_first_surname
        )
        cr_year = None
        for date_key in ("published-print", "published-online", "issued", "created"):
            dp = it.get(date_key, {}).get("date-parts")
            if dp and dp[0]:
                cr_year = dp[0][0]
                break
        year_close = cr_year is not None and abs(cr_year - year) <= 1
        doi = it.get("DOI")
        candidates.append({
            "doi": doi,
            "title": ct,
            "sim": sim,
            "author_match": author_match,
            "cr_year": cr_year,
            "year_close": year_close,
        })

    candidates.sort(key=lambda c: (c["sim"] + (0.15 if c["author_match"] else 0)
                                    + (0.05 if c["year_close"] else 0)), reverse=True)
    best = candidates[0]

    accept = best["sim"] >= 0.60 or (best["sim"] >= 0.40 and best["author_match"] and best["year_close"])
    note = (f"crossref sim={best['sim']:.2f} author_match={best['author_match']} "
            f"year_close={best['year_close']} matched_title=\"{best['title']}\"")
    if not accept:
        return None, candidates, "no confident crossref match: " + note
    return best, candidates, note


# --------------------------------------------------------------------------
# Unpaywall / Europe PMC OA location lookup
# --------------------------------------------------------------------------

def unpaywall_lookup(doi):
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={CONTACT_EMAIL}"
    data = http_get_json(url)
    if not data:
        return []
    candidates = []
    best = data.get("best_oa_location") or {}
    if best.get("url_for_pdf"):
        candidates.append((best["url_for_pdf"], "unpaywall:best_oa_location"))
    elif best.get("url"):
        candidates.append((best["url"], "unpaywall:best_oa_location(url)"))
    for loc in (data.get("oa_locations") or []):
        u = loc.get("url_for_pdf") or None
        if u and (u, "unpaywall:oa_locations") not in [(c[0], c[1]) for c in candidates]:
            candidates.append((u, "unpaywall:oa_locations"))
    # de-dup preserving order
    seen = set()
    out = []
    for u, src in candidates:
        if u not in seen:
            seen.add(u)
            out.append((u, src))
    return out


def europepmc_lookup(doi):
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
           f"query=DOI:{urllib.parse.quote(doi)}&format=json")
    data = http_get_json(url)
    if not data:
        return []
    results = data.get("resultList", {}).get("result", [])
    if not results:
        return []
    candidates = []
    for r in results:
        pmcid = r.get("pmcid")
        is_oa = r.get("isOpenAccess") == "Y"
        if pmcid and is_oa:
            candidates.append((f"https://europepmc.org/articles/{pmcid}?pdf=render",
                                "europepmc:pmcid_render"))
        ftl = r.get("fullTextUrlList", {}).get("fullTextUrl", [])
        for ft in ftl:
            if ft.get("documentStyle") == "pdf" and ft.get("availability", "").lower().startswith("open"):
                candidates.append((ft.get("url"), "europepmc:fullTextUrlList"))
    seen = set()
    out = []
    for u, src in candidates:
        if u and u not in seen:
            seen.add(u)
            out.append((u, src))
    return out


LEGACY_NCBI_PMC_RE = re.compile(r"ncbi\.nlm\.nih\.gov/pmc/articles/(?:PMC)?(\d+)", re.IGNORECASE)


def resolve_legacy_pmc_pdf(url):
    """Unpaywall sometimes returns the old
    https://www.ncbi.nlm.nih.gov/pmc/articles/NNNNNNN style URL for a PMC
    open-access (often NIH-manuscript / nihms) copy. That URL now serves an
    HTML landing page (redirects to pmc.ncbi.nlm.nih.gov) rather than a PDF
    directly, and the real PDF lives at an unpredictable
    pdf/nihmsNNNNNN.pdf filename on that landing page. Fetch the landing
    page and pull out the real PDF link so we can still get the legitimate,
    freely-hosted copy rather than giving up on a technicality.
    Returns a full https URL, or None.
    """
    m = LEGACY_NCBI_PMC_RE.search(url)
    if not m:
        return None
    pmcid = m.group(1)
    landing_url = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/"
    host = urllib.parse.urlparse(landing_url).netloc
    _throttle(host)
    req = urllib.request.Request(landing_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    [warn] could not fetch legacy-PMC landing page {landing_url}: {e}", file=sys.stderr)
        return None
    pdf_match = re.search(r'href="(pdf/[^"]+\.pdf)"', html)
    if not pdf_match:
        return None
    return urllib.parse.urljoin(landing_url, pdf_match.group(1))


# --------------------------------------------------------------------------
# Download + verification
# --------------------------------------------------------------------------

def curl_download(url, dest_path):
    host = urllib.parse.urlparse(url).netloc
    _throttle(host)
    cmd = [
        "curl", "-L", "--silent", "--show-error", "--fail",
        "--connect-timeout", "20", "--max-time", str(REQUEST_TIMEOUT),
        "-A", USER_AGENT,
        "-H", "Accept: application/pdf,*/*",
        "-o", dest_path,
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=REQUEST_TIMEOUT + 10)
    except subprocess.TimeoutExpired:
        return False, "curl timed out"
    if result.returncode != 0:
        return False, f"curl exit {result.returncode}: {result.stderr.strip()[:200]}"
    if not os.path.exists(dest_path):
        return False, "curl produced no file"
    return True, None


def is_pdf_file(path):
    try:
        out = subprocess.run(["file", "--brief", path], capture_output=True, text=True, timeout=15)
        return "PDF" in out.stdout
    except Exception:
        return False


def pdftotext_excerpt(path, max_chars=6000):
    try:
        out = subprocess.run(["pdftotext", "-l", "3", path, "-"], capture_output=True, text=True, timeout=30)
        return out.stdout[:max_chars]
    except Exception:
        return ""


def verify_pdf_content(path, title, author):
    size = os.path.getsize(path) if os.path.exists(path) else 0
    if size < MIN_PDF_BYTES:
        return False, f"file too small ({size} bytes)"
    if not is_pdf_file(path):
        return False, "`file` did not report PDF"
    text = pdftotext_excerpt(path)
    norm_text = normalize_for_compare(text)
    surname = surname_last_token(author)
    author_hit = bool(surname) and surname in norm_text

    words = significant_words(title)
    if words:
        hits = sum(1 for w in words if w in norm_text)
        frac = hits / len(words)
    else:
        frac = 0.0
    title_hit = frac >= 0.5

    if author_hit or title_hit:
        return True, f"verified (author_hit={author_hit}, title_word_frac={frac:.2f})"
    return False, f"content verification failed (author_hit={author_hit}, title_word_frac={frac:.2f})"


def try_download_candidates(candidates, dest_path, title, author):
    """Try up to MAX_DOWNLOAD_ATTEMPTS candidate URLs; return (ok, used_url, used_source, note)."""
    attempts = 0
    notes = []
    for url, source in candidates:
        if attempts >= MAX_DOWNLOAD_ATTEMPTS:
            break
        attempts += 1
        ok, err = curl_download(url, dest_path)
        if not ok:
            notes.append(f"{source} {url} -> download failed: {err}")
            continue
        verified, vnote = verify_pdf_content(dest_path, title, author)
        if verified:
            return True, url, source, "; ".join(notes + [vnote])
        notes.append(f"{source} {url} -> {vnote}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
    return False, None, None, "; ".join(notes) if notes else "no OA candidates found"


# --------------------------------------------------------------------------
# Preprint host detection (for status=preprint_downloaded)
# --------------------------------------------------------------------------

PREPRINT_HOSTS = ("biorxiv.org", "psyarxiv.com", "osf.io", "arxiv.org", "researchsquare.com")


def is_preprint_url(url):
    host = urllib.parse.urlparse(url).netloc.lower()
    return any(p in host for p in PREPRINT_HOSTS)


# --------------------------------------------------------------------------
# Manifest I/O (resumable)
# --------------------------------------------------------------------------

MANIFEST_FIELDS = ["author", "year", "title", "doi", "status", "file", "source_url", "notes"]


def load_existing_manifest():
    existing = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (normalize_for_compare(row["author"]), normalize_for_compare(row["title"]))
                existing[key] = row
    return existing


def write_manifest(rows):
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print dedup'd/filtered study list and exit")
    ap.add_argument("--dry-run", action="store_true", help="do DOI/OA lookups only, no downloads")
    ap.add_argument("--only", help="comma-separated 'AuthorSubstring,Year' to process a single study")
    ap.add_argument("--force", action="store_true", help="reprocess studies even if already in MANIFEST.csv")
    args = ap.parse_args()

    all_studies = load_studies()
    keep, skipped_already_have = filter_studies(all_studies)
    keep, dup_notes = merge_title_duplicates(keep)
    keep.sort(key=lambda t: (t[2], surname_last_token(t[0])))

    if args.list:
        print(f"Total unique (author,title,year) rows after union+unpublished-filter: {len(all_studies)}")
        print(f"Skipped (already have): {len(skipped_already_have)}")
        for a, t, y in skipped_already_have:
            print(f"  SKIP  {y}  {a}  -- {t}")
        print(f"To process: {len(keep)}")
        for a, t, y in keep:
            print(f"  {y}  {a}  -- {t}")
        return

    if args.only:
        sub, yr = args.only.split(",")
        yr = int(yr.strip())
        keep = [s for s in keep if sub.strip().lower() in s[0].lower() and s[2] == yr]
        if not keep:
            print("No study matched --only filter", file=sys.stderr)
            sys.exit(1)

    existing = {} if args.force else load_existing_manifest()
    used_filenames = {}
    for row in existing.values():
        if row.get("file"):
            used_filenames[row["file"]] = normalize_for_compare(row.get("title", ""))
    results = []
    n_downloaded = n_preprint = n_paywalled = n_not_found = 0

    for idx, (author, title, year) in enumerate(keep, 1):
        key = (normalize_for_compare(author), normalize_for_compare(title))
        print(f"[{idx}/{len(keep)}] {author} ({year}) -- {title}")

        if key in existing and existing[key].get("status") in (
            "downloaded", "preprint_downloaded", "paywalled"
        ):
            print(f"    already in manifest with status={existing[key]['status']}, skipping")
            results.append(existing[key])
            st = existing[key]["status"]
            if st == "downloaded":
                n_downloaded += 1
            elif st == "preprint_downloaded":
                n_preprint += 1
            elif st == "paywalled":
                n_paywalled += 1
            continue

        row = {
            "author": author, "year": year, "title": title,
            "doi": "", "status": "not_found", "file": "", "source_url": "", "notes": "",
        }
        if key in dup_notes:
            row["notes"] = dup_notes[key]

        best, candidates, cr_note = crossref_lookup(title, author, year)
        if not best:
            row["notes"] = (row["notes"] + "; " if row["notes"] else "") + cr_note
            print(f"    DOI: NOT FOUND -- {cr_note}")
            results.append(row)
            n_not_found += 1
            continue

        doi = best["doi"]
        row["doi"] = doi
        print(f"    DOI: {doi}  ({cr_note})")

        # Prefer Crossref's own publication year for the on-disk filename and
        # manifest, since the two source spreadsheets sometimes disagree with
        # each other (epub-ahead-of-print vs. print/issue date).
        final_year = best["cr_year"] if best.get("cr_year") else year
        if final_year != year:
            row["notes"] = (row["notes"] + "; " if row["notes"] else "") + (
                f"spreadsheet year={year}, Crossref year={final_year} (using Crossref year)"
            )
        row["year"] = final_year

        if best["sim"] < 0.85:
            row["notes"] = (row["notes"] + "; " if row["notes"] else "") + "TITLE MISMATCH RISK: " + cr_note

        if args.dry_run:
            row["status"] = "not_found"
            row["notes"] = (row["notes"] + "; " if row["notes"] else "") + "[dry-run: OA lookup skipped]"
            results.append(row)
            continue

        # Query both OA-discovery services and pool their candidate URLs
        # (Unpaywall first, since it usually points straight at a publisher
        # PDF). We used to only consult Europe PMC when Unpaywall returned
        # zero candidates, but Unpaywall sometimes returns a stale/legacy
        # NCBI PMC URL (e.g. https://www.ncbi.nlm.nih.gov/pmc/articles/NNNN)
        # that now serves an HTML landing page instead of a PDF -- in that
        # case we still want Europe PMC's modern PMC render URL as a
        # fallback candidate, not just a dead end.
        up_candidates = unpaywall_lookup(doi)
        epmc_candidates = europepmc_lookup(doi)
        seen_urls = set()
        oa_candidates = []
        for u, src in up_candidates + epmc_candidates:
            if u and u not in seen_urls:
                seen_urls.add(u)
                oa_candidates.append((u, src))
                if LEGACY_NCBI_PMC_RE.search(u):
                    resolved = resolve_legacy_pmc_pdf(u)
                    if resolved and resolved not in seen_urls:
                        seen_urls.add(resolved)
                        oa_candidates.append((resolved, "legacy_pmc_resolved"))

        if not oa_candidates:
            row["status"] = "paywalled"
            row["notes"] = (row["notes"] + "; " if row["notes"] else "") + "no OA location via Unpaywall or Europe PMC"
            print("    OA: none found -> paywalled")
            results.append(row)
            n_paywalled += 1
            continue

        base = f"{filename_author(author)}_{final_year}"
        fname = f"{base}.pdf"
        norm_title_key = normalize_for_compare(title)
        suffix_n = 2
        while fname in used_filenames and used_filenames[fname] != norm_title_key:
            fname = f"{base}_{suffix_n}.pdf"
            suffix_n += 1
        used_filenames[fname] = norm_title_key
        dest = os.path.join(HERE, fname)
        ok, used_url, used_source, dl_note = try_download_candidates(oa_candidates, dest, title, author)

        if ok:
            status = "preprint_downloaded" if is_preprint_url(used_url) else "downloaded"
            row["status"] = status
            row["file"] = fname
            row["source_url"] = used_url
            row["notes"] = (row["notes"] + "; " if row["notes"] else "") + f"[{used_source}] {dl_note}"
            print(f"    OA: downloaded ({status}) <- {used_url}")
            if status == "downloaded":
                n_downloaded += 1
            else:
                n_preprint += 1
        else:
            row["status"] = "paywalled"
            row["notes"] = (row["notes"] + "; " if row["notes"] else "") + f"OA candidates found but none verified: {dl_note}"
            print(f"    OA: candidates found but download/verify failed -> paywalled ({dl_note})")
            n_paywalled += 1

        results.append(row)

        # Write manifest incrementally so an interruption doesn't lose progress.
        write_manifest(results)

    write_manifest(results)
    print()
    print("=== Summary ===")
    print(f"downloaded:          {n_downloaded}")
    print(f"preprint_downloaded: {n_preprint}")
    print(f"paywalled:           {n_paywalled}")
    print(f"not_found:           {n_not_found}")
    print(f"total processed:     {len(results)}")
    print(f"Manifest written to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
