"""Finding the papers, when the user would rather ask a question than upload a folder.

This package chooses which papers a review reads. It never reads them: extraction, verification
and pooling are the pipeline's job and are untouched by anything here. The seam between the two
is a directory of PDFs and a protocol — exactly what a person uploads by hand — which is why
`server/make_run.py` serves both doors and neither can drift from the other.
"""
from .dedupe import dedupe, normalise_doi, normalise_title
from .models import (COUNT_KEYS, KEY_RE, PHASES, Candidate, CandidateState, SearchRecord,
                     counts_of, new_key, project)
from .queries import build_queries, template_queries

__all__ = ["COUNT_KEYS", "KEY_RE", "PHASES", "Candidate", "CandidateState", "SearchRecord",
           "build_queries", "counts_of", "dedupe", "new_key", "normalise_doi", "normalise_title",
           "project", "template_queries"]
