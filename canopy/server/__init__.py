"""The local review UI: a FastAPI app and a static SPA, both meant for one person's laptop.

`canopy serve` binds loopback, runs the pipeline in a background worker, and serves the run
directory it wrote. Nothing here talks to anything except the browser on the same machine and
(through `canopy.llm`) the Anthropic API.

The security baseline (plan amendment I) lives in three places and is tested in
`tests/test_server.py`:

* `uploads.py` — a file is a paper only if it starts with `%PDF-`, fits the size cap and can be
  ingested **in a child process with a timeout**; it is then stored under its own sha256.
* `security.py` — every served path is resolved and must still be inside the run directory, with
  a suffix on a short allow-list; every run endpoint needs that run's bearer token.
* `app.py` — the API key is never echoed (only "configured: yes/no"), and the SPA writes every
  string that came from a PDF or a model with `textContent`.
"""
from .app import create_app, serve

__all__ = ["create_app", "serve"]
