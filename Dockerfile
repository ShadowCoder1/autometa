# AutoMeta as a hosted service.
#
# The same server `canopy serve` runs locally, bound to every interface instead of loopback.
# Nothing about the pipeline changes: runs still execute in a background thread, PDFs are still
# ingested in a subprocess, and every run is still private to the bearer token issued when it
# was created (off loopback the run list stops handing those tokens out).
#
# Build:  docker build -t autometa .
# Run:    docker run -p 7860:7860 -e ANTHROPIC_API_KEY=… -v autometa-runs:/data/runs autometa

# 3.12 or newer: the report and CLI modules use backslashes inside f-string expressions,
# which is a syntax error before PEP 701 landed in 3.12.
FROM python:3.12-slim

# PyMuPDF and opencv-python-headless need no GL, but both want glib; matplotlib wants a writable
# cache dir. Keep this list minimal — every package here is attack surface on a public host.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# A non-root user: the server accepts uploads from the public internet and shells out to ingest
# them, so it should never be able to write outside its own tree.
RUN useradd --create-home --uid 1000 app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/home/app

WORKDIR /app

# Dependencies first, so a code edit does not re-resolve the whole tree on rebuild.
COPY pyproject.toml README.md ./
COPY canopy ./canopy
RUN pip install --upgrade pip && pip install .

# Runs live on a volume: page rasters, figure crops and the model cache are what make a resumed
# run free, and they must survive a restart.
RUN mkdir -p /data/runs && chown -R app:app /data /app
USER app
VOLUME ["/data/runs"]

# 7860 is what Hugging Face Spaces expects; PORT overrides it for Fly, Render and Railway.
ENV PORT=7860
EXPOSE 7860

# Shell form so ${PORT} is expanded by the shell at start-up.
CMD canopy serve --host 0.0.0.0 --port ${PORT} --runs /data/runs
