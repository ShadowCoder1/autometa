# AutoMeta as a hosted service.
#
# The same server `canopy serve` runs locally, bound to every interface instead of loopback.
# Nothing about the pipeline changes: runs still execute in a background thread, PDFs are still
# ingested in a subprocess, and every run is still private to the bearer token issued when it
# was created (off loopback the run list stops handing those tokens out). Set CANOPY_ACCESS_CODE
# to put a password on the whole site — without it, anyone who finds the URL can start a run on
# your key.
#
# Build:  docker build -t autometa .
# Run:    docker run -p 7860:7860 -e ANTHROPIC_API_KEY=… -e CANOPY_ACCESS_CODE=… \
#             -v autometa-runs:/data/runs autometa
#
# deploy/cloudrun.sh does the same on Google Cloud Run, with the runs on a bucket.

# 3.12 or newer: the report and CLI modules use backslashes inside f-string expressions,
# which is a syntax error before PEP 701 landed in 3.12.
FROM python:3.12-slim

# PyMuPDF and opencv-python-headless need no GL, but both want glib; matplotlib wants a writable
# cache dir. R draws the forest plot the report links (meta::forest). Debian ships R and nearly
# everything `meta` stands on — lme4, Matrix, RcppEigen, metafor, ggplot2 and the rest are taken
# from apt as built packages, because compiling that stack from CRAN takes longer than a cloud
# build allows (and Posit has no prebuilt binaries for Debian's R 4.2). `meta` itself is not
# packaged and comes from CRAN in the next step; cmake, libxml2-dev and r-base-dev are there so
# the handful of small packages that still have to compile can. Keep this list minimal — every
# package here is attack surface on a public host.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 ca-certificates cmake libxml2-dev \
        r-base-core r-base-dev r-recommended \
        r-cran-jsonlite r-cran-svglite r-cran-systemfonts r-cran-ragg \
        r-cran-rcpp r-cran-rcppeigen r-cran-matrix r-cran-minqa r-cran-nloptr r-cran-lme4 \
        r-cran-numderiv r-cran-mvtnorm r-cran-mathjaxr r-cran-metadat r-cran-metafor \
        r-cran-xml2 r-cran-magrittr r-cran-stringi r-cran-stringr r-cran-tibble r-cran-purrr \
        r-cran-dplyr r-cran-readr r-cran-ggplot2 r-cran-pbapply \
    && rm -rf /var/lib/apt/lists/*

# `meta` from CRAN, on top of the apt-installed stack: only what is still missing is fetched,
# and the few that compile (CompQuadForm) are small. The build fails loudly if the package is not
# importable at the end: a missing renderer would otherwise surface as a warning in every report,
# which is exactly the kind of thing a reader takes for an error.
RUN Rscript -e ' \
    options(repos = c(CRAN = "https://cloud.r-project.org"), \
            Ncpus = max(1L, parallel::detectCores())); \
    install.packages("meta"); \
    if (!requireNamespace("meta", quietly = TRUE)) stop("the R package meta did not install")'

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

# 7860 is what Hugging Face Spaces expects; PORT overrides it for Fly, Render, Railway and
# Cloud Run (which injects PORT=8080).
ENV PORT=7860
EXPOSE 7860

# Shell form so ${PORT} is expanded by the shell at start-up.
CMD canopy serve --host 0.0.0.0 --port ${PORT} --runs /data/runs
