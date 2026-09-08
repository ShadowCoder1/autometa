# Hosting AutoMeta

The server that runs locally is the server that runs in production; the container just binds it to
every interface instead of loopback. Nothing about the pipeline changes.

```
docker build -t autometa .
docker run -p 7860:7860 -e ANTHROPIC_API_KEY=… -v autometa-runs:/data/runs autometa
```

Verified: the image builds on `python:3.12-slim` and serves the UI and `/api/settings` with
`loopback_only: false`.

## What the host has to provide

A **persistent container with a disk** — not a serverless function. Three requirements rule the
serverless platforms out:

1. **Runs are long.** The 23-paper handedness review took 34 minutes wall-clock. A run is started
   by `POST /api/runs`, which returns immediately and leaves a background thread working; the
   process must stay alive long after the response. Vercel, Netlify Functions and Lambda all stop
   at 5 minutes or less.
2. **Runs need a filesystem.** Page rasters, figure crops and the model cache are written under
   the runs directory, and they are what make a resumed run free instead of re-billed.
3. **Ingest shells out.** Each PDF is parsed in a subprocess so a malformed file cannot take the
   server down with it.

Sizing: roughly 1–2 GB of RAM (PyMuPDF rasterising pages, OpenCV, matplotlib). CPU is not the
constraint; the wall-clock is spent waiting on model calls.

| host | fits | notes |
|---|---|---|
| Hugging Face Spaces (Docker SDK) | yes | free CPU tier is 2 vCPU / 16 GB, no card. Sleeps when idle and wakes on request; free storage is ephemeral, so finished runs do not survive a restart. |
| Fly.io | yes | small VM plus a persistent volume; needs a card. The best "real" option. |
| Railway / Render (paid tiers) | yes | Render's free tier sleeps after 15 minutes of no traffic, which kills a run in progress. |
| Vercel / Netlify / Lambda | **no** | execution-time limit; see above. |

## Who pays for the models

This is the decision that matters, not the hosting. A review costs roughly **$1 per paper** on a
clean pass and several dollars for a paper whose numbers are only in figures — the 23-paper
validation corpus came to **$18.68**. Every run billed to whatever key sits in the server's
environment.

- **One shared key (the server's own).** Simplest, and every visitor spends the operator's money.
  Only sane behind a gate — a shared password, or an allow-list — so the people running reviews
  are people you meant to invite.
- **Bring your own key.** The visitor supplies their own credential and pays for their own run.
  This is what makes a genuinely public instance safe. It needs a code change: the key is read
  once from the environment (`canopy/config.py:46`), so accepting a per-run key means threading it
  through job creation and never writing it to the run record.

Budget caps already exist per run (`budget_usd`, `max_usd_per_paper`) and are the right backstop
under either model, but they cap a single run, not the number of runs a stranger may start.

## Security posture as it stands

- Every run and search is private to a **bearer token** minted when it is created, and off
  loopback the listing endpoints stop returning those tokens — so one visitor cannot read, resume
  or cancel another's run.
- Uploads are PDF-sniffed by magic bytes, size-capped, and ingested in a subprocess with a timeout.
- The container runs as a non-root user and the image carries no `.env`, corpus or run data.
- There are **no user accounts**. Anyone who can reach the port can start a run. That is exactly
  why the key question above has to be answered before the URL is shared.
