#!/usr/bin/env bash
# Put AutoMeta on Google Cloud Run, with the runs on a bucket and every secret in Secret Manager.
#
#   gcloud auth login                       # once, in your own terminal
#   gcloud config set project <PROJECT_ID>  # a project with billing enabled
#   CANOPY_ENV_FILE=~/Projects/canopy/.env deploy/cloudrun.sh [service] [region]
#
# What it does, in order: enables the APIs; makes a bucket for /data/runs; asks for each secret
# ONCE, silently, at the terminal (paste from the clipboard — nothing is echoed, logged or put in
# shell history) and stores it in Secret Manager, keeping any that already exist; grants the
# service account the bucket and the secrets; builds the image with Cloud Build (30-minute
# timeout — the R packages can take a while when no binary exists); deploys it always-on, one
# instance, CPU never throttled, so a review's background thread keeps running after the page
# that started it has been answered; prints the URL.
#
# Non-secret settings (ANTHROPIC_BASE_URL, CANOPY_LLM_EXTRA_BODY, CANOPY_CONTACT_EMAIL) are copied
# from the .env file named by CANOPY_ENV_FILE when present, so the hosted server talks to the same
# model endpoint the local one does. Re-running the script is safe: every step is idempotent.
set -euo pipefail

SERVICE="${1:-autometa}"
REGION="${2:-us-central1}"
ENV_FILE="${CANOPY_ENV_FILE:-.env}"

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
if [ -z "$PROJECT" ]; then
    echo "No project selected. Run:  gcloud config set project <PROJECT_ID>" >&2
    exit 1
fi
if ! gcloud auth print-access-token >/dev/null 2>&1; then
    echo "Not logged in. Run:  gcloud auth login" >&2
    exit 1
fi
echo "project $PROJECT · region $REGION · service $SERVICE"

# ----------------------------------------------------------------------------- APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
    secretmanager.googleapis.com storage.googleapis.com --quiet

NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
SA="${NUMBER}-compute@developer.gserviceaccount.com"

# ----------------------------------------------------------------------------- bucket
BUCKET="${SERVICE}-runs-${PROJECT}"
if ! gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
    gcloud storage buckets create "gs://$BUCKET" --location="$REGION" \
        --uniform-bucket-level-access --quiet
fi
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
    --member="serviceAccount:$SA" --role=roles/storage.objectAdmin --quiet >/dev/null

# ----------------------------------------------------------------------------- secrets
# Read silently, from the terminal, never from an argument: an argument lands in shell history
# and `ps`. An existing secret is kept — delete it in the console to rotate.
secret() {
    local name="$1" prompt="$2" optional="${3:-}" value
    if gcloud secrets describe "$name" >/dev/null 2>&1; then
        echo "  $name: already stored, keeping it"
    else
        read -r -s -p "  $prompt" value; echo
        if [ -z "$value" ]; then
            [ -n "$optional" ] && { echo "  $name: skipped"; return 0; }
            echo "  $name is required" >&2; exit 1
        fi
        printf '%s' "$value" | gcloud secrets create "$name" --data-file=- --quiet
        value=""
    fi
    gcloud secrets add-iam-policy-binding "$name" --member="serviceAccount:$SA" \
        --role=roles/secretmanager.secretAccessor --quiet >/dev/null
    SECRETS="${SECRETS:+$SECRETS,}$name=$name:latest"
}
SECRETS=""
echo "secrets (paste from the clipboard; nothing is shown):"
secret ANTHROPIC_API_KEY "API key — the OpenRouter key if ANTHROPIC_BASE_URL points there: "
secret CANOPY_ACCESS_CODE "Access code visitors will type to open the site: "
secret CANOPY_OPENALEX_KEY "OpenAlex key (Enter to skip): " optional

# ----------------------------------------------------------------------------- plain settings
ENVFILE="$(mktemp)"
trap 'rm -f "$ENVFILE"' EXIT
# YAML, one key per line, values JSON-quoted — a JSON string is a valid YAML string, and
# CANOPY_LLM_EXTRA_BODY is JSON with commas that --set-env-vars would split on.
{
    echo "CANOPY_MAX_UPLOAD_MB: \"30\""          # Cloud Run refuses request bodies over 32 MB
    if [ -f "$ENV_FILE" ]; then
        for key in ANTHROPIC_BASE_URL CANOPY_LLM_EXTRA_BODY CANOPY_CONTACT_EMAIL; do
            val="$(grep -E "^${key}=" "$ENV_FILE" | head -1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
            [ -n "$val" ] && printf '%s: %s\n' "$key" "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$val")"
        done
    else
        echo "  (no $ENV_FILE — set CANOPY_ENV_FILE to copy ANTHROPIC_BASE_URL etc. from your local .env)" >&2
    fi
} > "$ENVFILE"
echo "settings copied: $(grep -oE '^[A-Z_]+' "$ENVFILE" | tr '\n' ' ')"

# ----------------------------------------------------------------------------- build
REPO="autometa"
if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1; then
    gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" --quiet
fi
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:$(date -u +%Y%m%d-%H%M%S)"
echo "building $IMAGE"
gcloud builds submit --tag "$IMAGE" --timeout=1800s --quiet .

# ----------------------------------------------------------------------------- deploy
gcloud run deploy "$SERVICE" --image "$IMAGE" --region "$REGION" --platform managed \
    --allow-unauthenticated \
    --execution-environment gen2 \
    --cpu 2 --memory 4Gi --min-instances 1 --max-instances 1 --no-cpu-throttling \
    --timeout 3600 --concurrency 40 \
    --env-vars-file "$ENVFILE" \
    --set-secrets "$SECRETS" \
    --add-volume "name=runs,type=cloud-storage,bucket=$BUCKET" \
    --add-volume-mount "volume=runs,mount-path=/data/runs" \
    --quiet

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo
echo "  $URL"
echo
echo "  Open it, enter the access code, and you are on the same UI as localhost:8000."
echo "  Runs persist in gs://$BUCKET. Always-on at 2 vCPU / 4 GiB is roughly \$100/month;"
echo "  --cpu 1 --memory 2Gi halves that at the cost of slower PDF ingest."
