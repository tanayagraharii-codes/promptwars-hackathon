#!/usr/bin/env bash
# One-command deploy of Evidentia to Google Cloud Run.
#
#   PROJECT_ID=my-project ./deploy.sh                 # Vertex AI (recommended, no API key)
#   PROJECT_ID=my-project AUTH_MODE=apikey ./deploy.sh  # Gemini API key from $GEMINI_API_KEY
#
# Idempotent: safe to re-run. Requires the gcloud CLI and a billing-enabled project.
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-evidentia}"
AUTH_MODE="${AUTH_MODE:-vertex}" # vertex | apikey
SA_NAME="${SERVICE}-runner"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SECRET="gemini-api-key"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "Using project ${PROJECT_ID} (${REGION}), auth mode: ${AUTH_MODE}"
gcloud config set project "${PROJECT_ID}" >/dev/null

step "Enabling Google Cloud APIs"
apis=(run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
      firestore.googleapis.com logging.googleapis.com aiplatform.googleapis.com)
[[ "${AUTH_MODE}" == "apikey" ]] && apis+=(secretmanager.googleapis.com)
gcloud services enable "${apis[@]}"

step "Creating least-privilege service account ${SA_EMAIL}"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SA_NAME}" --display-name "Evidentia Cloud Run runtime"
fi
for role in roles/datastore.user roles/logging.logWriter roles/aiplatform.user; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${SA_EMAIL}" --role "${role}" --condition=None >/dev/null
done

step "Ensuring a Firestore (Native mode) database exists"
if ! gcloud firestore databases describe --database="(default)" >/dev/null 2>&1; then
  gcloud firestore databases create --location="${REGION}" --type=firestore-native
fi

# GOOGLE_CLOUD_PROJECT also enables Cloud Trace correlation of log lines.
env_vars="FIRESTORE_ENABLED=true,SEARCH_BACKEND=auto,GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"
secret_flags=()
if [[ "${AUTH_MODE}" == "vertex" ]]; then
  env_vars+=",GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_LOCATION=${REGION}"
else
  : "${GEMINI_API_KEY:?Set GEMINI_API_KEY for AUTH_MODE=apikey}"
  step "Storing the Gemini API key in Secret Manager"
  if gcloud secrets describe "${SECRET}" >/dev/null 2>&1; then
    printf '%s' "${GEMINI_API_KEY}" | gcloud secrets versions add "${SECRET}" --data-file=- >/dev/null
  else
    printf '%s' "${GEMINI_API_KEY}" | gcloud secrets create "${SECRET}" --data-file=- >/dev/null
  fi
  # Access is granted on this one secret only, not project-wide.
  gcloud secrets add-iam-policy-binding "${SECRET}" \
    --member "serviceAccount:${SA_EMAIL}" --role roles/secretmanager.secretAccessor >/dev/null
  secret_flags=(--set-secrets "GEMINI_API_KEY=${SECRET}:latest")
fi

step "Building and deploying to Cloud Run"
gcloud run deploy "${SERVICE}" \
  --source . \
  --region "${REGION}" \
  --service-account "${SA_EMAIL}" \
  --allow-unauthenticated \
  --set-env-vars "${env_vars}" \
  ${secret_flags[@]+"${secret_flags[@]}"} \
  --memory 512Mi --cpu 1 --timeout 300 --concurrency 20 --max-instances 3

url="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format 'value(status.url)')"
step "Deployed: ${url}"
curl -fsS "${url}/healthz" && echo
