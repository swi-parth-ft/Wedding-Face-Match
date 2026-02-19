# Drive Face Finder

GitHub Pages frontend + Firebase Functions backend to:
1. Crawl a public Google Drive folder recursively (including nested folders).
2. Extract face embeddings for each image (InsightFace).
3. Search similar faces from an uploaded query image.
4. Download selected matched images as a zip.

This repo also keeps your local/offline script at `scripts/cluster_faces.py`.

## Project Structure

- `functions/main.py`: Firebase HTTP APIs.
- `functions/requirements.txt`: Python backend dependencies.
- `docs/`: Static site for GitHub Pages.
- `.github/workflows/deploy-pages.yml`: Pages deployment workflow.
- `scripts/cluster_faces.py`: optional local-only clustering script.

## Backend APIs

Deployed Firebase function names:
- `health`
- `index_chunk`
- `search`
- `download_matches_zip`

## 1) Firebase Setup

### Prerequisites

- Firebase project (Blaze plan recommended for long-running indexing)
- Firestore database enabled
- `firebase-tools` installed
- Python 3.11 available on your machine for local validation

```bash
npm install -g firebase-tools
firebase login
```

### Configure project

```bash
cp .firebaserc.example .firebaserc
```

Edit `.firebaserc` and set your real Firebase project id.

### Optional env config

```bash
cp functions/.env.example functions/.env
```

You can leave defaults, or set `DRIVE_API_KEY` for public Drive API access.

### Deploy functions

```bash
firebase deploy --only functions
```

After deploy, your base URL will be:

`https://us-central1-<your-project-id>.cloudfunctions.net`

Important: Cloud Functions deploy requires the Firebase project on the Blaze plan.

## 2) GitHub Pages Setup

1. Push this repo to GitHub.
2. In repo settings, enable Pages with `GitHub Actions` as source.
3. Push to `main` branch to trigger `.github/workflows/deploy-pages.yml`.
4. Open the published Pages URL.

## 3) Use the Web App

In the page:
1. Paste Functions base URL.
2. Keep or replace Drive folder link.
3. Click `Start Indexing` (chunked; can resume by running again).
4. Upload a query face image and click `Search`.
5. Select matches and click `Download Selected ZIP`.

Default folder prefilled in UI:

`https://drive.google.com/drive/folders/1EF9ddrU0dWt3H5c43Rt_yS54TZRK_TgB?usp=sharing`

## Local Emulator (macOS)

For local testing on macOS with OpenCV/InsightFace in Python functions:

```bash
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES firebase emulators:start --only functions,firestore
```

## Notes and Limits

- First cold start is slower because InsightFace model loads.
- For very large datasets, indexing takes multiple chunks/runs.
- Search currently scans indexed faces in Firestore directly (simple MVP behavior).
- Ensure Drive items are shared publicly (`Anyone with link`).

## Local Offline Script (Optional)

If you still want local folder clustering by person:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/cluster_faces.py --input-dir /absolute/path/to/downloaded_images
```

## Local Drive -> Firestore Indexing (Faster)

Use this when you want indexing on your Mac instead of Cloud Functions:

```bash
cd /Users/parthantala/Desktop/Face
functions/venv/bin/python scripts/local_index_drive_to_firestore.py \
  --folder "https://drive.google.com/drive/folders/1EF9ddrU0dWt3H5c43Rt_yS54TZRK_TgB?usp=sharing" \
  --max-files-per-chunk 80 \
  --credentials "/Users/parthantala/Desktop/Face/.secrets/local-indexer-key.json"
```

Notes:
- Resume is automatic via `.local_index_state.json`.
- Stop anytime with `Ctrl+C`; rerun same command to continue.
- Keep `--skip-existing` enabled (default) so already indexed files are skipped.
