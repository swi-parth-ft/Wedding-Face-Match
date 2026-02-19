#!/usr/bin/env python3
"""
Quick profiler: compare download vs face-extraction cost for buffered files.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS_DIR = REPO_ROOT / "functions"
STATE_FILE = REPO_ROOT / ".local_index_state.json"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


def main() -> int:
    load_env(FUNCTIONS_DIR / ".env")
    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(REPO_ROOT / ".secrets" / "local-indexer-key.json"),
    )
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "facefinder-260219-2674")

    if not STATE_FILE.exists():
        print(f"State file not found: {STATE_FILE}")
        return 1

    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    buffer = state.get("buffer")
    if not isinstance(buffer, list):
        print("State buffer is missing.")
        return 1

    samples = [
        item
        for item in buffer
        if isinstance(item, dict) and str(item.get("mimeType", "")).startswith("image/")
    ][:3]

    if not samples:
        print("No image samples in current buffer.")
        return 0

    sys.path.insert(0, str(FUNCTIONS_DIR))
    import main as backend  # type: ignore

    for item in samples:
        file_id = str(item.get("id") or "")
        if not file_id:
            continue

        t0 = time.time()
        blob = backend._download_drive_file_bytes(file_id)
        t1 = time.time()
        faces = backend._extract_face_embeddings(blob, min_face_size=60)
        t2 = time.time()

        print(
            json.dumps(
                {
                    "fileId": file_id,
                    "fileName": str(item.get("name") or ""),
                    "bytes": len(blob),
                    "downloadSec": round(t1 - t0, 3),
                    "extractSec": round(t2 - t1, 3),
                    "totalSec": round(t2 - t0, 3),
                    "faces": len(faces),
                }
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
