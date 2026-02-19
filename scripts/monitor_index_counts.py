#!/usr/bin/env python3
"""
Log indexed/remaining image counts at a fixed interval.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import firebase_admin
from firebase_admin import firestore


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CREDENTIALS_PATH = REPO_ROOT / ".secrets" / "local-indexer-key.json"
DEFAULT_COLLECTION = "indexed_files"

_STOP = False


def _handle_signal(_sig: int, _frame: object) -> None:
    global _STOP
    _STOP = True


def _load_default_project_id() -> str:
    cfg = REPO_ROOT / ".firebaserc"
    if not cfg.exists():
        return ""
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return ""
    projects = data.get("projects")
    if not isinstance(projects, dict):
        return ""
    project_id = projects.get("default")
    if isinstance(project_id, str):
        return project_id.strip()
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor indexed/remaining counts.")
    p.add_argument("--total-images", type=int, required=True, help="Total image count.")
    p.add_argument(
        "--interval-seconds",
        type=int,
        default=60,
        help="How often to print counts (default: 60).",
    )
    p.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Firestore collection to count (default: {DEFAULT_COLLECTION}).",
    )
    p.add_argument(
        "--credentials",
        type=Path,
        default=DEFAULT_CREDENTIALS_PATH,
        help="Service account JSON path.",
    )
    p.add_argument(
        "--project-id",
        default="",
        help="Project id (defaults from .firebaserc).",
    )
    return p.parse_args()


def _count_indexed(db: firestore.Client, collection: str) -> int:
    return int(db.collection(collection).count().get()[0][0].value)


def main() -> int:
    args = parse_args()
    interval = max(5, int(args.interval_seconds))
    total = max(0, int(args.total_images))

    creds = args.credentials.expanduser().resolve()
    if not creds.exists():
        print(f"[ERROR] Credentials not found: {creds}")
        return 1

    project_id = (args.project_id or _load_default_project_id()).strip()
    if project_id:
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds)

    firebase_admin.initialize_app()
    db = firestore.client()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    prev_count: int | None = None
    prev_ts: float | None = None

    while not _STOP:
        now = time.time()
        count = _count_indexed(db, args.collection)
        remaining = max(0, total - count)
        pct = 0.0 if total <= 0 else min(100.0, (count / total) * 100.0)

        per_min = None
        eta_min = None
        if prev_count is not None and prev_ts is not None and now > prev_ts:
            delta_count = max(0, count - prev_count)
            delta_min = (now - prev_ts) / 60.0
            if delta_min > 0:
                per_min = delta_count / delta_min
                if per_min > 0:
                    eta_min = remaining / per_min

        row = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "indexed": count,
            "remaining": remaining,
            "total": total,
            "progressPct": round(pct, 2),
        }
        if per_min is not None:
            row["imagesPerMin"] = round(per_min, 2)
        if eta_min is not None:
            row["etaMin"] = round(eta_min, 1)

        print(json.dumps(row), flush=True)

        if remaining <= 0:
            break

        prev_count = count
        prev_ts = now

        for _ in range(interval):
            if _STOP:
                break
            time.sleep(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
