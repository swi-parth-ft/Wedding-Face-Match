#!/usr/bin/env python3
"""
Run Drive -> Firestore face indexing locally on your machine.

This reuses the same indexing internals and Firestore schema as functions/main.py
so the web search API keeps working with the indexed data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS_DIR = REPO_ROOT / "functions"
DEFAULT_CREDENTIALS_PATH = REPO_ROOT / ".secrets" / "local-indexer-key.json"
DEFAULT_STATE_PATH = REPO_ROOT / ".local_index_state.json"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1]
        if k not in os.environ:
            os.environ[k] = v


def _load_default_project_id() -> str:
    cfg = REPO_ROOT / ".firebaserc"
    if not cfg.exists():
        return ""
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    projects = data.get("projects")
    if not isinstance(projects, dict):
        return ""
    project_id = projects.get("default")
    if isinstance(project_id, str):
        return project_id.strip()
    return ""


def _read_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(raw, dict):
        return raw
    return None


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _belongs_to_shard(file_id: str, shard_count: int, shard_index: int) -> bool:
    if shard_count <= 1:
        return True
    digest = hashlib.blake2b(file_id.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % shard_count
    return bucket == shard_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local Drive -> Firestore face indexing runner."
    )
    p.add_argument(
        "--folder",
        required=True,
        help="Drive folder URL or ID.",
    )
    p.add_argument(
        "--max-files-per-chunk",
        type=int,
        default=80,
        help="Images processed per local chunk iteration.",
    )
    p.add_argument(
        "--min-face-size",
        type=int,
        default=40,
        help="Ignore faces smaller than this many pixels.",
    )
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files already present in indexed_files collection.",
    )
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Pause between chunks (default: 0).",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Stop after N chunks (0 means no limit).",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Stop after processing N images (0 means no limit).",
    )
    p.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Checkpoint file path for resume support.",
    )
    p.add_argument(
        "--reset-state",
        action="store_true",
        help="Ignore existing state file and start from root folder.",
    )
    p.add_argument(
        "--credentials",
        type=Path,
        default=DEFAULT_CREDENTIALS_PATH,
        help="Service account JSON path for Firestore Admin writes.",
    )
    p.add_argument(
        "--project-id",
        default="",
        help="Firebase/Google Cloud project id (auto from .firebaserc if omitted).",
    )
    p.add_argument(
        "--drive-api-key",
        default="",
        help="Optional Drive API key override (defaults to functions/.env).",
    )
    p.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total shard count for parallel workers (default: 1).",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index for this worker.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    _load_env_file(FUNCTIONS_DIR / ".env")
    if args.drive_api_key:
        os.environ["DRIVE_API_KEY"] = args.drive_api_key

    project_id = (args.project_id or _load_default_project_id()).strip()
    if project_id:
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)

    creds = args.credentials.expanduser().resolve()
    if not creds.exists():
        print(f"[ERROR] Credentials file not found: {creds}")
        return 1
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds)

    sys.path.insert(0, str(FUNCTIONS_DIR))
    try:
        import main as backend  # type: ignore
    except Exception as exc:
        print(f"[ERROR] Failed to import backend module: {exc}")
        return 1

    folder_id = backend._parse_drive_folder_id(str(args.folder))
    if not folder_id:
        print("[ERROR] Invalid --folder value. Provide a Drive folder URL or ID.")
        return 1

    shard_count = max(1, int(args.shard_count))
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        print(
            f"[ERROR] Invalid shard settings: shard_index={shard_index}, "
            f"shard_count={shard_count}"
        )
        return 1

    state_file = args.state_file.expanduser().resolve()
    if args.reset_state and state_file.exists():
        state_file.unlink()

    state = _read_state(state_file)
    if state and str(state.get("rootFolderId") or "") != folder_id:
        print(
            f"[INFO] Existing state belongs to another folder; resetting: {state_file}"
        )
        state = None

    if state is None:
        state = backend._new_index_state(folder_id)
        _write_state(state_file, state)
        print(f"[INFO] Started new indexing state: {state_file}")
    else:
        print(f"[INFO] Resuming from state file: {state_file}")

    max_files = max(1, min(int(args.max_files_per_chunk), 1000))
    min_face = max(8, min(int(args.min_face_size), 512))

    if shard_count > 1:
        orig_index_single = backend._index_single_image_file

        def _sharded_index_single(
            root_folder_id: str,
            item: dict[str, Any],
            min_face_size: int,
            skip_existing: bool,
        ) -> tuple[int, bool]:
            file_id = str(item.get("id") or "")
            if not file_id:
                return 0, False
            if not _belongs_to_shard(file_id, shard_count=shard_count, shard_index=shard_index):
                # Not this worker's shard; skip quickly without download/model work.
                return 0, False
            return orig_index_single(
                root_folder_id=root_folder_id,
                item=item,
                min_face_size=min_face_size,
                skip_existing=skip_existing,
            )

        backend._index_single_image_file = _sharded_index_single

    chunk = 0
    total_processed = 0
    total_faces = 0
    total_skipped = 0
    total_errors = 0
    t0 = time.time()

    print(
        "[INFO] Running local indexing",
        json.dumps(
            {
                "folderId": folder_id,
                "maxFilesPerChunk": max_files,
                "minFaceSize": min_face,
                "skipExisting": bool(args.skip_existing),
                "shardCount": shard_count,
                "shardIndex": shard_index,
                "projectId": os.getenv("GOOGLE_CLOUD_PROJECT", ""),
                "credentials": str(creds),
            }
        ),
    )

    try:
        while state.get("queue") or state.get("buffer"):
            if args.max_chunks > 0 and chunk >= args.max_chunks:
                print("[INFO] Hit --max-chunks limit; pausing.")
                break
            if args.max_images > 0 and total_processed >= args.max_images:
                print("[INFO] Hit --max-images limit; pausing.")
                break

            chunk += 1
            result, state = backend._run_index_chunk(
                state=state,
                max_files=max_files,
                min_face_size=min_face,
                skip_existing=bool(args.skip_existing),
            )
            _write_state(state_file, state)

            processed = int(result.get("processedFiles", 0))
            faces = int(result.get("indexedFaces", 0))
            skipped = int(result.get("skippedFiles", 0))
            errs = int(result.get("errorCount", 0))

            total_processed += processed
            total_faces += faces
            total_skipped += skipped
            total_errors += errs

            elapsed = time.time() - t0
            rate = 0.0 if elapsed <= 0 else (total_processed / elapsed) * 60.0
            print(
                json.dumps(
                    {
                        "chunk": chunk,
                        "processed": processed,
                        "indexedFaces": faces,
                        "skipped": skipped,
                        "errorCount": errs,
                        "queueSize": int(result.get("queueSize", 0)),
                        "bufferSize": int(result.get("bufferSize", 0)),
                        "totals": {
                            "processed": total_processed,
                            "indexedFaces": total_faces,
                            "skipped": total_skipped,
                            "errors": total_errors,
                        },
                        "elapsed": _format_duration(elapsed),
                        "imagesPerMinute": round(rate, 2),
                    }
                )
            )

            errors = result.get("errors")
            if isinstance(errors, list):
                for msg in errors[:3]:
                    print(f"[WARN] {msg}")

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. State saved.")
        return 130
    except Exception as exc:
        print(f"[ERROR] Indexing failed: {exc}")
        return 1

    done = not state.get("queue") and not state.get("buffer")
    elapsed = time.time() - t0
    summary = {
        "done": done,
        "chunks": chunk,
        "processed": total_processed,
        "indexedFaces": total_faces,
        "skipped": total_skipped,
        "errors": total_errors,
        "elapsed": _format_duration(elapsed),
        "imagesPerMinute": round(
            0.0 if elapsed <= 0 else (total_processed / elapsed) * 60.0, 2
        ),
    }
    print("[INFO] Summary", json.dumps(summary))

    if done:
        try:
            state_file.unlink()
        except OSError:
            pass
        print("[INFO] Index complete. Cleared state file.")
    else:
        print(f"[INFO] Resume later using state file: {state_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
