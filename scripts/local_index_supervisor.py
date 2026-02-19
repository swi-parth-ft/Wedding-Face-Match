#!/usr/bin/env python3
"""
Watchdog supervisor for local Drive -> Firestore indexing.

It runs `local_index_drive_to_firestore.py` one chunk at a time with a hard
timeout. If a chunk hangs, it kills that chunk process and drops one buffered
file from the saved state so indexing can keep moving.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_SCRIPT = REPO_ROOT / "scripts" / "local_index_drive_to_firestore.py"
DEFAULT_STATE_PATH = REPO_ROOT / ".local_index_state.json"
DEFAULT_SKIPPED_PATH = REPO_ROOT / ".local_index_skipped.json"
DEFAULT_CREDENTIALS_PATH = REPO_ROOT / ".secrets" / "local-indexer-key.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Keep local indexing alive with chunk timeout + auto-recovery."
    )
    p.add_argument("--folder", required=True, help="Drive folder URL or ID.")
    p.add_argument(
        "--python-bin",
        default=str(REPO_ROOT / "functions" / "venv" / "bin" / "python"),
        help="Python binary used to run local index script.",
    )
    p.add_argument(
        "--credentials",
        type=Path,
        default=DEFAULT_CREDENTIALS_PATH,
        help="Service account JSON path.",
    )
    p.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Path to local index state file.",
    )
    p.add_argument(
        "--skipped-file",
        type=Path,
        default=DEFAULT_SKIPPED_PATH,
        help="Where skipped/timeout file ids are logged.",
    )
    p.add_argument(
        "--max-files-per-chunk",
        type=int,
        default=20,
        help="Images per chunk run.",
    )
    p.add_argument(
        "--chunks-per-run",
        type=int,
        default=5,
        help="How many chunks each child process should run before it exits.",
    )
    p.add_argument(
        "--chunk-timeout-seconds",
        type=int,
        default=600,
        help="Hard timeout for one chunk process.",
    )
    p.add_argument(
        "--min-face-size",
        type=int,
        default=40,
        help="Minimum face size passed to chunk runner.",
    )
    p.add_argument(
        "--project-id",
        default="",
        help="Optional project id override.",
    )
    p.add_argument(
        "--drive-api-key",
        default="",
        help="Optional DRIVE_API_KEY override.",
    )
    p.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total shard count for parallel workers.",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index for this worker.",
    )
    p.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=4.0,
        help="Sleep between retries/chunks.",
    )
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _append_skipped(path: Path, row: dict[str, Any]) -> None:
    history: list[dict[str, Any]]
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                history = existing
            else:
                history = []
        except Exception:
            history = []
    else:
        history = []
    history.append(row)
    _write_json(path, history[-2000:])


def _drop_one_buffer_item(
    state_file: Path, skipped_file: Path, reason: str
) -> dict[str, Any] | None:
    state = _read_json(state_file)
    if not state:
        return None
    buf = state.get("buffer")
    if not isinstance(buf, list) or not buf:
        return None
    item = buf.pop(0)
    _write_json(state_file, state)
    if isinstance(item, dict):
        row = {
            "ts": int(time.time()),
            "reason": reason,
            "id": str(item.get("id") or ""),
            "name": str(item.get("name") or ""),
            "mimeType": str(item.get("mimeType") or ""),
        }
    else:
        row = {
            "ts": int(time.time()),
            "reason": reason,
            "id": "",
            "name": "",
            "mimeType": "",
        }
    _append_skipped(skipped_file, row)
    return row


def _build_chunk_cmd(args: argparse.Namespace) -> list[str]:
    chunks_per_run = max(1, min(args.chunks_per_run, 1000))
    shard_count = max(1, int(args.shard_count))
    shard_index = int(args.shard_index)
    cmd = [
        args.python_bin,
        str(INDEX_SCRIPT),
        "--folder",
        args.folder,
        "--max-files-per-chunk",
        str(max(1, min(args.max_files_per_chunk, 1000))),
        "--max-chunks",
        str(chunks_per_run),
        "--min-face-size",
        str(max(8, min(args.min_face_size, 512))),
        "--state-file",
        str(args.state_file),
        "--credentials",
        str(args.credentials),
        "--skip-existing",
        "--shard-count",
        str(shard_count),
        "--shard-index",
        str(shard_index),
    ]
    if args.project_id:
        cmd.extend(["--project-id", args.project_id])
    if args.drive_api_key:
        cmd.extend(["--drive-api-key", args.drive_api_key])
    return cmd


def _state_done(state_file: Path) -> bool:
    state = _read_json(state_file)
    if not state:
        return True
    queue = state.get("queue")
    buf = state.get("buffer")
    if isinstance(queue, list) and isinstance(buf, list):
        return len(queue) == 0 and len(buf) == 0
    return False


def main() -> int:
    args = parse_args()
    state_file = args.state_file.expanduser().resolve()
    skipped_file = args.skipped_file.expanduser().resolve()
    credentials = args.credentials.expanduser().resolve()

    if not credentials.exists():
        print(f"[ERROR] Credentials file not found: {credentials}")
        return 1
    if not Path(args.python_bin).exists():
        print(f"[ERROR] Python binary not found: {args.python_bin}")
        return 1
    if not INDEX_SCRIPT.exists():
        print(f"[ERROR] Index script not found: {INDEX_SCRIPT}")
        return 1

    print(
        "[INFO] Supervisor started",
        json.dumps(
            {
                "folder": args.folder,
                "chunkTimeoutSec": args.chunk_timeout_seconds,
                "maxFilesPerChunk": args.max_files_per_chunk,
                "chunksPerRun": max(1, min(args.chunks_per_run, 1000)),
                "shardCount": max(1, int(args.shard_count)),
                "shardIndex": int(args.shard_index),
                "stateFile": str(state_file),
                "skippedFile": str(skipped_file),
            }
        ),
    )

    while True:
        cmd = _build_chunk_cmd(args)
        t0 = time.time()
        print(f"[INFO] Running chunk command: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=max(60, args.chunk_timeout_seconds),
                check=False,
            )
            out = proc.stdout or ""
            if out.strip():
                print(out.rstrip())
            elapsed = round(time.time() - t0, 2)
            if proc.returncode != 0:
                print(
                    f"[WARN] Chunk runner exited with code {proc.returncode} after {elapsed}s"
                )
            else:
                print(f"[INFO] Chunk runner finished in {elapsed}s")
        except subprocess.TimeoutExpired:
            elapsed = round(time.time() - t0, 2)
            print(
                f"[WARN] Chunk runner timed out after {elapsed}s "
                f"(limit={args.chunk_timeout_seconds}s)"
            )
            dropped = _drop_one_buffer_item(
                state_file=state_file,
                skipped_file=skipped_file,
                reason="chunk-timeout",
            )
            if dropped:
                print(
                    "[WARN] Dropped one buffered file to unblock progress: "
                    f"{json.dumps(dropped)}"
                )
            else:
                print("[WARN] No buffered file available to drop.")

        if _state_done(state_file):
            print("[INFO] State is done. Exiting supervisor.")
            break

        time.sleep(max(0.0, args.retry_sleep_seconds))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
