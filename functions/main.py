#!/usr/bin/env python3
"""
Firebase Functions backend for:
1) Indexing images from a public Google Drive folder (recursive, chunked).
2) Face recognition search against indexed embeddings.
3) Downloading matched files as a zip.

Expected deploy target: Cloud Functions (Firebase gen2, Python 3.11+).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
import zipfile
import zlib
from typing import Any

import firebase_admin
import google.auth
import requests
from firebase_admin import firestore
from firebase_functions import https_fn
from firebase_functions.options import MemoryOption
from google.auth.transport.requests import AuthorizedSession
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

firebase_admin.initialize_app()
_DB = None

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
FOLDER_MIME = "application/vnd.google-apps.folder"

FACE_INDEX_COLLECTION = os.getenv("FACE_INDEX_COLLECTION", "face_index")
INDEXED_FILES_COLLECTION = os.getenv("INDEXED_FILES_COLLECTION", "indexed_files")

DEFAULT_MAX_FILES_PER_CHUNK = int(os.getenv("DEFAULT_MAX_FILES_PER_CHUNK", "40"))
MAX_FILES_PER_CHUNK_CAP = int(os.getenv("MAX_FILES_PER_CHUNK_CAP", "200"))
DEFAULT_MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "40"))
DEFAULT_SEARCH_MAX_DISTANCE = float(os.getenv("SEARCH_MAX_DISTANCE", "0.35"))
DEFAULT_SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "0"))
SEARCH_CACHE_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "1800"))
SEARCH_INDEX_PAGE_SIZE = int(os.getenv("SEARCH_INDEX_PAGE_SIZE", "250"))
SEARCH_INDEX_PAGE_RETRIES = int(os.getenv("SEARCH_INDEX_PAGE_RETRIES", "4"))
SEARCH_INDEX_RETRY_BASE_SLEEP_SECONDS = float(
    os.getenv("SEARCH_INDEX_RETRY_BASE_SLEEP_SECONDS", "0.35")
)
DOWNLOAD_MAX_FILES_PER_ZIP = int(os.getenv("DOWNLOAD_MAX_FILES_PER_ZIP", "80"))
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_l")
FACE_DET_SIZE = int(os.getenv("FACE_DET_SIZE", "640"))
FACE_MODEL_MODULES = [
    part.strip()
    for part in os.getenv("FACE_MODEL_MODULES", "detection,recognition").split(",")
    if part.strip()
]

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
}

_DRIVE_SERVICE = None
_DRIVE_API_KEY = os.getenv("DRIVE_API_KEY")
_AUTH_SESSION: AuthorizedSession | None = None
_FACE_APP: Any | None = None
_FACE_INDEX_CACHE_ROWS: list[dict[str, Any]] | None = None
_FACE_INDEX_CACHE_TS = 0.0
_FACE_INDEX_CACHE_LOCK = threading.Lock()


def _np() -> Any:
    import numpy as np  # type: ignore

    return np


def _cv2() -> Any:
    import cv2  # type: ignore

    return cv2


def _face_analysis_class() -> Any:
    from insightface.app import FaceAnalysis  # type: ignore

    return FaceAnalysis


def _options_response() -> https_fn.Response:
    return https_fn.Response("", status=204, headers=CORS_HEADERS)


def _json_response(payload: dict[str, Any], status: int = 200) -> https_fn.Response:
    headers = {**CORS_HEADERS, "Content-Type": "application/json"}
    return https_fn.Response(json.dumps(payload), status=status, headers=headers)


def _binary_response(
    data: bytes, content_type: str, filename: str | None = None, status: int = 200
) -> https_fn.Response:
    headers: dict[str, str] = {**CORS_HEADERS, "Content-Type": content_type}
    if filename:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return https_fn.Response(data, status=status, headers=headers)


def _read_json(req: https_fn.Request) -> dict[str, Any]:
    payload = req.get_json(silent=True)
    if isinstance(payload, dict):
        return payload
    return {}


def _db():
    global _DB
    if _DB is None:
        _DB = firestore.client()
    return _DB


def _parse_drive_folder_id(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", raw)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", raw)
    if m:
        return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", raw):
        return raw
    return ""


def _decode_state_token(token: str) -> dict[str, Any] | None:
    try:
        raw = base64.urlsafe_b64decode(token.encode("utf-8"))
    except Exception:
        return None

    candidates = [raw]
    try:
        candidates.insert(0, zlib.decompress(raw))
    except Exception:
        pass

    for payload in candidates:
        try:
            data = json.loads(payload.decode("utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


def _encode_state_token(state: dict[str, Any]) -> str:
    raw = json.dumps(state, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    return base64.urlsafe_b64encode(compressed).decode("utf-8")


def _new_index_state(root_folder_id: str) -> dict[str, Any]:
    return {
        "rootFolderId": root_folder_id,
        "queue": [
            {
                "id": root_folder_id,
                "pageToken": None,
                "doneAfterBuffer": False,
            }
        ],
        "buffer": [],
    }


def _get_drive_service():
    global _DRIVE_SERVICE, _AUTH_SESSION
    if _DRIVE_SERVICE is not None:
        return _DRIVE_SERVICE

    if _DRIVE_API_KEY:
        _DRIVE_SERVICE = build(
            "drive",
            "v3",
            developerKey=_DRIVE_API_KEY,
            cache_discovery=False,
        )
        _AUTH_SESSION = None
        return _DRIVE_SERVICE

    creds, _ = google.auth.default(scopes=[DRIVE_SCOPE])
    _DRIVE_SERVICE = build("drive", "v3", credentials=creds, cache_discovery=False)
    _AUTH_SESSION = AuthorizedSession(creds)
    return _DRIVE_SERVICE


def _list_drive_folder_page(folder_id: str, page_token: str | None) -> dict[str, Any]:
    service = _get_drive_service()
    return (
        service.files()
        .list(
            q=f"'{folder_id}' in parents and trashed=false",
            pageSize=100,
            pageToken=page_token,
            fields="nextPageToken,files(id,name,mimeType)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )


def _get_drive_file_meta(file_id: str) -> dict[str, Any]:
    service = _get_drive_service()
    return (
        service.files()
        .get(
            fileId=file_id,
            fields="id,name,mimeType,webViewLink,webContentLink,thumbnailLink",
            supportsAllDrives=True,
        )
        .execute()
    )


def _download_drive_file_bytes(file_id: str) -> bytes:
    if _DRIVE_API_KEY:
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
        try:
            resp = requests.get(
                url, params={"alt": "media", "key": _DRIVE_API_KEY}, timeout=120
            )
            resp.raise_for_status()
            return resp.content
        except requests.HTTPError:
            # Some publicly shared files reject API-key media download. Fall back to
            # the direct public download URL.
            public = requests.get(
                _public_download_url(file_id),
                timeout=120,
                allow_redirects=True,
            )
            public.raise_for_status()
            return public.content

    _get_drive_service()
    if _AUTH_SESSION is None:
        raise RuntimeError("Auth session is not initialized")
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    resp = _AUTH_SESSION.get(
        url,
        params={"alt": "media", "supportsAllDrives": "true"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.content


def _get_face_app() -> Any:
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP

    FaceAnalysis = _face_analysis_class()
    kwargs: dict[str, Any] = {
        "name": FACE_MODEL_NAME,
        "providers": ["CPUExecutionProvider"],
    }
    if FACE_MODEL_MODULES:
        kwargs["allowed_modules"] = FACE_MODEL_MODULES

    app = FaceAnalysis(**kwargs)
    det_size = max(320, min(int(FACE_DET_SIZE), 1280))
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    _FACE_APP = app
    return app


def _normalize_embedding(vec: Any) -> Any:
    np = _np()
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _extract_face_embeddings(image_bytes: bytes, min_face_size: int) -> list[dict[str, Any]]:
    np = _np()
    cv2 = _cv2()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    app = _get_face_app()
    faces = app.get(img)
    records: list[dict[str, Any]] = []
    for face in faces:
        x1, y1, x2, y2 = [float(v) for v in face.bbox.tolist()]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w < min_face_size or h < min_face_size:
            continue
        emb = _normalize_embedding(face.embedding.astype(np.float32))
        records.append(
            {
                "embedding": emb.tolist(),
                "bbox": [x1, y1, x2, y2],
                "detScore": float(face.det_score),
            }
        )
    return records


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
    return cleaned or "file"


def _public_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _public_preview_url(file_id: str) -> str:
    return f"https://drive.google.com/thumbnail?id={file_id}&sz=w1200"


def _compact_drive_item(item: dict[str, Any]) -> dict[str, str] | None:
    file_id = str(item.get("id") or "").strip()
    if not file_id:
        return None
    return {
        "id": file_id,
        "name": str(item.get("name") or ""),
        "mimeType": str(item.get("mimeType") or ""),
    }


def _is_file_indexed(file_id: str) -> bool:
    ref = _db().collection(INDEXED_FILES_COLLECTION).document(file_id).get()
    return ref.exists


def _write_indexed_file(
    root_folder_id: str,
    item: dict[str, Any],
    faces: list[dict[str, Any]],
) -> None:
    file_id = item["id"]
    file_name = item.get("name", "")
    web_view_link = item.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view"
    web_content_link = item.get("webContentLink") or _public_download_url(file_id)
    thumbnail_link = item.get("thumbnailLink") or _public_preview_url(file_id)

    batch = _db().batch()

    for idx, face in enumerate(faces):
        face_doc_id = f"{file_id}_{idx}"
        face_ref = _db().collection(FACE_INDEX_COLLECTION).document(face_doc_id)
        batch.set(
            face_ref,
            {
                "faceId": face_doc_id,
                "fileId": file_id,
                "fileName": file_name,
                "rootFolderId": root_folder_id,
                "embedding": face["embedding"],
                "bbox": face["bbox"],
                "detScore": face["detScore"],
                "webViewLink": web_view_link,
                "webContentLink": web_content_link,
                "thumbnailLink": thumbnail_link,
                "downloadUrl": _public_download_url(file_id),
                "previewUrl": _public_preview_url(file_id),
                "indexedAt": firestore.SERVER_TIMESTAMP,
            },
        )

    indexed_ref = _db().collection(INDEXED_FILES_COLLECTION).document(file_id)
    batch.set(
        indexed_ref,
        {
            "fileId": file_id,
            "fileName": file_name,
            "rootFolderId": root_folder_id,
            "faceCount": len(faces),
            "webViewLink": web_view_link,
            "webContentLink": web_content_link,
            "thumbnailLink": thumbnail_link,
            "indexedAt": firestore.SERVER_TIMESTAMP,
        },
    )
    batch.commit()


def _index_single_image_file(
    root_folder_id: str,
    item: dict[str, Any],
    min_face_size: int,
    skip_existing: bool,
) -> tuple[int, bool]:
    file_id = item["id"]
    if skip_existing and _is_file_indexed(file_id):
        return 0, True

    blob = _download_drive_file_bytes(file_id)
    faces = _extract_face_embeddings(blob, min_face_size=min_face_size)
    _write_indexed_file(root_folder_id=root_folder_id, item=item, faces=faces)
    return len(faces), False


def _maybe_pop_finished_folder(queue: list[dict[str, Any]], buffer: list[dict[str, Any]]) -> None:
    if queue and not buffer and queue[0].get("doneAfterBuffer"):
        queue.pop(0)


def _run_index_chunk(
    state: dict[str, Any],
    max_files: int,
    min_face_size: int,
    skip_existing: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    queue = state.get("queue")
    buffer = state.get("buffer")
    if not isinstance(queue, list) or not isinstance(buffer, list):
        raise ValueError("Invalid state token")

    root_folder_id = str(state.get("rootFolderId") or "")
    if not root_folder_id:
        raise ValueError("State is missing rootFolderId")

    processed_files = 0
    indexed_faces = 0
    skipped_files = 0
    errors: list[str] = []

    while processed_files < max_files and (buffer or queue):
        _maybe_pop_finished_folder(queue, buffer)
        if not queue and not buffer:
            break

        if buffer:
            item = buffer.pop(0)
            item_id = str(item.get("id") or "")
            if not item_id:
                _maybe_pop_finished_folder(queue, buffer)
                continue
            mime = str(item.get("mimeType", ""))

            if mime == FOLDER_MIME:
                queue.append(
                    {
                        "id": item_id,
                        "pageToken": None,
                        "doneAfterBuffer": False,
                    }
                )
                _maybe_pop_finished_folder(queue, buffer)
                continue

            if not mime.startswith("image/"):
                _maybe_pop_finished_folder(queue, buffer)
                continue

            processed_files += 1
            try:
                face_count, skipped = _index_single_image_file(
                    root_folder_id=root_folder_id,
                    item=item,
                    min_face_size=min_face_size,
                    skip_existing=skip_existing,
                )
                indexed_faces += face_count
                if skipped:
                    skipped_files += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{item_id}: {exc}")

            _maybe_pop_finished_folder(queue, buffer)
            continue

        current = queue[0]
        folder_id = str(current.get("id", ""))
        page_token = current.get("pageToken")

        try:
            resp = _list_drive_folder_page(folder_id=folder_id, page_token=page_token)
        except HttpError as exc:
            errors.append(f"{folder_id}: {exc}")
            queue.pop(0)
            continue

        page_items = resp.get("files", [])
        if not isinstance(page_items, list):
            page_items = []
        for raw_item in page_items:
            if not isinstance(raw_item, dict):
                continue
            compact = _compact_drive_item(raw_item)
            if compact is not None:
                buffer.append(compact)

        next_page = resp.get("nextPageToken")
        current["pageToken"] = next_page
        current["doneAfterBuffer"] = next_page is None

        _maybe_pop_finished_folder(queue, buffer)

    new_state = {
        "rootFolderId": root_folder_id,
        "queue": queue,
        "buffer": buffer,
    }
    result = {
        "processedFiles": processed_files,
        "indexedFaces": indexed_faces,
        "skippedFiles": skipped_files,
        "errors": errors[:25],
        "errorCount": len(errors),
        "queueSize": len(queue),
        "bufferSize": len(buffer),
    }
    return result, new_state


def _decode_image_payload(payload: dict[str, Any]) -> bytes:
    if payload.get("imageBase64"):
        raw = str(payload["imageBase64"])
        if "," in raw and raw.strip().startswith("data:"):
            raw = raw.split(",", 1)[1]
        return base64.b64decode(raw)

    if payload.get("imageUrl"):
        url = str(payload["imageUrl"])
        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
        return resp.content

    raise ValueError("Provide imageBase64 or imageUrl")


def _iter_face_index_docs():
    page_size = max(50, min(int(SEARCH_INDEX_PAGE_SIZE), 1000))
    retries = max(0, min(int(SEARCH_INDEX_PAGE_RETRIES), 8))
    retry_base_sleep = max(float(SEARCH_INDEX_RETRY_BASE_SLEEP_SECONDS), 0.05)

    last_doc = None
    while True:
        query = _db().collection(FACE_INDEX_COLLECTION).order_by("__name__").limit(page_size)
        if last_doc is not None:
            query = query.start_after(last_doc)

        docs = None
        for attempt in range(retries + 1):
            try:
                docs = list(query.stream())
                break
            except Exception as exc:  # noqa: BLE001
                # Firestore can transiently fail with transport/unavailable errors
                # while scanning large collections. Retry the same page with backoff.
                if attempt >= retries:
                    raise RuntimeError(f"face index page query failed: {exc}") from exc
                sleep_for = retry_base_sleep * (2**attempt)
                print(
                    "Retrying face index page read "
                    f"(attempt {attempt + 1}/{retries}) after error: {exc}"
                )
                time.sleep(min(sleep_for, 5.0))

        if not docs:
            break
        for doc in docs:
            yield doc
        last_doc = docs[-1]


def _clear_face_index_cache() -> None:
    global _FACE_INDEX_CACHE_ROWS, _FACE_INDEX_CACHE_TS
    with _FACE_INDEX_CACHE_LOCK:
        _FACE_INDEX_CACHE_ROWS = None
        _FACE_INDEX_CACHE_TS = 0.0


def _build_face_index_cache_rows() -> list[dict[str, Any]]:
    np = _np()
    rows: list[dict[str, Any]] = []
    for doc in _iter_face_index_docs():
        data = doc.to_dict() or {}
        emb = data.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue
        try:
            cand = _normalize_embedding(np.array(emb, dtype=np.float32))
        except Exception:
            continue
        rows.append(
            {
                "fileId": str(data.get("fileId") or ""),
                "fileName": data.get("fileName"),
                "faceId": data.get("faceId", doc.id),
                "webViewLink": data.get("webViewLink"),
                "downloadUrl": data.get("downloadUrl"),
                "previewUrl": data.get("previewUrl"),
                "embeddingVec": cand,
            }
        )
    return rows


def _get_face_index_rows_for_search() -> list[dict[str, Any]]:
    global _FACE_INDEX_CACHE_ROWS, _FACE_INDEX_CACHE_TS

    ttl = max(0, int(SEARCH_CACHE_TTL_SECONDS))
    if ttl <= 0:
        return _build_face_index_cache_rows()

    now = time.time()
    if _FACE_INDEX_CACHE_ROWS is not None and (now - _FACE_INDEX_CACHE_TS) < ttl:
        return _FACE_INDEX_CACHE_ROWS

    with _FACE_INDEX_CACHE_LOCK:
        now = time.time()
        if _FACE_INDEX_CACHE_ROWS is not None and (now - _FACE_INDEX_CACHE_TS) < ttl:
            return _FACE_INDEX_CACHE_ROWS

        rows = _build_face_index_cache_rows()
        _FACE_INDEX_CACHE_ROWS = rows
        _FACE_INDEX_CACHE_TS = now
        return rows


def _cosine_distance(a: Any, b: Any) -> float:
    np = _np()
    return float(1.0 - float(np.dot(a, b)))


@https_fn.on_request(timeout_sec=540)
def health(req: https_fn.Request) -> https_fn.Response:
    if req.method == "OPTIONS":
        return _options_response()
    return _json_response({"ok": True})


@https_fn.on_request(timeout_sec=540, memory=MemoryOption.GB_2)
def index_chunk(req: https_fn.Request) -> https_fn.Response:
    if req.method == "OPTIONS":
        return _options_response()
    if req.method != "POST":
        return _json_response({"error": "Use POST"}, status=405)

    payload = _read_json(req)
    folder_raw = str(payload.get("folderId") or payload.get("folderLink") or "")
    folder_id = _parse_drive_folder_id(folder_raw)
    if not folder_id:
        return _json_response(
            {"error": "Invalid folderId/folderLink. Provide a Drive folder ID or URL."},
            status=400,
        )

    state_token = payload.get("stateToken")
    if state_token:
        state = _decode_state_token(str(state_token))
        if state is None:
            return _json_response({"error": "Invalid stateToken"}, status=400)
    else:
        state = _new_index_state(root_folder_id=folder_id)

    try:
        requested = int(payload.get("maxFiles", DEFAULT_MAX_FILES_PER_CHUNK))
    except (TypeError, ValueError):
        requested = DEFAULT_MAX_FILES_PER_CHUNK
    max_files = max(1, min(requested, MAX_FILES_PER_CHUNK_CAP))

    try:
        min_face_size = int(payload.get("minFaceSize", DEFAULT_MIN_FACE_SIZE))
    except (TypeError, ValueError):
        min_face_size = DEFAULT_MIN_FACE_SIZE
    min_face_size = max(8, min(min_face_size, 512))

    skip_existing = bool(payload.get("skipExisting", True))

    try:
        chunk_result, next_state = _run_index_chunk(
            state=state,
            max_files=max_files,
            min_face_size=min_face_size,
            skip_existing=skip_existing,
        )
    except Exception as exc:  # noqa: BLE001
        return _json_response({"error": str(exc)}, status=500)

    if int(chunk_result.get("indexedFaces", 0)) > 0:
        _clear_face_index_cache()

    done = not next_state["queue"] and not next_state["buffer"]
    return _json_response(
        {
            "done": done,
            "stateToken": None if done else _encode_state_token(next_state),
            **chunk_result,
        }
    )


@https_fn.on_request(timeout_sec=540, memory=MemoryOption.GB_2)
def search(req: https_fn.Request) -> https_fn.Response:
    if req.method == "OPTIONS":
        return _options_response()
    if req.method != "POST":
        return _json_response({"error": "Use POST"}, status=405)

    try:
        payload = _read_json(req)
        try:
            image_blob = _decode_image_payload(payload)
        except Exception as exc:  # noqa: BLE001
            return _json_response({"error": str(exc)}, status=400)

        try:
            top_k = int(payload.get("topK", DEFAULT_SEARCH_TOP_K))
        except (TypeError, ValueError):
            top_k = DEFAULT_SEARCH_TOP_K
        # topK <= 0 means "return all matches".
        if top_k < 0:
            top_k = 0

        try:
            max_distance = float(payload.get("maxDistance", DEFAULT_SEARCH_MAX_DISTANCE))
        except (TypeError, ValueError):
            max_distance = DEFAULT_SEARCH_MAX_DISTANCE
        max_distance = max(0.05, min(max_distance, 0.95))

        try:
            query_faces = _extract_face_embeddings(
                image_blob, min_face_size=DEFAULT_MIN_FACE_SIZE
            )
        except Exception as exc:  # noqa: BLE001
            return _json_response({"error": f"Failed to process image: {exc}"}, status=400)

        if not query_faces:
            return _json_response({"error": "No face found in query image."}, status=400)

        np = _np()

        query_vectors = [
            _normalize_embedding(np.array(face["embedding"], dtype=np.float32))
            for face in query_faces
        ]

        best_by_file: dict[str, dict[str, Any]] = {}
        scanned = 0
        for row in _get_face_index_rows_for_search():
            cand = row["embeddingVec"]
            scanned += 1
            dist = min(_cosine_distance(q, cand) for q in query_vectors)
            if dist > max_distance:
                continue

            file_id = row["fileId"]
            if not file_id:
                continue

            current = best_by_file.get(file_id)
            if current is None or dist < current["distance"]:
                best_by_file[file_id] = {
                    "fileId": file_id,
                    "fileName": row.get("fileName") or file_id,
                    "faceId": row.get("faceId") or file_id,
                    "distance": round(dist, 6),
                    "score": round(max(0.0, 1.0 - dist), 6),
                    "webViewLink": row.get("webViewLink")
                    or f"https://drive.google.com/file/d/{file_id}/view",
                    "downloadUrl": row.get("downloadUrl") or _public_download_url(file_id),
                    "previewUrl": row.get("previewUrl") or _public_preview_url(file_id),
                }

        matches = sorted(best_by_file.values(), key=lambda row: row["distance"])
        if top_k > 0:
            matches = matches[:top_k]
        return _json_response(
            {
                "queryFaceCount": len(query_faces),
                "indexedFaceCountScanned": scanned,
                "matchCount": len(matches),
                "matches": matches,
            }
        )
    except Exception as exc:  # noqa: BLE001
        print(f"search failed unexpectedly: {exc}")
        return _json_response(
            {
                "error": (
                    "Search backend is temporarily unavailable. "
                    "Please try again in a few seconds."
                )
            },
            status=503,
        )


@https_fn.on_request(timeout_sec=540, memory=MemoryOption.GB_2)
def download_matches_zip(req: https_fn.Request) -> https_fn.Response:
    if req.method == "OPTIONS":
        return _options_response()
    if req.method != "POST":
        return _json_response({"error": "Use POST"}, status=405)

    payload = _read_json(req)
    raw_ids = payload.get("fileIds")
    if not isinstance(raw_ids, list) or not raw_ids:
        return _json_response({"error": "fileIds must be a non-empty array"}, status=400)

    unique_file_ids: list[str] = []
    for item in raw_ids:
        file_id = str(item).strip()
        if file_id and file_id not in unique_file_ids:
            unique_file_ids.append(file_id)
    max_files = max(1, min(int(DOWNLOAD_MAX_FILES_PER_ZIP), 200))
    if len(unique_file_ids) > max_files:
        return _json_response(
            {
                "error": (
                    f"Too many files selected ({len(unique_file_ids)}). "
                    f"Please select at most {max_files} files per zip."
                )
            },
            status=400,
        )

    zip_name = _safe_filename(str(payload.get("zipName") or "face-matches"))
    zip_bytes = io.BytesIO()
    errors: list[str] = []
    used_names: set[str] = set()

    with zipfile.ZipFile(zip_bytes, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_id in unique_file_ids:
            try:
                meta = _get_drive_file_meta(file_id)
                source_name = str(meta.get("name") or f"{file_id}.jpg")
                name = _safe_filename(source_name)
                if "." not in name:
                    name += ".jpg"
                if name in used_names:
                    base, ext = os.path.splitext(name)
                    n = 2
                    while f"{base}_{n}{ext}" in used_names:
                        n += 1
                    name = f"{base}_{n}{ext}"
                used_names.add(name)

                blob = _download_drive_file_bytes(file_id)
                zf.writestr(name, blob)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{file_id}: {exc}")

    zip_bytes.seek(0)
    resp = _binary_response(
        data=zip_bytes.read(),
        content_type="application/zip",
        filename=f"{zip_name}.zip",
    )
    resp.headers["X-Download-Error-Count"] = str(len(errors))
    if errors:
        resp.headers["X-Download-Error-Sample"] = "; ".join(errors[:3])[:900]
    return resp
