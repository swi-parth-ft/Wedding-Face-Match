#!/usr/bin/env python3
"""
Cluster photos by face identity.

Given an input directory with images, this script:
1. Detects faces and computes embeddings with InsightFace.
2. Clusters embeddings by similarity.
3. Creates output/person_### folders with matched images.

If one photo contains multiple people, it may appear in multiple person folders.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class FaceRecord:
    image_path: Path
    rel_path: Path
    bbox: list[float]
    det_score: float
    embedding: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group images into folders by person (face clustering)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root folder of downloaded images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_faces"),
        help="Output folder for grouped results.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.35,
        help="DBSCAN epsilon for cosine distance (lower=stricter, higher=looser).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="DBSCAN min samples for a cluster.",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=40,
        help="Ignore faces smaller than this many pixels (width or height).",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="How to place images in output folders.",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Also export unmatched/noise faces under output/noise.",
    )
    return parser.parse_args()


def iter_image_files(root: Path, exclude_roots: tuple[Path, ...] = ()) -> Iterable[Path]:
    for p in root.rglob("*"):
        if any(excl == p or excl in p.parents for excl in exclude_roots):
            continue
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def read_image(path: Path) -> np.ndarray | None:
    # Using imdecode allows broader path compatibility across platforms.
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def normalize_embedding(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def collect_faces(
    app: FaceAnalysis,
    input_dir: Path,
    min_face_size: int,
    exclude_roots: tuple[Path, ...],
) -> tuple[list[FaceRecord], int]:
    records: list[FaceRecord] = []
    no_face_count = 0

    image_paths = sorted(iter_image_files(input_dir, exclude_roots=exclude_roots))
    total = len(image_paths)
    print(f"Found {total} images. Detecting faces...")

    for idx, image_path in enumerate(image_paths, start=1):
        img = read_image(image_path)
        if img is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        faces = app.get(img)
        kept = 0
        rel_path = image_path.relative_to(input_dir)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w < min_face_size or h < min_face_size:
                continue
            emb = normalize_embedding(face.embedding.astype(np.float32))
            records.append(
                FaceRecord(
                    image_path=image_path,
                    rel_path=rel_path,
                    bbox=[x1, y1, x2, y2],
                    det_score=float(face.det_score),
                    embedding=emb,
                )
            )
            kept += 1

        if kept == 0:
            no_face_count += 1

        if idx % 100 == 0 or idx == total:
            print(f"Processed {idx}/{total} images")

    return records, no_face_count


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_generated_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    for child in output_dir.iterdir():
        if child.is_dir() and (
            child.name.startswith("person_") or child.name == "noise"
        ):
            shutil.rmtree(child)
        elif child.is_file() and child.name in {"summary.json", "faces.csv"}:
            child.unlink()


def make_output_name(rel_path: Path) -> str:
    return "__".join(rel_path.parts)


def materialize_file(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def export_clusters(
    labels: np.ndarray,
    records: list[FaceRecord],
    output_dir: Path,
    copy_mode: str,
    include_noise: bool,
) -> None:
    cluster_to_images: dict[int, set[tuple[Path, Path]]] = defaultdict(set)
    for label, rec in zip(labels, records):
        cluster_to_images[int(label)].add((rec.image_path, rec.rel_path))

    ordered_clusters = sorted(
        [c for c in cluster_to_images.keys() if c != -1],
        key=lambda c: len(cluster_to_images[c]),
        reverse=True,
    )

    cluster_name_map: dict[int, str] = {}
    for i, cluster_id in enumerate(ordered_clusters, start=1):
        cluster_name_map[cluster_id] = f"person_{i:03d}"

    for cluster_id in ordered_clusters:
        person_dir = output_dir / cluster_name_map[cluster_id]
        ensure_clean_dir(person_dir)
        used_names: set[str] = set()
        for image_path, rel_path in sorted(
            cluster_to_images[cluster_id], key=lambda item: str(item[1])
        ):
            candidate = make_output_name(rel_path)
            if candidate in used_names:
                stem, suffix = os.path.splitext(candidate)
                n = 2
                while f"{stem}__{n}{suffix}" in used_names:
                    n += 1
                candidate = f"{stem}__{n}{suffix}"
            used_names.add(candidate)
            materialize_file(image_path, person_dir / candidate, mode=copy_mode)

    if include_noise and -1 in cluster_to_images:
        noise_dir = output_dir / "noise"
        ensure_clean_dir(noise_dir)
        used_names: set[str] = set()
        for image_path, rel_path in sorted(
            cluster_to_images[-1], key=lambda item: str(item[1])
        ):
            candidate = make_output_name(rel_path)
            if candidate in used_names:
                stem, suffix = os.path.splitext(candidate)
                n = 2
                while f"{stem}__{n}{suffix}" in used_names:
                    n += 1
                candidate = f"{stem}__{n}{suffix}"
            used_names.add(candidate)
            materialize_file(image_path, noise_dir / candidate, mode=copy_mode)

    summary = []
    for cluster_id in ordered_clusters:
        folder = cluster_name_map[cluster_id]
        images = sorted(str(item[0]) for item in cluster_to_images[cluster_id])
        summary.append(
            {
                "cluster_id": cluster_id,
                "folder": folder,
                "image_count": len(images),
                "images": images,
            }
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with (output_dir / "faces.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["cluster_label", "image_path", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
        )
        for label, rec in zip(labels, records):
            writer.writerow([label, str(rec.image_path), *rec.bbox])

    print(f"Exported {len(ordered_clusters)} people to: {output_dir}")
    if include_noise and -1 in cluster_to_images:
        print(f"Exported noise images to: {output_dir / 'noise'}")


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_clean_dir(output_dir)
    reset_generated_outputs(output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    print("Loading InsightFace model...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    exclude_roots: tuple[Path, ...] = ()
    if output_dir == input_dir or input_dir in output_dir.parents:
        exclude_roots = (output_dir,)

    records, no_face_count = collect_faces(
        app=app,
        input_dir=input_dir,
        min_face_size=args.min_face_size,
        exclude_roots=exclude_roots,
    )

    if not records:
        raise SystemExit("No faces found. Try lowering --min-face-size.")

    X = np.stack([r.embedding for r in records])
    print(
        f"Collected {len(records)} faces from images. "
        f"Clustering (eps={args.eps}, min_samples={args.min_samples})..."
    )
    clusterer = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="cosine")
    labels = clusterer.fit_predict(X)

    clustered = int(np.sum(labels != -1))
    noise = int(np.sum(labels == -1))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"Clusters: {n_clusters}, clustered faces: {clustered}, noise faces: {noise}")
    print(f"Images without usable faces: {no_face_count}")

    export_clusters(
        labels=labels,
        records=records,
        output_dir=output_dir,
        copy_mode=args.copy_mode,
        include_noise=args.include_noise,
    )


if __name__ == "__main__":
    main()
