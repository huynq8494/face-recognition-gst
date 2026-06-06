"""
generate_face_db.py
───────────────────
Build / update a binary face-embedding database from a directory of images.

Directory layout expected
─────────────────────────
faces_root/
├── Alice/
│   ├── 01.jpg
│   ├── 02.jpg
│   └── ...
├── Bob/
│   └── 01.jpg
└── ...

Binary database format  (easy to mmap / read in C++)
─────────────────────────────────────────────────────
[8  bytes]  magic      : "FACEDB\0\0"
[4  bytes]  version    : uint32  = 1
[4  bytes]  num_ids    : uint32
[4  bytes]  emb_dim    : uint32
── per identity ──────────────────────────────────────
[4  bytes]  name_len   : uint32
[name_len]  name       : utf-8
[4  bytes]  num_embs   : uint32
[num_embs * emb_dim * 4]  embeddings : float32 row-major
──────────────────────────────────────────────────────

A companion  .json  index is also written for human inspection.

Usage
─────
# Build from scratch
python generate_face_db.py --faces_dir /data/faces --db_out /data/face_db.bin

# Add / update one person
python generate_face_db.py --faces_dir /data/faces --db_out /data/face_db.bin \
                           --add_person /data/new_faces/Charlie

# Inspect existing db
python generate_face_db.py --inspect /data/face_db.bin
"""

from __future__ import annotations
import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from my_scrfd    import SCRFD
from my_edgeface import EdgeFace


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
MAGIC        = b'FACEDB\x00\x00'   # 8 bytes
DB_VERSION   = 1
IMG_EXTS     = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_DET_THRESH  = 0.5
DEFAULT_INPUT_SIZE  = (112, 112)


# ─────────────────────────────────────────────
#  Binary I/O helpers
# ─────────────────────────────────────────────
def _write_uint32(f, v: int):
    f.write(struct.pack('<I', v))

def _read_uint32(f) -> int:
    return struct.unpack('<I', f.read(4))[0]

def _write_str(f, s: str):
    encoded = s.encode('utf-8')
    _write_uint32(f, len(encoded))
    f.write(encoded)

def _read_str(f) -> str:
    n = _read_uint32(f)
    return f.read(n).decode('utf-8')


# ─────────────────────────────────────────────
#  Database read / write
# ─────────────────────────────────────────────
def load_db(db_path: Path) -> dict[str, np.ndarray]:
    """
    Load binary database.
    Returns { name: np.ndarray [N, emb_dim] }
    """
    db: dict[str, np.ndarray] = {}
    if not db_path.exists():
        return db

    with open(db_path, 'rb') as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f'Not a valid FACEDB file: {db_path}')
        version = _read_uint32(f)
        if version != DB_VERSION:
            raise ValueError(f'Unsupported DB version: {version}')
        num_ids = _read_uint32(f)
        emb_dim = _read_uint32(f)

        for _ in range(num_ids):
            name     = _read_str(f)
            num_embs = _read_uint32(f)
            raw      = f.read(num_embs * emb_dim * 4)
            embs     = np.frombuffer(raw, dtype=np.float32).reshape(num_embs, emb_dim)
            db[name] = embs

    print(f'[DB] Loaded {len(db)} identities from {db_path}')
    return db


def save_db(db: dict[str, np.ndarray], db_path: Path):
    """Save database to binary file + companion JSON index."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine embedding dim from first entry
    emb_dim = next(iter(db.values())).shape[1] if db else 512

    with open(db_path, 'wb') as f:
        f.write(MAGIC)
        _write_uint32(f, DB_VERSION)
        _write_uint32(f, len(db))
        _write_uint32(f, emb_dim)

        for name, embs in sorted(db.items()):
            _write_str(f, name)
            _write_uint32(f, len(embs))
            f.write(embs.astype(np.float32).tobytes())

    # ── companion JSON index ──────────────────
    json_path = db_path.with_suffix('.json')
    index = {
        'version'    : DB_VERSION,
        'emb_dim'    : emb_dim,
        'num_ids'    : len(db),
        'identities' : {
            name: {'num_embeddings': len(embs)}
            for name, embs in sorted(db.items())
        },
    }
    json_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))

    total_embs = sum(len(v) for v in db.values())
    print(f'[DB] Saved {len(db)} identities / {total_embs} embeddings → {db_path}')
    print(f'[DB] JSON index → {json_path}')


# ─────────────────────────────────────────────
#  Median embedding
# ─────────────────────────────────────────────
def compute_median_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Element-wise median across all embeddings, then L2-normalise.
    More robust than mean — outlier images (bad pose/lighting) 
    have less influence.
    """
    stacked = np.stack(embeddings, axis=0)       # [N, D]
    median  = np.median(stacked, axis=0)         # [D]
    norm    = np.linalg.norm(median)
    return median / norm if norm > 0 else median


# ─────────────────────────────────────────────
#  Embedding extraction
# ─────────────────────────────────────────────
def extract_embeddings_from_dir(
    person_dir : Path,
    detector   : SCRFD,
    recognizer : EdgeFace,
    det_thresh : float = DEFAULT_DET_THRESH,
    input_size : tuple = DEFAULT_INPUT_SIZE,
) -> list[np.ndarray]:
    """
    Process all images in `person_dir`.
    Returns list of L2-normalised embeddings (one per valid face found).
    """
    embeddings = []
    img_files  = sorted(
        p for p in person_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS
    )

    if not img_files:
        print(f'  [!] No images found in {person_dir}')
        return embeddings

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'  [!] Cannot read {img_path.name} — skipping')
            continue

        bboxes, kpss = detector.detect(img, det_thresh, input_size=input_size)

        if bboxes is None or bboxes.shape[0] == 0:
            print(f'  [!] No face detected in {img_path.name} — skipping')
            continue

        if bboxes.shape[0] > 1:
            # Pick largest face
            areas  = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            idx    = int(np.argmax(areas))
            print(f'  [~] {img_path.name}: {bboxes.shape[0]} faces, '
                  f'using largest (idx={idx})')
        else:
            idx = 0

        kps = kpss[idx] if kpss is not None else None
        emb = recognizer.get_embedding_from_det(img, bboxes[idx], kps)
        embeddings.append(emb)
        print(f'  [+] {img_path.name}: embedding extracted  '
              f'(norm={np.linalg.norm(emb):.4f})')

    return embeddings


# ─────────────────────────────────────────────
#  Build full database
# ─────────────────────────────────────────────
def build_database(
    faces_root : Path,
    db_path    : Path,
    detector   : SCRFD,
    recognizer : EdgeFace,
    det_thresh : float = DEFAULT_DET_THRESH,
    input_size : tuple = DEFAULT_INPUT_SIZE,
) -> dict[str, np.ndarray]:
    """
    Walk `faces_root`, process every sub-directory as one identity.
    Overwrites any existing database.
    """
    db: dict[str, np.ndarray] = {}

    person_dirs = sorted(
        p for p in faces_root.iterdir() if p.is_dir()
    )
    if not person_dirs:
        print(f'[!] No sub-directories found in {faces_root}')
        return db

    print(f'\n[Build] Found {len(person_dirs)} identities in {faces_root}\n')

    for person_dir in person_dirs:
        name = person_dir.name
        print(f'── {name} ──────────────────────────')
        t0   = time.perf_counter()
        embs = extract_embeddings_from_dir(
            person_dir, detector, recognizer, det_thresh, input_size)
        elapsed = (time.perf_counter() - t0) * 1000

        if not embs:
            print(f'  [!] No valid embeddings for "{name}" — skipped')
            continue

        median_emb = compute_median_embedding(embs)
        db[name]   = median_emb[np.newaxis, :]
        print(f'  → {len(embs)} images → 1 median embedding  ({elapsed:.0f} ms)\n')

    return db


# ─────────────────────────────────────────────
#  Add / update one person
# ─────────────────────────────────────────────
def add_person(
    person_dir : Path,
    db_path    : Path,
    detector   : SCRFD,
    recognizer : EdgeFace,
    det_thresh : float = DEFAULT_DET_THRESH,
    input_size : tuple = DEFAULT_INPUT_SIZE,
):
    """
    Extract embeddings from `person_dir` and merge into existing database.
    If the person already exists, their embeddings are replaced.
    """
    name = person_dir.name
    print(f'\n[Add] Processing "{name}" from {person_dir}\n')

    embs = extract_embeddings_from_dir(
        person_dir, detector, recognizer, det_thresh, input_size)

    if not embs:
        print(f'[!] No valid embeddings extracted — database not updated.')
        return

    db = load_db(db_path)
    action = 'Updated' if name in db else 'Added'
    median_emb     = compute_median_embedding(embs)
    db[name]       = median_emb[np.newaxis, :]       # shape [1, D]

    save_db(db, db_path)
    print(f'[Add] {action} "{name}" — {len(embs)} images → 1 median embedding.')

# ─────────────────────────────────────────────
#  Inspect existing database
# ─────────────────────────────────────────────
def inspect_db(db_path: Path):
    db = load_db(db_path)
    if not db:
        print('[Inspect] Database is empty or does not exist.')
        return

    print(f'\n{"─"*50}')
    print(f'  Database : {db_path}')
    print(f'  Identities : {len(db)}')
    print(f'{"─"*50}')
    for name, embs in sorted(db.items()):
        print(f'  {name:<30}  {embs.shape[0]:>3} embeddings  '
              f'dim={embs.shape[1]}')
    print(f'{"─"*50}\n')


# ─────────────────────────────────────────────
#  Init models (shared)
# ─────────────────────────────────────────────
def init_models(det_model: str, rec_model: str) -> tuple[SCRFD, EdgeFace]:
    print(f'[Model] Loading detector  : {det_model}')
    detector = SCRFD(model_file=det_model)
    detector.prepare(-1)

    print(f'[Model] Loading recognizer: {rec_model}')
    recognizer = EdgeFace(model_file=rec_model)
    recognizer.prepare(-1)

    return detector, recognizer


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Generate / update face embedding database.')

    p.add_argument('--db_out', type=str,
                   default='face_db.bin',
                   help='Output binary database path')

    p.add_argument('--det_model', type=str,
                   default='/home/huynq/models/scrfd/scrfd_500m_bnkps_shape112x112_bs1.onnx')
                #    default='/home/huynq/models/scrfd/scrfd_500m_bnkps_shape640x640.onnx')
    p.add_argument('--rec_model', type=str,
                   default='/home/huynq/models/edgeface/edgeface_xs_gamma_06.onnx')

    p.add_argument('--det_thresh',  type=float, default=DEFAULT_DET_THRESH)
    p.add_argument('--input_size',  type=int,   nargs=2, default=[112, 112])

    # ── modes (mutually exclusive) ────────────
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--faces_dir',  type=str,
                     help='Root directory of person sub-dirs (build full DB)')
    grp.add_argument('--add_person', type=str,
                     help='Directory for ONE person to add/update in DB')
    grp.add_argument('--inspect',    action='store_true',
                     help='Print contents of existing DB (no model needed)')

    return p.parse_args()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    args       = parse_args()
    db_path    = Path(args.db_out)
    input_size = tuple(args.input_size)

    # ── inspect mode (no model needed) ────────
    if args.inspect:
        inspect_db(db_path)
        sys.exit(0)

    # ── init models ───────────────────────────
    detector, recognizer = init_models(args.det_model, args.rec_model)

    # ── build full database ───────────────────
    if args.faces_dir:
        faces_root = Path(args.faces_dir)
        if not faces_root.is_dir():
            print(f'[!] faces_dir not found: {faces_root}')
            sys.exit(1)

        db = build_database(
            faces_root, db_path, detector, recognizer,
            det_thresh=args.det_thresh,
            input_size=input_size,
        )
        if db:
            save_db(db, db_path)

    # ── add / update one person ───────────────
    elif args.add_person:
        person_dir = Path(args.add_person)
        if not person_dir.is_dir():
            print(f'[!] add_person directory not found: {person_dir}')
            sys.exit(1)

        add_person(
            person_dir, db_path, detector, recognizer,
            det_thresh=args.det_thresh,
            input_size=input_size,
        )
