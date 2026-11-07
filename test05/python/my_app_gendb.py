# Full updated script with blur-skip support
import argparse
import json
import struct
import time
import threading
from pathlib import Path
from sklearn.cluster import KMeans
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "11"

import cv2
import numpy as np
from my_scrfd import SCRFD
from my_edgeface import EdgeFace

MAGIC = b'FACEDB\x00\x00'
VERSION = 1
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

def cluster_embeddings(embs: list[np.ndarray], n_clusters: int) -> list[np.ndarray]:
    if len(embs) == 0:
        return []
    X = np.stack(embs)
    n_clusters = min(n_clusters, len(X))
    if n_clusters <= 0:
        return []
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    clustered_medians = []
    for c in range(n_clusters):
        cluster_embs = X[labels == c]
        if len(cluster_embs) == 0:
            continue
        m = np.median(cluster_embs, axis=0)
        m = m / np.linalg.norm(m)
        clustered_medians.append(m)
    return clustered_medians

def load_db(path: Path) -> dict[str, np.ndarray]:
    if not path.exists(): return {}
    with open(path, 'rb') as f:
        if f.read(8) != MAGIC: raise ValueError("Invalid DB")
        if struct.unpack('<I', f.read(4))[0] != VERSION: raise ValueError("Bad version")
        num_ids, emb_dim = struct.unpack('<II', f.read(8))
        db = {}
        for _ in range(num_ids):
            nlen = struct.unpack('<I', f.read(4))[0]
            name = f.read(nlen).decode()
            num_embs = struct.unpack('<I', f.read(4))[0]
            raw = f.read(num_embs * emb_dim * 4)
            db[name] = np.frombuffer(raw, np.float32).reshape(num_embs, emb_dim)
        return db

def save_db(db: dict[str, np.ndarray], path: Path):
    emb_dim = next(iter(db.values())).shape[1] if db else 512
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<III', VERSION, len(db), emb_dim))
        for name, embs in db.items():
            enc = name.encode()
            f.write(struct.pack('<I', len(enc)))
            f.write(enc)
            f.write(struct.pack('<I', len(embs)))
            f.write(embs.astype(np.float32).tobytes())
    path.with_suffix('.json').write_text(json.dumps({
        "version": VERSION,
        "emb_dim": emb_dim,
        "num_ids": len(db),
        "identities": {n: {"num_embeddings": len(e)} for n, e in db.items()}
    }, indent=2))

def median_embedding(embs: list[np.ndarray]) -> np.ndarray:
    m = np.median(np.stack(embs), axis=0)
    return m / np.linalg.norm(m)

# ---------- New: blur detection helpers ----------
def crop_bbox(img: np.ndarray, bbox: np.ndarray, pad: float = 0.1) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox[:4].astype(int)
    bw = x2 - x1
    bh = y2 - y1
    # optional padding
    px = int(bw * pad)
    py = int(bh * pad)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)
    return img[y1:y2, x1:x2]

def is_blurry(face_crop: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Return True if the face_crop is considered blurry.
    Uses variance of Laplacian; lower variance => blurrier.
    """
    if face_crop is None or face_crop.size == 0:
        return True
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    # If the crop is very small, treat as blurry
    if gray.shape[0] < 10 or gray.shape[1] < 10:
        return True
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < threshold

# ---------- Extraction functions (images + video) ----------
def extract_embeddings_from_images(person_dir: Path, detector: SCRFD, recognizer: EdgeFace,
                                   min_sharpness: float = 100.0) -> list[np.ndarray]:
    embs = []
    for img_path in sorted(p for p in person_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        img = cv2.imread(str(img_path))
        if img is None: continue
        bboxes, kpss = detector.detect(img, 0.5, input_size=(112,112))
        if bboxes is None or len(bboxes) == 0: continue
        idx = np.argmax((bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1]))
        face_crop = crop_bbox(img, bboxes[idx])
        if is_blurry(face_crop, threshold=min_sharpness):
            print(f"[Info] Skipping blurry face in {img_path}, variance below {min_sharpness}.")
            continue
        emb = recognizer.get_embedding_from_det(img, bboxes[idx], kpss[idx] if kpss is not None else None)
        if emb is not None:
            embs.append(emb)
    return embs

def extract_embeddings_from_video(video_path: Path, detector: SCRFD, recognizer: EdgeFace,
                                  frame_sample_interval: float = 0.5, min_confidence: float = 0.5,
                                  min_sharpness: float = 100.0) -> list[np.ndarray]:
    """
    Open video and sample frames every `frame_sample_interval` seconds.
    For each sampled frame, detect faces and extract embedding for the largest face,
    skipping faces that are too blurry.
    """
    embs = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Cannot open video file: {video_path}")
        return embs

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = max(1, int(round(frame_sample_interval * fps)))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            bboxes, kpss = detector.detect(frame, min_confidence, input_size=(640,640))
            if bboxes is None or len(bboxes) == 0:
                frame_idx += 1
                continue
            areas = (bboxes[:,2]-bboxes[:,0]) * (bboxes[:,3]-bboxes[:,1])
            largest_idx = int(np.argmax(areas))
            face_crop = crop_bbox(frame, bboxes[largest_idx])
            if is_blurry(face_crop, threshold=min_sharpness):
                print(f"[Info] Skipping blurry face in {video_path}:{frame_idx}, variance below {min_sharpness}.")
                frame_idx += 1
                continue
            try:
                kps = kpss[largest_idx] if kpss is not None else None
                emb = recognizer.get_embedding_from_det(frame, bboxes[largest_idx], kps)
                if emb is not None:
                    embs.append(emb)
            except Exception as e:
                print("[Error] embedding extraction failed on video frame:", e)
        frame_idx += 1

    cap.release()
    return embs

def extract_embeddings(person_path: Path, detector: SCRFD, recognizer: EdgeFace, min_sharpness: float = 100.0) -> list[np.ndarray]:
    """
    Unified extractor: if person_path is a directory, use images; if it's a video file, use video extraction.
    """
    if person_path.is_dir():
        return extract_embeddings_from_images(person_path, detector, recognizer, min_sharpness=min_sharpness)
    elif person_path.is_file() and person_path.suffix.lower() in VIDEO_EXTS:
        return extract_embeddings_from_video(person_path, detector, recognizer, min_sharpness=min_sharpness)
    else:
        print(f"[Info] Path {person_path} is neither a directory of images nor a supported video file.")
        return []

# ---------- DB builders and add_person ----------
def build_db(root: Path, path: Path, det: SCRFD, rec: EdgeFace, n_clusters: int = 1, min_sharpness: float = 100.0):
    db = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        embs = extract_embeddings(person_dir, det, rec, min_sharpness=min_sharpness)
        if not embs:
            continue
        clustered = cluster_embeddings(embs, n_clusters)
        if clustered:
            db[person_dir.name] = np.stack(clustered)
    save_db(db, path)

def add_person(person_path: Path, path: Path, det: SCRFD, rec: EdgeFace, n_clusters: int = 1, min_sharpness: float = 100.0):
    """
    Add a person from either:
      - a directory of images (original behavior), or
      - a single video file (new behavior).
    The identity name used in the DB is:
      - directory name for image folder, or
      - video filename stem for video file.
    """
    embs = extract_embeddings(person_path, det, rec, min_sharpness=min_sharpness)
    if not embs:
        print("[Info] No embeddings extracted, aborting add_person.")
        return

    clustered = cluster_embeddings(embs, n_clusters)
    if not clustered:
        print("[Info] Clustering produced no clusters, aborting add_person.")
        return

    db = load_db(path)
    name = person_path.name if person_path.is_dir() else person_path.stem
    new_embs = np.stack(clustered)
    if name in db:
        try:
            db[name] = np.vstack([db[name], new_embs])
        except Exception:
            db[name] = new_embs
    else:
        db[name] = new_embs
    save_db(db, path)
    print(f"[Done] Stored {len(new_embs)} embeddings for '{name}' in {path}")

# ---------- remove, inspect, init_models (unchanged) ----------
def remove_user(name: str, path: Path):
    db = load_db(path)
    if name in db:
        del db[name]
        save_db(db, path)
        print(f"[Done] Removed user '{name}' from {path}")
    else:
        print(f"[Info] User '{name}' not found in DB {path}")

def inspect(path: Path):
    db = load_db(path)
    for name, embs in db.items():
        print(f"{name}: {len(embs)} embeddings, dim={embs.shape[1]}")

def init_models(det_model: str, rec_model: str):
    det = SCRFD(model_file=det_model); det.prepare(-1)
    rec = EdgeFace(model_file=rec_model); rec.prepare(-1)
    return det, rec

# ---------- camera_add_user: skip blurry faces when sampling ----------
def camera_add_user(path: Path, det: SCRFD, rec: EdgeFace, n_clusters: int, cam_index: int = 0, min_sharpness: float = 100.0):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[Error] Cannot open camera index", cam_index)
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution: {int(actual_width)}x{int(actual_height)}")

    name_lock = threading.Lock()
    name_holder = {"name": None, "entered": False}

    def input_thread():
        nm = input("Enter new user name (or leave empty to cancel): ").strip()
        with name_lock:
            name_holder["name"] = nm
            name_holder["entered"] = True

    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    collected_embs = []
    last_sample_time = 0.0
    last_warn_time = 0.0

    window_name = "Add user - Press 'n' to prompt name, 'q' or ESC to finish"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] Failed to read frame from camera.")
                break

            bboxes, kpss = det.detect(frame, 0.5, input_size=(640,640))
            if bboxes is None:
                bboxes = np.zeros((0,4))
            n_faces = 0 if bboxes is None else len(bboxes)

            largest_idx = None
            if n_faces > 0:
                areas = (bboxes[:,2]-bboxes[:,0]) * (bboxes[:,3]-bboxes[:,1])
                largest_idx = int(np.argmax(areas))
                for i, bb in enumerate(bboxes):
                    x1, y1, x2, y2 = map(int, bb[:4])
                    if i == largest_idx:
                        color = (0,255,0)
                        thickness = 2
                    else:
                        color = (0,0,255)
                        thickness = 1
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

            now = time.time()
            if n_faces > 1 and now - last_warn_time >= 1.0:
                print("[Warning] More than one face detected.")
                last_warn_time = now

            with name_lock:
                name_entered = name_holder["entered"]
                name_val = name_holder["name"]

            if largest_idx is not None and name_entered and name_val:
                if now - last_sample_time >= 0.5:
                    try:
                        face_crop = crop_bbox(frame, bboxes[largest_idx])
                        if is_blurry(face_crop, threshold=min_sharpness):
                            # indicate skipped sample (red dot)
                            cv2.circle(frame, (20,20), 8, (0,0,255), -1)
                        else:
                            kps = kpss[largest_idx] if kpss is not None else None
                            emb = rec.get_embedding_from_det(frame, bboxes[largest_idx], kps)
                            if emb is not None:
                                collected_embs.append(emb)
                                cv2.circle(frame, (20,20), 8, (0,255,0), -1)
                    except Exception as e:
                        print("[Error] embedding extraction failed:", e)
                    last_sample_time = now

            if not name_entered:
                cv2.putText(frame, "Enter name in console or press 'n' to prompt", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Collecting for: {name_val}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            if key == ord('n'):
                if not t.is_alive():
                    t = threading.Thread(target=input_thread, daemon=True)
                    t.start()

    finally:
        cap.release()
        cv2.destroyWindow(window_name)

    with name_lock:
        name = name_holder["name"]
        entered = name_holder["entered"]

    if not entered or not name:
        print("[Info] No name entered or empty name, aborting save.")
        return

    if not collected_embs:
        print("[Info] No embeddings collected, nothing to store.")
        return

    clustered = cluster_embeddings(collected_embs, n_clusters)
    if not clustered:
        print("[Info] Clustering produced no clusters, aborting save.")
        return

    db = load_db(path)
    new_embs = np.stack(clustered)
    if name in db:
        try:
            db[name] = np.vstack([db[name], new_embs])
        except Exception:
            db[name] = new_embs
    else:
        db[name] = new_embs
    save_db(db, path)
    print(f"[Done] Stored {len(new_embs)} embeddings for '{name}' in {path}")

# ---------- CLI ----------
if __name__ == "__main__":
    det_model = '/home/hw/models/scrfd/scrfd_500m_bnkps_shape640x640.onnx'
    rec_model = '/home/hw/models/edgeface/edgeface_xs_gamma_06.onnx'
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--db_out", default="face_db.bin")
    p.add_argument("-n", "--n_clusters", type=int, default=1)
    p.add_argument("-c", "--camera", action="store_true", help="Start realtime camera to add a new user")
    p.add_argument("-r", "--remove", metavar="USER", help="Remove USER from the database")
    p.add_argument("--min_sharpness", type=float, default=100.0, help="Minimum Laplacian variance to accept a face (skip blurry)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("-d", "--faces_dir")
    g.add_argument("-a", "--add_person", help="Path to person images directory or a video file (e.g., USER.mp4)")
    g.add_argument("-i", "--inspect", action="store_true")
    args = p.parse_args()
    path = Path(args.db_out)

    if args.inspect:
        inspect(path)
    elif args.remove:
        remove_user(args.remove, path)
    else:
        det, rec = init_models(det_model, rec_model)
        if args.faces_dir:
            build_db(Path(args.faces_dir), path, det, rec, args.n_clusters, min_sharpness=args.min_sharpness)
        elif args.add_person:
            add_person(Path(args.add_person), path, det, rec, args.n_clusters, min_sharpness=args.min_sharpness)
        elif args.camera:
            camera_add_user(path, det, rec, args.n_clusters, cam_index=0, min_sharpness=args.min_sharpness)

    print("[Done] Output DB:", path)
