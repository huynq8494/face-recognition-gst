from __future__ import division
import numpy as np
import cv2
import struct
import time
from my_scrfd import SCRFD
from my_edgeface import EdgeFace
from my_tracker import IoUTracker

# ─────────────────────────────────────────────
#  Binary DB loader
# ─────────────────────────────────────────────
_FACEDB_MAGIC = b'FACEDB\x00\x00'

def load_face_db(db_path: str) -> dict[str, list]:
    """
    Load face_db.bin produced by generate_face_db.py.
    Returns { name: [embedding, ...] }  — same shape as self.gallery.
    """
    gallery: dict[str, list] = {}
    with open(db_path, 'rb') as f:
        if f.read(8) != _FACEDB_MAGIC:
            raise ValueError(f'Not a valid FACEDB file: {db_path}')

        def _u32():
            return struct.unpack('<I', f.read(4))[0]

        version = _u32()
        if version != 1:
            raise ValueError(f'Unsupported FACEDB version: {version}')

        num_ids = _u32()
        emb_dim = _u32()

        for _ in range(num_ids):
            name_len = _u32()
            name     = f.read(name_len).decode('utf-8')
            num_embs = _u32()
            raw      = f.read(num_embs * emb_dim * 4)
            embs     = np.frombuffer(raw, dtype=np.float32).reshape(num_embs, emb_dim)
            gallery[name] = [embs[i] for i in range(num_embs)]

    print(f'[DB] Loaded {len(gallery)} identities  '
          f'(dim={emb_dim}) from {db_path}')
    return gallery

class FaceRecognitionSystem:
    """
    Integrates SCRFD detector, IoU tracker, and EdgeFace recognizer.

    Performance strategy
    ────────────────────
    • Detector runs every frame.
    • Tracker is ALWAYS updated (even on empty detection) so tracks age correctly.
    • Embeddings computed once per confirmed track, refreshed every re_id_interval frames.
    • min_hits in tracker prevents flash detections from surfacing.
    """

    def __init__(
        self,
        detector,
        recognizer,
        rec_threshold  = 0.5,
        re_id_interval = 30,
        iou_thresh     = 0.4,
        max_lost       = 10,
        min_hits       = 3,
    ):
        self.detector       = detector
        self.recognizer     = recognizer
        self.rec_threshold  = rec_threshold
        self.re_id_interval = re_id_interval
        self.tracker        = IoUTracker(
            iou_thresh = iou_thresh,
            max_lost   = max_lost,
            min_hits   = min_hits,
        )

        self.gallery: dict[str, list] = {}
        self._track_cache: dict[int, dict] = {}
        self._frame_idx = 0

    def load_gallery(self, db_path: str):
        """
        Populate gallery from a pre-built face_db.bin.
        Can be called at any time; merges with any already-registered faces.
        """
        loaded = load_face_db(db_path)
        for name, embs in loaded.items():
            self.gallery.setdefault(name, []).extend(embs)
            print(f'[Gallery] "{name}"  {len(embs)} embedding(s) loaded')
        print(f'[Gallery] Total identities: {len(self.gallery)}')

    # ── gallery ───────────────────────────────
    def register_face(self, label: str, img, bbox=None, kps=None):
        if bbox is not None:
            emb = self.recognizer.get_embedding_from_det(img, bbox, kps)
        else:
            emb = self.recognizer.get_embedding(img)
        self.gallery.setdefault(label, []).append(emb)
        print(f'[Gallery] Registered "{label}" '
              f'(total samples: {len(self.gallery[label])})')

    def _match_gallery(self, embedding):
        best_label, best_score = None, -1.0
        for label, embs in self.gallery.items():
            score = max(self.recognizer.cosine_similarity(embedding, e) for e in embs)
            if score > best_score:
                best_score = score
                best_label = label
        if best_score >= self.rec_threshold:
            return best_label, best_score
        return None, best_score

    # ── main pipeline ─────────────────────────
    def process_frame(self, img, det_thresh=0.5, input_size=(640, 640)):
        """
        Returns list of dicts:
            { track_id, bbox, score, embedding, label, rec_score, is_new }
        """
        self._frame_idx += 1

        # 1. Detect
        bboxes, kpss = self.detector.detect(img, det_thresh, input_size=input_size)

        # ── ALWAYS update tracker even when detection is empty ──
        if bboxes is None or bboxes.shape[0] == 0:
            bboxes = np.empty((0, 5), dtype=np.float32)
            kpss   = None

        # 2. Track
        track_results = self.tracker.update(bboxes)

        if not track_results:
            return []

        # 3. Map each confirmed track to nearest raw detection
        tid_to_det: dict[int, int] = {}
        for tid, t_bbox, _ in track_results:
            best_di, best_iou = -1, 0.0
            for di, d_bbox in enumerate(bboxes):
                iou = IoUTracker._iou(t_bbox, d_bbox)
                if iou > best_iou:
                    best_iou, best_di = iou, di
            tid_to_det[tid] = best_di if best_iou > 0.3 else -1

        # 4. Recognize (selective)
        outputs = []
        for tid, t_bbox, is_new in track_results:
            det_idx = tid_to_det.get(tid, -1)
            kps     = kpss[det_idx] if (kpss is not None and det_idx >= 0) else None
            cached  = self._track_cache.get(tid)

            need_recognition = (
                is_new
                or cached is None
                or (self._frame_idx - cached['frame']) >= self.re_id_interval
            )

            if need_recognition and det_idx >= 0:
                emb          = self.recognizer.get_embedding_from_det(
                                   img, bboxes[det_idx], kps)
                label, score = self._match_gallery(emb)
                self._track_cache[tid] = {
                    'embedding' : emb,
                    'label'     : label,
                    'rec_score' : score,
                    'frame'     : self._frame_idx,
                }
            elif cached:
                emb   = cached['embedding']
                label = cached['label']
                score = cached['rec_score']
            else:
                emb, label, score = None, None, 0.0

            if tid in self.tracker.tracks:
                self.tracker.tracks[tid].label     = label
                self.tracker.tracks[tid].embedding = emb

            outputs.append({
                'track_id' : tid,
                'bbox'     : t_bbox,
                'score'    : float(t_bbox[4]),
                'embedding': emb,
                'label'    : label,
                'rec_score': score,
                'is_new'   : is_new,
            })

        return outputs

    # ── visualisation ─────────────────────────
    @staticmethod
    def draw(img, results, fps=0.0, avg_fps=0.0):
        vis = img.copy()

        # ── face boxes ────────────────────────
        for r in results:
            x1, y1, x2, y2, _ = r['bbox'].astype(np.int32)
            tid   = r['track_id']
            label = r['label'] or 'Unknown'
            score = r['rec_score']
            color = (0, 255, 0) if r['label'] else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            text  = f'ID:{tid} {label} ({score:.2f})'
            cv2.putText(vis, text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # ── FPS overlay ───────────────────────
        fps_color     = (0, 255, 255)
        avg_fps_color = (0, 200, 255)
        cv2.putText(vis, f'FPS: {fps:6.1f}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)
        cv2.putText(vis, f'Avg FPS: {avg_fps:6.1f}',
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, avg_fps_color, 2)

        return vis


# ─────────────────────────────────────────────
#  FPS Calculator
# ─────────────────────────────────────────────
class FPSCounter:
    """
    Tracks per-frame FPS and rolling average FPS.
    Average is calculated only after `warmup_frames` to exclude
    initial model loading / JIT overhead.
    """
    def __init__(self, warmup_frames=100, avg_window=200):
        self.warmup_frames = warmup_frames
        self.avg_window    = avg_window        # rolling window size
        self._frame        = 0
        self._t_last       = None
        self._fps_history  = []               # rolling buffer after warmup
        self.fps           = 0.0
        self.avg_fps       = 0.0

    def tick(self):
        """Call once per frame. Updates fps and avg_fps."""
        now = time.perf_counter()
        self._frame += 1

        if self._t_last is not None:
            self.fps = 1.0 / max(now - self._t_last, 1e-9)

            if self._frame > self.warmup_frames:
                self._fps_history.append(self.fps)
                # keep rolling window
                if len(self._fps_history) > self.avg_window:
                    self._fps_history.pop(0)
                self.avg_fps = float(np.mean(self._fps_history))

        self._t_last = now

    @property
    def is_warmed_up(self):
        return self._frame > self.warmup_frames


if __name__ == '__main__':

    detector = SCRFD(
        model_file='/home/huynq/models/scrfd/scrfd_500m_bnkps_shape640x640.onnx')
    detector.prepare(-1)

    recognizer = EdgeFace(
        model_file='/home/huynq/models/edgeface/edgeface_xs_gamma_06.onnx')
    recognizer.prepare(-1)

    system = FaceRecognitionSystem(
        detector       = detector,
        recognizer     = recognizer,
        rec_threshold  = 0.5,
        re_id_interval = 30,
        iou_thresh     = 0.4,
        max_lost       = 10,
        min_hits       = 3,
    )

    # ── load pre-built gallery ─────────────────
    system.load_gallery('face_db.bin')

    # ── (optional) register extra face at runtime
    # ref = cv2.imread('/home/huynq/data/gallery/charlie.jpg')
    # system.register_face('Charlie', ref)

    fps_counter = FPSCounter(warmup_frames=100, avg_window=200)

    cap = cv2.VideoCapture('/home/huynq/data/videos/test_domain.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = system.process_frame(frame, det_thresh=0.5, input_size=(640, 640))

        fps_counter.tick()

        warmup_tag = '' if fps_counter.is_warmed_up else '  [warming up]'
        print(f'[Frame {system._frame_idx:5d}] '
              f'faces={len(results):2d}  '
              f'FPS={fps_counter.fps:6.1f}  '
              f'AvgFPS={fps_counter.avg_fps:6.1f}'
              f'{warmup_tag}')

        vis = FaceRecognitionSystem.draw(
            frame, results,
            fps     = fps_counter.fps,
            avg_fps = fps_counter.avg_fps,
        )

        cv2.imshow('EdgeFace + Tracker', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f'\n{"─"*45}')
    print(f'  Total frames  : {system._frame_idx}')
    print(f'  Warmup frames : {fps_counter.warmup_frames}')
    print(f'  Average FPS   : {fps_counter.avg_fps:.2f}')
    print(f'{"─"*45}')

    cap.release()
    cv2.destroyAllWindows()

