from __future__ import division
import numpy as np
import cv2
from collections import OrderedDict


# ─────────────────────────────────────────────
#  Kalman Filter per track
#  State  : [cx, cy, w, h, vcx, vcy, vw, vh]
#  Measure: [cx, cy, w, h]
# ─────────────────────────────────────────────
class KalmanBoxTracker:
    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(8, 4)

        # State transition  (constant-velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0,  1, 0, 0, 0],
            [0, 1, 0, 0,  0, 1, 0, 0],
            [0, 0, 1, 0,  0, 0, 1, 0],
            [0, 0, 0, 1,  0, 0, 0, 1],
            [0, 0, 0, 0,  1, 0, 0, 0],
            [0, 0, 0, 0,  0, 1, 0, 0],
            [0, 0, 0, 0,  0, 0, 1, 0],
            [0, 0, 0, 0,  0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix  (observe cx, cy, w, h only)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0,  0, 0, 0, 0],
            [0, 1, 0, 0,  0, 0, 0, 0],
            [0, 0, 1, 0,  0, 0, 0, 0],
            [0, 0, 0, 1,  0, 0, 0, 0],
        ], dtype=np.float32)

        # Process noise  (velocity terms get higher noise)
        Q = np.eye(8, dtype=np.float32) * 0.01
        Q[4:, 4:] *= 5.0
        self.kf.processNoiseCov = Q

        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5

        # Initial error covariance  (high uncertainty in velocity)
        P = np.eye(8, dtype=np.float32) * 10.0
        P[4:, 4:] *= 100.0
        self.kf.errorCovPost = P

        # Initialise state from first detection
        cx, cy, w, h = self._to_z(bbox)
        self.kf.statePost = np.array(
            [cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)

        self.last_score = float(bbox[4]) if len(bbox) > 4 else 1.0

    # ── conversions ───────────────────────────
    @staticmethod
    def _to_z(bbox):
        x1, y1, x2, y2 = bbox[:4]
        return (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1

    @staticmethod
    def _to_bbox(state, score=1.0):
        cx, cy, w, h = state[:4].ravel()
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2, score],
                        dtype=np.float32)

    # ── kalman steps ──────────────────────────
    def predict(self):
        """Call every frame to get predicted bbox."""
        state = self.kf.predict()
        return self._to_bbox(state, self.last_score)

    def update(self, bbox):
        """Call when a detection is matched to this track."""
        cx, cy, w, h = self._to_z(bbox)
        z = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)
        self.kf.correct(z)
        self.last_score = float(bbox[4]) if len(bbox) > 4 else 1.0


# ─────────────────────────────────────────────
#  Face Track
# ─────────────────────────────────────────────
class FaceTrack:
    def __init__(self, track_id, bbox, embedding=None):
        self.track_id      = track_id
        self.kalman        = KalmanBoxTracker(bbox)
        self.bbox          = bbox           # last raw detection bbox
        self.pred_bbox     = bbox           # kalman-predicted bbox
        self.embedding     = embedding
        self.hits          = 1
        self.no_loss_count = 0
        self.age           = 0
        self.label         = None


# ─────────────────────────────────────────────
#  IoU + Kalman Tracker
# ─────────────────────────────────────────────
class IoUTracker:
    def __init__(self, iou_thresh=0.3, max_lost=10, min_hits=3):
        """
        iou_thresh : IoU threshold for matching (lower = tolerates more movement)
        max_lost   : frames before a track is removed
        min_hits   : frames before a track is confirmed (avoids flash detections)
        """
        self.iou_thresh = iou_thresh
        self.max_lost   = max_lost
        self.min_hits   = min_hits
        self.tracks     = OrderedDict()
        self._next_id   = 1

    # ── helpers ───────────────────────────────
    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        if inter == 0:
            return 0.0
        aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return inter / float(aA + aB - inter)

    @staticmethod
    def _center_distance(boxA, boxB):
        """Normalised centre-to-centre distance (fallback for low-IoU frames)."""
        cxA = (boxA[0]+boxA[2])/2;  cyA = (boxA[1]+boxA[3])/2
        cxB = (boxB[0]+boxB[2])/2;  cyB = (boxB[1]+boxB[3])/2
        wA  = boxA[2]-boxA[0];      hA  = boxA[3]-boxA[1]
        return np.sqrt((cxA-cxB)**2+(cyA-cyB)**2) / (np.sqrt(wA*hA) + 1e-6)

    def _match_score(self, det_bbox, pred_bbox):
        """
        Combined score: IoU + inverse-distance bonus.
        Handles cases where fast motion drops IoU below threshold.
        """
        iou  = self._iou(det_bbox, pred_bbox)
        dist = self._center_distance(det_bbox, pred_bbox)
        # distance penalty: 0 when overlapping, grows with motion
        dist_score = max(0.0, 1.0 - dist)
        return 0.7 * iou + 0.3 * dist_score

    # ── update ────────────────────────────────
    def update(self, bboxes):
        """
        bboxes      : np.ndarray [N, 5]  — may be empty
        Returns     : list of (track_id, predicted_bbox, is_new)
                      only for CONFIRMED tracks (hits >= min_hits)
        """
        # ── 1. Kalman predict for every track ─
        for t in self.tracks.values():
            t.pred_bbox = t.kalman.predict()
            t.age      += 1

        unmatched_dets = list(range(len(bboxes)))
        matched_tracks = set()

        # ── 2. Greedy matching on combined score ─
        if self.tracks and len(bboxes):
            track_ids = list(self.tracks.keys())
            score_mat = np.zeros((len(bboxes), len(track_ids)), dtype=np.float32)
            for di, det in enumerate(bboxes):
                for ti, tid in enumerate(track_ids):
                    score_mat[di, ti] = self._match_score(
                        det, self.tracks[tid].pred_bbox)

            while True:
                idx = np.argmax(score_mat)
                di, ti = divmod(idx, len(track_ids))
                if score_mat[di, ti] < self.iou_thresh:
                    break
                tid = track_ids[ti]
                track = self.tracks[tid]
                track.kalman.update(bboxes[di])     # correct kalman
                track.bbox          = bboxes[di]
                track.hits         += 1
                track.no_loss_count = 0
                matched_tracks.add(tid)
                if di in unmatched_dets:
                    unmatched_dets.remove(di)
                score_mat[di, :] = -1
                score_mat[:, ti] = -1

        # ── 3. Create new tracks ──────────────
        new_ids = []
        for di in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = FaceTrack(tid, bboxes[di])
            new_ids.append(tid)
            matched_tracks.add(tid)

        # ── 4. Age unmatched, remove stale ────
        for tid in list(self.tracks):
            if tid not in matched_tracks:
                self.tracks[tid].no_loss_count += 1
                if self.tracks[tid].no_loss_count > self.max_lost:
                    del self.tracks[tid]

        # ── 5. Return confirmed tracks only ───
        results = []
        for tid, track in self.tracks.items():
            if track.hits < self.min_hits:
                continue                            # not confirmed yet
            # Use Kalman predicted bbox (smooth even on missed frames)
            out_bbox = track.pred_bbox.copy()
            out_bbox[4] = track.bbox[4]            # keep original score
            is_new = (tid in new_ids) and track.hits == 1
            results.append((tid, out_bbox, is_new))

        return results
