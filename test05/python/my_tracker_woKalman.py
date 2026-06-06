from __future__ import division
import numpy as np
from collections import OrderedDict


class FaceTrack:
    def __init__(self, track_id, bbox, embedding=None):
        self.track_id       = track_id
        self.bbox           = bbox          # [x1,y1,x2,y2,score]
        self.embedding      = embedding
        self.hits           = 1
        self.no_loss_count  = 0
        self.age            = 0
        self.label          = None


class IoUTracker:
    def __init__(self, iou_thresh=0.4, max_lost=10, min_hits=3):
        """
        iou_thresh : minimum IoU to match detection to track
        max_lost   : frames to keep a track alive without matching detection
        min_hits   : minimum consecutive hits before a track is confirmed/visible
        """
        self.iou_thresh  = iou_thresh
        self.max_lost    = max_lost
        self.min_hits    = min_hits
        self.tracks      = OrderedDict()
        self._next_id    = 1

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return inter / float(aA + aB - inter)

    def update(self, bboxes):
        """
        bboxes : np.ndarray [N, 5] (x1,y1,x2,y2,score) — can be empty
        Returns list of (track_id, bbox, is_new) for CONFIRMED tracks only
        """
        for t in self.tracks.values():
            t.age += 1

        unmatched_dets = list(range(len(bboxes)))
        matched_tracks = set()

        # ── greedy IoU matching ───────────────────
        if self.tracks and len(bboxes):
            track_ids  = list(self.tracks.keys())
            iou_matrix = np.zeros((len(bboxes), len(track_ids)), dtype=np.float32)
            for di, det in enumerate(bboxes):
                for ti, tid in enumerate(track_ids):
                    iou_matrix[di, ti] = self._iou(det, self.tracks[tid].bbox)

            while True:
                idx = np.argmax(iou_matrix)
                di, ti = divmod(idx, len(track_ids))
                if iou_matrix[di, ti] < self.iou_thresh:
                    break
                tid = track_ids[ti]
                self.tracks[tid].bbox          = bboxes[di]
                self.tracks[tid].hits         += 1
                self.tracks[tid].no_loss_count = 0
                matched_tracks.add(tid)
                if di in unmatched_dets:
                    unmatched_dets.remove(di)
                iou_matrix[di, :] = -1
                iou_matrix[:, ti] = -1

        # ── new tracks for unmatched detections ──
        new_ids = []
        for di in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = FaceTrack(tid, bboxes[di])
            new_ids.append(tid)
            matched_tracks.add(tid)

        # ── age unmatched tracks ──────────────────
        lost_ids = [tid for tid in self.tracks if tid not in matched_tracks]
        for tid in lost_ids:
            self.tracks[tid].no_loss_count += 1

        # ── remove stale tracks ───────────────────
        stale = [tid for tid, t in self.tracks.items()
                 if t.no_loss_count > self.max_lost]
        for tid in stale:
            del self.tracks[tid]

        # ── return only CONFIRMED tracks ──────────
        # A track is confirmed when hits >= min_hits
        # Unconfirmed new tracks are kept internally but not returned yet
        results = []
        for tid, track in self.tracks.items():
            if track.hits >= self.min_hits or track.no_loss_count == 0 and track.hits > 1:
                is_new = tid in new_ids and track.hits == 1
                # Only surface as "new" after it clears min_hits on first appearance
                is_confirmed_new = (tid in new_ids is False) and track.hits == self.min_hits
                results.append((tid, track.bbox, is_confirmed_new or is_new))

        return results
