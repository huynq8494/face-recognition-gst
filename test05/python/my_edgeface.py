from __future__ import division
import numpy as np
import onnxruntime
import os.path as osp
import cv2
from utils import FACE_ALIGN_REF_POINTS

class EdgeFace:
    def __init__(self, model_file=None, session=None, input_size=(112, 112)):
        self.model_file = model_file
        self.session    = session
        self.input_size = input_size
        self.taskname   = 'recognition'

        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        self._init_vars()

    def _init_vars(self):
        input_cfg          = self.session.get_inputs()[0]
        self.input_name    = input_cfg.name
        outputs            = self.session.get_outputs()
        self.output_names  = [o.name for o in outputs]
        self.embedding_size = outputs[0].shape[-1]

    def prepare(self, ctx_id):
        provider = 'CUDAExecutionProvider' if ctx_id >= 0 else 'CPUExecutionProvider'
        self.session.set_providers([provider])

    # ── face alignment ────────────────────────
    @staticmethod
    def align_face(img, kps, output_size=(112, 112)):
        """
        Align face using 5 keypoints via affine transform.
        kps : np.ndarray [5, 2]
        """
        ref = FACE_ALIGN_REF_POINTS.copy()
        # scale reference to output_size
        ref[:, 0] *= output_size[0] / 112.0
        ref[:, 1] *= output_size[1] / 112.0
        M, _ = cv2.estimateAffinePartial2D(kps, ref, method=cv2.LMEDS)
        aligned = cv2.warpAffine(img, M, output_size, flags=cv2.INTER_LINEAR)
        return aligned

    @staticmethod
    def crop_face(img, bbox, margin=0.0):
        """
        Crop face from image using bbox [x1,y1,x2,y2,...].
        Used when kps is not available.
        """
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        w, h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - w * margin))
        y1 = max(0, int(y1 - h * margin))
        x2 = min(img.shape[1], int(x2 + w * margin))
        y2 = min(img.shape[0], int(y2 + h * margin))
        return img[y1:y2, x1:x2]

    # ── preprocessing ─────────────────────────
    def preprocess(self, face_img):
        img = cv2.resize(face_img, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))          # HWC → CHW
        img = np.expand_dims(img, axis=0)            # add batch
        return img

    # ── inference ─────────────────────────────
    def get_embedding(self, face_img):
        blob      = self.preprocess(face_img)
        net_out   = self.session.run(self.output_names, {self.input_name: blob})
        embedding = net_out[0].flatten()
        return self._l2_normalize(embedding)

    def get_embedding_from_det(self, img, bbox, kps=None):
        """
        Extract embedding directly from full image + detector output.
        img   : full BGR image
        bbox  : [x1,y1,x2,y2,score]
        kps   : [5, 2] or None
        """
        if kps is not None:
            face = self.align_face(img, kps, output_size=self.input_size)
        else:
            face = self.crop_face(img, bbox)
        return self.get_embedding(face)

    def get_embeddings_batch(self, img, bboxes, kpss=None):
        """
        Extract embeddings for all detections in one call.
        Returns list of embeddings aligned with bboxes.
        """
        embeddings = []
        for i, bbox in enumerate(bboxes):
            kps = kpss[i] if kpss is not None else None
            emb = self.get_embedding_from_det(img, bbox, kps)
            embeddings.append(emb)
        return embeddings

    # ── similarity helpers ────────────────────
    @staticmethod
    def _l2_normalize(v):
        n = np.linalg.norm(v)
        return v / n if n != 0 else v

    @staticmethod
    def cosine_similarity(e1, e2):
        return float(np.dot(e1, e2))          # works on L2-normalised vectors

    def is_same_person(self, e1, e2, threshold=0.5):
        score = self.cosine_similarity(e1, e2)
        return score >= threshold, score
