"""Pose-based fall rules using keypoint geometry.

COCO keypoint indices (YOLOv8-Pose):
    5/6  = left/right shoulder
    11/12 = left/right hip
"""
import numpy as np
from collections import deque


class PoseFallDetector:
    def __init__(self, angle_th=55.0, aspect_th=1.25, drop_th=20.0, consec=4):
        self.angle_th = float(angle_th)
        self.aspect_th = float(aspect_th)
        self.drop_th = float(drop_th)
        self.consec = int(consec)
        self.history = {}  # track_id -> deque of (hip_y, aspect, angle)

    @classmethod
    def from_config(cls, cfg):
        """Build from config.yaml `pose:` block keys."""
        return cls(
            angle_th=cfg.get("angle_threshold", 55.0),
            aspect_th=cfg.get("aspect_threshold", 1.25),
            drop_th=cfg.get("drop_speed_threshold", 20.0),
            consec=cfg.get("consecutive_frames", 4),
        )

    def _torso_angle(self, keypoints):
        shoulders = keypoints[[5, 6]]
        hips = keypoints[[11, 12]]
        if np.any(shoulders <= 0) or np.any(hips <= 0):
            return None, None
        shoulder_c = shoulders.mean(axis=0)
        hip_c = hips.mean(axis=0)
        torso = hip_c - shoulder_c
        angle = float(np.degrees(np.arctan2(abs(torso[0]), abs(torso[1] + 1e-6))))
        return angle, float(hip_c[1])

    def update(self, track_id, keypoints, bbox):
        if keypoints is None:
            return False
        keypoints = np.asarray(keypoints)
        if keypoints.shape[0] < 17:
            return False

        angle_hip = self._torso_angle(keypoints)
        if angle_hip[0] is None:
            return False
        angle, hip_y = angle_hip

        x1, y1, x2, y2 = (float(v) for v in bbox)
        aspect = (x2 - x1) / max(y2 - y1, 1.0)

        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=30)
        self.history[track_id].append((hip_y, aspect, angle))

        hist = list(self.history[track_id])
        if len(hist) < 5:
            return False

        drop = hist[-1][0] - hist[-5][0]  # positive = moving down in image
        falling_now = (angle > self.angle_th and aspect > self.aspect_th) or \
                      (drop > self.drop_th and aspect > 1.1)

        # Require sustained evidence over consecutive frames.
        count = 0
        for i in range(len(hist) - self.consec + 1, len(hist) + 1):
            # i is 1-based end index; need window [i-consec, i) — simplify:
            pass
        # Straightforward check: last `consec` frames each look like a fall.
        votes = 0
        for j in range(len(hist) - self.consec, len(hist)):
            if j < 4:
                d = 0.0
            else:
                d = hist[j][0] - hist[j - 4][0]
            _, a_j, ang_j = hist[j]
            if (ang_j > self.angle_th and a_j > self.aspect_th) or (d > self.drop_th):
                votes += 1

        return bool(falling_now and votes >= self.consec - 1)

    def reset(self, track_id=None):
        if track_id is None:
            self.history.clear()
        else:
            self.history.pop(track_id, None)
