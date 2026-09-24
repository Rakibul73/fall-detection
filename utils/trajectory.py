"""Trajectory-based fall scoring over a sliding bbox window."""
import numpy as np
from collections import deque, defaultdict


class TrajectoryFallDetector:
    def __init__(self, window=16, min_frames=8, score_th=0.65):
        self.window = int(window)
        self.min_frames = int(min_frames)
        self.score_th = float(score_th)
        self.hist = defaultdict(lambda: deque(maxlen=self.window))

    @classmethod
    def from_config(cls, cfg):
        """Build from config.yaml `trajectory:` block keys."""
        return cls(
            window=cfg.get("window_size", 16),
            min_frames=cfg.get("min_frames", 8),
            score_th=cfg.get("score_threshold", 0.65),
        )

    def update(self, track_id, bbox):
        x1, y1, x2, y2 = (float(v) for v in bbox)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        h = y2 - y1
        aspect = (x2 - x1) / max(h, 1.0)
        self.hist[track_id].append([cx, cy, aspect, h])

        seq = np.array(self.hist[track_id], dtype=np.float64)
        if len(seq) < self.min_frames:
            return False

        # 1. vertical velocity of center (positive = downwards)
        vy = np.diff(seq[:, 1])
        # 2. aspect ratio change (becomes more horizontal)
        aspect_delta = float(seq[-1, 2] - seq[0, 2])
        # 3. height collapse
        height_ratio = float(seq[-1, 3] / max(seq[0, 3], 1.0))

        score = 0.0
        if float(np.mean(vy)) > 8.0:      # falling down
            score += 0.4
        if aspect_delta > 0.4:            # becoming wider
            score += 0.3
        if height_ratio < 0.65:           # height collapsed
            score += 0.3

        return bool(score >= self.score_th)

    def reset(self, track_id=None):
        if track_id is None:
            self.hist.clear()
        else:
            self.hist.pop(track_id, None)
