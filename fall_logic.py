import math
import time
from collections import deque

# COCO pose keypoint indices (Ultralytics yolov8-pose output order)
KP = {
    "nose": 0, "left_shoulder": 5, "right_shoulder": 6,
    "left_hip": 11, "right_hip": 12,
    "left_ankle": 15, "right_ankle": 16,
}


class TrackState:
    def __init__(self, history_len):
        self.bbox_history = deque(maxlen=history_len)
        self.angle_history = deque(maxlen=history_len)
        self.fall_start_time = None
        self.alerted = False


class FallDetector:
    def __init__(self, history_len=20, angle_threshold_deg=45, confirm_seconds=1.5):
        self.history_len = history_len
        self.angle_threshold_deg = angle_threshold_deg
        self.confirm_seconds = confirm_seconds
        self.tracks = {}

    def _get_state(self, track_id):
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(self.history_len)
        return self.tracks[track_id]

    def _body_angle(self, keypoints):
        # angle of torso line (shoulder midpoint -> hip midpoint) vs vertical
        try:
            ls, rs = keypoints[KP["left_shoulder"]], keypoints[KP["right_shoulder"]]
            lh, rh = keypoints[KP["left_hip"]], keypoints[KP["right_hip"]]
            if min(ls[2], rs[2], lh[2], rh[2]) < 0.3:
                return None
            sx, sy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
            hx, hy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
            dx, dy = hx - sx, hy - sy
            angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
            return angle
        except Exception:
            return None

    def update(self, track_id, bbox, keypoints=None):
        """
        bbox: (x1, y1, x2, y2)
        keypoints: Nx3 array (x, y, conf) or None
        Returns: (is_fall_confirmed: bool, confidence: float)
        """
        state = self._get_state(track_id)
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        aspect = w / (h + 1e-6)
        state.bbox_history.append((aspect, y2, time.time()))

        angle = self._body_angle(keypoints) if keypoints is not None else None
        if angle is not None:
            state.angle_history.append(angle)

        is_falling_now = False
        confidence = 0.0

        if angle is not None and len(state.angle_history) >= 3:
            if angle > self.angle_threshold_deg:
                is_falling_now = True
                confidence = min(angle / 90.0, 1.0)
        elif len(state.bbox_history) >= self.history_len:
            aspects = [a for a, _, _ in state.bbox_history]
            was_tall = aspects[0] < 0.8
            is_wide = aspects[-1] > 1.3
            if was_tall and is_wide:
                is_falling_now = True
                confidence = min(aspects[-1] / 2.0, 1.0)

        if is_falling_now:
            if state.fall_start_time is None:
                state.fall_start_time = time.time()
            elapsed = time.time() - state.fall_start_time
            if elapsed >= self.confirm_seconds and not state.alerted:
                state.alerted = True
                return True, confidence
        else:
            state.fall_start_time = None
            state.alerted = False

        return False, confidence

    def reset_track(self, track_id):
        if track_id in self.tracks:
            del self.tracks[track_id]
