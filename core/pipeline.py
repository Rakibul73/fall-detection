"""Main Detect -> Track -> Fall pipeline (fully offline after model download)."""
import json
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from utils.geometry import PoseFallDetector
from utils.trajectory import TrajectoryFallDetector
from utils.visualizer import draw_skeleton, draw_box_and_label, draw_zone
from core.detector import PersonPoseTracker


def point_in_polygon(point, polygon):
    """Ray-casting point-in-polygon. point=(x,y), polygon=Nx2 array-like."""
    if polygon is None:
        return True
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or len(poly) < 3:
        return True
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


class FallDetectionPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        device = cfg.get("device", None)
        # YAML `device: 0` parses as int; ultralytics accepts int/str/None.
        self.detector = PersonPoseTracker(
            model_path=cfg.get("model", "yolov8n-pose.pt"),
            device=device,
            conf=float(cfg.get("conf", 0.40)),
            iou=float(cfg.get("iou", 0.45)),
            tracker=cfg.get("tracker", "botsort.yaml"),
        )
        fall_cfg = cfg.get("fall", {}) or {}
        self.pose_det = PoseFallDetector.from_config(cfg.get("pose", {}) or {}) \
            if fall_cfg.get("pose", True) else None
        self.traj_det = TrajectoryFallDetector.from_config(cfg.get("trajectory", {}) or {}) \
            if fall_cfg.get("trajectory", True) else None
        zone_cfg = cfg.get("zone", {}) or {}
        self.zone = zone_cfg.get("polygon", None)
        self.events = []

    def _in_zone(self, bbox):
        if not self.zone:
            return True
        x1, y1, x2, y2 = bbox
        cx, cy = (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0
        return point_in_polygon((cx, cy), self.zone)

    def process(self, source, output_dir=None):
        output_dir = Path(output_dir or self.cfg.get("output_dir", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        src = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:  # NaN guard for webcams
            fps = 25.0

        writer = None
        if self.cfg.get("save_video", True):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = output_dir / f"fall_out_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            print(f"Saving annotated video -> {out_path}")

        frame_idx = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            boxes, ids, kpts, confs = self.detector.track(frame)

            for i, tid in enumerate(ids):
                bbox = boxes[i]
                kp = kpts[i] if i < len(kpts) else None
                conf = confs[i] if i < len(confs) else None

                is_fall = False
                if self._in_zone(bbox):
                    if self.pose_det is not None:
                        is_fall |= bool(self.pose_det.update(int(tid), kp, bbox))
                    if self.traj_det is not None:
                        is_fall |= bool(self.traj_det.update(int(tid), bbox))

                if is_fall:
                    self.events.append({
                        "frame": frame_idx,
                        "track_id": int(tid),
                        "time": round(frame_idx / fps, 3),
                        "bbox": [float(v) for v in np.asarray(bbox).tolist()],
                    })

                color = (0, 0, 255) if is_fall else (0, 255, 0)
                draw_box_and_label(frame, bbox, int(tid), bool(is_fall), conf)
                if kp is not None:
                    draw_skeleton(frame, kp, color)

            draw_zone(frame, self.zone)

            fps_cur = frame_idx / (time.time() - t0 + 1e-6)
            cv2.putText(frame, f"FPS: {fps_cur:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if writer:
                writer.write(frame)

            if self.cfg.get("show", True):
                try:
                    scale = float(self.cfg.get("imshow_scale", 0.8))
                except (TypeError, ValueError):
                    scale = 0.8
                disp = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
                cv2.imshow("Fall Detection Pipeline (Offline)", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        if self.cfg.get("save_json", True) and self.events:
            json_path = output_dir / "fall_events.json"
            with open(json_path, "w") as f:
                json.dump(self.events, f, indent=2)
            print(f"Saved events -> {json_path}")
        elif self.cfg.get("save_json", True):
            print("No falls detected; no JSON written.")

        print(f"Processed {frame_idx} frames, {len(self.events)} fall events.")
        return self.events
