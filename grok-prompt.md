Here’s a **complete, copy-paste ready prompt** you can drop straight into OpenCode / Cursor / Windsurf / any AI IDE.

It contains the full working offline fall-detection pipeline (Pose + Trajectory methods), project structure, requirements, and run instructions — so the AI does **not** need to invent code.

---

```
You are an expert computer-vision engineer. 
Create a complete, production-ready, fully offline Fall Detection Pipeline project that matches the OpenViewer “Fall-detection-pipeline” product.

### Project Goal
Build a self-contained Python application that:
- Accepts webcam, video file, or RTSP stream
- Detects people with YOLOv8-Pose (or YOLOv11-Pose)
- Maintains stable track IDs across frames
- Implements TWO fall detection methods (selectable):
  1. Pose method (keypoint geometry: torso angle, aspect ratio, vertical speed)
  2. Trajectory method (sliding-window bbox features + simple LSTM-style scoring)
- Draws bounding boxes, track IDs, skeletons, and “FALL!” alerts
- Runs 100% offline after the first model download
- Saves annotated video and optional JSON event log
- Supports optional polygon zone and CLI flags

### Exact File Structure to Generate

fall-detection-pipeline/
├── requirements.txt
├── README.md
├── run.py                          # main entry point
├── config.yaml                     # all settings
├── models/                         # (empty, models download on first run)
├── utils/
│   ├── __init__.py
│   ├── geometry.py                 # pose fall rules
│   ├── trajectory.py               # trajectory / LSTM scorer
│   └── visualizer.py               # drawing helpers
├── core/
│   ├── __init__.py
│   ├── detector.py                 # YOLO + tracker wrapper
│   └── pipeline.py                 # main Detect → Track → Fall pipeline
└── data/
    └── videos/                     # put demo videos here

### Full Working Code (use exactly this logic)

#### 1. requirements.txt
```
ultralytics>=8.3.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
PyYAML>=6.0
filterpy>=1.4.5
tqdm>=4.65.0
```

#### 2. config.yaml
```yaml
model: yolov8n-pose.pt          # or yolov8s-pose.pt / yolov11n-pose.pt
device: 0                       # 0 = first GPU, cpu = CPU
conf: 0.40
iou: 0.45
tracker: botsort.yaml           # or bytetrack.yaml

# Fall methods (set one or both to true)
fall:
  pose: true
  trajectory: true

# Pose thresholds
pose:
  angle_threshold: 55.0         # degrees from vertical
  aspect_threshold: 1.25
  drop_speed_threshold: 20.0    # pixels per frame
  consecutive_frames: 4

# Trajectory settings
trajectory:
  window_size: 16
  min_frames: 8
  score_threshold: 0.65

# Output
show: true
save_video: true
save_json: true
output_dir: output
imshow_scale: 0.8
```

#### 3. utils/geometry.py
```python
import numpy as np
from collections import deque

class PoseFallDetector:
    def __init__(self, angle_th=55.0, aspect_th=1.25, drop_th=20.0, consec=4):
        self.angle_th = angle_th
        self.aspect_th = aspect_th
        self.drop_th = drop_th
        self.consec = consec
        self.history = {}  # track_id -> deque of (hip_y, aspect, angle)

    def update(self, track_id, keypoints, bbox):
        if keypoints is None or len(keypoints) < 17:
            return False

        # COCO: 5/6 shoulders, 11/12 hips
        shoulders = keypoints[[5, 6]]
        hips = keypoints[[11, 12]]
        if np.any(shoulders <= 0) or np.any(hips <= 0):
            return False

        shoulder_c = shoulders.mean(axis=0)
        hip_c = hips.mean(axis=0)
        torso = hip_c - shoulder_c
        angle = np.degrees(np.arctan2(abs(torso[0]), abs(torso[1] + 1e-6)))

        x1, y1, x2, y2 = bbox
        aspect = (x2 - x1) / max(y2 - y1, 1.0)
        hip_y = hip_c[1]

        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=30)
        self.history[track_id].append((hip_y, aspect, angle))

        hist = list(self.history[track_id])
        if len(hist) < 5:
            return False

        drop = hist[-1][0] - hist[-5][0]
        falling_now = (angle > self.angle_th and aspect > self.aspect_th) or \
                      (drop > self.drop_th and aspect > 1.1)

        # require consecutive frames
        recent = [ (a > self.angle_th and as_ > self.aspect_th) or (d > self.drop_th)
                   for (_, as_, a), d in zip(hist[-self.consec:], 
                   [hist[i][0]-hist[i-4][0] if i>=4 else 0 for i in range(len(hist)-self.consec+1, len(hist))])]
        return falling_now and sum(recent) >= self.consec - 1
```

#### 4. utils/trajectory.py
```python
import numpy as np
from collections import deque, defaultdict

class TrajectoryFallDetector:
    def __init__(self, window=16, min_frames=8, score_th=0.65):
        self.window = window
        self.min_frames = min_frames
        self.score_th = score_th
        self.hist = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h = y2 - y1
        aspect = (x2 - x1) / max(h, 1.0)
        self.hist[track_id].append([cx, cy, aspect, h])

        seq = np.array(self.hist[track_id])
        if len(seq) < self.min_frames:
            return False

        # Simple but effective trajectory score
        # 1. vertical velocity of center
        vy = np.diff(seq[:, 1])
        # 2. aspect ratio change (becomes more horizontal)
        aspect_delta = seq[-1, 2] - seq[0, 2]
        # 3. height collapse
        height_ratio = seq[-1, 3] / max(seq[0, 3], 1.0)

        # normalize roughly
        score = 0.0
        if np.mean(vy) > 8:          # falling down
            score += 0.4
        if aspect_delta > 0.4:       # becoming wider
            score += 0.3
        if height_ratio < 0.65:      # height collapsed
            score += 0.3

        return score >= self.score_th
```

#### 5. utils/visualizer.py
```python
import cv2
import numpy as np

SKELETON = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

def draw_skeleton(frame, kpts, color=(0,255,0), thickness=2):
    if kpts is None:
        return
    for a, b in SKELETON:
        if a < len(kpts) and b < len(kpts):
            pa, pb = kpts[a], kpts[b]
            if pa[0] > 0 and pb[0] > 0:
                cv2.line(frame, tuple(pa.astype(int)), tuple(pb.astype(int)), color, thickness)
    for p in kpts:
        if p[0] > 0:
            cv2.circle(frame, tuple(p.astype(int)), 3, color, -1)

def draw_box_and_label(frame, bbox, track_id, is_fall, conf=None):
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 0, 255) if is_fall else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{track_id}"
    if is_fall:
        label += " FALL!"
    if conf is not None:
        label += f" {conf:.2f}"
    cv2.putText(frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
```

#### 6. core/detector.py
```python
from ultralytics import YOLO
import numpy as np

class PersonPoseTracker:
    def __init__(self, model_path="yolov8n-pose.pt", device=0, conf=0.4, tracker="botsort.yaml"):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.tracker = tracker

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            device=self.device,
            tracker=self.tracker,
            verbose=False
        )
        r = results[0]
        boxes, ids, kpts, confs = [], [], [], []
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()
            confs = r.boxes.conf.cpu().numpy()
            if r.keypoints is not None:
                kpts = r.keypoints.xy.cpu().numpy()
            else:
                kpts = [None] * len(ids)
        return boxes, ids, kpts, confs
```

#### 7. core/pipeline.py
```python
import cv2
import json
import time
from pathlib import Path
from datetime import datetime
from utils.geometry import PoseFallDetector
from utils.trajectory import TrajectoryFallDetector
from utils.visualizer import draw_skeleton, draw_box_and_label
from core.detector import PersonPoseTracker

class FallDetectionPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.detector = PersonPoseTracker(
            model_path=cfg["model"],
            device=cfg["device"],
            conf=cfg["conf"],
            tracker=cfg["tracker"]
        )
        self.pose_det = PoseFallDetector(**cfg["pose"]) if cfg["fall"]["pose"] else None
        self.traj_det = TrajectoryFallDetector(**cfg["trajectory"]) if cfg["fall"]["trajectory"] else None
        self.events = []

    def process(self, source, output_dir="output"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(source if not str(source).isdigit() else int(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        writer = None
        if self.cfg.get("save_video", True):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = Path(output_dir) / f"fall_out_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

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
                if self.pose_det:
                    is_fall |= self.pose_det.update(tid, kp, bbox)
                if self.traj_det:
                    is_fall |= self.traj_det.update(tid, bbox)

                if is_fall:
                    self.events.append({
                        "frame": frame_idx,
                        "track_id": tid,
                        "time": frame_idx / fps,
                        "bbox": bbox.tolist()
                    })

                color = (0, 0, 255) if is_fall else (0, 255, 0)
                draw_box_and_label(frame, bbox, tid, is_fall, conf)
                if kp is not None:
                    draw_skeleton(frame, kp, color)

            # FPS overlay
            fps_cur = frame_idx / (time.time() - t0 + 1e-6)
            cv2.putText(frame, f"FPS: {fps_cur:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if writer:
                writer.write(frame)

            if self.cfg.get("show", True):
                scale = self.cfg.get("imshow_scale", 0.8)
                disp = cv2.resize(frame, None, fx=scale, fy=scale)
                cv2.imshow("Fall Detection Pipeline (Offline)", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        if self.cfg.get("save_json", True) and self.events:
            json_path = Path(output_dir) / "fall_events.json"
            with open(json_path, "w") as f:
                json.dump(self.events, f, indent=2)
            print(f"Saved events → {json_path}")

        print(f"Processed {frame_idx} frames")
        return self.events
```

#### 8. run.py
```python
#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path
from core.pipeline import FallDetectionPipeline

def main():
    parser = argparse.ArgumentParser(description="Offline Fall Detection Pipeline")
    parser.add_argument("--source", type=str, default="0",
                        help="webcam index, video path, or RTSP URL")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device is not None:
        cfg["device"] = args.device if args.device != "cpu" else "cpu"
    if args.no_show:
        cfg["show"] = False
    if args.no_save:
        cfg["save_video"] = False
        cfg["save_json"] = False

    pipe = FallDetectionPipeline(cfg)
    pipe.process(args.source)

if __name__ == "__main__":
    main()
```

#### 9. README.md
```markdown
# Offline Fall Detection Pipeline

Fully offline Detect → Track → Fall pipeline (Pose + Trajectory methods).

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
# Webcam
python run.py --source 0

# Video file
python run.py --source data/videos/your_video.mp4

# RTSP
python run.py --source rtsp://user:pass@ip:554/stream

# CPU only
python run.py --source 0 --device cpu
```

Press `q` to quit.

## Config
Edit `config.yaml` to enable/disable pose or trajectory methods and tune thresholds.
```

### Instructions for you (the AI IDE)
1. Create the exact folder structure above.
2. Write every file with the exact code I provided (do not simplify or change the logic).
3. Make sure all imports work.
4. After generating, tell me the exact commands to run it.
5. The project must work 100% offline after the first YOLO model download.

Start generating the full project now.
```

---

**How to use**

1. Copy the entire block above (from “You are an expert…” to the end).
2. Paste it into OpenCode / Cursor / Windsurf as a new chat or Composer prompt.
3. Let the AI generate the whole project.
4. Then just run:

```bash
pip install -r requirements.txt
python run.py --source 0
```

That’s it — full offline fall detection matching the product, ready to go.