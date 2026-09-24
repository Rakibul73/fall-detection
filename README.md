# Fall Detection (YOLOv8-Pose)

Real-time human fall detection from webcam, video file, or RTSP camera stream.
Uses **YOLOv8-pose** for person detection + tracking + pose estimation, a two-stage
fall classifier (body-angle → bbox-aspect fallback), SQLite event logging,
snapshot clips, optional annotated video output, and optional Telegram alerts.

## How it works

```
camera / video file / RTSP
        │
        ▼
YOLOv8-pose + ByteTrack   ──►  per-person track_id, bbox, 17 keypoints
        │
        ▼
FallDetector.update() per track
        │
        ├── Method 1 (primary): torso angle
        │     shoulder-midpoint → hip-midpoint vs vertical.
        │     If angle > angle_threshold_deg (default 45°) for
        │     fall_confirm_seconds (default 1.5 s) → FALL.
        │     Confidence = angle / 90°.
        │
        ├── Method 2 (fallback, when keypoint confidence < 0.3):
        │     bbox aspect-ratio trajectory over history_len frames.
        │     Tall (w/h < 0.8) → wide (w/h > 1.3) = person went horizontal.
        │     Confidence = aspect / 2.0.
        │
        ▼
on confirmed fall: save snapshot → log to SQLite → Telegram alert (optional)
```

**Files:**

| File | Purpose |
|---|---|
| `main.py` | Pipeline: load config → YOLO track → per-track fall check → annotate → display / save video → log + alert |
| `fall_logic.py` | `FallDetector` — angle + aspect logic, per-track history, confirmation timer |
| `db.py` | SQLite (`data/events.db`, table `falls`) — `init_db()`, `log_fall()` |
| `telegram_alert.py` | `send_telegram_alert()` — message or photo via Bot API |
| `config.yaml` | All tunables (source, model, thresholds, telegram, clip saving) |

**Key design points:**

- Each tracked person gets independent state (`TrackState`): bbox history, angle
  history, fall-start timer, alerted flag — so one fall doesn't re-trigger every frame.
- Confirmation delay (`fall_confirm_seconds`) filters out brief bends/crouches.
- Pose-first, bbox-fallback: works even when keypoints are occluded or low-confidence.
- ByteTrack (`tracker="bytetrack.yaml"`) keeps `track_id` stable across frames.

## Requirements

- Python 3.10+ (tested on 3.14), Windows / Linux / macOS
- ~6 MB model download on first run (`yolov8n-pose.pt`, then fully offline)
- CPU works; GPU optional (see config)

## Installation

```powershell
# 1. Clone / open the folder
cd fall-detection

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

### Webcam (with display window)

```powershell
.\venv\Scripts\python.exe main.py
# press 'q' in the window to quit
```

### Video file (headless — no window needed)

```powershell
.\venv\Scripts\python.exe main.py --source data/sample.mp4 --headless
```

Video-file sources auto-enable headless mode; `--headless` makes it explicit
(required on servers, SSH, Docker where there is no display).

### Save annotated output video

```powershell
.\venv\Scripts\python.exe main.py --source data/sample.mp4 --headless --output data/annotated.mp4
```

### RTSP camera

```powershell
.\venv\Scripts\python.exe main.py --source "rtsp://user:pass@cam-ip:554/stream" --headless
```

Or set it permanently in `config.yaml`:
```yaml
source: "rtsp://user:pass@cam-ip:554/stream"
```

### All CLI options

```
--source      Override config source: webcam index (0), video path, or RTSP URL
--headless    No GUI window (servers / SSH / Docker)
--show        Force GUI window even for video files
--output      Save annotated video (e.g. data/annotated.mp4)
--max-frames  Stop after N frames (0 = until stream ends / 'q'). Good for tests
--config      Config file path (default: config.yaml)
```

`--source` accepts a webcam index as string too: `--source 0`.

## Configuration (`config.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `source` | `0` | Webcam index, video path, or RTSP URL |
| `model` | `yolov8n-pose.pt` | Pose model (`yolov8s/m/l-pose.pt` = slower but more accurate) |
| `device` | `cpu` | `cpu` or `0` (first GPU) |
| `imgsz` | `640` | Inference image size (larger = slower, more accurate) |
| `conf_threshold` | `0.4` | Person detection confidence cutoff |
| `fall_confirm_seconds` | `1.5` | How long the fall pose must persist before alerting |
| `angle_threshold_deg` | `45` | Torso-from-vertical angle that counts as "fallen" |
| `history_len` | `20` | Frames of bbox history for the aspect-ratio fallback |
| `telegram.enabled` | `false` | Set `true` + token/chat_id to enable alerts |
| `save_clips` / `clips_dir` | `true` / `data/clips` | Save a JPEG snapshot per fall event |

**Tuning tips:**

- Too many false alarms → raise `fall_confirm_seconds` (e.g. `2.5`) or
  `angle_threshold_deg` (e.g. `55`).
- Missing real falls → lower them, or use a bigger model (`yolov8s-pose.pt`).
- Crowded scenes → raise `conf_threshold` (e.g. `0.5`).

## Telegram alerts

1. Chat with `@BotFather` on Telegram → `/newbot` → copy the token.
2. Chat with your bot (send any message), then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your `chat_id`.
3. In `config.yaml`:
   ```yaml
   telegram:
     enabled: true
     bot_token: "123456:ABC..."
     chat_id: "987654321"
   ```

Each confirmed fall sends the snapshot photo with track id, confidence, timestamp.

## Event database

SQLite at `data/events.db`, table `falls`:

| Column | Type | Meaning |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `track_id` | INTEGER | ByteTrack person id |
| `timestamp` | TEXT | ISO-8601 detection time |
| `confidence` | REAL | 0–1 fall confidence |
| `clip_path` | TEXT | Snapshot JPEG path (nullable) |

Query examples:
```sql
SELECT * FROM falls ORDER BY id DESC LIMIT 10;
SELECT date(timestamp), count(*) FROM falls GROUP BY date(timestamp);
```

## Tested

- `python main.py --help` → argparse UI correct
- Headless run on a synthetic 30-frame MP4 with `--max-frames 10 --output ...` →
  `frames=10 fall_events=0`, annotated MP4 written, no GUI opened, model
  auto-downloaded on first run
- `fall_logic` unit check: upright keypoints → no fall; horizontal torso →
  angle ≈ 84° > 45° threshold
- `db.py` round-trip: init → insert → count → delete verified

## Limitations

- Single-camera 2D only — no depth, so lying on a bed/sofa can look like a fall.
- Confirmation delay means very short clips (< `fall_confirm_seconds`) never alert.
- CPU inference ≈ real-time at 640px for 1–3 people; crowded scenes may lag —
  lower `imgsz` to 480/320 if needed.
- `track_id`s reset if a person leaves and re-enters the frame.
