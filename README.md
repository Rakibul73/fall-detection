# Offline Fall Detection Pipeline

Fully offline Detect → Track → Fall pipeline (Pose + Trajectory methods).
YOLOv8-Pose detects people, BoT-SORT/ByteTrack keeps stable IDs, then two
selectable fall detectors vote per track.

## Setup (Windows, venv recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

First run downloads the pose model (`yolov8n-pose.pt`, ~6 MB) automatically.
After that, everything runs 100% offline.

## Run

```powershell
# Webcam
.\.venv\Scripts\python.exe run.py --source 0

# Video file
.\.venv\Scripts\python.exe run.py --source data\videos\your_video.mp4

# RTSP
.\.venv\Scripts\python.exe run.py --source rtsp://user:pass@ip:554/stream

# CPU only / headless / pose-only
.\.venv\Scripts\python.exe run.py --source 0 --device cpu
.\.venv\Scripts\python.exe run.py --source data\videos\demo.mp4 --no-show
.\.venv\Scripts\python.exe run.py --source 0 --method pose
.\.venv\Scripts\python.exe run.py --source 0 --method trajectory
```

Press `q` to quit the preview window.

## Flags

| Flag | What it does | Example |
|------|--------------|---------|
| `--source` | Input: webcam index (`0`), video file path, or RTSP URL | `--source 0`, `--source data\videos\demo.mp4` |
| `--device` | Override `config.yaml` device: `cpu` forces CPU, `0` uses first GPU | `--device cpu` |
| `--method` | Which fall detector to run: `pose`, `trajectory`, or `both` (default from config) | `--method pose` |
| `--conf` | Override YOLO confidence threshold | `--conf 0.5` |
| `--zone` | Polygon zone as JSON; falls outside are ignored | `--zone "[[10,10],[500,10],[500,400],[10,400]]"` |
| `--output-dir` | Where annotated video + JSON go | `--output-dir output` |
| `--no-show` | Headless mode: no preview window (for servers / file processing) | `--no-show` |
| `--no-save` | Don't write video or JSON | `--no-save` |
| `--config` | Use a different config file | `--config config.yaml` |

```powershell
.\.venv\Scripts\python.exe run.py --help
```

## Config

Edit `config.yaml` to enable/disable pose or trajectory methods and tune thresholds:

- `pose.angle_threshold / aspect_threshold / drop_speed_threshold / consecutive_frames`
- `trajectory.window_size / min_frames / score_threshold`
- `zone.polygon`: optional `[[x,y], ...]` list; falls outside are ignored.

## Outputs

- Annotated video → `output/fall_out_<timestamp>.mp4`
- Fall events → `output/fall_events.json` (`frame`, `track_id`, `time`, `bbox`)

## Layout

```text
run.py / config.yaml / requirements.txt
core/detector.py   # YOLO + tracker wrapper
core/pipeline.py   # Detect → Track → Fall loop
utils/geometry.py  # pose fall rules
utils/trajectory.py# sliding-window bbox scorer
utils/visualizer.py# boxes, skeletons, zones
models/            # model cache (downloaded on first run)
data/videos/       # put demo videos here
output/            # results
```
