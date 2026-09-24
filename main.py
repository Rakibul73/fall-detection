import argparse
import cv2
import yaml
import time
import os
from datetime import datetime
from ultralytics import YOLO

from fall_logic import FallDetector
from db import init_db, log_fall
from telegram_alert import send_telegram_alert


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8-pose fall detection (webcam / video file / RTSP).")
    p.add_argument("--source", default=None,
                   help="Override config source: webcam index (e.g. 0), video file path, or RTSP URL.")
    p.add_argument("--headless", action="store_true",
                   help="No GUI window (required on servers / SSH / Docker). Also auto-enabled for video files unless --show is passed.")
    p.add_argument("--show", action="store_true",
                   help="Force showing the GUI window even for video files.")
    p.add_argument("--output", default=None,
                   help="Save annotated output video here (e.g. data/annotated.mp4).")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames (0 = run until stream ends / 'q' pressed). Useful for testing.")
    p.add_argument("--config", default="config.yaml", help="Path to config file.")
    return p.parse_args()


def is_video_file(source):
    if isinstance(source, (int, float)):
        return False
    s = str(source)
    if s.isdigit():
        return False
    if s.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return False
    return os.path.isfile(s)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    init_db()

    os.makedirs(cfg["clips_dir"], exist_ok=True)

    source = args.source if args.source is not None else cfg["source"]
    # Accept webcam index passed as string ("0") from CLI
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    file_mode = is_video_file(source)
    # Headless when: flag passed, or video file without --show, or no display available
    headless = args.headless or (file_mode and not args.show)
    if not headless and os.name != "nt" and not os.environ.get("DISPLAY"):
        print("[info] No DISPLAY detected, switching to headless mode.")
        headless = True

    model = YOLO(cfg["model"])
    detector = FallDetector(
        history_len=cfg["history_len"],
        angle_threshold_deg=cfg["angle_threshold_deg"],
        confirm_seconds=cfg["fall_confirm_seconds"],
    )

    results_gen = model.track(
        source=source,
        conf=cfg["conf_threshold"],
        imgsz=cfg["imgsz"],
        device=cfg["device"],
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    )

    frame_count = 0
    fall_count = 0
    writer = None

    print(f"[info] source={source} headless={headless} output={args.output}")

    for result in results_gen:
        frame = result.orig_img.copy()
        frame_count += 1

        if result.boxes is None or result.boxes.id is None:
            annotated = frame
        else:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            keypoints_all = None
            if result.keypoints is not None:
                keypoints_all = result.keypoints.data.cpu().numpy()

            for i, (box, tid) in enumerate(zip(boxes, track_ids)):
                kp = keypoints_all[i] if keypoints_all is not None else None
                is_fall, conf = detector.update(tid, tuple(box), kp)

                x1, y1, x2, y2 = box.astype(int)
                color = (0, 0, 255) if is_fall else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"id{tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if is_fall:
                    cv2.putText(frame, f"FALL {conf:.2f}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if is_fall:
                    fall_count += 1
                    timestamp = datetime.now().isoformat()
                    clip_path = None
                    if cfg["save_clips"]:
                        clip_path = os.path.join(
                            cfg["clips_dir"], f"fall_{tid}_{int(time.time())}.jpg"
                        )
                        cv2.imwrite(clip_path, frame)

                    log_fall(int(tid), timestamp, float(conf), clip_path)
                    print(f"[ALERT] Fall detected: track_id={tid} conf={conf:.2f}")

                    if cfg["telegram"]["enabled"]:
                        send_telegram_alert(
                            cfg["telegram"]["bot_token"],
                            cfg["telegram"]["chat_id"],
                            f"Fall detected (track {tid}, confidence {conf:.2f}) at {timestamp}",
                            image_path=clip_path,
                        )
            annotated = frame

        if args.output:
            if writer is None:
                h, w = annotated.shape[:2]
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                writer = cv2.VideoWriter(
                    args.output, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
                )
                if not writer.isOpened():
                    print(f"[warn] Could not open VideoWriter for {args.output}")
                    writer = None
            if writer is not None:
                writer.write(annotated)

        if not headless:
            cv2.imshow("Fall Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        elif frame_count % 30 == 0:
            print(f"[info] processed {frame_count} frames, falls={fall_count}")

        if args.max_frames and frame_count >= args.max_frames:
            print(f"[info] Reached --max-frames={args.max_frames}, stopping.")
            break

    if writer is not None:
        writer.release()
        print(f"[info] Saved annotated video to {args.output}")
    if not headless:
        cv2.destroyAllWindows()
    print(f"[done] frames={frame_count} fall_events={fall_count}")


if __name__ == "__main__":
    main()
