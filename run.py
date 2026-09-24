#!/usr/bin/env python3
"""Offline Fall Detection Pipeline — entry point.

Examples:
    python run.py --source 0
    python run.py --source data/videos/demo.mp4 --device cpu --no-show
    python run.py --source rtsp://user:pass@ip:554/stream --method pose
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

# Ensure project root is importable when invoked as `python run.py`.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline import FallDetectionPipeline


def parse_args():
    p = argparse.ArgumentParser(description="Offline Fall Detection Pipeline")
    p.add_argument("--source", type=str, default="0",
                   help="webcam index, video path, or RTSP URL")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--device", type=str, default=None,
                   help="override config device (e.g. cpu, 0)")
    p.add_argument("--method", type=str, default=None,
                   choices=["pose", "trajectory", "both"],
                   help="fall detection method(s) to use")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--zone", type=str, default=None,
                   help='polygon zone as JSON, e.g. "[[10,10],[500,10],[500,400],[10,400]]"')
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def coerce_device(value):
    if value is None:
        return None
    v = str(value).strip()
    if v.lower() in ("cpu", "none", "null", "auto", ""):
        return "cpu" if v.lower() == "cpu" else None
    if v.isdigit():
        return int(v)
    return v


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device is not None:
        cfg["device"] = coerce_device(args.device)
    if args.conf is not None:
        cfg["conf"] = float(args.conf)
    if args.method == "pose":
        cfg["fall"] = {"pose": True, "trajectory": False}
    elif args.method == "trajectory":
        cfg["fall"] = {"pose": False, "trajectory": True}
    elif args.method == "both":
        cfg["fall"] = {"pose": True, "trajectory": True}
    if args.zone is not None:
        cfg.setdefault("zone", {})["polygon"] = json.loads(args.zone)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.no_show:
        cfg["show"] = False
    if args.no_save:
        cfg["save_video"] = False
        cfg["save_json"] = False

    pipe = FallDetectionPipeline(cfg)
    pipe.process(args.source, output_dir=cfg.get("output_dir", "output"))


if __name__ == "__main__":
    main()
