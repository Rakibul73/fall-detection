"""YOLO-Pose + tracker wrapper."""
from ultralytics import YOLO
import numpy as np


class PersonPoseTracker:
    def __init__(self, model_path="yolov8n-pose.pt", device=None, conf=0.4, iou=0.45,
                 tracker="botsort.yaml", imgsz=640):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.tracker = tracker
        self.imgsz = imgsz

    def track(self, frame):
        kwargs = dict(
            persist=True,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker,
            verbose=False,
            imgsz=self.imgsz,
        )
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.track(frame, **kwargs)
        r = results[0]
        boxes, ids, kpts, confs = [], [], [], []
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()
            confs = r.boxes.conf.cpu().numpy()
            if r.keypoints is not None and r.keypoints.xy is not None:
                kpts = r.keypoints.xy.cpu().numpy()
            else:
                kpts = [None] * len(ids)
        return np.asarray(boxes), list(ids), list(kpts), list(confs)
