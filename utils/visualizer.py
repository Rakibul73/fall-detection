"""Drawing helpers: skeletons, boxes, zones."""
import cv2
import numpy as np

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_skeleton(frame, kpts, color=(0, 255, 0), thickness=2):
    if kpts is None:
        return
    kpts = np.asarray(kpts)
    for a, b in SKELETON:
        if a < len(kpts) and b < len(kpts):
            pa, pb = kpts[a], kpts[b]
            if pa[0] > 0 and pb[0] > 0:
                cv2.line(frame, tuple(np.asarray(pa[:2]).astype(int)),
                         tuple(np.asarray(pb[:2]).astype(int)), color, thickness)
    for p in kpts:
        if p[0] > 0:
            cv2.circle(frame, tuple(np.asarray(p[:2]).astype(int)), 3, color, -1)


def draw_box_and_label(frame, bbox, track_id, is_fall, conf=None):
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 0, 255) if is_fall else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{track_id}"
    if is_fall:
        label += " FALL!"
    if conf is not None:
        try:
            label += f" {float(conf):.2f}"
        except (TypeError, ValueError):
            pass
    cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_zone(frame, polygon, color=(255, 0, 0), thickness=2):
    """Draw optional polygon zone. polygon: Nx2 array-like or None."""
    if polygon is None:
        return
    pts = np.asarray(polygon, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return
    cv2.polylines(frame, [pts.reshape((-1, 1, 2))], True, color, thickness)
