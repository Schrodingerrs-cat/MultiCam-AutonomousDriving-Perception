"""
Small Object Detector — Phase 3.

COCO fallback for small road objects: when YOLOE misses something,
YOLO12x COCO detections are reclassified by appearance (color + shape).

Primary detection is handled by YOLOE full-frame in run_detection.py.
This class only provides the COCO appearance-based fallback.
"""

from __future__ import annotations

import cv2
import numpy as np


class SmallObjectDetector:
    """COCO appearance-based fallback for small road objects."""

    def detect_with_coco_fallback(self, frame: np.ndarray,
                                    yolo_results,
                                    existing_bboxes: list[list[float]]) -> list[dict]:
        """Reclassify YOLO12x COCO detections as small road objects by appearance.

        Args:
            frame: BGR frame (for crop-based color/shape classification)
            yolo_results: pre-computed ultralytics Results from YOLO12x Step 1
            existing_bboxes: bboxes to skip (vehicles, peds, YOLOE objects, etc.)
        """
        if yolo_results is None or yolo_results.boxes is None:
            return []

        H, W = frame.shape[:2]
        frame_area = H * W
        detections = []

        skip_classes = {"car", "truck", "bus", "person", "bicycle",
                        "motorcycle", "traffic light", "stop sign"}

        for box in yolo_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w_box = x2 - x1
            h_box = y2 - y1
            area = w_box * h_box
            conf = float(box.conf.item())
            cls_name = yolo_results.names.get(int(box.cls.item()), "")

            if area > frame_area * 0.05 or area < 400:
                continue
            if cls_name in skip_classes:
                continue

            ar = w_box / max(h_box, 1)
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            label = _classify_by_appearance(crop, ar)
            if label is None:
                continue

            if _overlaps_any([x1, y1, x2, y2], existing_bboxes, 0.5):
                continue

            detections.append({
                "label": label,
                "type": label,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": conf * 0.7,
                "position_3d": [0.0, 0.0, 0.0],
            })

        return detections


def _classify_by_appearance(crop, aspect_ratio):
    """Classify small object by color and shape."""
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Orange → traffic cone
    orange_mask = cv2.inRange(hsv, (5, 150, 150), (25, 255, 255))
    if orange_mask.mean() > 15:
        return "traffic_cone"

    # Tall thin → traffic pole
    if aspect_ratio < 0.4:
        return "traffic_pole"

    # Dark + roughly square → dustbin
    v_mean = hsv[:, :, 2].mean()
    if v_mean < 80 and 0.4 < aspect_ratio < 2.0:
        return "dustbin"

    # Cylinder shape
    if 0.5 < aspect_ratio < 1.5:
        return "traffic_cylinder"

    return None


def _overlaps_any(bbox, bboxes, iou_thresh=0.3) -> bool:
    x1, y1, x2, y2 = bbox[:4]
    for bb in bboxes:
        bx1, by1, bx2, by2 = bb[:4]
        ix1 = max(x1, bx1)
        iy1 = max(y1, by1)
        ix2 = min(x2, bx2)
        iy2 = min(y2, by2)
        if ix1 >= ix2 or iy1 >= iy2:
            continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        area = (x2 - x1) * (y2 - y1)
        if inter / max(area, 1) > iou_thresh:
            return True
    return False
