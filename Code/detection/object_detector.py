"""
Object Detector — YOLO12 based.

Phase 1: Detects vehicles (generic), pedestrians, traffic lights, stop signs.
Phase 2: Adds vehicle sub-classification, orientation, additional objects.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ─── Detection result dataclass (plain dict for JSON serialisability) ─────────
def make_detection(
    bbox: list[float],
    label: str,
    confidence: float,
    subclass: Optional[str] = None,
    orientation: Optional[float] = None,
    position_3d: Optional[list[float]] = None,
    extra: Optional[dict] = None,
) -> dict:
    d = {
        "bbox": [float(x) for x in bbox],   # [x1, y1, x2, y2] in pixels
        "label": label,
        "confidence": float(confidence),
        "subclass": subclass,
        "orientation": orientation,          # degrees, None = unknown
        "position_3d": position_3d,          # [x, y, z] in meters relative to ego
    }
    if extra:
        d.update(extra)
    return d


class ObjectDetector:
    """
    Wraps YOLO12 for multi-class detection.

    Returns separate lists:
        vehicles, pedestrians, traffic_lights, road_signs, objects
    """

    def __init__(self, phase: int = 1, device: Optional[str] = None):
        from ultralytics import YOLO

        self.phase = phase
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[ObjectDetector] Loading {cfg.YOLO_MODEL} on {self.device}…")
        self.model = YOLO(cfg.YOLO_MODEL)
        self.model.to(self.device)

        # Phase 2 vehicle classifier (loaded lazily)
        self._vehicle_clf = None

        # Store last raw YOLO results for COCO fallback (small object detection)
        self.last_raw_results = None

    # ─── Public API ───────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run detection on a single BGR frame.

        Returns:
            {
              "vehicles":       [...],
              "pedestrians":    [...],
              "traffic_lights": [...],
              "road_signs":     [...],
              "objects":        [...],   # Phase 2 only
            }
        """
        results = self.model(
            frame,
            conf=cfg.YOLO_CONF,
            iou=cfg.YOLO_IOU,
            imgsz=cfg.YOLO_IMG_SIZE,
            verbose=False,
        )[0]

        # Store for COCO fallback small object detection
        self.last_raw_results = results

        vehicles, pedestrians, traffic_lights, road_signs, objects_ = [], [], [], [], []

        for box in results.boxes:
            cid  = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [x1, y1, x2, y2]
            cx   = (x1 + x2) / 2
            cy   = (y1 + y2) / 2

            # ── Vehicles ─────────────────────────────────────────────────────
            if cid in cfg.VEHICLE_IDS:
                label = self.model.names[cid]   # car / truck / bus / motorcycle / bicycle
                subclass  = None
                subclass = None
                if self.phase >= 2:
                    subclass = self._classify_vehicle(frame, bbox, label)
                det = make_detection(bbox, label, conf,
                                     subclass=subclass, orientation=None)
                vehicles.append(det)

            # ── Pedestrians ───────────────────────────────────────────────────
            elif cid in cfg.PEDESTRIAN_IDS:
                det = make_detection(bbox, "person", conf)
                pedestrians.append(det)

            # ── Traffic lights ────────────────────────────────────────────────
            elif cid in cfg.LIGHT_IDS:
                det = make_detection(bbox, "traffic_light", conf)
                traffic_lights.append(det)

            # ── Road signs ────────────────────────────────────────────────────
            elif cid in cfg.SIGN_IDS:
                det = make_detection(bbox, "stop_sign", conf)
                road_signs.append(det)

        return {
            "vehicles":       vehicles,
            "pedestrians":    pedestrians,
            "traffic_lights": traffic_lights,
            "road_signs":     road_signs,
            "objects":        objects_,
        }

    # ─── Phase 2: Vehicle Sub-Classification ──────────────────────────────────

    def _classify_vehicle(self, frame: np.ndarray, bbox: list, broad_label: str) -> str:
        """
        Classify vehicle into fine-grained subtype using aspect ratio heuristics
        and (optionally) a secondary CNN classifier.

        Heuristic rules work surprisingly well as a baseline:
          - bicycle / motorcycle → keep as-is
          - truck / bus          → keep as-is
          - car: use bounding box aspect ratio + height normalisation
            tall & narrow  → sedan or hatchback
            wide & low     → pickup
            large & square → suv
        """
        if broad_label in ("motorcycle", "bicycle", "truck", "bus"):
            return broad_label

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        ar = w / max(h, 1)      # aspect ratio (width / height)
        area = w * h

        # Very rough heuristics — will improve with real data
        if ar > 1.8:
            return "pickup"
        elif area > 80_000:
            return "suv"
        elif h > w * 0.85:
            return "hatchback"
        else:
            return "sedan"

