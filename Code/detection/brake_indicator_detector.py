"""
Brake Light + Turn Indicator Detector — HSV analysis on vehicle rear crop.
Phase 3 clean rewrite.
"""

from __future__ import annotations

from collections import Counter, deque

import cv2
import numpy as np


class BrakeLightDetector:
    """Detect brake lights and turn indicators from rear crop HSV analysis."""

    def __init__(self):
        self._brake_ema: dict[int, float] = {}
        self._indicator_buf: dict[int, deque] = {}

    def detect(self, frame_bgr: np.ndarray, vehicle_det: dict) -> dict:
        default = {"brake_light": False, "indicator": None}

        cam = vehicle_det.get("camera", "front")
        if cam == "front":
            # Check orientation: vehicle must be roughly facing away (rear visible)
            ori = vehicle_det.get("orientation_deg", 0.0)
            if not (-60.0 <= ori <= 60.0):
                return default
        # Caller (run_detection.py) restricts to front-cam vehicles only.
        # Front cam sees rear of vehicles we're following.

        x1, y1, x2, y2 = [int(v) for v in vehicle_det["bbox"]]
        H, W = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        # For back camera, use full bbox (we see front of vehicles ahead)
        if cam == "back":
            rear = frame_bgr[y1:y2, x1:x2]
        else:
            rear_top = int(y1 + 0.50 * (y2 - y1))
            rear = frame_bgr[rear_top:y2, x1:x2]

        if rear.shape[0] < 6 or rear.shape[1] < 10:
            return default

        hsv = cv2.cvtColor(rear, cv2.COLOR_BGR2HSV)
        total_px = rear.shape[0] * rear.shape[1]
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]

        # LED brake lights: wider range for hue, lower sat/val thresholds
        red_mask = ((h_ch < 15) | (h_ch > 160)) & (s_ch > 60) & (v_ch > 60)
        # Also catch bright white-red (overexposed LEDs)
        bright_mask = (v_ch > 200) & (s_ch < 80)
        red_mask = red_mask | bright_mask
        orange_mask = (h_ch >= 8) & (h_ch <= 30) & (s_ch > 80) & (v_ch > 60)

        # Brake light — check symmetry: brake lights are on BOTH sides
        w = rear.shape[1]
        red_ratio = float(red_mask.sum()) / max(total_px, 1)
        left_red = int(red_mask[:, :w // 2].sum())
        right_red = int(red_mask[:, w // 2:].sum())

        # Suppress brake detection if red is heavily one-sided (likely indicator)
        red_symmetric = True
        if left_red > 0 or right_red > 0:
            ratio_lr = min(left_red, right_red) / max(left_red, right_red, 1)
            if ratio_lr < 0.08:  # one side has <8% of the other → one-sided
                red_symmetric = False

        tid = vehicle_det.get("track_id", id(vehicle_det))
        if tid not in self._brake_ema:
            self._brake_ema[tid] = 0.0
        # Faster response: higher alpha catches transient braking
        self._brake_ema[tid] = 0.6 * red_ratio + 0.4 * self._brake_ema[tid]
        brake_light = self._brake_ema[tid] > 0.02 and red_symmetric

        # Indicator — use orange mask, check left/right asymmetry
        left_orange = int(orange_mask[:, :w // 2].sum())
        right_orange = int(orange_mask[:, w // 2:].sum())
        min_px = 20

        if left_orange > 3 * right_orange and left_orange > min_px:
            raw_indicator = "left"
        elif right_orange > 3 * left_orange and right_orange > min_px:
            raw_indicator = "right"
        else:
            raw_indicator = "none"

        if tid not in self._indicator_buf:
            self._indicator_buf[tid] = deque(maxlen=5)
        self._indicator_buf[tid].append(raw_indicator)
        counts = Counter(self._indicator_buf[tid])
        best = counts.most_common(1)[0]
        indicator = best[0] if best[0] != "none" and best[1] >= 3 else None

        return {"brake_light": brake_light, "indicator": indicator}

    def reset(self):
        self._brake_ema.clear()
        self._indicator_buf.clear()

    def prune(self, active_ids: set[int]):
        for tid in list(self._brake_ema.keys()):
            if tid not in active_ids:
                del self._brake_ema[tid]
                self._indicator_buf.pop(tid, None)
