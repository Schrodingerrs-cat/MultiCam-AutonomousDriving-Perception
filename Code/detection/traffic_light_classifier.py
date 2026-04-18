"""
Traffic Light Classifier — position brightness + HSV verification.
Phase 3 clean rewrite. No SigLIP2. No Qwen.
"""

from __future__ import annotations

import cv2
import numpy as np


class TrafficLightClassifier:
    """Classify traffic light colour from a BGR crop using brightness bands + HSV."""

    def classify(self, frame_bgr: np.ndarray, bbox: list[float]) -> dict:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        H, W = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        crop = frame_bgr[y1:y2, x1:x2]

        if crop.shape[0] < 15 or crop.shape[1] < 8:
            return {"tl_color": "unknown", "tl_arrow": None, "confidence": 0.30}

        # --- Step 2: vertical position brightness ---
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        third = max(h // 3, 1)
        bands = [
            gray[:third, :],
            gray[third:2 * third, :],
            gray[2 * third:, :],
        ]
        brightness = [float(b.mean()) for b in bands]
        pos_idx = int(np.argmax(brightness))
        pos_map = {0: "red", 1: "yellow", 2: "green"}
        position_result = pos_map[pos_idx]

        # --- Step 3: HSV verification ---
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]

        red_mask = ((h_ch < 12) | (h_ch > 168)) & (s_ch > 80) & (v_ch > 80)
        yellow_mask = (h_ch >= 15) & (h_ch <= 35) & (s_ch > 100) & (v_ch > 100)
        green_mask = (h_ch >= 55) & (h_ch <= 95) & (s_ch > 80) & (v_ch > 80)

        counts = {
            "red": int(red_mask.sum()),
            "yellow": int(yellow_mask.sum()),
            "green": int(green_mask.sum()),
        }
        max_count = max(counts.values())
        if max_count > 20:
            hsv_result = max(counts, key=counts.get)
        else:
            hsv_result = "unknown"

        # --- Step 4: decision ---
        if hsv_result == position_result:
            confidence = 0.85
        elif hsv_result == "unknown":
            confidence = 0.55
        else:
            confidence = 0.50

        return {
            "tl_color": position_result,
            "tl_arrow": None,
            "confidence": confidence,
        }

    @staticmethod
    def is_in_fov(bbox: list[float], frame_width: int) -> bool:
        cx = (bbox[0] + bbox[2]) / 2.0
        return abs(cx - frame_width / 2.0) < frame_width * 0.53
