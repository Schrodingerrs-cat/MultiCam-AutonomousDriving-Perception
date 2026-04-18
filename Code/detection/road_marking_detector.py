"""
Road Marking Detector — ground arrow detection via drivable mask + Canny contours
+ adaptive thresholding fallback.
Phase 3 clean rewrite.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def _direction_from_approx(approx: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    pts = approx.reshape(-1, 2).astype(np.float64)
    cx = pts[:, 0].mean()
    rect_cx = x + w / 2.0
    offset = cx - rect_cx
    if offset < -0.15 * w:
        return "left"
    if offset > 0.15 * w:
        return "right"
    return "straight"


class RoadMarkingDetector:
    """Detect ground arrows using drivable mask + edge contours + adaptive threshold."""

    def __init__(self, qwen_model=None, qwen_processor=None):
        self._arrow_history: dict[int, deque] = {}
        self._frame_idx = 0
        self._qwen = qwen_model
        self._qwen_proc = qwen_processor

    def detect(self, frame_bgr: np.ndarray, drivable_mask: np.ndarray | None) -> list[dict]:
        self._frame_idx += 1

        detections = []

        # Step 1 — White paint detection (works WITHOUT drivable mask)
        white_dets = self._detect_white_paint(frame_bgr)
        detections.extend(white_dets)

        # Step 2 — Mask-based methods (if mask available)
        if drivable_mask is not None:
            if drivable_mask.max() <= 1:
                mask_u8 = (drivable_mask * 255).astype(np.uint8)
            else:
                mask_u8 = drivable_mask.astype(np.uint8)

            road_frame = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_u8)

            # Step 2a — Canny edge detection
            canny_dets = self._detect_canny(road_frame, mask_u8, frame_bgr)
            for cd in canny_dets:
                if not any(_iou(cd["bbox"], d["bbox"]) > 0.4 for d in detections):
                    detections.append(cd)

            # Step 2b — Adaptive threshold
            adaptive_dets = self._detect_adaptive(road_frame, mask_u8, frame_bgr)
            for ad in adaptive_dets:
                if not any(_iou(ad["bbox"], d["bbox"]) > 0.4 for d in detections):
                    detections.append(ad)
        else:
            # No mask: CLAHE+adaptive on road region as fallback
            clahe_dets = self._detect_clahe_road(frame_bgr)
            for cd in clahe_dets:
                if not any(_iou(cd["bbox"], d["bbox"]) > 0.4 for d in detections):
                    detections.append(cd)

        # Step 3 — temporal filter
        filtered = self._temporal_filter(detections)
        return filtered[:3]

    def _detect_white_paint(self, frame_bgr: np.ndarray) -> list[dict]:
        """Detect white road paint markings via HSV thresholding (no mask needed)."""
        H = frame_bgr.shape[0]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # White paint: low saturation, high value (tuned for daylight + dusk)
        white_mask = cv2.inRange(hsv, (0, 0, 140), (180, 60, 255))
        # Only road area (bottom 60%)
        white_mask[:int(H * 0.4), :] = 0
        # Cleanup
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (1500 < area < 50000):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / max(h, 1)
            if not (0.3 < ar < 5.0):
                continue
            hull = cv2.convexHull(cnt)
            solidity = area / max(cv2.contourArea(hull), 1)
            if not (0.15 < solidity < 0.90):
                continue
            # Must have arrow-like vertex count
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if not (4 <= len(approx) <= 25):
                continue

            bbox = [float(x), float(y), float(x + w), float(y + h)]
            direction = self._classify_direction(frame_bgr, bbox, approx, x, y, w, h)

            detections.append({
                "label": "ground_arrow",
                "sign_type": "ground_arrow",
                "direction": direction,
                "bbox": bbox,
                "confidence": 0.50,
                "camera": "front",
            })
        return detections

    def _detect_clahe_road(self, frame_bgr: np.ndarray) -> list[dict]:
        """CLAHE + adaptive threshold on road region (fallback when no mask)."""
        H = frame_bgr.shape[0]
        road = frame_bgr[int(H * 0.4):, :]
        y_off = int(H * 0.4)
        gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, -2
        )
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (2000 < area < 40000):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / max(h, 1)
            if not (0.3 < ar < 4.0):
                continue
            hull = cv2.convexHull(cnt)
            solidity = area / max(cv2.contourArea(hull), 1)
            if not (0.15 < solidity < 0.85):
                continue
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            bbox = [float(x), float(y + y_off), float(x + w), float(y + h + y_off)]
            direction = self._classify_direction(frame_bgr, bbox, approx, x, y, w, h)

            detections.append({
                "label": "ground_arrow",
                "sign_type": "ground_arrow",
                "direction": direction,
                "bbox": bbox,
                "confidence": 0.45,
                "camera": "front",
            })
        return detections

    def _detect_canny(self, road_frame, mask_u8, frame_bgr) -> list[dict]:
        """Canny-based contour detection for ground arrows."""
        gray = cv2.cvtColor(road_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        mean_brightness = float(gray[mask_u8 > 0].mean()) if mask_u8.sum() > 0 else 128
        if mean_brightness < 80:
            low_thresh, high_thresh = 25, 80
        else:
            low_thresh, high_thresh = 40, 120
        edges = cv2.Canny(blur, low_thresh, high_thresh)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (500 < area < 30000):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / max(h, 1)
            if not (0.2 < ar < 4.0):
                continue

            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            n_verts = len(approx)
            if not (5 <= n_verts <= 20):
                continue

            hull = cv2.convexHull(cnt)
            solidity = area / max(cv2.contourArea(hull), 1)
            if not (0.15 < solidity < 0.90):
                continue

            roi_mask = mask_u8[y:y + h, x:x + w]
            if roi_mask.size == 0:
                continue
            if roi_mask.mean() < 150:
                continue

            bbox = [float(x), float(y), float(x + w), float(y + h)]
            direction = self._classify_direction(frame_bgr, bbox, approx, x, y, w, h)

            detections.append({
                "label": "ground_arrow",
                "sign_type": "ground_arrow",
                "direction": direction,
                "bbox": bbox,
                "confidence": 0.55,
                "camera": "front",
            })

        return detections

    def _detect_adaptive(self, road_frame, mask_u8, frame_bgr) -> list[dict]:
        """Adaptive threshold: find regions brighter than local road surface."""
        gray = cv2.cvtColor(road_frame, cv2.COLOR_BGR2GRAY)

        # Only process where drivable mask is active
        if mask_u8.sum() == 0:
            return []

        # Adaptive threshold: detect locally bright regions (paint markings)
        adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, -15  # negative C = detect brighter-than-local
        )
        # Mask to drivable area only
        adapt = cv2.bitwise_and(adapt, mask_u8)

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, kernel, iterations=2)
        adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        H = frame_bgr.shape[0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (600 < area < 25000):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / max(h, 1)
            if not (0.3 < ar < 3.5):
                continue

            # Must be in lower 60% of frame (road level)
            if y < H * 0.4:
                continue

            hull = cv2.convexHull(cnt)
            solidity = area / max(cv2.contourArea(hull), 1)
            if not (0.2 < solidity < 0.85):
                continue

            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            bbox = [float(x), float(y), float(x + w), float(y + h)]
            direction = self._classify_direction(frame_bgr, bbox, approx, x, y, w, h)

            detections.append({
                "label": "ground_arrow",
                "sign_type": "ground_arrow",
                "direction": direction,
                "bbox": bbox,
                "confidence": 0.50,
                "camera": "front",
            })

        return detections

    def _temporal_filter(self, detections: list[dict]) -> list[dict]:
        cur_frame = self._frame_idx

        self._arrow_history[cur_frame] = detections

        old_keys = [k for k in self._arrow_history if cur_frame - k > 5]
        for k in old_keys:
            del self._arrow_history[k]

        if len(self._arrow_history) < 2:
            return detections  # First frame: pass through without filter

        confirmed = []
        for det in detections:
            match_count = 0
            for fid, prev_dets in self._arrow_history.items():
                if fid == cur_frame:
                    continue
                for prev in prev_dets:
                    if _iou(det["bbox"], prev["bbox"]) > 0.3:
                        match_count += 1
                        break
            if match_count >= 1:
                confirmed.append(det)

        return confirmed

    def _classify_direction(self, frame_bgr, bbox, approx, x, y, w, h):
        """Classify arrow direction: Qwen if available, else centroid fallback."""
        if self._qwen is not None and self._qwen_proc is not None:
            result = self._classify_direction_qwen(frame_bgr, bbox)
            if result is not None:
                return result
        return _direction_from_approx(approx, x, y, w, h)

    def _classify_direction_qwen(self, frame_bgr, bbox):
        """Use Qwen2.5-VL to classify arrow direction from crop."""
        try:
            import torch
            from PIL import Image

            x1, y1, x2, y2 = [int(v) for v in bbox]
            H, W = frame_bgr.shape[:2]
            pad = max(int((x2 - x1) * 0.3), 20)
            x1c = max(0, x1 - pad)
            y1c = max(0, y1 - pad)
            x2c = min(W, x2 + pad)
            y2c = min(H, y2 + pad)
            crop = frame_bgr[y1c:y2c, x1c:x2c]
            if crop.shape[0] < 20 or crop.shape[1] < 20:
                return None

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)

            prompt = ("What direction does this road arrow point? "
                      "Answer with exactly one word: straight, left, or right.")

            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ]}]

            text = self._qwen_proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                image_inputs = [pil_img]
                video_inputs = None

            inputs = self._qwen_proc(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(self._qwen.device)

            with torch.no_grad():
                generated = self._qwen.generate(**inputs, max_new_tokens=8)

            answer = self._qwen_proc.batch_decode(
                generated[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0].strip().lower()

            for d in ("left", "right", "straight"):
                if d in answer:
                    return d
            return None
        except Exception:
            return None
