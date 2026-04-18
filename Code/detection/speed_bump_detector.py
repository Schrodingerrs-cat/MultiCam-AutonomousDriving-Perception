"""
Speed Bump Detector — SBP-YOLO based detection.
Phase 3 clean rewrite. Replaces old depth-dip heuristic with trained model.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np


class SpeedBumpDetector:
    """
    Speed bump detector using SBP-YOLO pretrained model.
    Falls back to YOLOv11n if SBP-YOLO weights not available.
    """

    def __init__(self, weights_path: str = "weights/bump/sbp-yolo.pt"):
        import torch
        from ultralytics import YOLO

        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._disabled = False

        # Try multiple candidate paths for SBP-YOLO weights
        project_root = Path(__file__).resolve().parent.parent.parent
        candidates = [
            Path(weights_path),
            Path("Code_p3/weights/sbp-yolo.pt"),
            Path("Code_p3/weights/bump/sbp-yolo.pt"),
            project_root / "weights" / "bump" / "sbp-yolo.pt",
            project_root / "weights" / "sbp-yolo.pt",
            Path(__file__).resolve().parent.parent / "weights" / "bump" / "sbp-yolo.pt",
            Path(__file__).resolve().parent.parent / "weights" / "sbp-yolo.pt",
        ]
        weights = next((p for p in candidates if p.exists()), None)

        if weights is None:
            print("[SpeedBumpDetector] sbp-yolo.pt not found in any candidate path")
            print(f"  Searched: {[str(c) for c in candidates]}")
            print("[SpeedBumpDetector] Falling back to yolo11n.pt (COCO — speed bump detection will be limited)")
            # yolo11n won't detect speed bumps by class name, but we keep it
            # for potential future use / so the module doesn't crash

        try:
            if weights is not None:
                self._model = YOLO(str(weights))
                print(f"[SpeedBumpDetector] SBP-YOLO loaded from {weights}")
            else:
                self._model = YOLO("yolo11n.pt")
                print("[SpeedBumpDetector] Using yolo11n fallback")
            self._model.to(self._device)
            self._using_sbp = weights is not None
        except Exception as e:
            print(f"[SpeedBumpDetector] Failed to load: {e} — disabled")
            self._disabled = True

    def detect(
        self,
        frame_bgr: np.ndarray | None,
        drivable_mask: np.ndarray | None,
        K: np.ndarray,
        R_cam2ego: np.ndarray,
        t_cam2ego: np.ndarray,
        frame_id: int,
        depth_map: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Detect speed bumps using SBP-YOLO on the front camera frame.

        Returns list of dicts:
        {"type": "speed_bump", "bbox": [x1,y1,x2,y2],
         "position_3d": [x,y,z], "confidence": float}
        """
        if self._disabled or self._model is None:
            return []
        if frame_bgr is None:
            return []

        H, W = frame_bgr.shape[:2]

        try:
            results = self._model(
                frame_bgr,
                conf=0.35,
                iou=0.45,
                imgsz=640,
                verbose=False,
            )[0]
        except Exception as e:
            print(f"[SpeedBumpDetector] Inference error: {e}")
            return []

        if results.boxes is None or len(results.boxes) == 0:
            return []

        detections = []
        K_inv = np.linalg.inv(K) if K is not None else None

        for box in results.boxes:
            conf = float(box.conf.item())
            cls_id = int(box.cls.item())
            cls_name = results.names.get(cls_id, "").lower()

            # SBP-YOLO classes: speed_bump, pothole — only want speed bumps
            if self._using_sbp and "pothole" in cls_name:
                continue

            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            bbox = [x1, y1, x2, y2]

            # Filter: speed bump should be in lower 70% of frame (road level)
            cy = (y1 + y2) / 2
            if cy < H * 0.3:
                continue

            # Filter: bbox should not span more than 60% of frame width
            # (prevents crosswalk / road marking false positives)
            if (x2 - x1) > 0.6 * W:
                continue

            # Filter: aspect ratio — speed bumps are wider than tall
            bump_w = x2 - x1
            bump_h = y2 - y1
            if bump_h > 0 and bump_w / bump_h < 1.5:
                continue  # too square/tall to be a speed bump

            # Drivable mask check
            if drivable_mask is not None:
                u_c = max(0, min(W - 1, int((x1 + x2) / 2)))
                v_c = max(0, min(H - 1, int((y1 + y2) / 2)))
                mask_val = drivable_mask[v_c, u_c]
                if isinstance(mask_val, np.ndarray):
                    mask_val = mask_val.max()
                if mask_val < 0.1:
                    continue

            # 3D position via depth map
            position_3d = [0.0, 0.0, 0.0]
            if depth_map is not None and K_inv is not None:
                u_d = max(0, min(depth_map.shape[1] - 1, int((x1 + x2) / 2)))
                v_d = max(0, min(depth_map.shape[0] - 1, int((y1 + y2) / 2)))
                d = float(depth_map[v_d, u_d])
                if 1.0 < d < 30.0:
                    pt_cam = d * (K_inv @ np.array([u_d, v_d, 1.0]))
                    pt_ego = R_cam2ego @ pt_cam + t_cam2ego
                    position_3d = pt_ego.tolist()

            # Depth filter: only 3-25m from ego
            if position_3d != [0.0, 0.0, 0.0]:
                if not (3.0 <= position_3d[1] <= 25.0):
                    continue

            detections.append({
                "type": "speed_bump",
                "bbox": bbox,
                "position_3d": position_3d,
                "confidence": conf,
            })

        return detections

    def detect_from_ego_motion(self, frame_id: int, depth_map: np.ndarray | None) -> list[dict]:
        """
        Detect speed bumps via ego-vehicle vertical oscillation.

        Bump signature: camera pitches UP then DOWN within 2-5 frames.
        This causes mean road depth to DIP then RECOVER quickly.

        When detected, backtrack to estimate bump position from the
        depth map at the dip frame (the frame where bump was under car).
        """
        if depth_map is None:
            return []

        H, W = depth_map.shape
        # Sample bottom-center strip (road ahead, close range)
        road_strip = depth_map[int(H * 0.7):int(H * 0.85), int(W * 0.3):int(W * 0.7)]
        mean_depth = float(np.nanmean(road_strip))

        if not hasattr(self, '_depth_history'):
            self._depth_history: deque = deque(maxlen=20)
            self._last_bump_frame = -999

        self._depth_history.append((frame_id, mean_depth))

        if len(self._depth_history) < 6:
            return []

        # Dedup: skip if bump detected within last 45 frames
        if frame_id - self._last_bump_frame < 45:
            return []

        hist = list(self._depth_history)
        n = len(hist)

        for win_size in range(3, min(7, n)):
            window = hist[n - win_size:]
            depths_w = [d for _, d in window]
            frames_w = [f for f, _ in window]

            min_idx = int(np.argmin(depths_w))
            if min_idx == 0 or min_idx == len(depths_w) - 1:
                continue

            baseline_before = np.mean(depths_w[:min(2, min_idx)])
            baseline_after = np.mean(depths_w[min_idx + 1:])
            dip_val = depths_w[min_idx]

            drop_before = baseline_before - dip_val
            recovery = baseline_after - dip_val

            # Tighter oscillation thresholds: need clear V-shape
            if drop_before > 0.6 and recovery > 0.4:
                span = frames_w[-1] - frames_w[0]
                if not (2 <= span <= 6):
                    continue

                # Both sides must be roughly similar height (symmetric V)
                if abs(drop_before - recovery) > max(drop_before, recovery) * 0.6:
                    continue

                self._last_bump_frame = frame_id

                dip_depth = max(1.0, dip_val)
                frames_since_dip = frame_id - frames_w[min_idx]
                est_y = max(1.0, dip_depth - frames_since_dip * 0.3)

                amplitude = (drop_before + recovery) / 2.0
                conf = min(0.70, 0.30 + amplitude * 0.1)

                # Tight bbox: center-road strip, not full width
                # Speed bumps span ~lane width, not full frame
                bbox_x1 = float(W * 0.30)
                bbox_x2 = float(W * 0.70)
                bbox_y1 = float(H * 0.62)
                bbox_y2 = float(H * 0.72)

                return [{
                    "type": "speed_bump",
                    "bbox": [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                    "position_3d": [0.0, float(est_y), 0.0],
                    "confidence": conf,
                    "source": "ego_motion",
                    "oscillation_amplitude_m": round(amplitude, 2),
                    "dip_frame": frames_w[min_idx],
                }]

        return []

    def reset(self):
        if hasattr(self, '_depth_history'):
            self._depth_history.clear()
            self._last_bump_frame = -999
