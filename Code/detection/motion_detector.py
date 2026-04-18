"""
Motion Detector — RAFT optical flow + Sampson distance for parked/moving
classification and vehicle moving direction.
Phase 3.

Uses torchvision's raft_small for dense optical flow between consecutive
frames. Sampson distance (epipolar geometry) separates ego-induced flow
from independently moving objects.

Pipeline integration:
  - Called at Step 11 (after VehicleOrientationEstimator, before BrakeLightDetector)
  - Overrides is_moving when flow evidence is strong
  - Computes moving_direction_deg for moving vehicles
  - Front camera only (most reliable, saves compute)
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np


# ── Thresholds ───────────────────────────────────────────────────────────────

FLOW_MOVING_THRESH = 1.5     # avg pixel displacement to consider moving
FLOW_PARKED_THRESH = 0.4     # below this → definitely parked
FLOW_EMA_ALPHA = 0.35        # smoothing for per-track flow magnitude
RAFT_RESIZE_H = 520          # downscale for speed (divisible by 8)
RAFT_RESIZE_W = 960           # downscale for speed (divisible by 8)
RAFT_NUM_UPDATES = 6         # fewer iterations = faster on CPU (default 12)

SAMPSON_MOVING_THRESH = 1.5  # Sampson distance above this → independent motion (was 2.0)
SAMPSON_PARKED_THRESH = 0.5  # below this → consistent with ego motion (parked)
SAMPSON_SAMPLE_POINTS = 200  # number of points to sample for fundamental matrix
DIRECTION_EMA_ALPHA = 0.3    # smoothing for per-track moving direction


class MotionDetector:
    """Dense optical flow motion detector using torchvision RAFT + Sampson distance."""

    def __init__(self, cameras: list[str] | None = None):
        self._cameras = cameras or ["front"]
        self._model = None
        self._device = None
        self._loaded = False
        self._prev_tensors: dict[str, "torch.Tensor"] = {}
        self._prev_frames: dict[str, np.ndarray] = {}  # for keypoint matching
        self._flow_ema: dict[int, float] = {}
        self._sampson_ema: dict[int, float] = {}
        self._direction_ema: dict[int, float] = {}  # track_id → smoothed direction (deg)
        self._fundamental_cache: dict[str, np.ndarray | None] = {}  # cam → F matrix
        self._traffic_light_red = False  # set externally per frame

    # ── Lazy model loading ────────────────────────────────────────────────

    def _load(self) -> bool:
        if self._loaded:
            return self._model is not None
        self._loaded = True

        try:
            import torch
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = raft_small(weights=Raft_Small_Weights.DEFAULT)
            self._model.to(self._device)
            self._model.eval()
            print(f"[MotionDetector] RAFT-Small loaded on {self._device}")
        except Exception as e:
            print(f"[MotionDetector] RAFT load failed: {e} — flow disabled")
            self._model = None

        return self._model is not None

    def is_available(self) -> bool:
        return self._load()

    def set_traffic_light_red(self, is_red: bool):
        """Call before update_vehicles each frame with current TL state."""
        self._traffic_light_red = is_red

    # ── Frame preprocessing ───────────────────────────────────────────────

    def _preprocess(self, frame_bgr: np.ndarray) -> "torch.Tensor":
        """BGR uint8 → float32 tensor [1, 3, H, W] normalized to [0, 1]."""
        import torch

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (RAFT_RESIZE_W, RAFT_RESIZE_H),
                         interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return tensor.to(self._device)

    # ── Optical flow computation ──────────────────────────────────────────

    def compute_flow(self, cam_id: str, frame_bgr: np.ndarray) -> np.ndarray | None:
        """
        Compute dense optical flow for camera.

        Returns:
            (H_orig, W_orig, 2) float32 flow in original frame coords,
            or None if no previous frame / model unavailable.
        """
        if not self._load():
            return None

        import torch

        curr_tensor = self._preprocess(frame_bgr)
        prev_tensor = self._prev_tensors.get(cam_id)
        self._prev_tensors[cam_id] = curr_tensor

        if prev_tensor is None:
            return None

        with torch.no_grad():
            flow_list = self._model(prev_tensor, curr_tensor,
                                    num_flow_updates=RAFT_NUM_UPDATES)

        # Last element = most refined flow estimate
        flow = flow_list[-1].squeeze(0).cpu().numpy()  # (2, H_small, W_small)
        flow = flow.transpose(1, 2, 0)  # (H_small, W_small, 2)

        # Scale flow back to original resolution
        H_orig, W_orig = frame_bgr.shape[:2]
        scale_x = W_orig / RAFT_RESIZE_W
        scale_y = H_orig / RAFT_RESIZE_H
        flow_full = cv2.resize(flow, (W_orig, H_orig),
                               interpolation=cv2.INTER_LINEAR)
        flow_full[:, :, 0] *= scale_x
        flow_full[:, :, 1] *= scale_y

        return flow_full

    # ── Fundamental matrix estimation ─────────────────────────────────────

    def _estimate_fundamental(self, cam_id: str, flow: np.ndarray,
                               frame_bgr: np.ndarray) -> np.ndarray | None:
        """
        Estimate fundamental matrix from background flow correspondences.

        Uses sparse sampling of flow vectors across frame, fits F via RANSAC.
        This captures the ego-motion model — points consistent with F are
        static background; points with high Sampson distance are independently
        moving.
        """
        H, W = flow.shape[:2]

        # Sample points on a grid (avoid edges)
        margin = 20
        n_pts = SAMPSON_SAMPLE_POINTS
        ys = np.random.randint(margin, H - margin, n_pts)
        xs = np.random.randint(margin, W - margin, n_pts)

        # Source points and their flow-displaced destinations
        pts1 = np.column_stack([xs, ys]).astype(np.float64)
        dx = flow[ys, xs, 0]
        dy = flow[ys, xs, 1]
        pts2 = pts1 + np.column_stack([dx, dy])

        # Filter out near-zero flow (uninformative)
        mag = np.sqrt(dx**2 + dy**2)
        valid = mag > 0.5
        if valid.sum() < 16:
            return None

        pts1 = pts1[valid]
        pts2 = pts2[valid]

        # RANSAC fundamental matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                                          ransacReprojThreshold=1.5,
                                          confidence=0.99)
        if F is None or F.shape != (3, 3):
            return None

        self._fundamental_cache[cam_id] = F
        return F

    # ── Sampson distance computation ──────────────────────────────────────

    def _compute_sampson_distance(self, F: np.ndarray,
                                   flow: np.ndarray,
                                   bbox: list[float]) -> float:
        """
        Compute median Sampson distance for points within a bounding box.

        Sampson distance measures how well a point correspondence fits the
        epipolar geometry (fundamental matrix). High Sampson distance =
        the point's motion is inconsistent with ego motion = independently
        moving object.

        d_S = (x2^T F x1)^2 / ( (Fx1)_1^2 + (Fx1)_2^2 + (F^Tx2)_1^2 + (F^Tx2)_2^2 )
        """
        H, W = flow.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if x2 - x1 < 5 or y2 - y1 < 5:
            return 0.0

        # Sample points within bbox
        n_sample = min(50, (x2 - x1) * (y2 - y1) // 4)
        n_sample = max(n_sample, 8)
        bys = np.random.randint(y1, y2, n_sample)
        bxs = np.random.randint(x1, x2, n_sample)

        pts1 = np.column_stack([bxs, bys]).astype(np.float64)
        dx = flow[bys, bxs, 0]
        dy = flow[bys, bxs, 1]
        pts2 = pts1 + np.column_stack([dx, dy])

        # Compute Sampson distance for each point pair
        # x1_h, x2_h: homogeneous coordinates [x, y, 1]
        ones = np.ones((len(pts1), 1), dtype=np.float64)
        x1_h = np.hstack([pts1, ones])  # (N, 3)
        x2_h = np.hstack([pts2, ones])  # (N, 3)

        # Fx1 and F^T x2
        Fx1 = (F @ x1_h.T).T       # (N, 3)
        Ftx2 = (F.T @ x2_h.T).T    # (N, 3)

        # Numerator: (x2^T F x1)^2
        num = np.sum(x2_h * Fx1, axis=1) ** 2

        # Denominator: sum of squared first two components
        denom = (Fx1[:, 0]**2 + Fx1[:, 1]**2 +
                 Ftx2[:, 0]**2 + Ftx2[:, 1]**2)
        denom = np.maximum(denom, 1e-10)

        sampson = num / denom
        return float(np.median(sampson))

    # ── Per-vehicle motion extraction ─────────────────────────────────────

    def get_bbox_motion(self, flow: np.ndarray, bbox: list[float],
                        F: np.ndarray | None = None) -> dict:
        """
        Extract motion magnitude, direction, and Sampson distance from flow.

        Returns:
            dict with flow_magnitude, flow_direction (degrees),
            sampson_distance.
        """
        H, W = flow.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if x2 <= x1 or y2 <= y1:
            return {"flow_magnitude": 0.0, "flow_direction": 0.0,
                    "sampson_distance": 0.0}

        roi = flow[y1:y2, x1:x2]  # (h, w, 2)
        mag = np.sqrt(roi[:, :, 0] ** 2 + roi[:, :, 1] ** 2)

        # Use median to be robust to background flow at bbox edges
        avg_mag = float(np.median(mag))
        avg_dx = float(np.median(roi[:, :, 0]))
        avg_dy = float(np.median(roi[:, :, 1]))

        # Flow direction: angle in image coords (0=right, 90=down)
        # Convert to ego-relative: image-down ≈ forward in ego frame
        direction = float(np.degrees(np.arctan2(avg_dx, avg_dy)))

        # Sampson distance
        sampson = 0.0
        if F is not None:
            sampson = self._compute_sampson_distance(F, flow, bbox)

        return {"flow_magnitude": avg_mag, "flow_direction": direction,
                "sampson_distance": sampson}

    # ── Update vehicles with flow-based motion ────────────────────────────

    def update_vehicles(
        self,
        vehicles: list[dict],
        frames: dict[str, np.ndarray | None],
    ) -> dict[str, np.ndarray | None]:
        """
        Compute optical flow for configured cameras and update
        is_moving + moving_direction on vehicles using flow + Sampson distance.

        Args:
            vehicles: list of vehicle dicts (modified in place)
            frames: cam_id → BGR frame

        Returns:
            dict of cam_id → flow array (for potential reuse by other modules)
        """
        if not self._load():
            return {}

        # Compute flow and fundamental matrix for each configured camera
        flows: dict[str, np.ndarray | None] = {}
        fundamentals: dict[str, np.ndarray | None] = {}
        for cam_id in self._cameras:
            frame = frames.get(cam_id)
            if frame is None:
                flows[cam_id] = None
                fundamentals[cam_id] = None
                continue
            flow = self.compute_flow(cam_id, frame)
            flows[cam_id] = flow
            if flow is not None:
                fundamentals[cam_id] = self._estimate_fundamental(
                    cam_id, flow, frame)
            else:
                fundamentals[cam_id] = None

        # Update each vehicle
        for v in vehicles:
            cam_id = v.get("camera", "front")
            flow = flows.get(cam_id)
            if flow is None:
                continue

            F = fundamentals.get(cam_id)
            motion = self.get_bbox_motion(flow, v["bbox"], F=F)
            raw_mag = motion["flow_magnitude"]
            raw_sampson = motion["sampson_distance"]
            raw_direction = motion["flow_direction"]

            # EMA smoothing per track
            tid = v.get("track_id")
            if tid is not None:
                # Flow magnitude EMA
                if tid not in self._flow_ema:
                    self._flow_ema[tid] = raw_mag
                self._flow_ema[tid] = (
                    FLOW_EMA_ALPHA * raw_mag
                    + (1.0 - FLOW_EMA_ALPHA) * self._flow_ema[tid]
                )
                smoothed = self._flow_ema[tid]

                # Sampson distance EMA
                if tid not in self._sampson_ema:
                    self._sampson_ema[tid] = raw_sampson
                self._sampson_ema[tid] = (
                    FLOW_EMA_ALPHA * raw_sampson
                    + (1.0 - FLOW_EMA_ALPHA) * self._sampson_ema[tid]
                )
                smoothed_sampson = self._sampson_ema[tid]

                # Direction EMA (angle smoothing)
                if tid not in self._direction_ema:
                    self._direction_ema[tid] = raw_direction
                else:
                    # Angle difference handling (wrap-around)
                    diff = raw_direction - self._direction_ema[tid]
                    diff = ((diff + 180) % 360) - 180
                    self._direction_ema[tid] += DIRECTION_EMA_ALPHA * diff
                smoothed_dir = self._direction_ema[tid]
            else:
                smoothed = raw_mag
                smoothed_sampson = raw_sampson
                smoothed_dir = raw_direction

            # Fusion: use both flow magnitude AND Sampson distance
            vel_moving = v.get("is_moving", False)

            # Scale flow thresholds by bbox size — far vehicles (small bbox)
            # have inherently lower pixel flow even when moving.
            bbox = v.get("bbox", [0, 0, 0, 0])
            bbox_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1.0)
            is_far = bbox_area < 8000  # roughly < 100x80 px

            # ── Traffic light awareness ──
            # If a red traffic light is detected and this vehicle is in the
            # front camera ahead of us, it's likely stopped at the light,
            # NOT parked. Don't override velocity-based moving → parked.
            at_red_light = (
                self._traffic_light_red
                and cam_id == "front"
                and v.get("position_3d", [0, 0, 0])[1] > 2.0  # ahead of ego
                and v.get("position_3d", [0, 0, 0])[1] < 40.0  # within range
            )

            # Sampson-aware classification:
            # High Sampson = independently moving (not ego-induced flow)
            if smoothed_sampson > SAMPSON_MOVING_THRESH and smoothed > FLOW_PARKED_THRESH:
                is_moving = True
            elif smoothed > FLOW_MOVING_THRESH:
                is_moving = True
            elif smoothed < FLOW_PARKED_THRESH and smoothed_sampson < SAMPSON_PARKED_THRESH:
                # Both flow and Sampson say stationary
                if at_red_light:
                    # Vehicle at a red light — keep velocity-based decision
                    is_moving = vel_moving
                elif is_far and vel_moving:
                    # Far vehicle with low pixel flow — trust velocity
                    is_moving = True
                elif vel_moving:
                    # Velocity says moving but flow says parked —
                    # only override if flow is very confidently parked
                    if smoothed < 0.15 and smoothed_sampson < 0.1:
                        is_moving = False
                    else:
                        is_moving = vel_moving
                else:
                    is_moving = False
            else:
                is_moving = vel_moving

            v["is_moving"] = is_moving
            v["flow_magnitude"] = round(smoothed, 2)
            v["sampson_distance"] = round(smoothed_sampson, 3)

            # Moving direction (only meaningful for moving vehicles)
            if is_moving and smoothed > FLOW_PARKED_THRESH:
                v["moving_direction_deg"] = round(smoothed_dir, 1)
            else:
                v["moving_direction_deg"] = None

        return flows

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def reset(self):
        """Call between sequences."""
        self._prev_tensors.clear()
        self._prev_frames.clear()
        self._flow_ema.clear()
        self._sampson_ema.clear()
        self._direction_ema.clear()
        self._fundamental_cache.clear()
        self._traffic_light_red = False

    def prune(self, active_ids: set[int]):
        for tid in list(self._flow_ema):
            if tid not in active_ids:
                del self._flow_ema[tid]
                self._sampson_ema.pop(tid, None)
                self._direction_ema.pop(tid, None)
