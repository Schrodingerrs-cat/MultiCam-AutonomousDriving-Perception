"""
Object Reconstructor — 3D from mask + depth + intrinsics
=========================================================

For each YOLO detection:
  1. Get corresponding mask (from Mask2Former) or fallback to bbox region
  2. Apply morphological erosion (3×3) to remove boundary noise
  3. Deterministic stride sampling (max 500 points)
  4. Median depth + 20% outlier filter for stability
  5. Backproject to 3D camera frame: P_cam = d * K_inv @ [u, v, 1]^T
  6. Transform to ego frame: P_ego = R @ P_cam + t
  7. Compute centroid (median), size
  8. Orientation set to 0.0 (placeholder — will be set by velocity or 3D-MOOD)
"""

from __future__ import annotations
import numpy as np
import cv2


# ─── Constants ──────────────────────────────────────────────────────────────

MAX_POINTS = 500
EROSION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
DEPTH_OUTLIER_RATIO = 0.20   # ±20% of median depth


# ─── Object Reconstructor ───────────────────────────────────────────────────

class ObjectReconstructor:
    """
    Converts 2D detections + masks + depth into 3D ego-frame objects.
    Orientation is set to 0.0 here — overridden by VehicleOrientationEstimator
    or 3D-MOOD downstream.
    """

    def __init__(self):
        pass

    def reconstruct(self,
                    detection: dict,
                    mask: np.ndarray | None,
                    depth_map: np.ndarray,
                    K_inv: np.ndarray,
                    R_cam2ego: np.ndarray,
                    t_cam2ego: np.ndarray) -> dict | None:
        """
        Reconstruct a single detection into 3D ego-frame coordinates.

        Args:
            detection: dict with 'bbox' [x1,y1,x2,y2], 'track_id', 'label'
            mask: (H, W) uint8 binary mask from Mask2Former, or None
            depth_map: (H, W) float32 metric depth from UniDepth
            K_inv: 3×3 inverse intrinsic matrix
            R_cam2ego: 3×3 rotation camera→ego
            t_cam2ego: (3,) translation camera→ego

        Returns:
            dict with centroid_3d, size_3d, orientation, median_depth, etc.
            or None if insufficient valid points.
        """
        H, W = depth_map.shape[:2]

        # 1. Apply morphological erosion to mask
        if mask is not None and mask.any():
            mask = cv2.erode(mask, EROSION_KERNEL, iterations=1)
            if not mask.any():
                mask = None  # erosion killed tiny mask — fallback to bbox

        # 2. Get pixel coordinates
        if mask is not None and mask.any():
            pixels = np.argwhere(mask > 0)  # (N, 2) — row, col
        else:
            # Fallback: sample from bbox
            x1, y1, x2, y2 = [int(v) for v in detection['bbox']]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            rows = np.arange(y1, y2)
            cols = np.arange(x1, x2)
            rr, cc = np.meshgrid(rows, cols, indexing='ij')
            pixels = np.stack([rr.ravel(), cc.ravel()], axis=1)

        if len(pixels) < 10:
            return None

        # 3. Deterministic stride sampling (NOT random — prevents temporal jitter)
        if len(pixels) > MAX_POINTS:
            stride = max(1, len(pixels) // MAX_POINTS)
            pixels = pixels[::stride][:MAX_POINTS]

        # 4. Extract depths and compute MEDIAN for stability
        row_idx = np.clip(pixels[:, 0], 0, H - 1)
        col_idx = np.clip(pixels[:, 1], 0, W - 1)
        depths = depth_map[row_idx, col_idx]
        valid = depths > 0.1
        if valid.sum() < 10:
            return None

        median_depth = float(np.median(depths[valid]))

        # 5. Filter outliers: keep only points within 20% of median
        depth_mask = np.abs(depths - median_depth) < DEPTH_OUTLIER_RATIO * median_depth
        final_mask = valid & depth_mask
        pixels = pixels[final_mask]
        depths = depths[final_mask]

        if len(pixels) < 10:
            return None

        # 6. Backproject to camera frame
        u = pixels[:, 1].astype(np.float64)  # col → u
        v = pixels[:, 0].astype(np.float64)  # row → v
        ones = np.ones_like(u)
        uvw = np.stack([u, v, ones], axis=1)           # (N, 3)
        points_cam = (K_inv @ uvw.T).T * depths[:, None]  # (N, 3)

        # 7. Transform to ego frame
        points_ego = (R_cam2ego @ points_cam.T).T + t_cam2ego  # (N, 3)

        # 8. Compute centroid using median (robust to remaining outliers)
        centroid = np.median(points_ego, axis=0)

        # 9. Size: axis-aligned extent
        size_3d = np.ptp(points_ego, axis=0)  # [width, length, height]

        # 10. Orientation: placeholder 0.0 — will be set by velocity or 3D-MOOD
        orientation = 0.0

        return {
            'centroid_3d': centroid.tolist(),
            'size_3d': size_3d.tolist(),
            'orientation': round(float(orientation), 1),
            'median_depth': median_depth,
            'num_points': len(pixels),
        }

    def prune_smoother(self, active_track_ids: set):
        """No-op — orientation handled by VehicleOrientationEstimator."""
        pass
