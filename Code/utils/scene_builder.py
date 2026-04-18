"""
Scene Builder — Drivable Area Mesh from HybridNets + UniDepth
==============================================================

Converts HybridNets drivable area segmentation + UniDepth metric depth
into a lightweight 3D point cloud (sparse road surface) for Blender rendering.

Key features:
  - Gaussian-smoothed depth before meshing (sigma=3.0 removes noise)
  - Grid sampling (every 16px) keeps mesh lightweight (~500-2000 vertices)
  - Output: (N, 3) ego-frame points for Blender vertex displacement
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter


def build_drivable_mesh(drivable_mask: np.ndarray,
                         depth_map: np.ndarray,
                         K_inv: np.ndarray,
                         R_cam2ego: np.ndarray,
                         t_cam2ego: np.ndarray,
                         grid_step: int = 16,
                         sigma: float = 3.0) -> np.ndarray:
    """
    Convert HybridNets drivable area to lightweight 3D ego-frame points.

    Args:
        drivable_mask: (H, W) binary uint8 mask from HybridNets
        depth_map: (H, W) float32 metric depth from UniDepth
        K_inv: 3×3 inverse intrinsic matrix
        R_cam2ego: 3×3 rotation camera→ego
        t_cam2ego: (3,) translation camera→ego
        grid_step: sample every Nth pixel (controls mesh density)
        sigma: Gaussian blur sigma for depth smoothing

    Returns:
        (N, 3) float64 array — sparse road surface points in ego frame.
        Returns empty (0, 3) array if no valid points.
    """
    if drivable_mask is None or depth_map is None:
        return np.zeros((0, 3), dtype=np.float64)

    H, W = drivable_mask.shape[:2]

    # 1. Smooth depth to remove per-pixel noise before meshing
    smoothed_depth = gaussian_filter(depth_map.astype(np.float64), sigma=sigma)

    # 2. Apply drivable mask (zero out non-drivable regions)
    smoothed_depth = smoothed_depth * drivable_mask.astype(np.float64)

    # 3. Sample on grid (every grid_step pixels)
    vs = np.arange(0, H, grid_step)
    us = np.arange(0, W, grid_step)
    uu, vv = np.meshgrid(us, vs)

    # 4. Filter to drivable pixels only
    v_flat = vv.ravel()
    u_flat = uu.ravel()
    valid_drivable = drivable_mask[v_flat, u_flat] > 0
    u_flat = u_flat[valid_drivable]
    v_flat = v_flat[valid_drivable]
    d_flat = smoothed_depth[v_flat, u_flat]

    # Filter invalid / too-close depth
    depth_valid = d_flat > 0.5
    u_flat = u_flat[depth_valid]
    v_flat = v_flat[depth_valid]
    d_flat = d_flat[depth_valid]

    if len(u_flat) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # 5. Backproject to camera 3D
    ones = np.ones_like(u_flat, dtype=np.float64)
    uvw = np.stack([u_flat.astype(np.float64), v_flat.astype(np.float64), ones], axis=1)
    points_cam = (K_inv @ uvw.T).T * d_flat[:, None]

    # 6. Transform to ego frame
    points_ego = (R_cam2ego @ points_cam.T).T + t_cam2ego

    return points_ego
