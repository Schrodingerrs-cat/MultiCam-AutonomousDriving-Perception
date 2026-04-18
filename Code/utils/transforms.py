"""
Coordinate Transforms — image space ↔ 3D world space.

Conventions:
  Camera frame : X=right, Y=down,     Z=forward   (OpenCV)
  Blender world: X=right, Y=forward,  Z=up        (right-hand, Z-up)

Ego car sits at the Blender world origin (0, 0, 0).
All detected objects are expressed relative to the ego camera position,
then converted to Blender world coordinates so the renderer can place them.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

sys_path_added = False
try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config as _cfg
    _EXTRINSICS = _cfg.CAMERA_EXTRINSICS
except Exception:
    _EXTRINSICS = {
        "front": {"position": [ 0.0,  1.5, 1.0], "yaw_deg":    0.0, "pitch_deg":  7.0},
        "back":  {"position": [ 0.0, -1.8, 0.9], "yaw_deg":  180.0, "pitch_deg":  5.0},
        "left":  {"position": [-0.9,  0.1, 1.3], "yaw_deg":  -90.0, "pitch_deg":  8.0},
        "right": {"position": [ 0.9,  0.1, 1.3], "yaw_deg":   90.0, "pitch_deg":  8.0},
    }


# ─── Camera → Blender world ───────────────────────────────────────────────────

def cam_to_world(
    cam_xyz: list | np.ndarray,
    camera_height: float = 1.2,
    camera_pitch_deg: float = 0.0,
) -> list[float]:
    """
    Convert a 3D point from camera coordinates to Blender world coordinates.

    Camera frame: Z forward, X right, Y down (OpenCV convention).
    Blender world: Y forward, X right, Z up.

    Args:
        cam_xyz:          [X, Y, Z] in camera frame (metres)
        camera_height:    height of camera above ground (metres)
        camera_pitch_deg: camera pitch (positive = looking down)
    Returns:
        [X, Y, Z] in Blender world frame relative to ego origin.
    """
    Xc, Yc, Zc = cam_xyz

    # Rotate for camera pitch
    pitch = np.deg2rad(camera_pitch_deg)
    Xc_r  = Xc
    Yc_r  = Yc * np.cos(pitch) - Zc * np.sin(pitch)
    Zc_r  = Yc * np.sin(pitch) + Zc * np.cos(pitch)

    # Map to Blender world:
    #   Blender X =  camera X   (right)
    #   Blender Y =  camera Z   (forward)
    #   Blender Z = -camera Y + camera_height   (up, accounting for cam mount)
    Xw =  Xc_r
    Yw =  Zc_r
    Zw = -Yc_r + camera_height

    # Clamp Z to ground plane (objects shouldn't go underground)
    Zw = max(0.0, Zw)

    return [float(Xw), float(Yw), float(Zw)]


def lane_points_to_3d(
    image_points: list[list],
    K: np.ndarray,
    camera_height: float = 1.2,
    camera_pitch_deg: float = 0.0,
    max_depth: float = 50.0,
) -> list[list[float]]:
    """
    Reproject lane line points (assumed on flat ground) into 3D Blender world coords.

    Uses the ground-plane constraint:
        depth = (camera_height * fy) / (dv_eff)
    where dv_eff accounts for the camera pitch so that even points near or
    above the image centre (distant lanes) are reprojected correctly.

    For points that would give physically impossible depths, a linear
    interpolation from the image row to a reasonable depth range is used
    as a fallback so we always get some 3D points for the visualisation.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    img_h = cy * 2   # approximate (principal point near centre)
    pitch = np.deg2rad(camera_pitch_deg)
    # Row at which the ground plane meets the horizon (below this = road)
    # horizon_row = cy - fy * tan(pitch)  (positive pitch = looking down)
    horizon_row = cy - fy * np.tan(pitch)

    world_pts = []

    for pt in image_points:
        u, v = float(pt[0]), float(pt[1])

        # Effective pixel offset from principal point, corrected for pitch
        dv = (v - cy) * np.cos(pitch) + fy * np.sin(pitch)

        if dv > 1.0:
            # Standard ground-plane reprojection
            depth = camera_height * fy / dv
        else:
            # Point is near or above horizon — use row-based linear fallback:
            # row=img_h → depth≈2m,  row=horizon_row → depth=max_depth
            row_frac = np.clip(
                (img_h - v) / max(1.0, img_h - horizon_row), 0.0, 1.0
            )
            depth = 2.0 + row_frac * (max_depth - 2.0)

        depth = float(np.clip(depth, 1.0, max_depth))

        Xc = (u - cx) * depth / fx
        Zc = depth

        world = cam_to_world([Xc, camera_height, Zc], camera_height, camera_pitch_deg)
        world[2] = 0.0   # lane markings are on the ground
        world_pts.append(world)

    return world_pts


def estimate_camera_pitch(calib_data: dict) -> float:
    """
    Estimate camera pitch from calibration data or vanishing point.
    Returns pitch in degrees (positive = looking down).
    A typical dashcam / front camera is pitched ~5-10° downward.
    """
    # Default for Tesla Model S front camera
    return 7.0


def compute_ground_homography(
    K: np.ndarray,
    camera_height: float,
    pitch_deg: float,
) -> np.ndarray:
    """
    Compute the 3×3 homography H that maps image pixel (u, v, 1)^T →
    ground-plane world point (X, Y, 1)^T  (Z = 0 by definition).

    World frame: X = right, Y = forward, Z = up.
    Camera is at height `camera_height` above ground, pitched downward
    by `pitch_deg` degrees.

    Derivation (analytical):
        Camera axes in world (X=right, Y=forward, Z=up), camera at (0,0,h),
        pitched down by θ:

            Cam X in world = (1, 0, 0)
            Cam Y in world = (0, −sin θ, −cos θ)   [down direction]
            Cam Z in world = (0,  cos θ, −sin θ)   [forward direction]

        For ground point P=(X, Y, 0), ray = P − camera_center = (X, Y, −h):

            Xc =  X
            Yc =  −Y·sin θ + h·cos θ
            Zc =   Y·cos θ + h·sin θ

        Projected pixel: u=cx+fx·Xc/Zc, v=cy+fy·Yc/Zc.
        Inverting gives:

            Zc = h·fy / (fy·sin θ + (v−cy)·cos θ)
            X  = (u−cx)·Zc / fx
            Y  = h·(fy·cos θ − (v−cy)·sin θ) / (fy·sin θ + (v−cy)·cos θ)

        Expressing as a projective H ([u,v,1]→[X·w, Y·w, w]):

            Row 0 (X·w): h·fy/fx · u  +  0 · v  +  (−h·fy·cx/fx)
            Row 1 (Y·w): 0 · u  +  (−h·sθ) · v  +  h·(fy·cθ + cy·sθ)
            Row 2 (w)  : 0 · u  +  cθ · v  +  (fy·sθ − cy·cθ)
    """
    θ  = np.deg2rad(pitch_deg)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h  = camera_height
    cθ, sθ = np.cos(θ), np.sin(θ)

    # H maps [u, v, 1]^T → [X·w, Y·w, w]^T, then divide by w.
    H = np.array([
        [h * fy / fx,  0.0,    -h * fy * cx / fx              ],
        [0.0,         -h * sθ,  h * (fy * cθ + cy * sθ)       ],
        [0.0,          cθ,      fy * sθ - cy * cθ              ],
    ], dtype=np.float64)
    return H


def project_to_ground(
    H: np.ndarray,
    u: float,
    v: float,
) -> tuple[float, float]:
    """
    Use ground-plane homography to map one image pixel to world (X, Y).

    Args:
        H: 3×3 matrix from compute_ground_homography()
        u, v: pixel coordinates (float)

    Returns:
        (X, Y) in metres, world frame.  Z = 0 (ground plane).
    """
    p = H @ np.array([u, v, 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-8:
        return 0.0, 0.0
    return float(p[0] / p[2]), float(p[1] / p[2])


def bbox_to_ground(
    H: np.ndarray,
    bbox: list,
    max_dist: float = 80.0,
) -> list[float]:
    """
    Project a detection bounding box to world coordinates using the
    ground-plane homography.

    Uses the bottom-centre of the bbox as the contact point with the ground.

    Args:
        H:        ground homography (3×3)
        bbox:     [x1, y1, x2, y2] in image pixels
        max_dist: discard detections further than this (metres)

    Returns:
        [X, Y, 0.0] in world frame, or None if out of range / behind camera.
    """
    x1, y1, x2, y2 = bbox
    u = (x1 + x2) / 2.0
    v = y2                   # bottom-centre = ground contact point

    X, Y = project_to_ground(H, u, v)

    if Y < 0.3 or Y > max_dist:   # behind camera or too far
        return None
    if abs(X) > max_dist:
        return None

    return [float(X), float(Y), 0.0]


def lane_point_to_ground(
    H: np.ndarray,
    u: float,
    v: float,
    max_dist: float = 50.0,
) -> Optional[list[float]]:
    """Project a single lane pixel to world [X, Y, 0]."""
    X, Y = project_to_ground(H, u, v)
    if Y < 0.3 or Y > max_dist:
        return None
    return [float(X), float(Y), 0.0]


def cam_to_world_multicam(
    cam_xyz: list | np.ndarray,
    camera_id: str,
    extrinsics: dict | None = None,
) -> list[float]:
    """
    Convert a 3D point from any camera's OpenCV frame to Blender world coords.

    Handles rotation (pitch + yaw) and translation offset for each camera
    in the multi-camera rig.

    Args:
        cam_xyz:    [X, Y, Z] in OpenCV camera frame (X=right, Y=down, Z=forward)
        camera_id:  one of 'front', 'back', 'left', 'right'
        extrinsics: override dict; defaults to config.CAMERA_EXTRINSICS
    Returns:
        [X, Y, Z] in Blender world frame relative to ego car origin.
    """
    ex = (extrinsics or _EXTRINSICS).get(camera_id, _EXTRINSICS["front"])
    Xc, Yc, Zc = cam_xyz

    # 1. Apply camera pitch (positive pitch_deg = tilted downward)
    pitch = np.deg2rad(ex["pitch_deg"])
    Xp =  Xc
    Yp =  Yc * np.cos(pitch) - Zc * np.sin(pitch)
    Zp =  Yc * np.sin(pitch) + Zc * np.cos(pitch)

    # 2. OpenCV cam frame → camera-body frame (Z-forward, Y-down → Y-forward, Z-up)
    Xb =  Xp
    Yb =  Zp                          # camera forward  → body forward
    Zb = -Yp + ex["position"][2]      # camera up (−Y)  + mount height

    # 3. Yaw rotation: camera body → ego body frame
    #    yaw=0 → front (forward), yaw=90 → right, yaw=180 → back, yaw=-90 → left
    #    Convention: Xe = cos(yaw)*Xb + sin(yaw)*Yb  (camera +Y rotated by yaw into ego frame)
    yaw = np.deg2rad(ex["yaw_deg"])
    Xe =  np.cos(yaw) * Xb + np.sin(yaw) * Yb
    Ye = -np.sin(yaw) * Xb + np.cos(yaw) * Yb
    Ze =  Zb

    # 4. Add camera position offset in ego frame
    pos = ex["position"]
    Xw = Xe + pos[0]
    Yw = Ye + pos[1]
    Zw = max(0.0, float(Ze))

    return [float(Xw), float(Yw), Zw]


def world_space_nms(detections: list[dict], radius_m: float = 2.5) -> list[dict]:
    """
    DEPRECATED — use fuse_multicam_detections() for the new pipeline.

    Non-maximum suppression in 2D world space (legacy, kept for compatibility).
    """
    if not detections:
        return []

    kept = []
    suppressed = [False] * len(detections)
    indexed = sorted(enumerate(detections), key=lambda x: x[1].get("confidence", 0), reverse=True)

    for i, (idx_i, det_i) in enumerate(indexed):
        if suppressed[idx_i]:
            continue
        kept.append(det_i)
        pi = det_i.get("position_3d", [0, 0, 0])
        for j, (idx_j, det_j) in enumerate(indexed):
            if j <= i or suppressed[idx_j]:
                continue
            pj = det_j.get("position_3d", [0, 0, 0])
            dist = np.hypot(pi[0] - pj[0], pi[1] - pj[1])
            if dist < radius_m:
                suppressed[idx_j] = True
    return kept


# ─── New pipeline: proper 3D projection ─────────────────────────────────────

def pixel_to_camera_3d(u: float, v: float, depth: float,
                        K_inv: np.ndarray) -> np.ndarray:
    """
    Single pixel + depth → camera-frame 3D point.

    Args:
        u, v: pixel coordinates
        depth: metric depth in metres
        K_inv: 3×3 inverse intrinsic matrix

    Returns:
        (3,) array — [X, Y, Z] in camera frame (OpenCV: X=right, Y=down, Z=forward)
    """
    return depth * (K_inv @ np.array([u, v, 1.0], dtype=np.float64))


def camera_to_ego(points_cam: np.ndarray,
                   R_cam2ego: np.ndarray,
                   t_cam2ego: np.ndarray) -> np.ndarray:
    """
    Batch transform camera-frame points to ego frame.

    Args:
        points_cam: (N, 3) array in camera frame
        R_cam2ego: (3, 3) rotation matrix
        t_cam2ego: (3,) translation vector

    Returns:
        (N, 3) array in ego frame (X=right, Y=forward, Z=up)
    """
    return (R_cam2ego @ points_cam.T).T + t_cam2ego


# ─── Multi-camera fusion ────────────────────────────────────────────────────

def fuse_multicam_detections(all_detections: list[dict],
                              dist_thresh: float = 1.5) -> list[dict]:
    """
    Merge detections across cameras in ego frame.

    Rules:
      - Same class required
      - Centroid distance < dist_thresh (metres)
      - Highest confidence wins (owns centroid, size, orientation, track_id)
      - Single-view detections are ALWAYS kept (never suppressed)
      - Source camera metadata preserved for all contributing detections

    Args:
        all_detections: list of dets from all cameras, each with:
            'label', 'confidence', 'camera', 'bbox', 'track_id',
            'reconstruction' (dict with 'centroid_3d')
        dist_thresh: max distance in metres to merge (default 1.5m)

    Returns:
        List of merged detections with added 'sources', 'num_views',
        'cameras_seen' fields.
    """
    if not all_detections:
        return []

    merged = []
    used = set()

    sorted_dets = sorted(
        all_detections, key=lambda d: d.get('confidence', 0), reverse=True
    )

    for i, det in enumerate(sorted_dets):
        if i in used:
            continue

        sources = [{
            'camera': det.get('camera', 'unknown'),
            'confidence': det.get('confidence', 0),
            'bbox': det.get('bbox'),
            'track_id': det.get('track_id'),
            'median_depth': det.get('reconstruction', {}).get('median_depth')
                if det.get('reconstruction') else None,
        }]

        for j in range(i + 1, len(sorted_dets)):
            if j in used:
                continue
            other = sorted_dets[j]
            if other.get('label') != det.get('label'):
                continue
            # Never merge detections from the same camera
            if other.get('camera') == det.get('camera'):
                continue

            # Get 3D centroids for distance check
            c_a = (det.get('reconstruction') or {}).get('centroid_3d')
            c_b = (other.get('reconstruction') or {}).get('centroid_3d')

            if c_a is None or c_b is None:
                continue

            c_a = np.array(c_a)
            c_b = np.array(c_b)
            dist = float(np.linalg.norm(c_a[:2] - c_b[:2]))

            if dist < dist_thresh:
                used.add(j)
                sources.append({
                    'camera': other.get('camera', 'unknown'),
                    'confidence': other.get('confidence', 0),
                    'bbox': other.get('bbox'),
                    'track_id': other.get('track_id'),
                    'median_depth': other.get('reconstruction', {}).get('median_depth')
                        if other.get('reconstruction') else None,
                })

        # ALWAYS keep — even if single view
        winner = dict(det)
        winner['sources'] = sources
        winner['num_views'] = len(sources)
        winner['cameras_seen'] = [s['camera'] for s in sources]
        merged.append(winner)

    return merged


def blender_to_image(
    world_xyz: list,
    K: np.ndarray,
    img_shape: tuple,
    camera_height: float = 1.2,
    camera_pitch_deg: float = 0.0,
) -> Optional[list[int]]:
    """
    Project a Blender world point back to image pixel coordinates.
    Returns [u, v] or None if behind the camera / out of frame.
    """
    Xw, Yw, Zw = world_xyz

    # Blender → Camera
    pitch = np.deg2rad(camera_pitch_deg)
    Xc =  Xw
    Yc = -(Zw - camera_height)   # invert Z-up
    Zc =  Yw                     # Blender Y → Camera Z

    # Rotate for pitch
    Xc_r =  Xc
    Yc_r =  Yc * np.cos(pitch) + Zc * np.sin(pitch)
    Zc_r = -Yc * np.sin(pitch) + Zc * np.cos(pitch)

    if Zc_r <= 0:
        return None

    u = int(K[0, 0] * Xc_r / Zc_r + K[0, 2])
    v = int(K[1, 1] * Yc_r / Zc_r + K[1, 2])

    h, w = img_shape[:2]
    if 0 <= u < w and 0 <= v < h:
        return [u, v]
    return None
