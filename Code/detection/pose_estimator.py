"""
Pedestrian Pose Estimator — Phase 3.

Primary:  RTMPose-l (17 COCO keypoints, 75.8 AP) via rtmlib
Fallback: MediaPipe Pose (if rtmlib unavailable)

Copied from Code_ph2_VLM+Dino/detection/pose_estimator.py with estimate_batch()
wrapper added for p3 integration. Per-crop inference (not full-frame YOLO Pose).

Walking vs Standing detection uses keypoint motion between frames:
  1. Compare lower-body keypoints (ankles, knees) relative to hip centroid
  2. If legs move relative to hips → walking (immune to ego/camera motion)
  3. Single-frame pose geometry (ankle spread, knee asymmetry) as secondary
  4. 9-frame majority vote for temporal stability

Outputs per pedestrian:
  - keypoints: (17, 2) array in frame pixel coordinates
  - keypoint_scores: (17,) confidence array
  - action: "standing" | "walking" | "crouching" | "unknown"
"""

from __future__ import annotations
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ─── COCO keypoint definitions ───────────────────────────────────────────────

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

KEYPOINT_INDICES = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# COCO skeleton connections for visualization
COCO_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),      # arms
    (5, 11), (6, 12), (11, 12),                      # torso
    (11, 13), (13, 15), (12, 14), (14, 16),          # legs
    (0, 1), (0, 2), (1, 3), (2, 4),                  # head
]

# Labels of objects guaranteed to be static in the world
_STATIC_LABELS = {"traffic_light", "stop_sign", "speed_limit",
                  "traffic_cone", "traffic_pole", "dustbin",
                  "traffic_cylinder"}


class PoseEstimator:
    """
    Estimates 2D pose keypoints and action labels for pedestrians.

    Walking detection uses RTMPose keypoint motion between frames:
      - Store previous keypoints per track
      - Compare ankle/knee movement relative to hip centroid
      - If lower body keypoints move significantly relative to torso → walking
      - Direction from the vector of ankle displacement
    """

    def __init__(self):
        self._backend = None        # "rtmpose" or "mediapipe"
        self._rtm_body = None
        self._mp_pose = None

        # ── Per-track keypoint history (for motion-based walking detection) ──
        self._prev_keypoints: dict[int, np.ndarray] = {}   # track_id → (17,2)
        self._prev_scores: dict[int, np.ndarray] = {}      # track_id → (17,)

        # ── Action smoothing: majority vote over last 9 frames ──
        self._action_buffer: dict[int, deque] = {}

        # ── Centroid history for movement direction ──
        self._centroid_history: dict[int, list] = {}

        # ── Ego velocity (kept for heading_3d but NOT used for walk detection) ──
        self._static_pos_history: dict[int, list] = {}
        self._ego_velocity: np.ndarray = np.zeros(3)
        self._ego_vel_alpha: float = 0.3  # EMA smoothing factor
        self._pos3d_history: dict[int, deque] = {}

        self._init_rtmpose()
        if self._backend is None:
            self._init_mediapipe()
        if self._backend is None:
            print("[PoseEstimator] WARNING: No pose backend available.")

    # ─── Initialization ──────────────────────────────────────────────────────

    def _init_rtmpose(self):
        """Try loading RTMPose-l via rtmlib."""
        try:
            from rtmlib import Body
        except ImportError:
            print("[PoseEstimator] rtmlib not installed — "
                  "install with: pip install rtmlib")
            return

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        try:
            self._rtm_body = Body(
                mode=cfg.RTMPOSE_MODE,
                to_openpose=False,
                backend=cfg.RTMPOSE_BACKEND,
                device=device,
            )
            self._backend = "rtmpose"
            print(f"[PoseEstimator] RTMPose loaded (mode={cfg.RTMPOSE_MODE}, "
                  f"{cfg.RTMPOSE_BACKEND}, {device}).")
        except Exception as e:
            print(f"[PoseEstimator] RTMPose init failed: {e}")
            self._rtm_body = None

    def _init_mediapipe(self):
        """Fallback: load MediaPipe Pose."""
        try:
            import mediapipe as mp
        except ImportError:
            print("[PoseEstimator] mediapipe not installed — "
                  "install with: pip install mediapipe")
            return

        try:
            self._mp_module = mp.solutions.pose
            self._mp_pose = self._mp_module.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=cfg.POSE_MIN_DETECTION,
            )
            self._backend = "mediapipe"
            print("[PoseEstimator] MediaPipe Pose loaded (fallback).")
        except Exception as e:
            print(f"[PoseEstimator] MediaPipe init failed: {e}")
            self._mp_pose = None

    # ─── Ego Velocity Estimation ─────────────────────────────────────────────

    def update_ego_velocity(self, frame_detections: dict):
        """
        Estimate ego vehicle velocity from static reference objects.
        Uses traffic lights, signs, cones, poles — objects guaranteed
        to be static in the world.
        """
        displacements = []

        for category in ("traffic_lights", "road_signs", "objects"):
            for det in frame_detections.get(category, []):
                tid = det.get("track_id")
                pos = det.get("position_3d")
                label = det.get("label", "")

                if tid is None or pos is None or len(pos) < 3:
                    continue
                if label not in _STATIC_LABELS and category != "traffic_lights":
                    continue

                pos_arr = np.array(pos[:3], dtype=np.float64)
                if tid in self._static_pos_history:
                    prev = np.array(self._static_pos_history[tid])
                    disp = pos_arr - prev
                    if np.linalg.norm(disp) < 5.0:
                        displacements.append(disp)
                self._static_pos_history[tid] = pos_arr.tolist()

        if len(displacements) < 2:
            for det in frame_detections.get("vehicles", []):
                tid = det.get("track_id")
                pos = det.get("position_3d")
                if tid is None or pos is None or len(pos) < 3:
                    continue
                pos_arr = np.array(pos[:3], dtype=np.float64)
                if tid in self._static_pos_history:
                    prev = np.array(self._static_pos_history[tid])
                    disp = pos_arr - prev
                    if np.linalg.norm(disp) < 5.0:
                        displacements.append(disp)
                self._static_pos_history[tid] = pos_arr.tolist()

        if displacements:
            median_disp = np.median(displacements, axis=0)
            self._ego_velocity = (
                (1 - self._ego_vel_alpha) * self._ego_velocity +
                self._ego_vel_alpha * median_disp
            )

    # ─── Public API (batch — p3 integration) ────────────────────────────────

    def estimate_batch(self, frames: dict, pedestrians: list) -> None:
        """
        Run RTMPose per-crop for each pedestrian, classify walking/standing.
        Mutates each pedestrian dict in-place.

        Args:
            frames:      dict mapping camera_id → BGR frame (np.ndarray)
            pedestrians: list of pedestrian dicts with 'bbox', 'camera', 'track_id', etc.
        """
        if self._backend is None or not pedestrians:
            return

        # Group by camera
        peds_by_cam: dict[str, list] = {}
        for p in pedestrians:
            cam = p.get("camera", "front")
            peds_by_cam.setdefault(cam, []).append(p)

        max_per_cam = getattr(cfg, "POSE_MAX_PEDS_PER_CAM", 8)

        for cam_id, cam_peds in peds_by_cam.items():
            frame = frames.get(cam_id)
            if frame is None:
                continue

            # Sort by bbox area descending (bigger peds = more keypoints visible)
            cam_peds.sort(key=lambda p: (p["bbox"][2] - p["bbox"][0]) *
                                        (p["bbox"][3] - p["bbox"][1]), reverse=True)

            for i, p in enumerate(cam_peds):
                if i >= max_per_cam:
                    # Over cap — default to standing
                    p["pose_label"] = "standing"
                    p["heading_3d"] = None
                    p["walking_direction"] = None
                    continue

                result = self.estimate(
                    frame, p["bbox"],
                    track_id=p.get("track_id", -1),
                    position_3d=p.get("position_3d"))

                p["pose_label"] = result["action"]
                p["heading_3d"] = result["heading_3d"]
                p["walking_direction"] = self._heading_to_cardinal(
                    result["heading_3d"]) if result["action"] == "walking" else None
                p["keypoints"] = result["keypoints"]
                p["keypoint_scores"] = result["keypoint_scores"]
                p["walk_direction"] = result["walk_direction"]
                p["body_heading"] = result["body_heading"]
                p["movement_direction"] = result["movement_direction"]

    # ─── Public API (single pedestrian) ──────────────────────────────────────

    def estimate(self, frame: np.ndarray, bbox: list,
                 track_id: int = -1,
                 position_3d: list = None) -> dict:
        """
        Run pose estimation on a single pedestrian crop.

        Args:
            frame:       Full BGR frame.
            bbox:        [x1, y1, x2, y2] of the pedestrian.
            track_id:    Tracker ID (for temporal history).
            position_3d: [x, y, z] in ego frame (for velocity-based walking).

        Returns:
            Pose result dict with keypoints, scores, action, etc.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        H, W = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        crop = frame[y1:y2, x1:x2]

        if crop.shape[0] < 32 or crop.shape[1] < 16:
            return self._empty_result()

        if self._backend == "rtmpose":
            keypoints, scores = self._estimate_rtmpose(crop)
        elif self._backend == "mediapipe":
            keypoints, scores = self._estimate_mediapipe(crop)
        else:
            return self._empty_result()

        if keypoints is None:
            return self._empty_result()

        # Convert keypoints from crop-local to frame-level coordinates
        keypoints_frame = keypoints.copy()
        keypoints_frame[:, 0] += x1
        keypoints_frame[:, 1] += y1

        # ── Action classification ────────────────────────────────────
        # 1. Crouching check (pose geometry)
        action = self._classify_action(keypoints, scores)

        # Save walking direction BEFORE _classify_walking overwrites prev_keypoints
        walk_direction = self._get_walking_direction(track_id, keypoints_frame, scores)

        # 2. Walking check (keypoint motion between frames)
        if action == "standing":
            action = self._classify_walking(
                track_id, position_3d, keypoints_frame, scores,
                x2 - x1, y2 - y1)

        # ── 9-frame majority vote (needs 5+ to switch) ──────────────
        if track_id >= 0:
            if track_id not in self._action_buffer:
                self._action_buffer[track_id] = deque(maxlen=9)
            self._action_buffer[track_id].append(action)
            votes = Counter(self._action_buffer[track_id])
            top_action, top_count = votes.most_common(1)[0]
            if top_count >= 5:
                action = top_action

        # Movement direction from bbox centroid delta
        movement_dir = self._estimate_movement_direction(track_id, x1, y1, x2, y2)

        # Body heading from shoulder orientation
        body_heading = self._estimate_body_heading(keypoints_frame, scores)

        # 3D heading from position history
        heading_3d = self._get_heading_3d(track_id, position_3d)

        return {
            "keypoints": keypoints_frame.tolist() if isinstance(keypoints_frame, np.ndarray)
                         else keypoints_frame,
            "keypoint_scores": scores.tolist() if isinstance(scores, np.ndarray)
                               else scores,
            "action": action,
            "pose_label": action,
            "movement_direction": movement_dir,
            "walk_direction": walk_direction,
            "body_heading": body_heading,
            "heading_3d": heading_3d,
        }

    # ─── Walking Classification (keypoint motion between frames) ──────────────

    def _classify_walking(self, track_id: int,
                           position_3d: list,
                           keypoints_frame: np.ndarray,
                           scores: np.ndarray,
                           body_w: int, body_h: int) -> str:
        """
        Determine walking vs standing by comparing RTMPose keypoints
        between consecutive frames. Lower-body keypoints move RELATIVE
        to hip centroid → immune to ego motion.
        """
        conf_thr = cfg.POSE_CONF_THRESHOLD
        walk_score = 0.0

        # Primary: keypoint motion between frames
        if track_id >= 0 and track_id in self._prev_keypoints:
            prev_kp = self._prev_keypoints[track_id]
            prev_sc = self._prev_scores[track_id]
            motion_score = self._compute_keypoint_motion(
                prev_kp, prev_sc, keypoints_frame, scores, body_h)
            walk_score += motion_score  # up to 0.7

        # Secondary: single-frame pose geometry
        pose_score = self._pose_geometry_score(keypoints_frame, scores, body_w)
        walk_score += pose_score * 0.3  # up to 0.3

        # Store current keypoints for next frame comparison
        if track_id >= 0:
            self._prev_keypoints[track_id] = keypoints_frame.copy()
            self._prev_scores[track_id] = scores.copy()

        return "walking" if walk_score >= 0.35 else "standing"

    def _compute_keypoint_motion(self,
                                  prev_kp: np.ndarray, prev_sc: np.ndarray,
                                  curr_kp: np.ndarray, curr_sc: np.ndarray,
                                  body_h: int) -> float:
        """
        Measure lower-body keypoint motion relative to hip centroid.
        Returns 0.0 to 0.7.
        """
        conf_thr = cfg.POSE_CONF_THRESHOLD
        body_h = max(body_h, 1)

        l_hip = KEYPOINT_INDICES["left_hip"]
        r_hip = KEYPOINT_INDICES["right_hip"]
        l_ankle = KEYPOINT_INDICES["left_ankle"]
        r_ankle = KEYPOINT_INDICES["right_ankle"]
        l_knee = KEYPOINT_INDICES["left_knee"]
        r_knee = KEYPOINT_INDICES["right_knee"]

        prev_hip_ok = prev_sc[l_hip] > conf_thr or prev_sc[r_hip] > conf_thr
        curr_hip_ok = curr_sc[l_hip] > conf_thr or curr_sc[r_hip] > conf_thr
        if not (prev_hip_ok and curr_hip_ok):
            return 0.0

        def _hip_center(kp, sc):
            if sc[l_hip] > conf_thr and sc[r_hip] > conf_thr:
                return (kp[l_hip] + kp[r_hip]) / 2.0
            elif sc[l_hip] > conf_thr:
                return kp[l_hip].copy()
            else:
                return kp[r_hip].copy()

        prev_hip = _hip_center(prev_kp, prev_sc)
        curr_hip = _hip_center(curr_kp, curr_sc)
        hip_disp = curr_hip - prev_hip

        lower_indices = [l_ankle, r_ankle, l_knee, r_knee]
        relative_movements = []

        for idx in lower_indices:
            if prev_sc[idx] > conf_thr and curr_sc[idx] > conf_thr:
                kp_disp = curr_kp[idx] - prev_kp[idx]
                rel_disp = kp_disp - hip_disp
                rel_dist = float(np.linalg.norm(rel_disp))
                relative_movements.append(rel_dist)

        if not relative_movements:
            return 0.0

        max_rel = max(relative_movements)
        avg_rel = sum(relative_movements) / len(relative_movements)
        max_ratio = max_rel / body_h
        avg_ratio = avg_rel / body_h

        score = 0.0
        if max_ratio > 0.06:
            score = 0.7
        elif max_ratio > 0.03:
            score = 0.5
        elif avg_ratio > 0.02:
            score = 0.3

        return score

    def _get_walking_direction(self, track_id: int,
                                keypoints_frame: np.ndarray,
                                scores: np.ndarray) -> float | None:
        """Walking direction from ankle movement between frames (degrees)."""
        if track_id < 0 or track_id not in self._prev_keypoints:
            return None

        conf_thr = cfg.POSE_CONF_THRESHOLD
        prev_kp = self._prev_keypoints[track_id]
        prev_sc = self._prev_scores[track_id]

        l_ankle = KEYPOINT_INDICES["left_ankle"]
        r_ankle = KEYPOINT_INDICES["right_ankle"]
        l_hip = KEYPOINT_INDICES["left_hip"]
        r_hip = KEYPOINT_INDICES["right_hip"]

        prev_hip_ok = prev_sc[l_hip] > conf_thr or prev_sc[r_hip] > conf_thr
        curr_hip_ok = scores[l_hip] > conf_thr or scores[r_hip] > conf_thr
        if not (prev_hip_ok and curr_hip_ok):
            return None

        disps = []
        for idx in [l_ankle, r_ankle]:
            if prev_sc[idx] > conf_thr and scores[idx] > conf_thr:
                disps.append(keypoints_frame[idx] - prev_kp[idx])

        if not disps:
            return None

        avg_disp = np.mean(disps, axis=0)
        dist = float(np.linalg.norm(avg_disp))
        if dist < 2.0:
            return None

        return float(np.degrees(np.arctan2(avg_disp[1], avg_disp[0])))

    def _get_heading_3d(self, track_id: int, position_3d: list) -> float | None:
        """Pedestrian heading from 3D position history (degrees)."""
        if track_id < 0 or position_3d is None or len(position_3d) < 3:
            return None

        pos = np.array(position_3d[:3], dtype=np.float64)

        if track_id not in self._pos3d_history:
            self._pos3d_history[track_id] = deque(maxlen=10)
            self._pos3d_history[track_id].append(pos)
            return None

        self._pos3d_history[track_id].append(pos)
        hist = list(self._pos3d_history[track_id])
        if len(hist) < 3:
            return None

        dx = hist[-1][0] - hist[0][0]
        dy = hist[-1][1] - hist[0][1]

        n = len(hist) - 1
        dx -= self._ego_velocity[0] * n
        dy -= self._ego_velocity[1] * n

        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 0.02 * n:
            return None

        heading = float(np.degrees(np.arctan2(dx, dy)))
        return round(heading, 1)

    def _pose_geometry_score(self, keypoints: np.ndarray,
                              scores: np.ndarray,
                              body_w: int) -> float:
        """Single-frame pose geometry score for walking (0.0-1.0)."""
        conf_thr = cfg.POSE_CONF_THRESHOLD
        score = 0.0
        body_w = max(body_w, 1)

        l_ankle = KEYPOINT_INDICES["left_ankle"]
        r_ankle = KEYPOINT_INDICES["right_ankle"]
        l_knee = KEYPOINT_INDICES["left_knee"]
        r_knee = KEYPOINT_INDICES["right_knee"]
        l_hip = KEYPOINT_INDICES["left_hip"]
        r_hip = KEYPOINT_INDICES["right_hip"]

        if scores[l_ankle] > conf_thr and scores[r_ankle] > conf_thr:
            spread = np.linalg.norm(keypoints[l_ankle] - keypoints[r_ankle])
            ratio = spread / body_w
            if ratio > 0.35:
                score += 0.5
            elif ratio > 0.20:
                score += 0.25

        if (scores[l_hip] > conf_thr and scores[l_knee] > conf_thr and
            scores[r_hip] > conf_thr and scores[r_knee] > conf_thr):
            l_dist = abs(keypoints[l_knee][1] - keypoints[l_hip][1])
            r_dist = abs(keypoints[r_knee][1] - keypoints[r_hip][1])
            if max(l_dist, r_dist) > 0:
                asym = abs(l_dist - r_dist) / max(l_dist, r_dist)
                if asym > 0.25:
                    score += 0.5
                elif asym > 0.15:
                    score += 0.25

        return min(score, 1.0)

    # ─── RTMPose Inference ───────────────────────────────────────────────────

    def _estimate_rtmpose(self, crop: np.ndarray) -> tuple:
        """Run RTMPose on a crop. Returns (keypoints (17,2), scores (17,))."""
        try:
            keypoints, scores = self._rtm_body(crop)
            if keypoints is None or len(keypoints) == 0:
                return None, None
            kps = np.array(keypoints[0], dtype=np.float32)
            scs = np.array(scores[0], dtype=np.float32)
            return kps, scs
        except Exception as e:
            print(f"[PoseEstimator] RTMPose error: {e}")
            return None, None

    # ─── MediaPipe Fallback ──────────────────────────────────────────────────

    def _estimate_mediapipe(self, crop: np.ndarray) -> tuple:
        """Run MediaPipe Pose on a crop, convert to 17 COCO keypoints."""
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = self._mp_pose.process(rgb)
            if not results.pose_landmarks:
                return None, None

            h, w = crop.shape[:2]
            mp_to_coco = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

            keypoints = np.zeros((17, 2), dtype=np.float32)
            scores = np.zeros(17, dtype=np.float32)

            lm = results.pose_landmarks.landmark
            for coco_idx, mp_idx in enumerate(mp_to_coco):
                if mp_idx < len(lm):
                    keypoints[coco_idx] = [lm[mp_idx].x * w, lm[mp_idx].y * h]
                    scores[coco_idx] = lm[mp_idx].visibility

            return keypoints, scores
        except Exception:
            return None, None

    # ─── Crouching Detection ─────────────────────────────────────────────────

    def _classify_action(self, keypoints: np.ndarray,
                          scores: np.ndarray) -> str:
        """Rule-based crouching detection from keypoints."""
        conf_thr = cfg.POSE_CONF_THRESHOLD

        l_hip = KEYPOINT_INDICES["left_hip"]
        l_knee = KEYPOINT_INDICES["left_knee"]

        if scores[l_hip] > conf_thr and scores[l_knee] > conf_thr:
            hip_knee_dist_y = abs(keypoints[l_hip][1] - keypoints[l_knee][1])
            valid_pts = keypoints[scores > conf_thr]
            if len(valid_pts) > 2:
                body_h = valid_pts[:, 1].max() - valid_pts[:, 1].min()
                if body_h > 0 and hip_knee_dist_y < 0.15 * body_h:
                    return "crouching"

        return "standing"

    # ─── Body Heading ────────────────────────────────────────────────────────

    def _estimate_body_heading(self, keypoints: np.ndarray,
                                scores: np.ndarray) -> float | None:
        """Estimate torso facing direction from shoulder-hip perpendicular."""
        conf_thr = cfg.POSE_CONF_THRESHOLD
        l_sh = KEYPOINT_INDICES["left_shoulder"]
        r_sh = KEYPOINT_INDICES["right_shoulder"]
        l_hip = KEYPOINT_INDICES["left_hip"]
        r_hip = KEYPOINT_INDICES["right_hip"]

        if scores[l_sh] <= conf_thr or scores[r_sh] <= conf_thr:
            return None

        sh_dx = keypoints[r_sh][0] - keypoints[l_sh][0]
        sh_dy = keypoints[r_sh][1] - keypoints[l_sh][1]
        face_dx = -sh_dy
        face_dy = sh_dx

        if scores[l_hip] > conf_thr and scores[r_hip] > conf_thr:
            hip_dx = keypoints[r_hip][0] - keypoints[l_hip][0]
            hip_dy = keypoints[r_hip][1] - keypoints[l_hip][1]
            face_dx = -(sh_dy + hip_dy) / 2.0
            face_dy = (sh_dx + hip_dx) / 2.0

        norm = (face_dx**2 + face_dy**2) ** 0.5
        if norm < 1.0:
            return None

        return float(np.degrees(np.arctan2(face_dy, face_dx)))

    # ─── Movement Direction ──────────────────────────────────────────────────

    def _estimate_movement_direction(self, track_id: int,
                                      x1: int, y1: int,
                                      x2: int, y2: int) -> float | None:
        """Estimate movement direction from bbox centroid delta."""
        if track_id < 0:
            return None

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if track_id in self._centroid_history:
            prev_cx, prev_cy = self._centroid_history[track_id]
            dx = cx - prev_cx
            dy = cy - prev_cy
            dist = (dx * dx + dy * dy) ** 0.5
            self._centroid_history[track_id] = [cx, cy]
            if dist > 2.0:
                return float(np.degrees(np.arctan2(dy, dx)))
            return None
        else:
            self._centroid_history[track_id] = [cx, cy]
            return None

    # ─── Direction Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _heading_to_cardinal(heading_deg: float | None) -> str | None:
        """
        Convert heading_3d (degrees) to cardinal direction string.
        Ego frame: 0°=forward(+Y), 90°=right(+X), -90°=left, ±180°=back.
        """
        if heading_deg is None:
            return None
        h = heading_deg % 360
        if h > 180:
            h -= 360
        if -45 <= h < 45:
            return "front"
        elif 45 <= h < 135:
            return "right"
        elif -135 <= h < -45:
            return "left"
        else:
            return "back"

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result() -> dict:
        return {
            "keypoints": [],
            "keypoint_scores": [],
            "action": "standing",
            "pose_label": "standing",
            "movement_direction": None,
            "walk_direction": None,
            "body_heading": None,
            "heading_3d": None,
        }

    def prune_ankle_history(self, active_ids: set):
        """Remove history for lost tracks."""
        for store in (self._centroid_history, self._action_buffer,
                      self._pos3d_history, self._static_pos_history,
                      self._prev_keypoints, self._prev_scores):
            stale = set(store.keys()) - active_ids
            for tid in stale:
                del store[tid]

    def close(self):
        """Release resources."""
        if self._mp_pose is not None:
            try:
                self._mp_pose.close()
            except Exception:
                pass
        self._rtm_body = None
