"""
Collision Predictor — linear-regression TTC within ego corridor.
Phase 3 rewrite — uses raw distance history + least-squares regression
instead of EMA, which killed the approach_speed signal.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class CollisionPredictor:
    EGO_CORRIDOR_W = 3.5   # metres, +/-1.75m lateral (was 1.8 — too narrow)
    MAX_DIST = 50.0         # predict for objects up to 50m forward (was 5m)
    MIN_HISTORY = 4         # minimum frames before predicting
    MIN_TTC = 0.2           # floor: below this = already crashed
    MIN_APPROACH = 0.01     # m/frame minimum approach speed (was 0.02)
    HISTORY_LEN = 20        # frames of position history to keep

    def __init__(self, fps: int = 30):
        self.fps = fps
        self._history: dict[int, deque] = {}
        self._ttc_ema: dict[int, float] = {}

    def update_and_assess(
        self,
        track_id: int,
        pos_3d: list[float],
        size_3d: list[float] | None,
        frame_idx: int,
    ) -> dict:
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.HISTORY_LEN)
        self._history[track_id].append((frame_idx, np.array(pos_3d, dtype=np.float64)))

        history = self._history[track_id]
        pos_arr = np.array(pos_3d, dtype=np.float64)
        fwd_dist = float(pos_arr[1])  # Y = forward in ego frame

        none_result = {"level": "none", "ttc_seconds": None, "distance_m": fwd_dist}

        # Only predict for objects AHEAD and within MAX_DIST
        if fwd_dist <= 1.0 or fwd_dist > self.MAX_DIST:
            return none_result

        # Lateral overlap check: object must be in ego corridor
        half_corr = self.EGO_CORRIDOR_W / 2.0
        obj_half_w = size_3d[0] / 2.0 if size_3d else 0.9
        lateral_overlap = (
            pos_arr[0] - obj_half_w < half_corr and
            pos_arr[0] + obj_half_w > -half_corr
        )
        if not lateral_overlap:
            return none_result

        # Need minimum history
        if len(history) < self.MIN_HISTORY:
            return none_result

        # ── Linear regression on raw forward distance vs frame index ──
        # approach_speed = -slope (positive slope = receding, negative = closing)
        frames_arr = np.array([f for f, _ in history], dtype=np.float64)
        dists_arr = np.array([float(p[1]) for _, p in history], dtype=np.float64)

        n = len(frames_arr)
        if n < 3:
            return none_result

        # Least-squares linear fit: dist = slope * frame + intercept
        # Using numpy polyfit degree 1
        try:
            slope, intercept = np.polyfit(frames_arr, dists_arr, 1)
        except (np.linalg.LinAlgError, np.RankWarning):
            return none_result

        # slope < 0 means distance is decreasing (approaching)
        approach_speed = -slope  # m/frame, positive = closing

        # Object receding — no collision
        if approach_speed < self.MIN_APPROACH:
            return none_result

        # TTC = current forward distance / (approach_speed * fps)
        speed_mps = approach_speed * self.fps  # m/s
        raw_ttc = fwd_dist / speed_mps

        if raw_ttc < self.MIN_TTC or raw_ttc > 5.0:
            return none_result

        # Smooth TTC
        ttc_out = self._smooth_ttc(track_id, raw_ttc)

        # Classification:
        # critical: TTC ≤ 1.5s AND within 15m
        # warning:  TTC ≤ 3.0s AND within 30m
        if ttc_out <= 1.5 and fwd_dist <= 15.0:
            return {"level": "critical", "ttc_seconds": round(ttc_out, 2),
                    "distance_m": round(fwd_dist, 2)}
        elif ttc_out <= 3.0 and fwd_dist <= 30.0:
            return {"level": "warning", "ttc_seconds": round(ttc_out, 2),
                    "distance_m": round(fwd_dist, 2)}

        return none_result

    def _smooth_ttc(self, track_id: int, raw_ttc: float) -> float:
        if track_id not in self._ttc_ema:
            self._ttc_ema[track_id] = raw_ttc
        self._ttc_ema[track_id] = 0.5 * raw_ttc + 0.5 * self._ttc_ema[track_id]
        return self._ttc_ema[track_id]

    def prune(self, active_ids: set[int]):
        for tid in list(self._history):
            if tid not in active_ids:
                del self._history[tid]
                self._ttc_ema.pop(tid, None)

    def reset(self):
        self._history.clear()
        self._ttc_ema.clear()
