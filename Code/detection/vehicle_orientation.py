"""
Vehicle Orientation Estimator — Phase 3 rewrite.

Multi-stage heading pipeline:
  1. Heading from total displacement vector (start→end, not per-frame average)
  2. Flip detection (only after heading is established for 10+ frames)
  3. Implausible jump rejection (>60° in one frame → hold previous)
  4. Conservative EMA blend (25% new, 75% prior)

Two separate position pipelines:
  - RAW positions → speed/is_moving (preserves true displacement signal)
  - Median-filtered positions → heading (damps depth jitter for stable orientation)
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


def _angle_diff(a: float, b: float) -> float:
    """Signed shortest-path difference a - b, in radians."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


def _normalize_rad(a: float) -> float:
    """Normalize angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class VehicleOrientationEstimator:
    """Estimate heading from multi-frame velocity with robust smoothing."""

    # ── Position history ──
    HISTORY_LEN = 15

    # ── Motion detection (uses raw positions) ──
    SPEED_EMA_ALPHA = 0.35
    SPEED_THRESH = 0.12       # m/frame

    # ── Heading computation ──
    HEADING_VEL_FRAMES = 7    # frames to compute displacement over
    HEADING_DISP_GUARD = 0.8  # min displacement (m) to trust heading
    FLIP_WARMUP = 10          # don't flip-check until this many heading updates

    # ── Smoothing ──
    ORI_EMA_ALPHA = 0.25      # 25% new, 75% prior (conservative)
    MAX_JUMP_RAD = math.radians(60.0)  # reject jumps > 60°

    def __init__(self):
        self._raw_history: dict[int, deque] = {}
        self._med_history: dict[int, deque] = {}
        self._ori_ema: dict[int, float] = {}
        self._speed_ema: dict[int, float] = {}
        self._raw_pos_buf: dict[int, deque] = {}
        self._heading_count: dict[int, int] = {}  # how many valid headings computed

    def update(self, track_id: int, frame_idx: int, pos_3d: list[float]) -> dict:
        raw_pos = np.array(pos_3d, dtype=np.float64)

        # ── Store raw position for speed computation ──
        if track_id not in self._raw_history:
            self._raw_history[track_id] = deque(maxlen=self.HISTORY_LEN)
        self._raw_history[track_id].append((frame_idx, raw_pos.copy()))

        # ── Median-filtered position for heading (damps depth jitter) ──
        if track_id not in self._raw_pos_buf:
            self._raw_pos_buf[track_id] = deque(maxlen=5)
        self._raw_pos_buf[track_id].append(raw_pos.copy())

        buf = np.array(list(self._raw_pos_buf[track_id]))
        med_pos = np.median(buf, axis=0)

        if track_id not in self._med_history:
            self._med_history[track_id] = deque(maxlen=self.HISTORY_LEN)
        self._med_history[track_id].append((frame_idx, med_pos.copy()))

        raw_hist = self._raw_history[track_id]
        med_hist = self._med_history[track_id]

        if len(raw_hist) < 4:
            return {
                "orientation_deg": 0.0,
                "orientation_source": "unknown",
                "is_moving": False,
                "velocity_ego": [0.0, 0.0, 0.0],
            }

        # ── Speed from RAW positions ──
        n_speed = min(5, len(raw_hist))
        raw_positions = np.array([p for _, p in list(raw_hist)[-n_speed:]])
        raw_diffs = np.diff(raw_positions, axis=0)
        raw_velocity = raw_diffs.mean(axis=0)
        speed = float(np.linalg.norm(raw_velocity[:2]))

        if track_id not in self._speed_ema:
            self._speed_ema[track_id] = speed
        self._speed_ema[track_id] = (
            self.SPEED_EMA_ALPHA * speed
            + (1.0 - self.SPEED_EMA_ALPHA) * self._speed_ema[track_id]
        )

        is_moving = self._speed_ema[track_id] > self.SPEED_THRESH

        # ── Heading from MEDIAN-filtered positions ──
        n_head = min(self.HEADING_VEL_FRAMES, len(med_hist))
        med_positions = np.array([p for _, p in list(med_hist)[-n_head:]])

        # Use TOTAL displacement vector (start → end), not per-frame average.
        # This cancels out jitter: noise in intermediate frames washes out.
        disp = med_positions[-1] - med_positions[0]
        total_disp = float(np.linalg.norm(disp[:2]))
        trust_heading = total_disp > self.HEADING_DISP_GUARD

        if is_moving and trust_heading:
            # atan2(x, y) where Y=forward in ego frame
            heading_rad = math.atan2(disp[0], disp[1])
            heading_rad = _normalize_rad(heading_rad)

            hcount = self._heading_count.get(track_id, 0)

            # ── Flip detection: only after heading is well-established ──
            if hcount >= self.FLIP_WARMUP and track_id in self._ori_ema:
                prev = self._ori_ema[track_id]
                diff_normal = abs(_angle_diff(heading_rad, prev))
                flipped = _normalize_rad(heading_rad + math.pi)
                diff_flipped = abs(_angle_diff(flipped, prev))
                if diff_flipped < diff_normal * 0.7:  # flipped must be clearly better
                    heading_rad = flipped

            # ── Implausible jump rejection ──
            if track_id in self._ori_ema:
                prev = self._ori_ema[track_id]
                jump = abs(_angle_diff(heading_rad, prev))
                if jump > self.MAX_JUMP_RAD:
                    heading_rad = prev  # hold previous

            # ── EMA blend ──
            if track_id not in self._ori_ema:
                self._ori_ema[track_id] = heading_rad
            else:
                prev = self._ori_ema[track_id]
                diff = _angle_diff(heading_rad, prev)
                self._ori_ema[track_id] = _normalize_rad(
                    prev + self.ORI_EMA_ALPHA * diff
                )

            self._heading_count[track_id] = hcount + 1
            heading_rad = self._ori_ema[track_id]
            source = "velocity"
        else:
            heading_rad = self._ori_ema.get(track_id, 0.0)
            source = "velocity" if track_id in self._ori_ema else "unknown"

        heading_deg = math.degrees(heading_rad)

        return {
            "orientation_deg": round(heading_deg, 1),
            "orientation_source": source,
            "is_moving": is_moving,
            "velocity_ego": raw_velocity.tolist(),
        }

    def reset(self):
        self._raw_history.clear()
        self._med_history.clear()
        self._ori_ema.clear()
        self._speed_ema.clear()
        self._raw_pos_buf.clear()
        self._heading_count.clear()

    def prune(self, active_ids: set[int]):
        for tid in list(self._raw_history.keys()):
            if tid not in active_ids:
                self._raw_history.pop(tid, None)
                self._med_history.pop(tid, None)
                self._ori_ema.pop(tid, None)
                self._speed_ema.pop(tid, None)
                self._raw_pos_buf.pop(tid, None)
                self._heading_count.pop(tid, None)
