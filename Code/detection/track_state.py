"""
Per-track real-time state smoother.
Runs during detection pipeline, one update per frame per track.
Smooths: 3D position, velocity, orientation.
No post-processing needed — data is already smooth when written to JSON.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

# ── Smoothing parameters ─────────────────────────────────────────────────────
POS_ALPHA    = 0.35   # position EMA — lower = smoother, more lag
VEL_ALPHA    = 0.25   # velocity EMA — very smooth
ORI_ALPHA    = 0.20   # orientation EMA — smooth but responsive
STALE_FRAMES = 45     # frames before clearing a track's state


@dataclass
class TrackState:
    """Per-track smoothed state."""
    # Position (ego frame, metres)
    pos: np.ndarray                                       # shape (3,) — smoothed [x, y, z]
    vel: np.ndarray                                       # shape (3,) — smoothed velocity m/frame

    # Orientation
    orientation: float                                    # smoothed heading degrees [-180, 180]

    # Subclass vote buffer
    subclass_votes: deque = field(default_factory=lambda: deque(maxlen=15))
    last_seen_frame: int = 0

    # Raw values (debugging)
    raw_pos_last: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_ori_last: float = 0.0


def _normalize(angle: float) -> float:
    """Wrap angle to [-180, 180]."""
    return ((angle + 180.0) % 360.0) - 180.0


class TrackStateManager:
    """
    Manages a dict of per-track states keyed by track_id (int).
    Call ``update()`` once per frame per detected track — the returned
    state already contains smoothed values ready for JSON output.
    """

    def __init__(self):
        self._states: dict[int, TrackState] = {}

    # ── core ──────────────────────────────────────────────────────────────

    def update(self,
               track_id: int,
               frame_idx: int,
               raw_pos: np.ndarray,
               raw_ori: float,
               raw_subclass: str | None = None) -> TrackState:
        """
        Call once per frame per detected track.
        Returns the smoothed state to use for this frame's JSON output.
        """
        state = self._states.get(track_id)

        # ── First sighting or stale track → initialise ────────────────
        if state is None or (frame_idx - state.last_seen_frame) > STALE_FRAMES:
            state = TrackState(
                pos=raw_pos.copy(),
                vel=np.zeros(3, dtype=np.float64),
                orientation=_normalize(raw_ori),
                subclass_votes=deque(maxlen=15),
                last_seen_frame=frame_idx,
                raw_pos_last=raw_pos.copy(),
                raw_ori_last=raw_ori,
            )
            if raw_subclass is not None:
                state.subclass_votes.append(raw_subclass)
            self._states[track_id] = state
            return state

        # ── Position EMA ──────────────────────────────────────────────
        new_vel = raw_pos - state.raw_pos_last   # raw→raw displacement (unbiased)
        state.vel = VEL_ALPHA * new_vel + (1.0 - VEL_ALPHA) * state.vel
        state.pos = POS_ALPHA * raw_pos + (1.0 - POS_ALPHA) * state.pos

        # ── Orientation EMA with 180-degree flip resolution ─────────────
        normalized_ori = _normalize(raw_ori)
        diff = _normalize(normalized_ori - state.orientation)
        # Resolve 180-degree flip (prevents sudden orientation jumps)
        if abs(diff) > 90.0:
            normalized_ori = _normalize(normalized_ori + 180.0)
            diff = _normalize(normalized_ori - state.orientation)
        state.orientation = _normalize(state.orientation + ORI_ALPHA * diff)

        # ── Subclass majority vote ────────────────────────────────────
        if raw_subclass is not None:
            state.subclass_votes.append(raw_subclass)

        # ── Book-keeping ──────────────────────────────────────────────
        state.raw_pos_last = raw_pos.copy()
        state.raw_ori_last = raw_ori
        state.last_seen_frame = frame_idx

        return state

    # ── orientation helpers ───────────────────────────────────────────────

    def get_velocity_heading(self, track_id: int) -> float | None:
        """
        Returns heading from velocity vector if speed > 0.3 m/frame.
        arctan2(vel_x, vel_y) in degrees (ego frame: Y = forward).
        Returns None if nearly stationary.
        """
        state = self._states.get(track_id)
        if state is None:
            return None
        speed = float(np.linalg.norm(state.vel))
        if speed < 0.3:
            return None
        return float(np.degrees(np.arctan2(state.vel[0], state.vel[1])))

    def get_display_orientation(self, track_id: int) -> float:
        """
        Best orientation for display:
        - velocity heading if speed > 0.5 m/frame (more reliable when moving)
        - smoothed orientation if stationary
        """
        vel_heading = self.get_velocity_heading(track_id)
        state = self._states.get(track_id)
        if state is None:
            return 0.0
        if vel_heading is not None and float(np.linalg.norm(state.vel)) > 0.5:
            return vel_heading
        return state.orientation

    def get_stable_subclass(self, track_id: int) -> str | None:
        """
        Returns majority-voted subclass if at least 3 votes agree,
        otherwise the most recent subclass.
        """
        state = self._states.get(track_id)
        if state is None or len(state.subclass_votes) == 0:
            return None
        majority = Counter(state.subclass_votes).most_common(1)[0]
        if majority[1] >= 7:
            return majority[0]
        # Not enough votes for strong majority — return most common still
        # (better than flickering with most recent)
        return majority[0]

    # ── lifecycle ─────────────────────────────────────────────────────────

    def reset(self, track_id: int | None = None):
        """Clear one track or all tracks (call between sequences)."""
        if track_id is not None:
            self._states.pop(track_id, None)
        else:
            self._states.clear()

    def prune(self, active_ids: set[int]):
        """Remove states for tracks no longer active."""
        stale = [tid for tid in self._states if tid not in active_ids]
        for tid in stale:
            del self._states[tid]
