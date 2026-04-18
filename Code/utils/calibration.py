"""
Camera Calibration Loader — Phase 3 clean rewrite.
Same interface as Phase 2 calibration module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

_PERCAM_INTRINSICS = {
    "front": {"fx": 1594.6574, "fy": 1607.6942, "cx": 655.2961, "cy": 414.3627},
    "left":  {"fx": 1004.7543, "fy": 1004.8449, "cx": 640.8121, "cy": 484.5352},
    "right": {"fx": 997.2817,  "fy": 997.5479,  "cx": 626.8848, "cy": 454.9555},
    "back":  {"fx": 496.9284,  "fy": 496.9284,  "cx": 643.8781, "cy": 476.7651},
}


def _make_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


class CameraCalibration:
    def __init__(self, K: Optional[np.ndarray] = None, D: Optional[np.ndarray] = None):
        if K is None:
            K = _make_K(cfg.DEFAULT_FX, cfg.DEFAULT_FY, cfg.DEFAULT_CX, cfg.DEFAULT_CY)
        if D is None:
            D = np.zeros((1, 5), dtype=np.float64)
        self.K = K
        self.D = D
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self.K_inv = np.linalg.inv(K)

        self.per_cam_K: dict[str, np.ndarray] = {}
        self.per_cam_K_inv: dict[str, np.ndarray] = {}
        for cam_id, intr in _PERCAM_INTRINSICS.items():
            Kc = _make_K(intr["fx"], intr["fy"], intr["cx"], intr["cy"])
            self.per_cam_K[cam_id] = Kc
            self.per_cam_K_inv[cam_id] = np.linalg.inv(Kc)

    def get_K(self, cam_id: str = "front") -> np.ndarray:
        return self.per_cam_K.get(cam_id, self.K)

    def get_K_inv(self, cam_id: str = "front") -> np.ndarray:
        return self.per_cam_K_inv.get(cam_id, self.K_inv)

    def get_extrinsics(self, cam_id: str) -> tuple[np.ndarray, np.ndarray]:
        ext = cfg.CAMERA_EXTRINSICS.get(cam_id, cfg.CAMERA_EXTRINSICS["front"])
        pos = np.array(ext["position"], dtype=np.float64)
        yaw   = np.radians(ext["yaw_deg"])
        pitch = np.radians(ext.get("pitch_deg", 0.0))

        Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                        [           0, 1,           0],
                        [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float64)
        Rp = np.array([[1,            0,             0],
                        [0,  np.cos(pitch), np.sin(pitch)],
                        [0, -np.sin(pitch), np.cos(pitch)]], dtype=np.float64)
        R_basis = np.array([[1,  0,  0],
                            [0,  0,  1],
                            [0, -1,  0]], dtype=np.float64)
        R_cam2ego = Ry @ Rp @ R_basis
        t_cam2ego = pos
        return R_cam2ego, t_cam2ego

    @classmethod
    def from_file(cls, calib_path: Path) -> "CameraCalibration":
        p = Path(calib_path)
        if not p.exists():
            return cls()
        if p.suffix == ".npz":
            data = np.load(p)
            K = data["K"] if "K" in data else None
            D = data["D"] if "D" in data else None
            return cls(K, D)
        if p.suffix in (".txt", ".yaml", ".yml"):
            try:
                mat = np.loadtxt(p)
                if mat.shape == (3, 3):
                    return cls(mat.astype(np.float64))
            except Exception:
                pass
        return cls()

    def save(self, path: Path):
        np.savez(str(path), K=self.K, D=self.D)
