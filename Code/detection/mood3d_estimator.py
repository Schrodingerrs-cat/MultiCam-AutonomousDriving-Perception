"""
mood3d_estimator.py — Wrapper for 3D-MOOD (ICCV 2025) monocular 3D detector.

Replaces DeepBox. Loads a Swin-T backbone checkpoint from HuggingFace and
runs per-vehicle 3D bounding-box estimation.  Falls back gracefully when
vis4d / opendet3d are not installed.

HuggingFace repo : RoyYang0714/3D-MOOD
Checkpoint file  : gdino3d_swin-t_120e_omni3d_699f69.pt
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Average KITTI dimensions fallback  [w, l, h]  (metres)
# ---------------------------------------------------------------------------
KITTI_DIMS: dict[str, list[float]] = {
    "car":        [1.76, 4.08, 1.53],
    "truck":      [2.63, 10.5, 3.07],
    "bus":        [2.80, 12.0, 3.25],
    "motorcycle": [0.73, 2.11, 1.44],
    "bicycle":    [0.58, 1.73, 1.25],
}

_CKPT_URL = (
    "https://huggingface.co/RoyYang0714/3D-MOOD/resolve/main/"
    "gdino3d_swin-t_120e_omni3d_699f69.pt"
)
_CKPT_CACHE = Path.home() / ".cache" / "einsteinvision" / "mood3d" / "gdino3d_swin-t_120e_omni3d_699f69.pt"


# ---------------------------------------------------------------------------
# Availability check (module-level, cached)
# ---------------------------------------------------------------------------
_vis4d_available: Optional[bool] = None
_opendet3d_available: Optional[bool] = None


def _check_availability() -> tuple[bool, bool]:
    global _vis4d_available, _opendet3d_available
    if _vis4d_available is None:
        try:
            import vis4d  # noqa: F401
            _vis4d_available = True
        except ImportError:
            _vis4d_available = False
    if _opendet3d_available is None:
        try:
            import opendet3d  # noqa: F401
            _opendet3d_available = True
        except ImportError:
            _opendet3d_available = False
    return _vis4d_available, _opendet3d_available


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------

class Mood3DEstimator:
    """
    Monocular 3-D object estimator backed by 3D-MOOD (ICCV 2025).

    Usage
    -----
    estimator = Mood3DEstimator()
    results   = estimator.estimate_batch(frame_bgr, detections, K=K)
    # results: {track_id: {"center_3d": [...], "dimensions": [...], "yaw_deg": float}}
    """

    def __init__(self) -> None:
        self._model = None          # lazy-loaded
        self._loaded: bool = False
        self._track_history: dict[int, list[dict]] = {}

        vis4d_ok, opendet3d_ok = _check_availability()
        if not vis4d_ok or not opendet3d_ok:
            print(
                "[Mood3DEstimator] vis4d / opendet3d not installed — "
                "3D-MOOD inference disabled.  Falling back to depth-only reconstruction."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the 3D-MOOD backend is importable."""
        vis4d_ok, opendet3d_ok = _check_availability()
        return vis4d_ok and opendet3d_ok

    def estimate(
        self,
        frame: np.ndarray,
        bbox: list | tuple,
        label: str = "car",
        K: Optional[np.ndarray] = None,
    ) -> dict | None:
        """
        Estimate 3D bounding box for a single vehicle crop.

        Parameters
        ----------
        frame : np.ndarray  — full BGR frame (H×W×3)
        bbox  : (x1, y1, x2, y2) in pixel coordinates
        label : COCO class name string
        K     : 3×3 camera intrinsic matrix

        Returns
        -------
        dict with keys center_3d, dimensions, yaw_deg  — or None on failure.
        """
        if not self.is_available():
            return None

        if not self._loaded:
            success = self._load_model()
            if not success:
                return None

        try:
            return self._run_inference(frame, [bbox], [label], K)[0]
        except Exception as exc:  # noqa: BLE001
            print(f"[Mood3DEstimator] estimate() failed: {exc}")
            return None

    def estimate_batch(
        self,
        frame: np.ndarray,
        detections: list[dict],
        K: Optional[np.ndarray] = None,
    ) -> dict[int, dict]:
        """
        Estimate 3D bounding boxes for a batch of tracked detections.

        Parameters
        ----------
        frame      : full BGR frame (H×W×3)
        detections : list of dicts with keys:
                       - "track_id" : int
                       - "bbox"     : (x1, y1, x2, y2)
                       - "label"    : str  (COCO class name)
        K          : 3×3 camera intrinsic matrix

        Returns
        -------
        dict mapping track_id → result_dict
        """
        if not detections:
            return {}

        if not self.is_available():
            return {}

        if not self._loaded:
            success = self._load_model()
            if not success:
                return {}

        bboxes = [d["bbox"] for d in detections]
        labels = [d.get("label", "car") for d in detections]
        track_ids = [d["track_id"] for d in detections]

        try:
            results_list = self._run_inference(frame, bboxes, labels, K)
        except Exception as exc:  # noqa: BLE001
            print(f"[Mood3DEstimator] estimate_batch() failed: {exc}")
            return {}

        out: dict[int, dict] = {}
        for tid, result in zip(track_ids, results_list):
            if result is not None:
                out[tid] = result
                # store in history for smoothing if desired
                self._track_history.setdefault(tid, []).append(result)
        return out

    def prune(self, active_ids: set) -> None:
        """Remove track history for IDs no longer active."""
        stale = [tid for tid in self._track_history if tid not in active_ids]
        for tid in stale:
            del self._track_history[tid]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        """
        Lazy-load the 3D-MOOD model.  Tries two strategies:
          1. vis4d pipeline config loader
          2. Direct checkpoint load onto a bare Swin-T backbone
        Sets self._model and self._loaded.
        """
        self._loaded = True   # set True even on failure to avoid repeated attempts

        # --- ensure checkpoint is present ---
        ckpt_path = self._ensure_checkpoint()
        if ckpt_path is None:
            print("[Mood3DEstimator] Could not download/locate checkpoint — disabling.")
            return False

        # Strategy 1: vis4d pipeline
        try:
            model = self._load_via_vis4d_pipeline(ckpt_path)
            if model is not None:
                self._model = model
                print("[Mood3DEstimator] Model loaded via vis4d pipeline.")
                return True
        except Exception as exc:  # noqa: BLE001
            print(f"[Mood3DEstimator] vis4d pipeline load failed ({exc}), trying direct load.")

        # Strategy 2: direct checkpoint
        try:
            model = self._load_direct(ckpt_path)
            if model is not None:
                self._model = model
                print("[Mood3DEstimator] Model loaded via direct checkpoint.")
                return True
        except Exception as exc:  # noqa: BLE001
            print(f"[Mood3DEstimator] Direct checkpoint load also failed ({exc}).")

        print("[Mood3DEstimator] All load strategies failed — inference disabled.")
        return False

    def _ensure_checkpoint(self) -> Optional[Path]:
        """Download the HuggingFace checkpoint if not already cached."""
        if _CKPT_CACHE.exists():
            return _CKPT_CACHE

        # Try huggingface_hub first (clean API)
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            dest = hf_hub_download(
                repo_id="RoyYang0714/3D-MOOD",
                filename="gdino3d_swin-t_120e_omni3d_699f69.pt",
                cache_dir=str(_CKPT_CACHE.parent),
            )
            return Path(dest)
        except Exception:  # noqa: BLE001
            pass

        # Fallback: raw urllib
        try:
            import urllib.request

            _CKPT_CACHE.parent.mkdir(parents=True, exist_ok=True)
            print(f"[Mood3DEstimator] Downloading checkpoint from {_CKPT_URL} …")
            urllib.request.urlretrieve(_CKPT_URL, str(_CKPT_CACHE))
            return _CKPT_CACHE
        except Exception as exc:  # noqa: BLE001
            print(f"[Mood3DEstimator] Checkpoint download failed: {exc}")
            return None

    @staticmethod
    def _load_via_vis4d_pipeline(ckpt_path: Path):
        """
        Build GroundingDINO3D via opendet3d zoo config + vis4d instantiation,
        then load the 3D-MOOD checkpoint weights.
        """
        import torch  # type: ignore

        from opendet3d.zoo.gdino3d.base.model import (  # type: ignore
            get_gdino3d_hyperparams_cfg,
            get_gdino3d_swin_tiny_cfg,
        )
        from vis4d.config import instantiate_classes  # type: ignore

        params = get_gdino3d_hyperparams_cfg()
        model_cfg, _box_coder = get_gdino3d_swin_tiny_cfg(
            params=params, pretrained=None, use_checkpoint=False,
        )
        model = instantiate_classes(model_cfg)

        # Load 3D-MOOD checkpoint
        state = torch.load(str(ckpt_path), map_location="cpu")
        weights = state.get("state_dict", state.get("model", state))
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing:
            print(f"[Mood3DEstimator] {len(missing)} missing keys (expected for backbone init)")
        model.eval()
        return model

    @staticmethod
    def _load_direct(ckpt_path: Path):
        """
        Fallback: attempt direct checkpoint load.
        Returns model object or None.
        """
        import torch  # type: ignore

        try:
            state = torch.load(str(ckpt_path), map_location="cpu")
            if callable(state):
                return state
            return None
        except Exception:  # noqa: BLE001
            return None

    def _run_inference(
        self,
        frame: np.ndarray,
        bboxes: list,
        labels: list[str],
        K: Optional[np.ndarray],
    ) -> list[dict | None]:
        """
        Run the loaded model and parse outputs.

        Returns a list (same length as bboxes) of result dicts or None.
        """
        import torch  # type: ignore

        if self._model is None:
            return [None] * len(bboxes)

        device = next(
            (p.device for p in getattr(self._model, "parameters", lambda: [])()),
            torch.device("cpu"),
        )

        # Build intrinsics tensor
        if K is not None:
            K_tensor = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            # sensible defaults if K not provided
            h, w = frame.shape[:2]
            fx = fy = max(h, w) * 1.2
            cx, cy = w / 2.0, h / 2.0
            K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            K_tensor = torch.tensor(K_np, dtype=torch.float32, device=device).unsqueeze(0)

        # Convert frame to tensor  (1, 3, H, W), RGB, float32 0-1
        import cv2  # type: ignore
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = (
            torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        )

        bboxes_tensor = torch.tensor(
            [[float(x) for x in bb] for bb in bboxes], dtype=torch.float32, device=device
        )

        results: list[dict | None] = []
        h, w = frame.shape[:2]

        with torch.no_grad():
            try:
                out = self._model(
                    images=img_tensor,
                    input_hw=[(h, w)],
                    intrinsics=K_tensor,
                    boxes2d=bboxes_tensor.unsqueeze(0),
                )
                results = self._parse_output(out, labels)
            except Exception as exc:  # noqa: BLE001
                print(f"[Mood3DEstimator] Inference forward pass failed: {exc}")
                results = [None] * len(bboxes)

        # Fill None entries with KITTI-dim fallback using bbox depth estimate
        for i, (res, bbox, label) in enumerate(zip(results, bboxes, labels)):
            if res is None:
                results[i] = self._fallback_from_bbox(bbox, label, K, frame.shape)

        return results

    @staticmethod
    def _parse_output(out, labels: list[str]) -> list[dict | None]:
        """
        Parse model output (Det3DOut namedtuple or dict) into result dicts.
        Det3DOut fields: boxes, boxes3d, scores, class_ids, depth_maps, categories
        boxes3d format: [x, y, z, w, l, h, yaw] per detection
        """
        import torch  # type: ignore

        results: list[dict | None] = []

        # Handle Det3DOut namedtuple (has _fields)
        if hasattr(out, '_fields') and hasattr(out, 'boxes3d'):
            boxes3d = out.boxes3d
            if boxes3d is None:
                return [None] * len(labels)

            # boxes3d may be batched — take first batch
            if isinstance(boxes3d, (list, tuple)):
                boxes3d = boxes3d[0]
            if isinstance(boxes3d, torch.Tensor):
                if boxes3d.dim() == 3:
                    boxes3d = boxes3d[0]  # remove batch dim
                boxes3d = boxes3d.detach().cpu().numpy()

            n = min(len(labels), len(boxes3d))
            for i in range(n):
                b = boxes3d[i]
                # Expected format: [x, y, z, w, l, h, yaw] or similar
                if len(b) >= 7:
                    results.append({
                        "center_3d":  [float(b[0]), float(b[1]), float(b[2])],
                        "dimensions": [float(b[3]), float(b[4]), float(b[5])],
                        "yaw_deg":    math.degrees(float(b[6])),
                    })
                elif len(b) >= 3:
                    # At minimum we get a center
                    results.append({
                        "center_3d":  [float(b[j]) for j in range(3)],
                        "dimensions": KITTI_DIMS.get(labels[i].lower(), KITTI_DIMS["car"]),
                        "yaw_deg":    0.0,
                    })
                else:
                    results.append(None)
            while len(results) < len(labels):
                results.append(None)
            return results

        # Fallback: dict-style output
        if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], dict):
            preds = out[0]
        elif isinstance(out, dict):
            preds = out
        else:
            return [None] * len(labels)

        centers   = preds.get("center_3d",   preds.get("centers_3d",   None))
        dims      = preds.get("dimensions",   preds.get("dims",         None))
        yaws_raw  = preds.get("yaw",          preds.get("rotation_y",   None))

        if centers is None or dims is None or yaws_raw is None:
            return [None] * len(labels)

        def _to_np(t):
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy()
            return np.asarray(t)

        centers_np = _to_np(centers)
        dims_np    = _to_np(dims)
        yaws_np    = _to_np(yaws_raw)

        n = min(len(labels), len(centers_np))
        for i in range(n):
            yaw_rad = float(yaws_np[i]) if yaws_np.ndim == 1 else float(yaws_np[i][0])
            results.append({
                "center_3d":  centers_np[i].tolist(),
                "dimensions": dims_np[i].tolist(),
                "yaw_deg":    math.degrees(yaw_rad),
            })

        while len(results) < len(labels):
            results.append(None)

        return results

    @staticmethod
    def _fallback_from_bbox(
        bbox: list | tuple,
        label: str,
        K: Optional[np.ndarray],
        frame_shape: tuple,
    ) -> dict:
        """
        Rough 3D estimate from 2D bbox + average KITTI dimensions.
        Used when model inference fails for a specific detection.
        """
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0
        h_px  = max(y2 - y1, 1.0)

        key = label.lower()
        w, l, h = KITTI_DIMS.get(key, KITTI_DIMS["car"])

        # Estimate depth from known object height and pixel height
        if K is not None:
            fy = float(K[1, 1])
        else:
            fy = max(frame_shape[:2]) * 1.2

        depth = (fy * h) / h_px  # metres

        # Back-project bbox centre to 3-D camera-frame point
        if K is not None:
            fx = float(K[0, 0])
            ppx = float(K[0, 2])
            ppy = float(K[1, 2])
        else:
            fx  = fy
            ppx = frame_shape[1] / 2.0
            ppy = frame_shape[0] / 2.0

        X = (cx_px - ppx) * depth / fx
        Y = (cy_px - ppy) * depth / fy
        Z = depth

        return {
            "center_3d":  [X, Y, Z],
            "dimensions": [w, l, h],
            "yaw_deg":    0.0,     # unknown — assume aligned with camera
        }
