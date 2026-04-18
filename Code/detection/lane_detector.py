"""
Lane Detector — TwinLiteNet+ Segmentation Pipeline
====================================================

Uses TwinLiteNet+ (pretrained on BDD100K) for both:
  - Lane line segmentation (pixel mask)
  - Drivable area segmentation (pixel mask)

Then extracts individual lane instances from the segmentation mask
via connected components + polyfit.

Output per frame — list of lane dicts:
    {
        "lane_id":     int,
        "points":      [[u, v], ...],
        "type":        "solid" | "dashed" | "double_yellow" | "double_white",
        "color":       "white" | "yellow",
    }

Also detects crosswalks from horizontal stripe patterns in the lane mask.
"""

from __future__ import annotations
import sys
from collections import deque
from pathlib import Path
from typing import Optional
from argparse import Namespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

# ─── TwinLiteNet+ repo path ──────────────────────────────────────────────────
_TWINLITE_DIR = str(Path(__file__).resolve().parent.parent.parent / "Code/TwinLiteNetPlus-main")
_TWINLITE_WEIGHT = str(Path(_TWINLITE_DIR) / "large.pth")
_TWINLITE_CONFIG = "large"
_IMG_SIZE = 640


# ─── Letterbox (from TwinLiteNet+ demoDataset.py) ───────────────────────────

def _letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize image to a 32-pixel-multiple rectangle with padding."""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


# ─── Main LaneDetector class ─────────────────────────────────────────────────

class LaneDetector:
    """
    TwinLiteNet+-based lane & drivable area detector.

    Loads the TwinLiteNet+ Large model once and runs inference per frame.
    Returns individual lane instances extracted from the segmentation mask.
    """

    # Lane extraction parameters
    _MIN_LANE_PX = 150          # min pixels for a connected component to count as a lane
    _MIN_LANE_HEIGHT_FRAC = 0.10  # lane must span at least 10% of frame height

    # Temporal smoothing window for lane type classification
    _TYPE_HISTORY_LEN = 15  # number of frames to keep per lane slot

    # Frame skip: only recompute lane geometry every N frames for stability
    _FRAME_SKIP = 3

    # Lane count persistence: require N consecutive frames with different
    # lane count before accepting the change (prevents flickering 2↔3 lanes)
    _LANE_COUNT_CONFIRM = 4

    def __init__(self):
        self._model = None
        self._device = None
        self._half = True

        # Cache for last-frame results
        self._last_da_mask: Optional[np.ndarray] = None
        self._last_ll_mask: Optional[np.ndarray] = None

        # Temporal smoothing: per-slot deque of recent type classifications
        # Keyed by lane slot index (left-to-right order)
        self._type_history: dict[int, deque] = {}

        # Frame skip state
        self._frame_counter: int = 0
        self._cached_lanes: list[dict] = []

        # Lane count persistence state
        self._stable_lane_count: int = 0
        self._pending_lane_count: int = 0
        self._pending_count_frames: int = 0

        # IPM matrix cache (computed on first call to get_ipm_matrices)
        self._ipm_cache: tuple | None = None

        # Crosswalk detection cache
        self._last_crosswalks: list[dict] = []

        # Lane dropout carryforward counter
        self._dropout_count: int = 0

        # Classical pipeline: track unmatched classical lanes across frames
        self._classical_unmatched: dict[int, int] = {}  # approx_x → frame_count

        self._load_model()

    def _load_model(self):
        """Load TwinLiteNet+ model with pretrained weights."""
        try:
            import torch
        except ImportError:
            print("[LaneDetector] torch not available — TwinLiteNet+ disabled")
            return

        print("[LaneDetector] Loading TwinLiteNet+ Large …")
        try:
            # Add TwinLiteNet+ to path temporarily
            saved_path = sys.path.copy()
            saved_mods = {}
            # Remove conflicting modules
            for k in list(sys.modules.keys()):
                if k == 'model' or k.startswith('model.'):
                    saved_mods[k] = sys.modules.pop(k)

            sys.path.insert(0, _TWINLITE_DIR)
            try:
                from model.model import TwinLiteNetPlus

                args = Namespace(config=_TWINLITE_CONFIG)
                model = TwinLiteNetPlus(args)

                self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = model.to(self._device)
                if self._half and self._device == 'cuda':
                    model.half()

                state_dict = torch.load(_TWINLITE_WEIGHT,
                                        map_location=self._device)
                model.load_state_dict(state_dict)
                model.eval()
                self._model = model

                # Warmup
                dummy = torch.zeros((1, 3, _IMG_SIZE, _IMG_SIZE),
                                    device=self._device)
                if self._half and self._device == 'cuda':
                    dummy = dummy.half()
                with torch.no_grad():
                    _ = model(dummy)

                print("[LaneDetector] TwinLiteNet+ Large loaded OK")
            finally:
                # Restore sys.path and modules
                sys.path = saved_path
                for k in list(sys.modules.keys()):
                    if k == 'model' or k.startswith('model.'):
                        sys.modules.pop(k, None)
                for k, v in saved_mods.items():
                    sys.modules[k] = v

        except Exception as e:
            print(f"[LaneDetector] TwinLiteNet+ failed to load: {e}")
            import traceback
            traceback.print_exc()
            self._model = None

    # ── Raw Inference ─────────────────────────────────────────────────────

    def _infer(self, frame_bgr: np.ndarray):
        """
        Run TwinLiteNet+ on a BGR frame.
        Returns (da_mask, ll_mask) as uint8 arrays at original resolution.
          da_mask: 0/1 drivable area
          ll_mask: 0/255 lane line pixels
        """
        import torch

        h0, w0 = frame_bgr.shape[:2]

        # Letterbox resize
        img_lb, ratio, (pad_w, pad_h) = _letterbox(frame_bgr, _IMG_SIZE)
        h_lb, w_lb = img_lb.shape[:2]

        # To tensor: BGR→RGB, HWC→CHW, normalize
        img_rgb = img_lb[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        img_rgb = np.ascontiguousarray(img_rgb)
        tensor = torch.from_numpy(img_rgb).unsqueeze(0).to(self._device)
        if self._half and self._device == 'cuda':
            tensor = tensor.half() / 255.0
        else:
            tensor = tensor.float() / 255.0

        pad_h_int = int(pad_h)
        pad_w_int = int(pad_w)

        with torch.no_grad():
            da_out, ll_out = self._model(tensor)

        # Remove padding and resize back to original
        da_pred = da_out[:, :, pad_h_int:(h_lb - pad_h_int),
                         pad_w_int:(w_lb - pad_w_int)]
        da_pred = torch.nn.functional.interpolate(
            da_pred, size=(h0, w0), mode='bilinear', align_corners=False)
        _, da_mask = torch.max(da_pred, 1)
        da_mask = da_mask.squeeze().cpu().numpy().astype(np.uint8)

        ll_pred = ll_out[:, :, pad_h_int:(h_lb - pad_h_int),
                         pad_w_int:(w_lb - pad_w_int)]
        ll_pred = torch.nn.functional.interpolate(
            ll_pred, size=(h0, w0), mode='bilinear', align_corners=False)
        _, ll_mask = torch.max(ll_pred, 1)
        ll_mask = ll_mask.squeeze().cpu().numpy().astype(np.uint8) * 255

        # DEBUG: save first frame's masks for visual inspection
        if not hasattr(self, '_debug_saved') or not self._debug_saved:
            self._debug_saved = True
            import os
            dbg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'debug_masks')
            os.makedirs(dbg_dir, exist_ok=True)
            cv2.imwrite(os.path.join(dbg_dir, 'da_mask.png'), da_mask * 255)
            cv2.imwrite(os.path.join(dbg_dir, 'll_mask.png'), ll_mask)
            # Also save swapped version for comparison
            cv2.imwrite(os.path.join(dbg_dir, 'll_mask_SWAPPED.png'), da_mask * 255)
            cv2.imwrite(os.path.join(dbg_dir, 'da_mask_SWAPPED.png'), ll_mask)
            print(f"[LaneDetector DEBUG] Saved mask debug images to {dbg_dir}")
            print(f"[LaneDetector DEBUG] da_mask: shape={da_mask.shape} nonzero={np.count_nonzero(da_mask)} unique={np.unique(da_mask)}")
            print(f"[LaneDetector DEBUG] ll_mask: shape={ll_mask.shape} nonzero={np.count_nonzero(ll_mask)} unique={np.unique(ll_mask)}")

        return da_mask, ll_mask

    # ── Public API: IPM Matrices (for ground arrow detection) ──────────────

    def get_ipm_matrices(self, K=None, R=None, t=None,
                         frame_shape: tuple = (720, 1280)):
        """
        Return (M, M_inv) for inverse perspective mapping.

        If camera intrinsics K and extrinsics (R, t) are provided,
        computes a proper IPM using the calibration. Otherwise returns
        (None, None) — caller should not attempt ground arrow detection
        without calibration since hardcoded points would be wrong for
        different camera setups.

        Args:
            K: 3x3 camera intrinsic matrix (optional)
            R: 3x3 rotation matrix camera→ego (optional)
            t: (3,) translation vector camera→ego (optional)
            frame_shape: (H, W) of the input frames.

        Returns:
            (M, M_inv) or (None, None)
        """
        if self._ipm_cache is not None:
            return self._ipm_cache

        if K is None:
            return None, None

        H, W = frame_shape[:2]
        if H < 100 or W < 100:
            return None, None

        # Compute IPM from calibration: project 4 ground-plane points
        # at known world coordinates through the camera model, then
        # compute the homography to map that trapezoid to a rectangle.
        #
        # Ground points in ego frame (x=lateral, y=forward, z=0):
        #   near-left, near-right, far-right, far-left
        ground_ego = np.array([
            [-3.0,  5.0, 0.0],   # near-left
            [ 3.0,  5.0, 0.0],   # near-right
            [ 3.0, 25.0, 0.0],   # far-right
            [-3.0, 25.0, 0.0],   # far-left
        ], dtype=np.float64)

        try:
            # Project to camera frame then to pixel coords
            if R is not None and t is not None:
                R_inv = R.T
                t_inv = -R.T @ t
                pts_cam = (R_inv @ ground_ego.T).T + t_inv
            else:
                pts_cam = ground_ego

            # Perspective projection
            pts_2d = (K @ pts_cam.T).T
            pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]

            # Check all points are within frame
            if (pts_2d[:, 0].min() < -W or pts_2d[:, 0].max() > 2*W or
                pts_2d[:, 1].min() < -H or pts_2d[:, 1].max() > 2*H):
                return None, None

            src = pts_2d.astype(np.float32)
            dst = np.float32([
                [W * 0.20, H],
                [W * 0.80, H],
                [W * 0.80, 0],
                [W * 0.20, 0],
            ])

            M = cv2.getPerspectiveTransform(src, dst)
            M_inv = cv2.getPerspectiveTransform(dst, src)
            self._ipm_cache = (M, M_inv)
            return M, M_inv

        except Exception:
            return None, None

    # ── Public API: Drivable Area ─────────────────────────────────────────

    def get_drivable_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Returns drivable area mask (0/1 uint8, same size as frame)."""
        if self._model is None:
            return None
        try:
            da_mask, ll_mask = self._infer(frame)
            self._last_da_mask = da_mask
            self._last_ll_mask = ll_mask
            return da_mask
        except Exception as e:
            print(f"[LaneDetector] Drivable mask failed: {e}")
            return None

    def get_crosswalks(self) -> list[dict]:
        """Return crosswalks detected in the last frame."""
        return self._last_crosswalks

    # ── Extract Individual Lanes from Segmentation Mask ───────────────────

    def _extract_lanes_from_mask(self, ll_mask: np.ndarray,
                                  frame_bgr: np.ndarray) -> list[dict]:
        """
        Extract individual lane line instances from the binary lane mask.

        Uses morphological thinning + connected components to separate lanes,
        then fits polylines and classifies color/type.
        """
        H, W = ll_mask.shape[:2]

        # Vertical-only close: bridge tiny vertical gaps for clean connected
        # components WITHOUT closing the horizontal dash gaps we need for
        # type classification.
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        cleaned = cv2.morphologyEx(ll_mask, cv2.MORPH_CLOSE, kernel_v)

        # Connected components to find individual lane segments
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8)

        lanes = []
        if self._frame_counter <= 5:
            print(f"[LaneDetector DEBUG] _extract: n_labels={n_labels} "
                  f"cleaned_nonzero={int(np.count_nonzero(cleaned))} "
                  f"H={H} W={W}")
        for label_id in range(1, n_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < self._MIN_LANE_PX:
                if self._frame_counter <= 5:
                    print(f"[LaneDetector DEBUG]   comp {label_id}: area={area} < {self._MIN_LANE_PX} SKIP")
                continue

            # Get bounding box
            y_top = stats[label_id, cv2.CC_STAT_TOP]
            y_bot = y_top + stats[label_id, cv2.CC_STAT_HEIGHT]
            height_span = y_bot - y_top

            if height_span < H * self._MIN_LANE_HEIGHT_FRAC:
                if self._frame_counter <= 5:
                    print(f"[LaneDetector DEBUG]   comp {label_id}: area={area} height={height_span} < {H * self._MIN_LANE_HEIGHT_FRAC:.0f} SKIP")
                continue

            # Extract pixels for this component
            ys, xs = np.where(labels == label_id)

            # Fit a polyline: sample points along the lane by averaging x per y-band
            n_bands = max(10, height_span // 8)
            y_bands = np.linspace(ys.min(), ys.max(), n_bands + 1)
            pts = []
            for i in range(n_bands):
                band_mask = (ys >= y_bands[i]) & (ys < y_bands[i + 1])
                if band_mask.sum() < 2:
                    continue
                mean_x = float(np.mean(xs[band_mask]))
                mean_y = float((y_bands[i] + y_bands[i + 1]) / 2)
                pts.append([mean_x, mean_y])

            if len(pts) < 3:
                if self._frame_counter <= 5:
                    print(f"[LaneDetector DEBUG]   comp {label_id}: area={area} height={height_span} pts={len(pts)} < 3 SKIP")
                continue

            pts = np.array(pts)
            if self._frame_counter <= 5:
                print(f"[LaneDetector DEBUG]   comp {label_id}: area={area} height={height_span} pts={len(pts)} ACCEPTED")

            # Classify color (white vs yellow)
            lane_color = self._classify_color(frame_bgr, pts)

            # Classify type (solid vs dashed)
            lane_type = self._classify_type_from_image(frame_bgr, pts)

            lanes.append({
                'points': pts.tolist(),
                'type': lane_type,
                'color': lane_color,
                'area': area,
            })

        # Sort by x position (left to right)
        lanes.sort(key=lambda l: np.mean([p[0] for p in l['points']]))

        # Assign lane IDs
        for i, lane in enumerate(lanes):
            lane['lane_id'] = i

        return lanes

    # ── Color Classification ──────────────────────────────────────────────

    def _classify_color(self, frame_bgr: np.ndarray,
                         pts: np.ndarray) -> str:
        """
        Classify lane color using RELATIVE comparisons to adjacent road.

        Key insight: at night, warm street lights raise saturation and b*
        for EVERYTHING (road + lanes). Only a true yellow lane will be
        distinctly warmer/more saturated than its adjacent road surface.

        Uses HSV saturation difference (lane vs road) as primary signal,
        with LAB b* excess as secondary.
        """
        H_img, W = frame_bgr.shape[:2]
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        patch_r = 4
        road_offset = 30

        lane_b_vals = []
        road_b_vals = []
        lane_hsv_s = []
        road_hsv_s = []
        lane_hsv_h = []

        for pt in pts[::2]:
            u, v = int(round(pt[0])), int(round(pt[1]))
            if u < patch_r or u >= W - patch_r or v < patch_r or v >= H_img - patch_r:
                continue

            # Lane patches
            lane_b_vals.append(lab[v-patch_r:v+patch_r+1,
                                   u-patch_r:u+patch_r+1, 2].flatten())
            lane_hsv_s.append(hsv[v-patch_r:v+patch_r+1,
                                   u-patch_r:u+patch_r+1, 1].flatten())
            lane_hsv_h.append(hsv[v-patch_r:v+patch_r+1,
                                   u-patch_r:u+patch_r+1, 0].flatten())

            # Road patches (try both sides, pick the one farther from image edge)
            for sign in (+1, -1):
                ru = u + sign * road_offset
                if patch_r <= ru < W - patch_r:
                    road_b_vals.append(lab[v-patch_r:v+patch_r+1,
                                          ru-patch_r:ru+patch_r+1, 2].flatten())
                    road_hsv_s.append(hsv[v-patch_r:v+patch_r+1,
                                          ru-patch_r:ru+patch_r+1, 1].flatten())
                    break

        if not lane_b_vals or not road_b_vals:
            return "white"

        med_lane_b = float(np.median(np.concatenate(lane_b_vals)))
        med_road_b = float(np.median(np.concatenate(road_b_vals)))
        med_lane_s = float(np.median(np.concatenate(lane_hsv_s)))
        med_road_s = float(np.median(np.concatenate(road_hsv_s)))
        med_lane_h = float(np.median(np.concatenate(lane_hsv_h)))

        b_excess = med_lane_b - med_road_b
        s_excess = med_lane_s - med_road_s

        # Yellow requires BOTH:
        #   1. Lane is warmer than road (LAB b* excess > 2)
        #   2. Lane is more saturated than road (HSV S excess > 10)
        #   3. Hue is in yellow range (HSV H 15-35)
        if (b_excess > 2 and s_excess > 10 and 10 <= med_lane_h <= 38):
            return "yellow"

        # Strong b* excess alone (daytime, clear yellow vs gray road)
        if b_excess > 5 and 10 <= med_lane_h <= 38:
            return "yellow"

        return "white"

    # ── Type Classification ───────────────────────────────────────────────

    def _classify_type_from_image(self, frame_bgr: np.ndarray,
                                   pts: np.ndarray) -> str:
        """
        Classify solid vs dashed using two complementary methods:

        Method 1 (intensity dip analysis):
          Along the lane extent, compare smoothed lane-centre intensity to
          nearby road surface. Periodic contrast dips indicate dashes.

        Method 2 (point gap analysis):
          Check spacing between consecutive lane points. Dashed lanes
          have irregular/larger gaps because TwinLiteNet+ mask fragments
          at dash boundaries.

        Either method voting "dashed" is sufficient.
        """
        if len(pts) < 4:
            return "solid"

        # ── Method 2: Point gap analysis ──
        # TwinLiteNet+ mask tends to break between dashes, creating
        # larger gaps between consecutive detected points.
        gap_vote_dashed = False
        if len(pts) >= 5:
            # Compute vertical gaps between consecutive points
            sorted_pts = pts[pts[:, 1].argsort()]
            gaps = np.diff(sorted_pts[:, 1])
            if len(gaps) >= 4:
                median_gap = float(np.median(gaps))
                # Large gaps relative to median indicate dashes
                if median_gap > 0:
                    large_gaps = gaps > median_gap * 2.5
                    n_large = int(large_gaps.sum())
                    gap_ratio = n_large / len(gaps)
                    # If >20% of gaps are large and we have at least 2 large gaps
                    if gap_ratio > 0.20 and n_large >= 2:
                        gap_vote_dashed = True

        # ── Method 1: Intensity dip analysis ──
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        try:
            poly = np.polyfit(pts[:, 1], pts[:, 0], 1)
        except (np.linalg.LinAlgError, np.RankWarning):
            return "dashed" if gap_vote_dashed else "solid"

        y_lo = int(pts[:, 1].min())
        y_hi = int(pts[:, 1].max())
        if y_hi - y_lo < 30:
            return "dashed" if gap_vote_dashed else "solid"

        pw = 5          # patch half-width (use max for robustness)
        road_off = 40   # road sample offset

        lane_vals = []
        road_vals = []
        for y in range(y_lo, y_hi + 1):
            xc = int(np.polyval(poly, y))
            if xc < pw + road_off or xc >= W - pw - road_off:
                continue
            if y < 0 or y >= H:
                continue

            lane_vals.append(float(gray[y, xc - pw:xc + pw + 1].max()))

            # Road: sample both sides, average
            r1 = float(gray[y, xc + road_off - pw:xc + road_off + pw + 1].mean())
            r2 = float(gray[y, xc - road_off - pw:xc - road_off + pw + 1].mean())
            road_vals.append((r1 + r2) / 2)

        if len(lane_vals) < 30:
            return "dashed" if gap_vote_dashed else "solid"

        # Smooth with moderate window to capture gap/dash pattern
        win = 11
        lane_s = np.convolve(lane_vals, np.ones(win) / win, mode='valid')
        road_s = np.convolve(road_vals, np.ones(win) / win, mode='valid')
        contrast = lane_s - road_s

        if len(contrast) < 20:
            return "dashed" if gap_vote_dashed else "solid"

        p75 = np.percentile(contrast, 75)
        p90 = np.percentile(contrast, 90)

        # Not enough contrast in the bright segments to judge —
        # use p90 (peak of painted segments), not median, since dashed
        # lanes have gaps that deliberately drag the median down.
        if p90 < 2:
            return "dashed" if gap_vote_dashed else "solid"

        # Dip threshold: contrast drops below 55% of p75.
        # Dashed lanes have periodic gaps where contrast ≈ 0.
        thresh = p75 * 0.55
        is_dip = contrast < thresh
        dip_frac = float(is_dip.sum()) / len(is_dip)

        # Count distinct dip runs >= 3 rows.
        # At distance, dashes subtend fewer pixels — 7 was too strict.
        min_run = 3
        dip_runs = []
        run = 0
        for d in is_dip:
            if d:
                run += 1
            else:
                if run >= min_run:
                    dip_runs.append(run)
                run = 0
        if run >= min_run:
            dip_runs.append(run)
        n_dips = len(dip_runs)

        # Dashed requires: enough dips with enough total gap fraction.
        intensity_vote_dashed = (dip_frac > 0.25 and n_dips >= 2)

        # Either method voting dashed is sufficient
        if intensity_vote_dashed or gap_vote_dashed:
            return "dashed"

        return "solid"

    # ── Temporal Smoothing ──────────────────────────────────────────────

    def _smooth_lane_types(self, lanes: list[dict]) -> list[dict]:
        """
        Apply temporal majority-vote smoothing to lane type classification.

        Lanes are matched to slots by left-to-right order (already sorted).
        Each slot keeps a deque of the last N type classifications. The
        output type is the majority vote, preventing frame-to-frame flicker.
        """
        # Prune stale slots (more slots than current lanes → old slots)
        active_slots = set(range(len(lanes)))
        stale = [k for k in self._type_history if k not in active_slots]
        for k in stale:
            del self._type_history[k]

        for lane in lanes:
            slot = lane['lane_id']
            raw_type = lane['type']

            # Double lanes are already merged — don't smooth them
            if raw_type.startswith("double_"):
                continue

            if slot not in self._type_history:
                self._type_history[slot] = deque(maxlen=self._TYPE_HISTORY_LEN)

            self._type_history[slot].append(raw_type)

            # Simple majority vote — 50% threshold.
            # Was 65% but that suppressed legitimate dashed detections.
            hist = self._type_history[slot]
            n_dashed = sum(1 for t in hist if t == 'dashed')
            n_total = len(hist)

            lane['type'] = 'dashed' if n_dashed / n_total > 0.50 else 'solid'

        return lanes

    # ── Double Lane Merging ─────────────────────────────────────────────

    @staticmethod
    def _merge_double_lanes(lanes: list[dict]) -> list[dict]:
        """
        Merge adjacent parallel lanes of the same color into double lanes.

        Two lanes are merged into "double_yellow" or "double_white" when:
          - Same color
          - Average lateral distance < 45 pixels
          - Both are solid type (dashed double lines are extremely rare)

        The merged lane keeps the midpoint of both lanes' points.
        """
        if len(lanes) < 2:
            return lanes

        merged_indices = set()
        result = []

        for i in range(len(lanes)):
            if i in merged_indices:
                continue
            for j in range(i + 1, len(lanes)):
                if j in merged_indices:
                    continue
                li, lj = lanes[i], lanes[j]

                # Same color required
                if li['color'] != lj['color']:
                    continue

                # Compute average lateral distance between the two lanes
                pts_i = np.array(li['points'])
                pts_j = np.array(lj['points'])

                # Sample at common y-values
                y_min = max(pts_i[:, 1].min(), pts_j[:, 1].min())
                y_max = min(pts_i[:, 1].max(), pts_j[:, 1].max())
                if y_max - y_min < 30:
                    continue  # not enough overlap

                # Interpolate x at shared y values
                n_samples = 10
                y_samples = np.linspace(y_min, y_max, n_samples)
                try:
                    x_i = np.interp(y_samples, pts_i[:, 1], pts_i[:, 0])
                    x_j = np.interp(y_samples, pts_j[:, 1], pts_j[:, 0])
                except Exception:
                    continue

                avg_dist = float(np.mean(np.abs(x_i - x_j)))

                # Double lanes are typically 10-45px apart in image space
                if avg_dist < 8 or avg_dist > 45:
                    continue

                # Merge: midpoint of both lanes
                mid_x = (x_i + x_j) / 2.0
                mid_pts = [[float(mx), float(my)]
                           for mx, my in zip(mid_x, y_samples)]

                double_type = f"double_{li['color']}"
                merged_lane = {
                    'points': mid_pts,
                    'type': double_type,
                    'color': li['color'],
                    'area': li.get('area', 0) + lj.get('area', 0),
                    'lane_id': li['lane_id'],
                }
                result.append(merged_lane)
                merged_indices.add(i)
                merged_indices.add(j)
                break  # lane i is consumed

        # Add unmerged lanes
        for i, lane in enumerate(lanes):
            if i not in merged_indices:
                result.append(lane)

        # Re-sort and re-assign IDs
        result.sort(key=lambda l: np.mean([p[0] for p in l['points']]))
        for idx, lane in enumerate(result):
            lane['lane_id'] = idx

        return result

    # ── Crosswalk Detection ──────────────────────────────────────────────

    def _detect_crosswalks(self, ll_mask: np.ndarray,
                            frame_bgr: np.ndarray) -> list[dict]:
        """
        Detect crosswalks from horizontal stripe patterns in the lane mask.

        Crosswalks (zebra crossings) appear as multiple thick horizontal
        white bars clustered within a vertical band. Strategy:
          1. Find horizontal connected components (wider than tall)
          2. Cluster nearby horizontal bars vertically
          3. If 3+ bars within ~100px vertical → crosswalk
        """
        H, W = ll_mask.shape[:2]

        # Focus on bottom 60% of frame (road region)
        y_start = int(H * 0.4)
        roi = ll_mask[y_start:, :]

        # Morphological close to merge thin horizontal structures
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel_h)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            closed, connectivity=8)

        # Find horizontal bars (wider than tall, minimum size)
        h_bars = []
        for label_id in range(1, n_labels):
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            area = stats[label_id, cv2.CC_STAT_AREA]

            # Horizontal bar: wider than tall, reasonable size
            if w > h * 2.0 and w > 40 and area > 200 and h < 30:
                h_bars.append({
                    'x': x, 'y': y + y_start,
                    'w': w, 'h': h,
                    'cy': y + y_start + h // 2,
                    'cx': x + w // 2,
                })

        if len(h_bars) < 3:
            return []

        # Cluster bars by vertical proximity (within 100px and similar x range)
        h_bars.sort(key=lambda b: b['cy'])
        clusters = []
        current_cluster = [h_bars[0]]

        for bar in h_bars[1:]:
            prev = current_cluster[-1]
            # Check vertical gap and horizontal overlap
            v_gap = abs(bar['cy'] - prev['cy'])
            x_overlap = (min(bar['x'] + bar['w'], prev['x'] + prev['w']) -
                         max(bar['x'], prev['x']))

            if v_gap < 60 and x_overlap > 20:
                current_cluster.append(bar)
            else:
                if len(current_cluster) >= 3:
                    clusters.append(current_cluster)
                current_cluster = [bar]

        if len(current_cluster) >= 3:
            clusters.append(current_cluster)

        # Convert clusters to crosswalk detections
        crosswalks = []
        for cluster in clusters:
            x_min = min(b['x'] for b in cluster)
            y_min = min(b['y'] for b in cluster)
            x_max = max(b['x'] + b['w'] for b in cluster)
            y_max = max(b['y'] + b['h'] for b in cluster)

            crosswalks.append({
                'type': 'crosswalk',
                'bbox': [float(x_min), float(y_min),
                         float(x_max), float(y_max)],
                'confidence': min(0.9, 0.3 + 0.15 * len(cluster)),
                'n_bars': len(cluster),
            })

        return crosswalks

    # ══════════════════════════════════════════════════════════════════════
    # CLASSICAL BEV PIPELINE (Sobel + histogram + sliding window + polyfit)
    # Runs in parallel with TwinLiteNet+ as a stabilizing/corrective signal.
    # ══════════════════════════════════════════════════════════════════════

    def _preprocess_for_bev(self, frame_bgr: np.ndarray,
                             da_mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Prepare frame for BEV edge detection.

        Shadow filtering: CLAHE on LAB L-channel normalizes local contrast
        so lane markings remain visible under tree/building shadows.

        Returns: preprocessed grayscale image (uint8).
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]

        clahe = cv2.createCLAHE(
            clipLimit=cfg.CLASSICAL_CLAHE_CLIP,
            tileGridSize=(cfg.CLASSICAL_CLAHE_GRID, cfg.CLASSICAL_CLAHE_GRID),
        )
        l_eq = clahe.apply(l_chan)

        # Mask to drivable area if available (ignores off-road shadow edges)
        if da_mask is not None:
            da_u8 = (da_mask * 255).astype(np.uint8) if da_mask.max() <= 1 else da_mask
            l_eq = cv2.bitwise_and(l_eq, da_u8)

        return l_eq

    def _sobel_edges_bev(self, bev_gray: np.ndarray,
                          bev_road: np.ndarray) -> np.ndarray:
        """
        Extract lane-relevant edges in BEV using Sobel.

        In BEV, lane edges are roughly vertical lines (left/right edges of
        a painted marking). Their gradient is horizontal. We keep only
        pixels with roughly horizontal gradient direction (±30°) and
        sufficient magnitude. This suppresses rain streaks (vertical
        gradients) and shadow boundaries (diagonal/soft gradients).

        Returns: binary edge map (uint8, 0 or 255).
        """
        ksize = cfg.CLASSICAL_SOBEL_KSIZE
        sobel_x = cv2.Sobel(bev_gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(bev_gray, cv2.CV_64F, 0, 1, ksize=ksize)

        mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        mag_norm = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8) \
            if mag.max() > 0 else np.zeros_like(bev_gray)

        # Gradient direction: 0 = horizontal edge, ±π/2 = vertical edge
        direction = np.arctan2(np.abs(sobel_y), np.abs(sobel_x))

        # Keep roughly horizontal gradients (lane edge perpendicular):
        # direction < 30° from horizontal means strong lane-edge signal
        horiz_mask = direction < np.radians(35)

        # Combine magnitude threshold + direction filter
        edge = np.zeros_like(bev_gray)
        edge[(mag_norm > cfg.CLASSICAL_EDGE_THRESH) & horiz_mask] = 255

        # Restrict to dilated road area
        if bev_road is not None and bev_road.any():
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            road_wide = cv2.dilate(bev_road, kern, iterations=1)
            edge = cv2.bitwise_and(edge, road_wide)

        # Morphological cleanup: remove tiny noise, bridge small gaps
        k_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, k_clean)

        return edge

    def _histogram_peaks(self, edge_binary: np.ndarray) -> list[int]:
        """
        Find lane starting x-positions from column histogram of edge pixels.

        Uses bottom quarter of BEV edge image. Smooths histogram with
        Gaussian, then finds peaks separated by ≥80 pixels.

        Returns: list of x-positions (peak columns).
        """
        H, W = edge_binary.shape
        # Bottom quarter — closest to car, most reliable edges
        bottom = edge_binary[3 * H // 4:, :]
        histogram = np.sum(bottom, axis=0).astype(np.float64)

        if histogram.max() == 0:
            return []

        # Gaussian smoothing to merge nearby edge clusters
        sigma = cfg.CLASSICAL_HIST_SMOOTH_SIG
        kernel_size = int(6 * sigma) | 1  # must be odd
        histogram = cv2.GaussianBlur(
            histogram.reshape(1, -1), (1, kernel_size), sigma
        ).flatten()

        # Find peaks: minimum height = 10% of max, minimum distance = 80px
        min_height = histogram.max() * 0.10
        min_distance = 80

        peaks = []
        # Simple peak detection (avoids scipy dependency)
        for i in range(min_distance, W - min_distance):
            if histogram[i] < min_height:
                continue
            # Local max in ±min_distance window
            window = histogram[max(0, i - min_distance // 2):
                               min(W, i + min_distance // 2 + 1)]
            if histogram[i] == window.max():
                # Check we're not too close to an existing peak
                if not peaks or abs(i - peaks[-1]) >= min_distance:
                    peaks.append(i)

        return peaks

    def _sliding_window_track(self, edge_binary: np.ndarray,
                                start_x: int) -> Optional[np.ndarray]:
        """
        Track a lane upward from a histogram starting point using
        sliding windows.

        Returns: (N, 2) array of (x, y) points in BEV coords, or None
                 if insufficient windows found evidence.
        """
        H, W = edge_binary.shape
        n_windows = cfg.CLASSICAL_SW_N_WINDOWS
        win_h = H // n_windows
        margin = cfg.CLASSICAL_SW_MARGIN
        min_pix = cfg.CLASSICAL_SW_MIN_PIX

        current_x = start_x
        pts = []
        good_windows = 0

        for w in range(n_windows):
            # Window bounds (bottom to top)
            y_lo = H - (w + 1) * win_h
            y_hi = H - w * win_h
            x_lo = max(0, current_x - margin)
            x_hi = min(W, current_x + margin)

            # Extract window
            window = edge_binary[y_lo:y_hi, x_lo:x_hi]
            nonzero_y, nonzero_x = np.nonzero(window)

            if len(nonzero_x) >= min_pix:
                # Recenter on mean of detected edge pixels
                new_x = int(np.mean(nonzero_x)) + x_lo
                current_x = new_x
                good_windows += 1

            mid_y = (y_lo + y_hi) // 2
            pts.append([current_x, mid_y])

        # Require at least 40% of windows to have found edge pixels
        if good_windows < n_windows * 0.4:
            return None

        pts_arr = np.array(pts, dtype=np.float64)
        # Sort by y ascending (top to bottom) for np.interp compatibility
        pts_arr = pts_arr[pts_arr[:, 1].argsort()]
        return pts_arr

    def _fit_lane_polynomial(self, pts_bev: np.ndarray,
                               H: int, W: int) -> Optional[dict]:
        """
        Fit polynomial x = f(y) to BEV lane points and generate
        smooth sample coordinates.

        Returns dict with:
            'coeffs': polynomial coefficients
            'pts_bev': (N, 2) smooth sample points in BEV
        or None if fit fails.
        """
        if pts_bev is None or len(pts_bev) < 3:
            return None

        degree = cfg.CLASSICAL_POLY_DEGREE
        try:
            # Fit x = f(y)
            coeffs = np.polyfit(pts_bev[:, 1], pts_bev[:, 0], degree)
        except (np.linalg.LinAlgError, np.RankWarning):
            return None

        # Generate smooth points along the polynomial
        y_min, y_max = int(pts_bev[:, 1].min()), int(pts_bev[:, 1].max())
        n_samples = max(20, (y_max - y_min) // 8)
        y_eval = np.linspace(y_min, y_max, n_samples)
        x_eval = np.polyval(coeffs, y_eval)

        # Clamp to image bounds
        valid = (x_eval >= 0) & (x_eval < W)
        if valid.sum() < 3:
            return None

        smooth_pts = np.column_stack([x_eval[valid], y_eval[valid]])

        return {'coeffs': coeffs, 'pts_bev': smooth_pts}

    def _classify_type_bev(self, bev_gray: np.ndarray,
                             poly_coeffs: np.ndarray,
                             H: int,
                             y_range: Optional[tuple] = None) -> str:
        """
        Classify solid vs dashed by sampling intensity along the
        polynomial in BEV. In BEV, dashes are uniform size — no
        perspective compression — so transition counting is reliable.

        Returns: 'solid' or 'dashed'.
        """
        # Sample within the polynomial's fitted range (not full image)
        y_lo = int(y_range[0]) if y_range else int(H * 0.05)
        y_hi = int(y_range[1]) if y_range else int(H * 0.95)
        y_vals = np.linspace(y_lo, y_hi, 200).astype(int)
        x_vals = np.polyval(poly_coeffs, y_vals).astype(int)
        W = bev_gray.shape[1]

        intensities = []
        for x, y in zip(x_vals, y_vals):
            if 2 <= x < W - 2 and 0 <= y < H:
                # Sample ±2px around center for robustness
                patch = bev_gray[y, max(0, x - 2):x + 3]
                intensities.append(float(patch.max()))
            else:
                intensities.append(0.0)

        if not intensities:
            return "solid"

        arr = np.array(intensities)
        # Adaptive threshold: marking vs road
        thresh = np.percentile(arr[arr > 0], 40) if (arr > 0).sum() > 20 else 128
        binary = (arr > thresh).astype(int)

        # Count transitions (0→1 or 1→0)
        transitions = int(np.sum(np.abs(np.diff(binary))))

        # Solid: ≤4 transitions (start + end + noise)
        # Dashed: ≥6 transitions (multiple mark/gap cycles)
        if transitions >= 6:
            # Find gap (0) runs
            gap_runs = []
            run = 0
            for b in binary:
                if b == 0:
                    run += 1
                else:
                    if run > 3:
                        gap_runs.append(run)
                    run = 0
            if run > 3:
                gap_runs.append(run)
            if len(gap_runs) >= 2:
                return "dashed"

        return "solid"

    def _detect_double_lines(self, lane_fits: list[dict],
                               frame_bgr: np.ndarray,
                               M_inv: np.ndarray,
                               H: int) -> list[tuple[int, int, str]]:
        """
        Detect double lines from parallel polynomial pairs in BEV.

        Returns list of (idx_i, idx_j, double_type) tuples.
        """
        doubles = []
        for i in range(len(lane_fits)):
            for j in range(i + 1, len(lane_fits)):
                ci = lane_fits[i]['coeffs']
                cj = lane_fits[j]['coeffs']

                # Evaluate at multiple y values
                y_test = np.linspace(H * 0.2, H * 0.8, 10)
                xi = np.polyval(ci, y_test)
                xj = np.polyval(cj, y_test)
                dists = np.abs(xi - xj)

                avg_dist = float(np.mean(dists))
                std_dist = float(np.std(dists))

                # Double line: constant distance 15-50px, low std
                if 15 < avg_dist < 50 and std_dist < 10:
                    # Determine color from perspective view
                    # Use midpoint of the pair, map back to perspective
                    mid_bev = np.column_stack([(xi + xj) / 2, y_test])
                    pts_persp = cv2.perspectiveTransform(
                        mid_bev.reshape(-1, 1, 2).astype(np.float32),
                        M_inv
                    ).reshape(-1, 2)
                    color = self._classify_color(frame_bgr, pts_persp)
                    double_type = f"double_{color}"
                    doubles.append((i, j, double_type))

        return doubles

    def _classical_detect_bev(self, frame_bgr: np.ndarray,
                                K: np.ndarray,
                                R_cam2ego: np.ndarray,
                                t_cam2ego: np.ndarray) -> list[dict]:
        """
        Classical lane detection pipeline in BEV.

        Steps:
          1. Preprocess (CLAHE shadow normalization)
          2. Warp to BEV using IPM
          3. Sobel edge extraction with direction filtering
          4. Histogram peaks → lane starting positions
          5. Sliding window tracking per lane
          6. Polynomial fit
          7. Type + double-line classification
          8. Map back to perspective

        Returns list of lane dicts (same format as _extract_lanes_from_mask).
        """
        H, W = frame_bgr.shape[:2]

        # 1. Get IPM matrices (cached)
        M, M_inv = self.get_ipm_matrices(K=K, R=R_cam2ego, t=t_cam2ego,
                                          frame_shape=(H, W))
        if M is None:
            return []

        # 2. Preprocess
        preprocessed = self._preprocess_for_bev(frame_bgr, self._last_da_mask)

        # 3. Warp to BEV
        bev_gray = cv2.warpPerspective(preprocessed, M, (W, H))

        # Warp drivable mask to BEV
        bev_road = None
        if self._last_da_mask is not None:
            da_u8 = ((self._last_da_mask * 255).astype(np.uint8)
                     if self._last_da_mask.max() <= 1 else self._last_da_mask)
            bev_road = cv2.warpPerspective(da_u8, M, (W, H))
            _, bev_road = cv2.threshold(bev_road, 127, 255, cv2.THRESH_BINARY)

        # 4. Sobel edges
        edges = self._sobel_edges_bev(bev_gray, bev_road)

        # 5. Histogram peaks
        peaks = self._histogram_peaks(edges)
        if not peaks:
            return []

        # 6. Sliding window tracking + polynomial fit
        lane_fits = []
        for start_x in peaks:
            sw_pts = self._sliding_window_track(edges, start_x)
            fit = self._fit_lane_polynomial(sw_pts, H, W)
            if fit is not None:
                lane_fits.append(fit)

        if not lane_fits:
            return []

        # 7. Detect double lines
        doubles = self._detect_double_lines(lane_fits, frame_bgr, M_inv, H)
        double_indices = set()
        double_results = []
        for i, j, dtype in doubles:
            double_indices.add(i)
            double_indices.add(j)
            # Merge into midpoint
            ci = lane_fits[i]['coeffs']
            cj = lane_fits[j]['coeffs']
            y_samp = np.linspace(0, H - 1, 20)
            xi = np.polyval(ci, y_samp)
            xj = np.polyval(cj, y_samp)
            mid_bev = np.column_stack([(xi + xj) / 2, y_samp])
            # Map to perspective
            pts_persp = cv2.perspectiveTransform(
                mid_bev.reshape(-1, 1, 2).astype(np.float32), M_inv
            ).reshape(-1, 2)
            # Clamp
            pts_persp[:, 0] = np.clip(pts_persp[:, 0], 0, W - 1)
            pts_persp[:, 1] = np.clip(pts_persp[:, 1], 0, H - 1)

            color = dtype.replace("double_", "")
            double_results.append({
                'points': pts_persp.tolist(),
                'type': dtype,
                'color': color,
                'area': 0,
            })

        # 8. Build single lane results (non-double)
        lanes = []
        for idx, fit in enumerate(lane_fits):
            if idx in double_indices:
                continue

            pts_bev = fit['pts_bev']
            coeffs = fit['coeffs']

            # Map BEV points back to perspective
            pts_persp = cv2.perspectiveTransform(
                pts_bev.reshape(-1, 1, 2).astype(np.float32), M_inv
            ).reshape(-1, 2)
            pts_persp[:, 0] = np.clip(pts_persp[:, 0], 0, W - 1)
            pts_persp[:, 1] = np.clip(pts_persp[:, 1], 0, H - 1)

            # Classify type in BEV (solid vs dashed)
            y_range = (pts_bev[:, 1].min(), pts_bev[:, 1].max())
            lane_type = self._classify_type_bev(bev_gray, coeffs, H, y_range)

            # Classify color from perspective points
            lane_color = self._classify_color(frame_bgr, pts_persp)

            lanes.append({
                'points': pts_persp.tolist(),
                'type': lane_type,
                'color': lane_color,
                'area': 0,
            })

        # Add double-line results
        lanes.extend(double_results)

        # Sort left-to-right
        lanes.sort(key=lambda l: np.mean([p[0] for p in l['points']]))
        for i, lane in enumerate(lanes):
            lane['lane_id'] = i

        return lanes

    def _detect_crosswalks_bev(self, bev_gray: np.ndarray,
                                 bev_road: Optional[np.ndarray],
                                 M_inv: np.ndarray,
                                 H: int, W: int) -> list[dict]:
        """
        Detect crosswalks in BEV as horizontal bar clusters.

        In BEV, crosswalk bars are horizontal and uniform width —
        more reliable than perspective-view detection.

        Returns list of crosswalk dicts with bbox in perspective coords.
        """
        # Horizontal Sobel to find horizontal edges (crosswalk bars)
        sobel_y = cv2.Sobel(bev_gray, cv2.CV_64F, 0, 1, ksize=3)
        mag_y = np.abs(sobel_y)
        mag_y = np.clip(mag_y / mag_y.max() * 255, 0, 255).astype(np.uint8) \
            if mag_y.max() > 0 else np.zeros_like(bev_gray)

        # Threshold
        _, edges_h = cv2.threshold(mag_y, 40, 255, cv2.THRESH_BINARY)

        # Restrict to road
        if bev_road is not None and bev_road.any():
            edges_h = cv2.bitwise_and(edges_h, bev_road)

        # Horizontal morphological close to connect bar fragments
        kern_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        edges_h = cv2.morphologyEx(edges_h, cv2.MORPH_CLOSE, kern_h)

        # Find horizontal connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            edges_h, connectivity=8)

        bars = []
        for label_id in range(1, n_labels):
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            area = stats[label_id, cv2.CC_STAT_AREA]
            if w > h * 2.5 and w > 50 and area > 300 and h < 40:
                bars.append({'x': x, 'y': y, 'w': w, 'h': h,
                             'cy': y + h // 2, 'cx': x + w // 2})

        if len(bars) < 3:
            return []

        # Cluster by vertical proximity
        bars.sort(key=lambda b: b['cy'])
        clusters = []
        cur = [bars[0]]
        for bar in bars[1:]:
            if (abs(bar['cy'] - cur[-1]['cy']) < 50 and
                min(bar['x'] + bar['w'], cur[-1]['x'] + cur[-1]['w']) -
                    max(bar['x'], cur[-1]['x']) > 20):
                cur.append(bar)
            else:
                if len(cur) >= 3:
                    clusters.append(cur)
                cur = [bar]
        if len(cur) >= 3:
            clusters.append(cur)

        crosswalks = []
        for cluster in clusters:
            x_min = min(b['x'] for b in cluster)
            y_min = min(b['y'] for b in cluster)
            x_max = max(b['x'] + b['w'] for b in cluster)
            y_max = max(b['y'] + b['h'] for b in cluster)

            # Map corners back to perspective
            corners_bev = np.array([
                [[x_min, y_min]], [[x_max, y_min]],
                [[x_max, y_max]], [[x_min, y_max]]
            ], dtype=np.float32)
            corners_persp = cv2.perspectiveTransform(corners_bev, M_inv)
            xs = corners_persp[:, 0, 0]
            ys = corners_persp[:, 0, 1]

            crosswalks.append({
                'type': 'crosswalk',
                'bbox': [float(max(0, xs.min())), float(max(0, ys.min())),
                         float(min(W - 1, xs.max())), float(min(H - 1, ys.max()))],
                'confidence': min(0.9, 0.3 + 0.15 * len(cluster)),
                'n_bars': len(cluster),
            })

        return crosswalks

    def _fuse_nn_and_classical(self, nn_lanes: list[dict],
                                 classical_lanes: list[dict]) -> list[dict]:
        """
        Fuse TwinLiteNet+ and classical BEV lanes.

        Matching: by average x-distance at shared y-values.
        Position: weighted average (NN × 0.6, classical × 0.4).
        Type: classical wins (BEV gap analysis is more reliable).
        Color: NN wins (LAB/HSV relative comparison already robust).
        Double lines: classical wins (detects two polynomials).
        """
        nn_w = cfg.CLASSICAL_FUSION_NN_W
        cl_w = 1.0 - nn_w

        # Convert points to arrays for matching
        def _get_x_at_y(lane, y_vals):
            pts = np.array(lane['points'])
            if len(pts) < 2:
                return np.full(len(y_vals), pts[0][0]) if len(pts) == 1 else None
            # Sort by y ascending for np.interp
            pts = pts[pts[:, 1].argsort()]
            return np.interp(y_vals, pts[:, 1], pts[:, 0],
                             left=pts[0, 0], right=pts[-1, 0])

        # Common y range for matching
        H_frame = 720  # approximate
        if nn_lanes:
            all_pts = [p for l in nn_lanes for p in l['points']]
            if all_pts:
                H_frame = max(p[1] for p in all_pts)

        y_match = np.linspace(H_frame * 0.5, H_frame * 0.9, 5)

        # Match NN lanes to classical lanes by lateral distance
        matched_nn = set()
        matched_cl = set()
        matches = []  # (nn_idx, cl_idx, avg_dist)

        for ni, nn_l in enumerate(nn_lanes):
            nn_x = _get_x_at_y(nn_l, y_match)
            if nn_x is None:
                continue
            best_ci, best_dist = -1, 999
            for ci, cl_l in enumerate(classical_lanes):
                if ci in matched_cl:
                    continue
                cl_x = _get_x_at_y(cl_l, y_match)
                if cl_x is None:
                    continue
                dist = float(np.mean(np.abs(nn_x - cl_x)))
                if dist < best_dist:
                    best_dist = dist
                    best_ci = ci
            if best_ci >= 0 and best_dist < 60:
                matches.append((ni, best_ci, best_dist))
                matched_nn.add(ni)
                matched_cl.add(best_ci)

        # Build fused lanes
        fused = []
        for ni, ci, _ in matches:
            nn_l = nn_lanes[ni]
            cl_l = classical_lanes[ci]

            nn_pts = np.array(nn_l['points'])
            cl_pts = np.array(cl_l['points'])

            # Sort both by y (ascending) — required by np.interp
            nn_pts = nn_pts[nn_pts[:, 1].argsort()]
            cl_pts = cl_pts[cl_pts[:, 1].argsort()]

            # Interpolate both to common y-values for weighted average
            y_min = max(nn_pts[:, 1].min(), cl_pts[:, 1].min())
            y_max = min(nn_pts[:, 1].max(), cl_pts[:, 1].max())
            if y_max - y_min < 20:
                fused.append(nn_l)
                continue

            n_pts = max(len(nn_pts), len(cl_pts))
            y_common = np.linspace(y_min, y_max, n_pts)
            nn_x = np.interp(y_common, nn_pts[:, 1], nn_pts[:, 0])
            cl_x = np.interp(y_common, cl_pts[:, 1], cl_pts[:, 0])

            # Weighted average positions
            fused_x = nn_w * nn_x + cl_w * cl_x
            fused_pts = np.column_stack([fused_x, y_common])

            # Type: classical wins (BEV gap analysis)
            # Color: NN wins (LAB+HSV)
            # Double: classical wins if it detected double
            fused_type = cl_l['type']  # classical type
            fused_color = nn_l['color']  # NN color
            # If classical says double, trust it
            if cl_l['type'].startswith('double_'):
                fused_type = cl_l['type']
                fused_color = cl_l['color']

            fused.append({
                'points': fused_pts.tolist(),
                'type': fused_type,
                'color': fused_color,
                'area': nn_l.get('area', 0),
            })

        # Unmatched NN lanes: keep as-is
        for ni, nn_l in enumerate(nn_lanes):
            if ni not in matched_nn:
                fused.append(nn_l)

        # Unmatched classical lanes: add only if persistent
        new_unmatched = {}
        for ci, cl_l in enumerate(classical_lanes):
            if ci in matched_cl:
                continue
            # Use approximate center x as key
            approx_x = int(np.mean([p[0] for p in cl_l['points']]))
            # Bucket to nearest 50px to allow slight drift
            bucket = (approx_x // 50) * 50
            prev_count = self._classical_unmatched.get(bucket, 0)
            new_unmatched[bucket] = prev_count + 1
            if new_unmatched[bucket] >= cfg.CLASSICAL_FUSION_MIN_FRAMES:
                fused.append(cl_l)
        self._classical_unmatched = new_unmatched

        # Re-sort left-to-right and assign IDs
        fused.sort(key=lambda l: np.mean([p[0] for p in l['points']]))
        for i, lane in enumerate(fused):
            lane['lane_id'] = i

        return fused

    # ── Lane 3D back-projection ─────────────────────────────────────────

    @staticmethod
    def _backproject_lane_points(
        pts_2d: list,
        depth_map: Optional[np.ndarray],
        K_inv: Optional[np.ndarray],
        R_cam2ego: Optional[np.ndarray],
        t_cam2ego: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Back-project 2D lane points to 3D ego frame.

        For each (u, v):
          - If depth available: use depth_map[v, u] for metric depth
          - Otherwise: assume flat ground (Z_ego=0) and solve for depth

        Returns (N, 3) array of [X_ego, Y_ego, Z_ego] points.
        """
        if not pts_2d or K_inv is None:
            return np.zeros((0, 3))

        pts_2d_arr = np.array(pts_2d, dtype=np.float64)
        if pts_2d_arr.ndim != 2 or pts_2d_arr.shape[1] < 2:
            return np.zeros((0, 3))

        H_depth = depth_map.shape[0] if depth_map is not None else 0
        W_depth = depth_map.shape[1] if depth_map is not None else 0

        pts_3d = []
        for u, v in pts_2d_arr[:, :2]:
            u_int, v_int = int(round(u)), int(round(v))

            # Get depth from depth map
            d = None
            if depth_map is not None and 0 <= v_int < H_depth and 0 <= u_int < W_depth:
                d = float(depth_map[v_int, u_int])
                if d <= 0.5 or d > 150:
                    d = None  # invalid depth

            if d is None:
                continue  # skip points without valid depth

            # Pixel → camera 3D: P_cam = d * K_inv @ [u, v, 1]
            p_hom = np.array([u, v, 1.0], dtype=np.float64)
            p_cam = d * (K_inv @ p_hom)  # [X_cam, Y_cam, Z_cam]

            # Camera → ego frame
            if R_cam2ego is not None and t_cam2ego is not None:
                p_ego = R_cam2ego @ p_cam + t_cam2ego.flatten()
            else:
                # No extrinsics: just swap axes (cam Z→ego Y, cam X→ego X, cam -Y→ego Z)
                p_ego = np.array([p_cam[0], p_cam[2], -p_cam[1]])

            pts_3d.append(p_ego)

        if not pts_3d:
            return np.zeros((0, 3))

        return np.array(pts_3d, dtype=np.float64)

    # ── Public API: Detect Lanes ──────────────────────────────────────────

    def detect(self, frame: np.ndarray,
               K: Optional[np.ndarray] = None,
               K_inv: Optional[np.ndarray] = None,
               R_cam2ego: Optional[np.ndarray] = None,
               t_cam2ego: Optional[np.ndarray] = None,
               depth_map: Optional[np.ndarray] = None) -> list[dict]:
        """
        Detect lane lines in a BGR frame.

        Returns list of lane dicts with keys:
            lane_id, points, type, color
        """
        if self._model is None:
            return []

        try:
            self._frame_counter += 1

            # Frame skip: reuse cached lanes between update frames
            if (self._frame_counter % self._FRAME_SKIP != 1
                    and self._cached_lanes):
                return self._cached_lanes

            # Use cached masks if available (from get_drivable_mask call)
            if self._last_ll_mask is not None and self._last_ll_mask.shape[:2] == frame.shape[:2]:
                ll_mask = self._last_ll_mask
                self._last_ll_mask = None  # consume cache
            else:
                _, ll_mask = self._infer(frame)

            # DEBUG: check TwinLiteNet+ mask output
            ll_nz = int(np.count_nonzero(ll_mask))
            if self._frame_counter <= 5 or self._frame_counter % 50 == 0:
                print(f"[LaneDetector DEBUG] frame={self._frame_counter} "
                      f"ll_mask shape={ll_mask.shape} dtype={ll_mask.dtype} "
                      f"nonzero={ll_nz} max={ll_mask.max()}")

            nn_lanes = self._extract_lanes_from_mask(ll_mask, frame)

            # Lane dropout carryforward: if 0 lanes detected but we had
            # lanes recently, carry forward the previous lanes for up to
            # 3 frames to avoid flickering
            if len(nn_lanes) == 0 and self._cached_lanes:
                self._dropout_count += 1
                if self._dropout_count <= 3:
                    return self._cached_lanes
            else:
                self._dropout_count = 0

            # DEBUG: extraction results
            if self._frame_counter <= 5 or self._frame_counter % 50 == 0:
                print(f"[LaneDetector DEBUG] frame={self._frame_counter} "
                      f"nn_lanes={len(nn_lanes)}")

            # Merge adjacent parallel lanes into double_yellow / double_white
            nn_lanes = self._merge_double_lanes(nn_lanes)

            # Classical BEV pipeline (stabilizer for TwinLiteNet+)
            if (cfg.CLASSICAL_LANE_ENABLED and K is not None
                    and R_cam2ego is not None and t_cam2ego is not None):
                classical_lanes = self._classical_detect_bev(
                    frame, K, R_cam2ego, t_cam2ego)
                if classical_lanes:
                    lanes = self._fuse_nn_and_classical(nn_lanes, classical_lanes)
                else:
                    lanes = nn_lanes
            else:
                lanes = nn_lanes

            # Detect crosswalks — prefer BEV if IPM available
            if (cfg.CLASSICAL_LANE_ENABLED and K is not None
                    and R_cam2ego is not None and t_cam2ego is not None):
                H, W = frame.shape[:2]
                M, M_inv = self.get_ipm_matrices(
                    K=K, R=R_cam2ego, t=t_cam2ego, frame_shape=(H, W))
                if M is not None:
                    preprocessed = self._preprocess_for_bev(frame, self._last_da_mask)
                    bev_gray = cv2.warpPerspective(preprocessed, M, (W, H))
                    da_u8 = None
                    if self._last_da_mask is not None:
                        da_u8 = ((self._last_da_mask * 255).astype(np.uint8)
                                 if self._last_da_mask.max() <= 1
                                 else self._last_da_mask)
                        da_u8 = cv2.warpPerspective(da_u8, M, (W, H))
                        _, da_u8 = cv2.threshold(da_u8, 127, 255, cv2.THRESH_BINARY)
                    crosswalks = self._detect_crosswalks_bev(
                        bev_gray, da_u8, M_inv, H, W)
                else:
                    crosswalks = self._detect_crosswalks(ll_mask, frame)
            else:
                crosswalks = self._detect_crosswalks(ll_mask, frame)
            self._last_crosswalks = crosswalks

            # Lane count persistence: if count changes, confirm for N frames
            if self._cached_lanes and len(lanes) != len(self._cached_lanes):
                self._lane_count_new = getattr(self, '_lane_count_new', 0) + 1
                if self._lane_count_new < self._LANE_COUNT_CONFIRM:
                    return self._cached_lanes
            else:
                self._lane_count_new = 0

            # Temporal smoothing for type classification
            lanes = self._smooth_lane_types(lanes)

            # Back-project lane points to 3D ego frame using depth
            for lane in lanes:
                pts_2d = lane.get("points", [])
                pts_3d = self._backproject_lane_points(
                    pts_2d, depth_map, K_inv, R_cam2ego, t_cam2ego)
                lane["points_ego"] = pts_3d
                lane["points_3d"] = pts_3d.tolist() if isinstance(pts_3d, np.ndarray) else pts_3d
                # Fit poly_ego: 2nd-degree polynomial Y_ego = f(X_ego)
                if isinstance(pts_3d, np.ndarray) and len(pts_3d) >= 3:
                    try:
                        lane["poly_ego"] = np.polyfit(
                            pts_3d[:, 0], pts_3d[:, 1], 2).tolist()
                    except Exception:
                        lane["poly_ego"] = [0.0, 0.0, 0.0]
                else:
                    lane["poly_ego"] = [0.0, 0.0, 0.0]

            self._cached_lanes = lanes
            return lanes

        except Exception as e:
            print(f"[LaneDetector] Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def detect_debug(self, frame: np.ndarray) -> dict:
        """Debug mode: return intermediate outputs."""
        if self._model is None:
            return {'lanes': [], 'll_mask': None, 'da_mask': None}

        da_mask, ll_mask = self._infer(frame)
        lanes = self._extract_lanes_from_mask(ll_mask, frame)

        for lane in lanes:
            lane.setdefault('poly_ego', [0.0, 0.0, 0.0])
            lane.setdefault('points_ego', np.zeros((0, 3)))
            lane.setdefault('points_3d', [])

        return {
            'lanes': lanes,
            'll_mask': ll_mask,
            'da_mask': da_mask,
            'bases': [],
            'n_candidates': len(lanes),
            'persp_mask': ll_mask,
            'mask_bev_raw': None,
            'road_bev': None,
            'mask_bev_filtered': None,
        }
