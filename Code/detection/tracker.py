"""
Multi-Class BoT-SORT Tracker (via BoxMOT)  +  Cross-Camera Global ID Manager
=============================================================================

Within-camera tracking:
    Wraps boxmot.BoTSORT to give stable track IDs across frames for
    vehicles and pedestrians.  Uses osnet_x0_25 ReID model.
    One BoT-SORT instance per camera × per category.

Cross-camera association:
    GlobalIDManager unifies per-camera track IDs into a single global
    namespace.  When multi-camera fusion merges two detections from
    different cameras (same class, 3D centroid within threshold), the
    manager creates a bidirectional mapping so both camera-local IDs
    resolve to the same global ID.

Install:
    pip install boxmot
"""

from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ─── Track categories that get stable IDs ────────────────────────────────────
# Traffic lights and road signs are static — they don't need tracking;
# raw detections are used directly with ephemeral IDs.
TRACKED_CATEGORIES = ["vehicles", "pedestrians"]

# Resolve osnet weights: prefer local copy next to this code, fallback to CWD
_CODE_DIR = Path(__file__).resolve().parent.parent
_OSNET_CANDIDATES = [
    _CODE_DIR / "osnet_x0_25_msmt17.pt",
    cfg.BASE_DIR / "Code _ph2_VLM+Dino" / "osnet_x0_25_msmt17.pt",
    Path("osnet_x0_25_msmt17.pt"),
]
OSNET_WEIGHTS = next((p for p in _OSNET_CANDIDATES if p.exists()),
                     Path("osnet_x0_25_msmt17.pt"))


class MultiClassTracker:
    """
    One BoT-SORT instance per category so that IDs don't collide across
    different object types.

    Usage:
        tracker = MultiClassTracker()
        ...
        tracked_dets = tracker.update(dets, frame_bgr)
    """

    def __init__(
        self,
        max_age: int        = 90,   # frames before a lost track is deleted
        min_hits: int       = 3,    # frames to confirm a new track (raised from 1 to reduce flicker)
        iou_threshold: float = 0.3, # IoU matching threshold
        device: str         = "",
    ):
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold

        if device:
            # boxmot expects "0" or "cpu", not "cuda" or "cuda:0"
            import torch
            if device in ("cuda", "cuda:0") and torch.cuda.is_available():
                self._device = "0"
            elif device.startswith("cuda:") and torch.cuda.is_available():
                self._device = device.split(":")[1]
            elif device == "cpu":
                self._device = "cpu"
            else:
                self._device = "0" if torch.cuda.is_available() else "cpu"
        else:
            import torch
            self._device = "0" if torch.cuda.is_available() else "cpu"

        self._trackers: dict = {}
        for cat in TRACKED_CATEGORIES:
            self._trackers[cat] = self._make_tracker()

        # running ID counter as fallback for categories not in TRACKED_CATEGORIES
        self._fallback_id: int = 10000

    def _make_tracker(self):
        """Create a fresh BoT-SORT instance with the configured params."""
        from boxmot import BotSort

        return BotSort(
            reid_weights=OSNET_WEIGHTS,
            device=self._device,
            half=False,
            per_class=True,
            track_buffer=self._max_age,       # max frames to keep lost tracks
            match_thresh=self._iou_threshold, # IoU matching threshold
            new_track_thresh=0.3,
            track_high_thresh=0.3,
            track_low_thresh=0.1,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        dets: dict[str, list],
        frame: np.ndarray,
    ) -> dict[str, list]:
        """
        Run BoT-SORT on each category and return detections with stable
        'track_id' fields attached.

        Args:
            dets:  output of ObjectDetector.detect() — dict of lists
            frame: current BGR frame (used for appearance re-ID)

        Returns:
            Same structure as dets but every detection dict now has:
                'track_id': int  (stable across frames for the same object)
        """
        result: dict[str, list] = {}

        for cat, det_list in dets.items():
            if cat not in self._trackers or not det_list:
                # Passthrough — assign ephemeral IDs
                for d in det_list:
                    if "track_id" not in d:
                        d["track_id"] = self._fallback_id
                        self._fallback_id += 1
                result[cat] = det_list
                continue

            # ── Format for BoxMOT: N×6 array [x1,y1,x2,y2,conf,cls_id] ──
            n = len(det_list)
            dets_array = np.zeros((n, 6), dtype=np.float32)
            for i, d in enumerate(det_list):
                x1, y1, x2, y2 = d["bbox"]
                conf = d.get("confidence", 0.5)
                dets_array[i] = [x1, y1, x2, y2, conf, 0]

            # ── Run tracker ──────────────────────────────────────────────
            try:
                tracks = self._trackers[cat].update(dets_array, frame)
            except Exception as e:
                print(f"[Tracker] BoT-SORT error on '{cat}': {e} — using fallback IDs")
                for d in det_list:
                    d["track_id"] = self._fallback_id
                    self._fallback_id += 1
                result[cat] = det_list
                continue

            # ── Map track results back to original detections ────────────
            # BoxMOT output: M×8 → [x1,y1,x2,y2, track_id, conf, cls_id, det_index]
            #                        0   1  2  3      4       5     6        7
            tracked_dets = []

            if len(tracks) > 0:
                for row in tracks:
                    tx1, ty1, tx2, ty2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    track_id = int(row[4])
                    det_idx = int(row[7]) if row.shape[0] > 7 else -1

                    # Match back to original detection by det_index or IoU
                    if 0 <= det_idx < n:
                        det_copy = dict(det_list[det_idx])
                    else:
                        # Fallback: IoU match against original detections
                        track_box = [tx1, ty1, tx2, ty2]
                        best_iou, best_idx = -1.0, -1
                        for idx, d in enumerate(det_list):
                            iou = _box_iou(track_box, d["bbox"])
                            if iou > best_iou:
                                best_iou, best_idx = iou, idx
                        if best_idx >= 0 and best_iou > 0.2:
                            det_copy = dict(det_list[best_idx])
                        else:
                            continue  # skip unmatched track predictions

                    det_copy["track_id"] = track_id
                    # Update bbox with tracker's smoothed estimate
                    det_copy["bbox"] = [tx1, ty1, tx2, ty2]
                    tracked_dets.append(det_copy)

            result[cat] = tracked_dets

        return result

    def reset(self):
        """Reset all trackers (call between sequences)."""
        for cat in TRACKED_CATEGORIES:
            self._trackers[cat] = self._make_tracker()


# ─── IoU helper ──────────────────────────────────────────────────────────────

def _box_iou(a: list, b: list) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Camera Global ID Manager
# ═══════════════════════════════════════════════════════════════════════════════

class GlobalIDManager:
    """
    Unifies per-camera BoT-SORT track IDs into a single global namespace.

    Design assumptions (made explicit):
      1. Each camera has an ID offset (front=0, back=20000, left=30000, right=40000)
         so camera-local IDs never collide numerically.
      2. Cross-camera association happens AFTER multi-camera fusion resolves
         which detections from different cameras correspond to the same physical
         object (via 3D centroid proximity + same class).
      3. When fusion says "camera-local ID 5 from front and camera-local ID 40003
         from right are the same object", GlobalIDManager maps both to a single
         global ID.
      4. Global IDs are persistent across frames within a sequence — once
         assigned, the mapping is kept until the track is lost from ALL cameras.

    Limitations:
      - Relies on accurate 3D centroid fusion (which depends on depth quality).
      - Cannot associate objects that never appear in overlapping FOV.
      - Does not use appearance embeddings across cameras (would require
        extracting ReID features from BotSort, which is not exposed by boxmot).

    Usage in run_detection.py:
        gid_mgr = GlobalIDManager()
        # After fuse_multicam_detections():
        gid_mgr.update_from_fusion(merged_detections)
        # Each detection now has 'global_id' field
    """

    def __init__(self):
        # camera_track_id → global_id
        self._cam_to_global: dict[int, int] = {}
        # global_id → set of camera_track_ids
        self._global_to_cams: dict[int, set] = defaultdict(set)
        # next available global ID
        self._next_global_id: int = 1

    def update_from_fusion(self, merged_detections: list[dict]) -> list[dict]:
        """
        After fuse_multicam_detections(), inspect 'sources' on each merged
        detection.  If multiple cameras contributed, unify their track_ids.
        Assign a global_id to every detection.

        Mutates detections in-place (adds 'global_id' field).
        Returns the same list for convenience.
        """
        for det in merged_detections:
            sources = det.get("sources", [])
            cam_track_ids = []
            for src in sources:
                tid = src.get("track_id")
                if tid is not None:
                    cam_track_ids.append(int(tid))

            # Also include the detection's own track_id
            own_tid = det.get("track_id")
            if own_tid is not None:
                own_tid = int(own_tid)
                if own_tid not in cam_track_ids:
                    cam_track_ids.append(own_tid)

            if not cam_track_ids:
                det["global_id"] = self._allocate_new()
                continue

            # Check if any of these camera track IDs already have a global ID
            existing_gids = set()
            for ctid in cam_track_ids:
                if ctid in self._cam_to_global:
                    existing_gids.add(self._cam_to_global[ctid])

            if len(existing_gids) == 0:
                # None mapped yet — assign new global ID
                gid = self._allocate_new()
            elif len(existing_gids) == 1:
                # All agree on one global ID
                gid = existing_gids.pop()
            else:
                # Multiple global IDs need merging — pick lowest, remap others
                gid = min(existing_gids)
                for old_gid in existing_gids:
                    if old_gid != gid:
                        self._merge_global(old_gid, gid)

            # Map all camera track IDs to this global ID
            for ctid in cam_track_ids:
                self._cam_to_global[ctid] = gid
                self._global_to_cams[gid].add(ctid)

            det["global_id"] = gid

        return merged_detections

    def resolve(self, camera_track_id: int) -> int:
        """Look up the global ID for a camera-local track ID."""
        return self._cam_to_global.get(camera_track_id, camera_track_id)

    def _allocate_new(self) -> int:
        gid = self._next_global_id
        self._next_global_id += 1
        return gid

    def _merge_global(self, old_gid: int, new_gid: int):
        """Remap all tracks from old_gid to new_gid."""
        if old_gid not in self._global_to_cams:
            return
        for ctid in self._global_to_cams[old_gid]:
            self._cam_to_global[ctid] = new_gid
            self._global_to_cams[new_gid].add(ctid)
        del self._global_to_cams[old_gid]

    def prune(self, active_cam_track_ids: set):
        """Remove mappings for tracks no longer active in any camera."""
        stale = set(self._cam_to_global.keys()) - active_cam_track_ids
        for ctid in stale:
            gid = self._cam_to_global.pop(ctid, None)
            if gid is not None and gid in self._global_to_cams:
                self._global_to_cams[gid].discard(ctid)
                if not self._global_to_cams[gid]:
                    del self._global_to_cams[gid]

    def reset(self):
        """Clear all state (call between sequences)."""
        self._cam_to_global.clear()
        self._global_to_cams.clear()
        self._next_global_id = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Camera ReID Feature Extractor
# ═══════════════════════════════════════════════════════════════════════════════

class ReIDFeatureExtractor:
    """
    Extracts 512-dim OSNet appearance embeddings for cross-camera matching.

    Loads the same OSNet weights that BotSort uses, but independently so
    embeddings can be compared across cameras.  Does NOT modify the tracking
    pipeline — it's an add-on for GlobalIDManager verification.

    Usage in run_detection.py:
        reid_ext = ReIDFeatureExtractor()
        embeddings = reid_ext.extract_batch(frame, tracked_dets)
        # embeddings: dict[int, np.ndarray]  (track_id → 512-dim L2-normalized)

        # After centroid-based fusion:
        reid_ext.verify_fusion(embeddings_cam_a, embeddings_cam_b, fused_pairs)
    """

    def __init__(self):
        self._model = None
        self._device = None
        self._loaded = False

    def _load(self):
        """Lazy-load OSNet model for ReID feature extraction."""
        if self._loaded:
            return
        self._loaded = True

        try:
            import torch
            import torch.nn.functional as F

            self._device = "cuda" if torch.cuda.is_available() else "cpu"  # ReID extractor

            # Try torchreid first (cleanest API)
            try:
                import torchreid
                self._model = torchreid.models.build_model(
                    name='osnet_x0_25', num_classes=1, pretrained=False
                )
                torchreid.utils.load_pretrained_weights(
                    self._model, str(OSNET_WEIGHTS)
                )
                self._model = self._model.to(self._device).eval()
                self._backend = "torchreid"
                print(f"[ReIDExtractor] OSNet loaded via torchreid from {OSNET_WEIGHTS}")
                return
            except ImportError:
                pass

            # Fallback: load OSNet directly via torch (works with any boxmot version)
            try:
                from boxmot.appearance.reid_model_factory import load_pretrained_weights as _load_bm
                import boxmot.appearance.backbones as _bb
                self._model = _bb.build_model('osnet_x0_25', num_classes=1, pretrained=False)
                _load_bm(self._model, str(OSNET_WEIGHTS))
                self._model = self._model.to(self._device).eval()
                self._backend = "boxmot"
                print(f"[ReIDExtractor] OSNet loaded via boxmot from {OSNET_WEIGHTS}")
                return
            except Exception:
                pass

            # Last resort: raw torch.load of the OSNet state dict
            try:
                from torchvision.models import mobilenet_v2  # just to confirm torch works
                ckpt = torch.load(str(OSNET_WEIGHTS), map_location=self._device, weights_only=False)
                if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                    # torchreid-style checkpoint — need the model architecture
                    print("[ReIDExtractor] OSNet checkpoint needs model architecture (torchreid or boxmot).")
                    print("[ReIDExtractor] Install torchreid: pip install torchreid")
                else:
                    print("[ReIDExtractor] Unrecognized checkpoint format.")
            except Exception as e:
                print(f"[ReIDExtractor] Direct load failed: {e}")

            print("[ReIDExtractor] Could not load OSNet — cross-camera ReID disabled.")

        except ImportError:
            print("[ReIDExtractor] PyTorch not available — ReID disabled.")

    def extract_batch(self, frame: np.ndarray,
                      tracked_dets: list[dict]) -> dict[int, np.ndarray]:
        """
        Extract ReID embeddings for all tracked detections in a frame.

        Args:
            frame: BGR frame
            tracked_dets: list of dets with 'bbox' and 'track_id'

        Returns:
            {track_id: 512-dim L2-normalized embedding} for each detection.
            Empty dict if model unavailable.
        """
        self._load()
        if self._model is None:
            return {}

        import torch
        import torch.nn.functional as F

        embeddings = {}
        H, W = frame.shape[:2]

        for det in tracked_dets:
            tid = det.get("track_id")
            if tid is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            crop = frame[y1:y2, x1:x2]

            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            try:
                if self._backend == "torchreid":
                    emb = self._extract_torchreid(crop)
                else:
                    emb = self._extract_boxmot(crop)

                if emb is not None:
                    embeddings[tid] = emb
            except Exception:
                continue

        return embeddings

    def _extract_torchreid(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using torchreid model."""
        import torch
        import torch.nn.functional as F

        # OSNet expects 256x128 input, ImageNet normalization
        crop = cv2.resize(crop_bgr, (128, 256))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(crop).float().permute(2, 0, 1) / 255.0

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        tensor = tensor.unsqueeze(0).to(self._device)

        with torch.no_grad():
            feat = self._model(tensor)

        feat = F.normalize(feat, dim=1)
        return feat.cpu().numpy().flatten()

    def _extract_boxmot(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using boxmot's ReID backend."""
        import torch
        import torch.nn.functional as F

        # boxmot's get_features expects a list of crops
        crops = [cv2.resize(crop_bgr, (128, 256))]
        feats = self._model.get_features(crops)

        if feats is not None and len(feats) > 0:
            feat = feats[0]
            if isinstance(feat, torch.Tensor):
                feat = feat.cpu().numpy()
            # L2 normalize
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat = feat / norm
            return feat.flatten()
        return None

    @staticmethod
    def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(emb_a, emb_b))

    def verify_fusion(self,
                      embeddings_by_cam: dict[str, dict[int, np.ndarray]],
                      fused_pairs: list[tuple[int, int]],
                      accept_thresh: float = 0.6,
                      reject_thresh: float = 0.35,
                      ) -> list[tuple[int, int, float, str]]:
        """
        Verify centroid-based fusion pairs using ReID appearance similarity.

        Args:
            embeddings_by_cam: {cam_id: {track_id: embedding}}
            fused_pairs: list of (track_id_a, track_id_b) from centroid fusion
            accept_thresh: cosine sim above this → confirmed match
            reject_thresh: cosine sim below this → rejected (coincidental proximity)

        Returns:
            list of (tid_a, tid_b, similarity, verdict) where verdict is
            "confirmed" | "rejected" | "uncertain"
        """
        all_embeddings = {}
        for cam_embs in embeddings_by_cam.values():
            all_embeddings.update(cam_embs)

        results = []
        for tid_a, tid_b in fused_pairs:
            emb_a = all_embeddings.get(tid_a)
            emb_b = all_embeddings.get(tid_b)

            if emb_a is None or emb_b is None:
                results.append((tid_a, tid_b, -1.0, "no_embedding"))
                continue

            sim = self.cosine_similarity(emb_a, emb_b)

            if sim >= accept_thresh:
                verdict = "confirmed"
            elif sim < reject_thresh:
                verdict = "rejected"
            else:
                verdict = "uncertain"

            results.append((tid_a, tid_b, sim, verdict))

        return results

    def find_cross_camera_matches(self,
                                   embeddings_by_cam: dict[str, dict[int, np.ndarray]],
                                   same_class_pairs: list[tuple[int, int]],
                                   similarity_thresh: float = 0.7,
                                   ) -> list[tuple[int, int, float]]:
        """
        Find new cross-camera matches not caught by centroid fusion.

        Only considers pairs where both are the same YOLO class (provided
        as same_class_pairs).

        Returns:
            list of (tid_a, tid_b, similarity) for high-confidence matches.
        """
        all_embeddings = {}
        for cam_embs in embeddings_by_cam.values():
            all_embeddings.update(cam_embs)

        matches = []
        for tid_a, tid_b in same_class_pairs:
            emb_a = all_embeddings.get(tid_a)
            emb_b = all_embeddings.get(tid_b)
            if emb_a is None or emb_b is None:
                continue

            sim = self.cosine_similarity(emb_a, emb_b)
            if sim >= similarity_thresh:
                matches.append((tid_a, tid_b, sim))

        return matches
