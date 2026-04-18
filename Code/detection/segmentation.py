"""
Instance Segmentation — Mask2Former (HuggingFace transformers)
==============================================================

Uses facebook/mask2former-swin-large-cityscapes-panoptic from HuggingFace.
Auto-downloads on first use (~1.2GB).

SECONDARY refinement model — NOT the primary detector.

Roles:
  - Instance mask refinement for better object shape
  - Improved 3D reconstruction (vs bbox-only)

Execution:
  - Front camera only (configurable via cfg.MASK2FORMER_CAMERAS)
  - Cropped YOLO regions (expanded by 25%) — preferred for performance
  - Class-filtered: only instances matching the YOLO label are considered
  - Best instance selected by IoU with YOLO bbox

Loaded ONCE at startup, reused across all frames.
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ─── Cityscapes label → YOLO label mapping ──────────────────────────────────
# Cityscapes panoptic class names (thing classes) that Mask2Former outputs.
# We map YOLO labels to the set of Cityscapes class names they should match.
_YOLO_TO_CITYSCAPES = {
    "car":           {"car"},
    "truck":         {"truck"},
    "bus":           {"bus"},
    "motorcycle":    {"motorcycle"},
    "bicycle":       {"bicycle"},
    "person":        {"person", "rider"},
    "traffic_light": {"traffic light"},
    "stop_sign":     {"traffic sign"},
}

# Track which classes we've warned about (warn once per unmapped class)
_warned_classes: set = set()
_logged_available_labels = False


class Mask2FormerSegmenter:
    """
    Mask2Former instance segmentation via HuggingFace transformers.

    Uses facebook/mask2former-swin-large-cityscapes-panoptic.
    Loaded once at startup. Runs on expanded YOLO crops (front camera only).
    Filters output by YOLO class and picks best IoU match.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        self._id2label = {}  # maps model class IDs to label strings
        self._load_model()

    def _load_model(self):
        """Load Mask2Former from HuggingFace transformers."""
        model_name = cfg.MASK2FORMER_HF_MODEL
        print(f"[Mask2Former] Loading {model_name} on {self.device} …")

        try:
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

            self._processor = AutoImageProcessor.from_pretrained(model_name)
            self._model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name
            ).to(self.device).eval()

            # Build id → label mapping from model config
            self._id2label = self._model.config.id2label
            print(f"[Mask2Former] Loaded: Swin-L Cityscapes panoptic "
                  f"({len(self._id2label)} classes)")

        except ImportError as e:
            print(f"[Mask2Former] ERROR: transformers not installed: {e}")
            print("[Mask2Former] Install with: pip install transformers")
            self._model = None
        except Exception as e:
            print(f"[Mask2Former] ERROR: Failed to load: {e}")
            self._model = None

    def is_available(self) -> bool:
        return self._model is not None

    def segment_cropped(self, frame: np.ndarray,
                        detections: list[dict],
                        expand_ratio: float = None) -> list[dict]:
        """
        Run instance segmentation on expanded YOLO crops.

        Args:
            frame: full BGR frame (H, W, 3)
            detections: list of dicts with 'bbox', 'label'
            expand_ratio: bbox expansion ratio (default from config)

        Returns:
            List of {mask, bbox, class_id, yolo_label, score} dicts.
            Each mask is in full-frame coordinates, clipped to YOLO bbox.
        """
        global _logged_available_labels

        if expand_ratio is None:
            expand_ratio = cfg.MASK2FORMER_EXPAND

        if self._model is None:
            return []

        H, W = frame.shape[:2]
        results = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            yolo_label = det['label']

            # 1. Expand bbox
            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ex1 = max(0, int(cx - bw * (1 + expand_ratio) / 2))
            ey1 = max(0, int(cy - bh * (1 + expand_ratio) / 2))
            ex2 = min(W, int(cx + bw * (1 + expand_ratio) / 2))
            ey2 = min(H, int(cy + bh * (1 + expand_ratio) / 2))

            # 2. Crop from frame
            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue

            # 3. Run Mask2Former on crop
            try:
                instances = self._run_inference(crop)
            except Exception as e:
                print(f"[Mask2Former] Inference error on crop: {e}")
                continue

            # Log available labels on first frame
            if not _logged_available_labels and instances:
                available = set(inst['label'] for inst in instances)
                print(f"[Mask2Former] Available labels in output: {sorted(available)}")
                _logged_available_labels = True

            # 4. Filter by YOLO class
            allowed_labels = _YOLO_TO_CITYSCAPES.get(yolo_label, set())
            if not allowed_labels and yolo_label not in _warned_classes:
                _warned_classes.add(yolo_label)
                print(f"[Mask2Former] WARNING: No class mapping for YOLO label "
                      f"'{yolo_label}'. Add to _YOLO_TO_CITYSCAPES.")

            class_matched = [
                inst for inst in instances
                if inst['label'] in allowed_labels
            ]

            if not class_matched:
                continue

            # 5. Find best IoU match with YOLO bbox
            crop_h, crop_w = crop.shape[:2]

            yolo_box_mask = np.zeros((H, W), dtype=np.uint8)
            ox1 = max(0, int(x1))
            oy1 = max(0, int(y1))
            ox2 = min(W, int(x2))
            oy2 = min(H, int(y2))
            yolo_box_mask[oy1:oy2, ox1:ox2] = 1

            best_mask = None
            best_iou = 0.0
            best_score = 0.0

            for inst in class_matched:
                # Resize mask to crop dimensions
                m = cv2.resize(
                    inst['mask'].astype(np.uint8), (crop_w, crop_h),
                    interpolation=cv2.INTER_NEAREST
                )

                # Place in full frame
                full = np.zeros((H, W), dtype=np.uint8)
                full[ey1:ey2, ex1:ex2] = m

                # IoU with YOLO bbox
                intersection = np.sum(full & yolo_box_mask)
                union = np.sum(full | yolo_box_mask)
                iou = intersection / max(union, 1)

                if iou > best_iou:
                    best_iou = iou
                    best_mask = full
                    best_score = inst['score']

            if best_mask is None:
                continue

            # 6. Clip to original YOLO bbox
            clipped = np.zeros_like(best_mask)
            clipped[oy1:oy2, ox1:ox2] = best_mask[oy1:oy2, ox1:ox2]

            results.append({
                'mask': clipped,
                'bbox': [ox1, oy1, ox2, oy2],
                'class_id': 0,  # not used downstream, kept for compat
                'yolo_label': yolo_label,
                'score': best_score,
            })

        return results

    def _run_inference(self, crop_bgr: np.ndarray) -> list[dict]:
        """Run Mask2Former panoptic segmentation on a BGR crop.

        Returns list of {mask, label, score, class_id} dicts.
        """
        from PIL import Image

        # Convert BGR → RGB PIL
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process panoptic segmentation (suppress label_ids_to_fuse warning)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*label_ids_to_fuse.*")
            result = self._processor.post_process_panoptic_segmentation(
                outputs, target_sizes=[pil_img.size[::-1]]
            )[0]

        segmentation = result["segmentation"].cpu().numpy()
        segments_info = result["segments_info"]

        instances = []
        for seg_info in segments_info:
            seg_id = seg_info["id"]
            class_id = seg_info["label_id"]
            score = seg_info.get("score", 1.0)
            label = self._id2label.get(class_id, f"class_{class_id}")

            mask = (segmentation == seg_id).astype(np.uint8)
            instances.append({
                'mask': mask,
                'label': label,
                'class_id': class_id,
                'score': score,
            })

        return instances


# ─── Spatial mask-to-detection matching ──────────────────────────────────────

def match_mask_to_detection(det: dict, masks: list[dict]) -> Optional[np.ndarray]:
    """
    Match a tracked detection to the best mask by bbox IoU + class.

    Args:
        det: detection dict with 'bbox' [x1,y1,x2,y2] and 'label'
        masks: list of mask dicts from segment_cropped(), each with
               'mask', 'bbox', 'yolo_label', 'score'

    Returns:
        Best-matching mask (H, W) uint8 array, or None if no match.
    """
    if not masks:
        return None

    det_label = det.get('label', '')
    det_bbox = det.get('bbox')
    if det_bbox is None:
        return None

    best_mask = None
    best_iou = 0.0

    for m in masks:
        # Class must match
        if m['yolo_label'] != det_label:
            continue

        iou = _bbox_iou(det_bbox, m['bbox'])
        if iou > best_iou:
            best_iou = iou
            best_mask = m['mask']

    # Require minimum IoU to avoid spurious matches
    if best_iou < 0.1:
        return None

    return best_mask


def _bbox_iou(a: list, b: list) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)
