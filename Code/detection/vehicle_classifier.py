"""
Vehicle Sub-Classifier — Phase 3.

For every vehicle detection, produces:
  - subclass: sedan | suv | hatchback | pickup | truck | motorcycle | bicycle
  - subclass_confidence: "high" | "medium" | "low"

Rules:
  - YOLO labels motorcycle/bicycle/truck/bus → trust directly, no DINOv2/Qwen
  - YOLO label "car" → DINOv2 probe → Qwen fallback → default sedan
  - If DINOv2/Qwen returns truck/motorcycle for a YOLO "car" → random sedan/suv
  - Pickup detection: run descriptive Qwen prompt on DINOv2-suv results
"""

from __future__ import annotations
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ─── Prompt template for car subclassification ─────────────────────────────

VEHICLE_PROMPT = """Look at this vehicle in the image.

What type of vehicle is it? Choose exactly one:
- sedan: a regular 4-door car with a separate trunk compartment, low roofline, three-box design
- suv: tall vehicle, high ground clearance, boxy shape, raised ride height, no open bed
- hatchback: small compact car with a sloped rear liftgate, two-box design, no separate trunk
- pickup: a truck with an open cargo bed behind the cabin. The bed is flat and exposed (e.g., Ford F-150, Toyota Tacoma, Ram 1500). Key feature: visible open bed at the rear.

Reply with ONLY a JSON object, no explanation:
{"subclass": "<your choice>", "confidence": "<high|medium|low>"}

If the image is too blurry, too small, or you cannot clearly identify \
the vehicle type, set confidence to "low"."""

PICKUP_PROMPT = """Look at this vehicle carefully.

A pickup truck is a rugged vehicle defined by two main sections: a forward \
enclosed cab for passengers and a rear open-top cargo bed with low sides and \
a tailgate. It typically features high ground clearance, large wheels, and a \
blunt, powerful front grille.

Is this vehicle a pickup truck? Answer with ONLY a JSON object:
{"is_pickup": true, "confidence": "<high|medium|low>"}
or
{"is_pickup": false, "confidence": "<high|medium|low>"}"""

# Only sedan/suv/hatchback/pickup for YOLO "car" subclassification
CAR_SUBCLASSES = {"sedan", "suv", "hatchback", "pickup"}
VALID_SUBCLASSES = {"sedan", "suv", "hatchback", "pickup", "truck", "motorcycle", "bicycle"}
VALID_CONFIDENCES = {"high", "medium", "low"}

MIN_CROP_AREA = 3000
MIN_CROP_DIM = 224
BBOX_EXPAND = 0.10


# ─── Geometric heuristic for vehicle subclass ─────────────────────────────

def _geometric_subclass(bbox: list, position_3d: list | None = None) -> tuple[str, str]:
    """
    Estimate vehicle subclass from bounding box geometry.

    Returns (subclass, confidence). Used as a prior before DINOv2/Qwen,
    and as a tiebreaker when DINOv2 confidence is low.

    Heuristics:
      - aspect_ratio = width / height
      - SUVs: taller relative to width (aspect_ratio < 1.4), larger bbox
      - Hatchbacks: compact, smaller bbox area
      - Pickups: wide, large area, moderate height
      - Sedans: typical aspect ratio 1.4-2.0
    """
    x1, y1, x2, y2 = bbox[:4]
    w = x2 - x1
    h = y2 - y1
    if h < 10 or w < 10:
        return "sedan", "low"

    aspect = w / h
    area = w * h

    # Very small bbox — far away, can't tell much
    if area < 2000:
        return "sedan", "low"

    # Tall and boxy → SUV
    if aspect < 1.3 and area > 15000:
        return "suv", "medium"
    if aspect < 1.1 and area > 8000:
        return "suv", "medium"

    # Small and compact → hatchback
    if area < 25000 and aspect < 1.6:
        return "hatchback", "low"

    # Wide and large → could be pickup or SUV
    if aspect > 2.0 and area > 40000:
        return "pickup", "low"

    # Medium aspect, large area → sedan or suv
    if aspect >= 1.3 and aspect <= 1.7 and area > 30000:
        # Use depth if available — closer vehicles appear larger
        if position_3d and position_3d[1] > 15:
            return "suv", "low"  # far + large bbox = probably bigger vehicle

    return "sedan", "low"

# ─── Blender asset mapping ──────────────────────────────────────────────────

ASSET_MAP = {
    "sedan":      "Vehicles/SedanAndHatchback.blend",
    "suv":        "Vehicles/SUV.blend",
    "hatchback":  "Vehicles/SedanAndHatchback.blend",
    "pickup":     "Vehicles/PickupTruck.blend",
    "truck":      "Vehicles/Truck.blend",
    "motorcycle": "Vehicles/Motorcycle.blend",
    "bicycle":    "Vehicles/Bicycle.blend",
}


# ─── Shared Qwen model singleton ────────────────────────────────────────────

_qwen_model = None
_qwen_processor = None


def get_qwen_model():
    """Return the shared (model, processor) tuple. Loads on first call."""
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return _qwen_model, _qwen_processor

    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"[VehicleClassifier] Loading {cfg.QWEN_VL_MODEL} …")
        model_dtype = getattr(torch, cfg.QWEN_VL_DTYPE, torch.float16)
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.QWEN_VL_MODEL,
            dtype=model_dtype,
            device_map=cfg.QWEN_VL_DEVICE_MAP,
        )
        _qwen_processor = AutoProcessor.from_pretrained(cfg.QWEN_VL_MODEL)
        print("[VehicleClassifier] Qwen2.5-VL loaded successfully.")
    except Exception as e:
        print(f"[VehicleClassifier] Failed to load Qwen2.5-VL: {e}")
        _qwen_model = None
        _qwen_processor = None

    return _qwen_model, _qwen_processor


# ─── Majority Vote Tracker ─────────────────────────────────────────────────

class SubclassVoteTracker:
    def __init__(self, vote_window: int = 7, min_votes_to_change: int = 3,
                 ema_alpha: float = 0.3, stale_frames: int = 30):
        self.vote_window = vote_window
        self.min_votes = min_votes_to_change
        self.alpha = ema_alpha
        self.stale_frames = stale_frames
        self._votes: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=vote_window)
        )
        self._current: dict[int, str] = {}
        self._yaw_ema: dict[int, float] = {}
        self._last_seen_frame: dict[int, int] = defaultdict(int)

    def update(self, track_id: int, subclass: str,
               confidence: str, yaw_deg: float,
               frame_idx: int = 0) -> tuple[str, float]:
        if frame_idx - self._last_seen_frame[track_id] > self.stale_frames:
            self._votes[track_id].clear()
            self._current.pop(track_id, None)
            self._yaw_ema.pop(track_id, None)
        self._last_seen_frame[track_id] = frame_idx

        self._votes[track_id].append(subclass)
        counts = Counter(self._votes[track_id])
        majority_subclass, majority_count = counts.most_common(1)[0]

        if track_id not in self._current:
            self._current[track_id] = majority_subclass
        elif majority_subclass != self._current[track_id]:
            if majority_count >= self.min_votes:
                self._current[track_id] = majority_subclass

        # Yaw EMA
        if track_id not in self._yaw_ema:
            self._yaw_ema[track_id] = yaw_deg
        else:
            prev = self._yaw_ema[track_id]
            delta = ((yaw_deg - prev + 180) % 360) - 180
            self._yaw_ema[track_id] = prev + self.alpha * delta

        return self._current[track_id], self._yaw_ema[track_id]

    def prune(self, active_ids: set):
        for store in (self._votes, self._current, self._yaw_ema,
                      self._last_seen_frame):
            stale = set(store.keys()) - active_ids
            for tid in stale:
                del store[tid]

    def reset(self):
        self._votes.clear()
        self._current.clear()
        self._yaw_ema.clear()
        self._last_seen_frame.clear()


# ─── Vehicle Classifier ─────────────────────────────────────────────────────

class VehicleClassifier:
    _dino_logged = False

    def __init__(self):
        self._qwen_available = None
        self._dino_clf = None
        self._dino_checked = False
        self.voter = SubclassVoteTracker(
            vote_window=7,
            min_votes_to_change=3,
            ema_alpha=cfg.SUBCLASS_EMA_ALPHA,
        )

    def _get_dino_classifier(self):
        if not self._dino_checked:
            self._dino_checked = True
            try:
                from detection.dino_vehicle_classifier import get_classifier
                self._dino_clf = get_classifier()
                if self._dino_clf.is_available():
                    if not VehicleClassifier._dino_logged:
                        print("[VehicleClassifier] Using DINOv2 probe (fast). "
                              "Fallback: Qwen -> default sedan")
                        VehicleClassifier._dino_logged = True
                else:
                    self._dino_clf = None
            except Exception:
                self._dino_clf = None
        return self._dino_clf

    def classify_batch(self, vehicle_dets: list[dict],
                       frames: dict[str, np.ndarray],
                       frame_idx: int = 0) -> list[dict]:
        if not vehicle_dets:
            return vehicle_dets

        # ── Step 0: Trust YOLO for non-car types (no DINOv2/Qwen needed) ──
        results: list[dict | None] = [None] * len(vehicle_dets)

        for i, det in enumerate(vehicle_dets):
            label = det.get("label", "car")
            if label in ("motorcycle", "bicycle"):
                results[i] = {"subclass": label, "confidence": "high"}
            elif label in ("truck", "bus"):
                results[i] = {"subclass": "truck", "confidence": "high"}

        # ── Step 1: Cars only — DINOv2 on closest 3 within 30m ──
        cars = [(i, det) for i, det in enumerate(vehicle_dets)
                if det.get("label") == "car" and results[i] is None]
        cars_with_depth = [
            (i, det, det.get("position_3d", [0, 999, 0])[1])
            for i, det in cars
        ]
        cars_with_depth.sort(key=lambda x: x[2])
        cars_to_classify = [(i, det) for i, det, d in cars_with_depth[:3] if d < 30.0]
        cars_to_classify_set = {i for i, _ in cars_to_classify}

        # Cars beyond top-3: use geometric heuristic instead of always sedan
        for i, det in cars:
            if i not in cars_to_classify_set:
                geo_sub, geo_conf = _geometric_subclass(
                    det["bbox"], det.get("position_3d"))
                results[i] = {"subclass": geo_sub, "confidence": geo_conf}

        # DINOv2 probe
        dino_clf = self._get_dino_classifier()
        if dino_clf is not None and cars_to_classify:
            crops = []
            crop_indices = []
            for i, det in cars_to_classify:
                cam_id = det.get("camera", "front")
                frame = frames.get(cam_id)
                if frame is not None:
                    crop = self._prepare_crop(frame, det["bbox"])
                    if crop is not None:
                        crops.append(crop)
                        crop_indices.append(i)

            if crops:
                batch_results = dino_clf.classify_batch(crops)
                for idx, br in zip(crop_indices, batch_results):
                    results[idx] = br

        # ── Step 2: Sanitize DINOv2 results for YOLO "car" ──
        needs_qwen = []
        for i, det in cars_to_classify:
            dr = results[i]
            if dr is not None and dr.get("confidence_label") != "low":
                subclass = dr["subclass"]
                # GUARD: DINOv2 says truck/motorcycle/bicycle for a YOLO "car"
                # → use geometric heuristic instead of random
                if subclass not in CAR_SUBCLASSES:
                    geo_sub, _ = _geometric_subclass(
                        det["bbox"], det.get("position_3d"))
                    subclass = geo_sub
                results[i] = {
                    "subclass": subclass,
                    "confidence": dr.get("confidence_label", dr.get("confidence", "medium")),
                }
            else:
                # DINOv2 low confidence — fall through to Qwen
                needs_qwen.append(i)

        # ── Step 3: Qwen fallback for DINOv2 failures ──
        if needs_qwen:
            qwen_results = self._classify_batch_qwen(vehicle_dets, frames)
            for i in needs_qwen:
                if qwen_results and qwen_results[i] is not None:
                    subclass = qwen_results[i]["subclass"]
                    # GUARD: Qwen says truck/motorcycle for YOLO "car"
                    if subclass not in CAR_SUBCLASSES:
                        geo_sub, _ = _geometric_subclass(
                            vehicle_dets[i]["bbox"],
                            vehicle_dets[i].get("position_3d"))
                        subclass = geo_sub
                    results[i] = {"subclass": subclass,
                                  "confidence": qwen_results[i].get("confidence", "medium")}
                else:
                    geo_sub, geo_conf = _geometric_subclass(
                        vehicle_dets[i]["bbox"],
                        vehicle_dets[i].get("position_3d"))
                    results[i] = {"subclass": geo_sub, "confidence": geo_conf}

        # ── Step 4: Pickup refinement — run descriptive prompt on SUVs ──
        for i, det in cars_to_classify:
            if results[i]["subclass"] == "suv":
                is_pickup = self._check_pickup(det, frames)
                if is_pickup:
                    results[i] = {"subclass": "pickup", "confidence": "medium"}

        # ── Step 5: Majority vote smoothing ──
        for i, det in enumerate(vehicle_dets):
            result = results[i]
            track_id = det.get("track_id", 0)
            subclass = result["subclass"]
            conf = result.get("confidence", "high")

            recon = det.get("reconstruction") or {}
            recon_ori = recon.get("orientation")
            if recon_ori is not None and not math.isnan(recon_ori):
                yaw = float(recon_ori)
            else:
                yaw = det.get("yaw_deg") or 0.0

            subclass, yaw = self.voter.update(track_id, subclass, conf, yaw,
                                              frame_idx=frame_idx)

            # Final guard: YOLO "car" can never become truck/motorcycle/bicycle
            if det.get("label") == "car" and subclass not in CAR_SUBCLASSES:
                subclass = "sedan"

            det["subclass"] = subclass
            det["subclass_confidence"] = conf
            det["yaw_deg"] = round(yaw, 1)

        return vehicle_dets

    def _check_pickup(self, det: dict, frames: dict) -> bool:
        """Run descriptive pickup prompt on a vehicle classified as SUV."""
        if self._qwen_available is False:
            return False

        model, processor = get_qwen_model()
        if model is None or processor is None:
            return False

        try:
            import torch
            from PIL import Image

            cam_id = det.get("camera", "front")
            frame = frames.get(cam_id)
            if frame is None:
                return False

            crop = self._prepare_crop(frame, det["bbox"])
            if crop is None:
                return False

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)

            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": PICKUP_PROMPT},
            ]}]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                image_inputs = [pil_img]
                video_inputs = None

            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=32)

            answer = processor.batch_decode(
                generated[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0].strip()

            # Parse response
            try:
                data = json.loads(answer)
            except json.JSONDecodeError:
                match = re.search(r'\{[^}]+\}', answer)
                if match:
                    try:
                        data = json.loads(match.group())
                    except json.JSONDecodeError:
                        return False
                else:
                    return False

            return bool(data.get("is_pickup", False))

        except Exception:
            return False

    def classify_single(self, frame: np.ndarray, bbox: list,
                        label: str, track_id: int = 0) -> dict:
        if label in ("motorcycle", "bicycle"):
            return {"subclass": label, "confidence": "high", "yaw_deg": 0.0}
        if label in ("truck", "bus"):
            return {"subclass": "truck", "confidence": "high", "yaw_deg": 0.0}

        result = self._run_qwen_single_safe(frame, bbox)
        if result is not None:
            subclass = result["subclass"]
            if subclass not in CAR_SUBCLASSES:
                subclass = random.choice(["sedan", "suv", "hatchback"])
            conf = result.get("confidence", "high")
        else:
            subclass = "sedan"
            conf = "low"

        yaw = 0.0
        subclass, yaw = self.voter.update(track_id, subclass, conf, yaw)
        return {"subclass": subclass, "confidence": conf, "yaw_deg": round(yaw, 1)}

    def prune(self, active_ids: set):
        self.voter.prune(active_ids)

    def reset_votes(self):
        self.voter.reset()

    # ─── Crop Preprocessing ─────────────────────────────────────────────────

    @staticmethod
    def _prepare_crop(frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w = x2 - x1
        h = y2 - y1
        area = w * h

        if area < MIN_CROP_AREA:
            return None

        H, W = frame.shape[:2]
        expand_w = int(w * BBOX_EXPAND)
        expand_h = int(h * BBOX_EXPAND)
        x1 = max(0, x1 - expand_w)
        y1 = max(0, y1 - expand_h)
        x2 = min(W, x2 + expand_w)
        y2 = min(H, y2 + expand_h)

        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch < 10 or cw < 10:
            return None

        short_side = min(ch, cw)
        if short_side < MIN_CROP_DIM:
            scale = MIN_CROP_DIM / short_side
            new_w = int(cw * scale)
            new_h = int(ch * scale)
            crop = cv2.resize(crop, (new_w, new_h),
                              interpolation=cv2.INTER_LANCZOS4)

        return crop

    # ─── Qwen Batch Inference ───────────────────────────────────────────────

    def _classify_batch_qwen(self, vehicle_dets: list[dict],
                             frames: dict) -> list[Optional[dict]]:
        if self._qwen_available is False:
            return [None] * len(vehicle_dets)

        model, processor = get_qwen_model()
        if model is None or processor is None:
            self._qwen_available = False
            return [None] * len(vehicle_dets)
        self._qwen_available = True

        results = []
        for det in vehicle_dets:
            label = det.get("label", "car")
            # Only run Qwen on "car" labels
            if label != "car":
                results.append(None)
                continue

            cam_id = det.get("camera", "front")
            frame = frames.get(cam_id)
            if frame is None:
                results.append(None)
                continue

            result = self._run_qwen_single(model, processor, frame, det["bbox"])
            results.append(result)

        return results

    def _run_qwen_single_safe(self, frame: np.ndarray,
                               bbox: list) -> Optional[dict]:
        if self._qwen_available is False:
            return None
        model, processor = get_qwen_model()
        if model is None or processor is None:
            self._qwen_available = False
            return None
        self._qwen_available = True
        try:
            return self._run_qwen_single(model, processor, frame, bbox)
        except Exception:
            return None

    def _run_qwen_single(self, model, processor,
                          frame: np.ndarray, bbox: list) -> Optional[dict]:
        try:
            import torch
            from PIL import Image

            crop = self._prepare_crop(frame, bbox)
            if crop is None:
                return None

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)

            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": VEHICLE_PROMPT},
            ]}]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                image_inputs = [pil_img]
                video_inputs = None

            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=64)

            output_text = processor.batch_decode(
                generated[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0].strip()

            return self._parse_qwen_response(output_text)

        except Exception as e:
            print(f"[VehicleClassifier] Qwen inference error: {e}")
            return None

    def _parse_qwen_response(self, text: str) -> Optional[dict]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        subclass = str(data.get("subclass", "")).lower().strip()
        if subclass not in VALID_SUBCLASSES:
            return None

        conf = str(data.get("confidence", "low")).lower().strip()
        if conf not in VALID_CONFIDENCES:
            conf = "low"

        return {"subclass": subclass, "confidence": conf}
