"""
Road Sign Detector — Phase 1 & 2.

Detection priority:
  1. bhaskrr traffic-sign-detection YOLOv11 model (15 classes: traffic lights,
     speed limits 20-120, stop sign) — SOLE detector
  2. EasyOCR fallback for speed limit reading on unclassified crops
  3. Qwen2.5-VL-3B fallback for sign type + speed value (if OCR fails)

Phase 2 also: ground arrow detection via bird's-eye view.

NOTE: bhaskrr model has no "Yellow Light" class.  Yellow light colour
      is handled by HSV fallback in traffic_light_classifier.py.
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from detection.object_detector import make_detection


# ─── Qwen 3B sign prompt ──────────────────────────────────────────────────────

SIGN_PROMPT = """You are analyzing a road sign crop from an autonomous driving camera.

Answer ONLY with a valid JSON object, no other text:
{
  "type": "<one of: stop, speed_limit, ground_arrow, other, unknown>",
  "value": "<speed limit number as string if speed_limit, else arrow direction (left/straight/right) if ground_arrow, else null>"
}
"""

# ─── bhaskrr model class mapping (15 classes) ────────────────────────────────
# Maps the bhaskrr model's class names to our schema.
def _bbox_iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


BHASKRR_CLASS_MAP = {
    # 15 classes total: Green Light, Red Light, Stop, Speed Limit 20-120
    # NOTE: No "Yellow Light" class — yellow is handled by HSV in traffic_light_classifier
    "Green Light":     {"category": "traffic_light", "label": "traffic_light", "tl_color": "green"},
    "Red Light":       {"category": "traffic_light", "label": "traffic_light", "tl_color": "red"},
    "Stop":            {"category": "road_sign",     "label": "stop_sign",     "sign_type": "stop", "speed_value": None},
    "Speed Limit 20":  {"category": "road_sign",     "label": "speed_limit",   "sign_type": "speed_limit", "speed_value": 20},
    "Speed Limit 30":  {"category": "road_sign",     "label": "speed_limit",   "sign_type": "speed_limit", "speed_value": 30},
    "Speed Limit 40":  {"category": "road_sign",     "label": "speed_limit",   "sign_type": "speed_limit", "speed_value": 40},
    "Speed Limit 50":  {"category": "road_sign",     "label": "speed_limit",   "sign_type": "speed_limit", "speed_value": 50},
    "Speed Limit 60":  {"category": "road_sign",     "label": "speed_limit",   "sign_type": "speed_limit", "speed_value": 60},
    "Speed Limit 70":  {"category": "road_sign",     "label": "speed_limit",   "sign_type": "speed_limit", "speed_value": 70},
    "Speed Limit 80":  {"category": "road_sign",     "label": "speed_limit",   "sign_type": "speed_limit", "speed_value": 80},
    "Speed Limit 90":  {"category": "road_sign",     "label": "speed_limit",   "sign_type": "speed_limit", "speed_value": 90},
    "Speed Limit 100": {"category": "road_sign",     "label": "speed_limit",   "sign_type": "speed_limit", "speed_value": 100},
    "Speed Limit 110": {"category": "road_sign",     "label": "speed_limit",   "sign_type": "speed_limit", "speed_value": 110},
    "Speed Limit 120": {"category": "road_sign",     "label": "speed_limit",   "sign_type": "speed_limit", "speed_value": 120},
}


# ─── Qwen 3B singleton (lighter than 7B, used only for sign OCR fallback) ────

_qwen3b_model = None
_qwen3b_processor = None


def _get_qwen3b():
    """Load Qwen2.5-VL-3B lazily (singleton). Used for sign crop classification."""
    global _qwen3b_model, _qwen3b_processor
    if _qwen3b_model is not None:
        return _qwen3b_model, _qwen3b_processor

    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"[RoadSignDetector] Loading {cfg.QWEN_VL_3B_MODEL} …")
        model_dtype = getattr(torch, cfg.QWEN_VL_3B_DTYPE, torch.float16)
        _qwen3b_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.QWEN_VL_3B_MODEL,
            dtype=model_dtype,
            device_map=cfg.QWEN_VL_3B_DEVICE_MAP,
        )
        _qwen3b_processor = AutoProcessor.from_pretrained(cfg.QWEN_VL_3B_MODEL)
        print("[RoadSignDetector] Qwen2.5-VL-3B loaded.")
    except Exception as e:
        print(f"[RoadSignDetector] Failed to load Qwen2.5-VL-3B: {e}")
        _qwen3b_model = None
        _qwen3b_processor = None

    return _qwen3b_model, _qwen3b_processor


class RoadSignDetector:
    """
    Detects and classifies road signs (stop, speed limits) and traffic lights.

    Sole model:  bhaskrr YOLOv11n (15 classes, traffic_sign_detector.pt).
    Fallback for unclassified COCO sign crops:  EasyOCR → Qwen2.5-VL-3B.
    """

    def __init__(self, phase: int = 1):
        self.phase = phase
        self._bhaskrr_model = None
        self._ocr: Optional[object] = None
        self._qwen3b_available = None  # lazy check
        self._yoloe = None               # lazy-loaded YOLOE for speed signs
        self._yoloe_loaded = False

        self._load_bhaskrr_model()
        if phase >= 2:
            self._load_ocr()

    # ─── Model Loading ──────────────────────────────────────────────────────

    def _load_bhaskrr_model(self):
        """Load the bhaskrr traffic sign YOLOv11 model (SOLE sign detector)."""
        candidates = [
            cfg.BASE_DIR / "traffic_sign_detector.pt",
            cfg.BASE_DIR / "weights" / "traffic_sign_detector.pt",
        ]

        model_path = None
        for p in candidates:
            if p.exists():
                model_path = p
                break

        if model_path is None:
            print("[RoadSignDetector] ERROR: bhaskrr model (traffic_sign_detector.pt) "
                  "not found at any expected location:")
            for p in candidates:
                print(f"  - {p}")
            print("[RoadSignDetector] Sign/light detection will rely on COCO classes + OCR only.")
            return

        try:
            from ultralytics import YOLO
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._bhaskrr_model = YOLO(str(model_path))
            self._bhaskrr_model.to(device)

            n_classes = len(self._bhaskrr_model.names)
            print(f"[RoadSignDetector] Model loaded: {model_path}")
            print(f"[RoadSignDetector]   Classes ({n_classes}): {self._bhaskrr_model.names}")
        except Exception as e:
            print(f"[RoadSignDetector] Failed to load bhaskrr model: {e}")
            self._bhaskrr_model = None

    def get_sign_model(self):
        """Return the bhaskrr model (for sharing with TrafficLightClassifier)."""
        return self._bhaskrr_model

    # ─── Primary: bhaskrr Model Detection ───────────────────────────────────

    def _detect_bhaskrr(self, frame: np.ndarray) -> dict:
        """Run the bhaskrr 15-class model on a full frame."""
        if self._bhaskrr_model is None:
            return {"traffic_lights": [], "road_signs": []}

        results = self._bhaskrr_model(
            frame,
            conf=cfg.SIGN_CONF_THRESHOLD,
            iou=cfg.SIGN_IOU_THRESHOLD,
            imgsz=cfg.SIGN_IMGSIZE,
            verbose=False,
        )[0]

        traffic_lights = []
        road_signs = []

        for box in results.boxes:
            cid = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [x1, y1, x2, y2]
            cls_name = self._bhaskrr_model.names[cid]

            mapping = BHASKRR_CLASS_MAP.get(cls_name)
            if mapping is None:
                continue

            # Skip low-confidence speed limit detections (bhaskrr over-detects)
            if mapping.get("sign_type") == "speed_limit" and conf < 0.45:
                continue

            det = make_detection(bbox, mapping["label"], conf)
            det["source_model"] = "bhaskrr_yolov11"

            if mapping["category"] == "traffic_light":
                det["tl_color"] = mapping["tl_color"]
                det["tl_arrow"] = None
                traffic_lights.append(det)

            elif mapping["category"] == "road_sign":
                det["sign_type"] = mapping["sign_type"]
                det["speed_value"] = mapping.get("speed_value")
                # Verify speed limit detections — bhaskrr was trained on
                # European circular signs and often misclassifies arbitrary
                # boards/signs as speed limits.
                if mapping["sign_type"] == "speed_limit":
                    verified = self._verify_speed_limit(frame, bbox,
                                                        mapping.get("speed_value"))
                    if verified is None:
                        # Not a speed limit — skip or reclassify as other
                        det["sign_type"] = "other"
                        det["label"] = "road_sign"
                        det["speed_value"] = None
                    else:
                        det["speed_value"] = verified
                road_signs.append(det)

        return {"traffic_lights": traffic_lights, "road_signs": road_signs}

    # ─── Speed Limit Verification ──────────────────────────────────────────

    def _verify_speed_limit(self, frame: np.ndarray, bbox: list,
                            claimed_speed: Optional[int]) -> Optional[int]:
        """
        Verify a bhaskrr speed-limit detection using OCR → Qwen.
        Returns the verified speed value, or None if it's not a speed limit.
        """
        # Try OCR first — fast and reliable when digits are present
        speed = self._read_speed(frame, bbox)
        if speed is not None:
            return speed

        # OCR found no digits — ask Qwen if available
        if self.phase >= 2:
            qwen_result = self._classify_qwen3b(frame, bbox)
            if qwen_result is not None:
                if qwen_result["type"] == "speed_limit":
                    return qwen_result.get("speed_value") or claimed_speed
                # Qwen says it's not a speed limit
                return None

        # No OCR, no Qwen — if OCR is loaded but found nothing, reject
        # (a real speed sign would have readable digits)
        if self._ocr is not None:
            return None

        # OCR not loaded at all (Phase 1) — trust bhaskrr
        return claimed_speed

    # ─── Combined Detection (public API) ────────────────────────────────────

    def detect_signs_and_lights(self, frame: np.ndarray) -> dict:
        """
        Run sign detection on a full frame.
        Uses bhaskrr model + YOLO-World for speed limit signs (Phase 2).
        """
        result = self._detect_bhaskrr(frame)

        # Phase 2: also run YOLOE for speed limit signs (catches US rectangular signs)
        if self.phase >= 2:
            speed_signs = self._detect_speed_signs_yoloe(frame)
            if speed_signs:
                # Deduplicate against bhaskrr results
                for ss in speed_signs:
                    overlap = any(
                        _bbox_iou(ss["bbox"], existing["bbox"]) > 0.4
                        for existing in result["road_signs"]
                    )
                    if not overlap:
                        result["road_signs"].append(ss)

        return result

    def _load_yoloe(self):
        """Lazy-load YOLOE for prompt-based speed sign detection."""
        if self._yoloe_loaded:
            return
        self._yoloe_loaded = True
        try:
            from ultralytics import YOLOE
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._yoloe = YOLOE("yoloe-11s-seg.pt")
            self._yoloe.set_classes([
                "speed limit sign",
                "road sign with numbers",
            ])
            self._yoloe.to(device)
            print("[RoadSignDetector] YOLOE loaded for speed sign detection.")
        except Exception as e:
            print(f"[RoadSignDetector] YOLOE unavailable: {e}")
            self._yoloe = None

    def _detect_speed_signs_yoloe(self, frame: np.ndarray) -> list[dict]:
        """Detect speed limit signs using YOLOE prompt-based detection."""
        self._load_yoloe()
        if self._yoloe is None:
            return []

        try:
            results = self._yoloe(
                frame, conf=0.30, iou=0.45, imgsz=640, verbose=False
            )[0]
        except Exception:
            return []

        signs = []
        for box in results.boxes:
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [x1, y1, x2, y2]
            area = (x2 - x1) * (y2 - y1)
            if area < 200:
                continue  # too small

            # Classify via OCR → Qwen VLM — must confirm it's a speed sign
            cls_result = self.classify_sign_crop(frame, bbox)
            sign_type = cls_result["type"]

            # Reject if OCR/Qwen couldn't confirm it's a speed limit
            if sign_type not in ("speed_limit", "stop"):
                continue

            label = "speed_limit" if sign_type == "speed_limit" else "stop_sign"
            det = make_detection(bbox, label, conf)
            det["sign_type"] = sign_type
            det["speed_value"] = cls_result.get("speed_value")
            det["source_model"] = "yoloe"
            signs.append(det)

        return signs

    # ─── IoU helper ────────────────────────────────────────────────────────

    # ─── Fallback: Enrich COCO-only detections ──────────────────────────────

    def classify(self, frame: np.ndarray, bbox: list, yolo_label: str) -> dict:
        """
        Enrich a YOLO12 COCO road-sign detection (fallback path).
        Chain: direct label → EasyOCR → Qwen 3B.
        """
        if yolo_label == "stop_sign":
            return {"type": "stop", "speed_value": None}

        # Try EasyOCR first (fast, lightweight)
        if yolo_label in ("speed_limit", "stop_sign"):
            speed = self._read_speed(frame, bbox)
            if speed is not None:
                return {"type": "speed_limit", "speed_value": speed}

        # EasyOCR failed or not a speed sign → try Qwen 3B
        if self.phase >= 2:
            qwen_result = self._classify_qwen3b(frame, bbox)
            if qwen_result is not None:
                return qwen_result

        return {"type": "other", "speed_value": None}

    def classify_sign_crop(self, frame: np.ndarray, bbox: list) -> dict:
        """Classify an arbitrary sign crop: EasyOCR → Qwen 3B."""
        # Try OCR first
        speed = self._read_speed(frame, bbox)
        if speed is not None:
            return {"type": "speed_limit", "speed_value": speed}

        # Fallback to Qwen 3B
        result = self._classify_qwen3b(frame, bbox)
        if result is not None:
            return result
        return {"type": "unknown", "speed_value": None}

    # ─── Qwen2.5-VL-3B Sign Classification ─────────────────────────────────

    def _classify_qwen3b(self, frame: np.ndarray, bbox: list) -> Optional[dict]:
        """Use Qwen2.5-VL-3B to classify sign crop (lighter than 7B)."""
        if self.phase < 2:
            return None
        if self._qwen3b_available is False:
            return None

        model, processor = _get_qwen3b()
        if model is None or processor is None:
            self._qwen3b_available = False
            return None
        self._qwen3b_available = True

        try:
            import torch
            from PIL import Image

            x1, y1, x2, y2 = [int(v) for v in bbox]
            H, W = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            crop = frame[y1:y2, x1:x2]

            if crop.shape[0] < 10 or crop.shape[1] < 10:
                return None

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": SIGN_PROMPT},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                image_inputs = [pil_img]
                video_inputs = None

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=64)

            output_text = processor.batch_decode(
                generated[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0].strip()

            return self._parse_qwen_sign_response(output_text)

        except Exception as e:
            print(f"[RoadSignDetector] Qwen-3B inference error: {e}")
            return None

    def _parse_qwen_sign_response(self, text: str) -> Optional[dict]:
        """Parse JSON from Qwen sign response."""
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

        sign_type = str(data.get("type", "unknown")).lower().strip()
        if sign_type not in ("stop", "speed_limit", "ground_arrow", "other", "unknown"):
            return None

        value = data.get("value")
        speed_value = None

        if sign_type == "speed_limit" and value is not None:
            try:
                speed_value = int(str(value).strip())
                valid_speeds = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80}
                if speed_value not in valid_speeds:
                    speed_value = None
            except (ValueError, TypeError):
                speed_value = None

        return {"type": sign_type, "speed_value": speed_value}

    # ─── Ground Arrow Detection (Phase 2) ───────────────────────────────────

    def detect_ground_arrows(self, frame: np.ndarray,
                              ipm_M: np.ndarray,
                              ipm_Minv: np.ndarray) -> list[dict]:
        """Detect painted ground arrows (Phase 2) via bird's-eye view analysis."""
        if self.phase < 2:
            return []

        h, w = frame.shape[:2]
        bev = cv2.warpPerspective(frame, ipm_M, (w, h))

        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        _, bin_ = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, kern)

        cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        arrows = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 2000:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            ar = bw / max(bh, 1)
            if ar > 3.0 or bh < 30:
                continue

            direction = self._arrow_direction_from_contour(cnt)
            if direction is None:
                continue

            cx_bev = float(x + bw // 2)
            cy_bev = float(y + bh // 2)
            pt = np.array([[[cx_bev, cy_bev]]], dtype=np.float32)
            pt_im = cv2.perspectiveTransform(pt, ipm_Minv)[0][0]
            arrows.append({
                "type": "ground_arrow",
                "direction": direction,
                "value": direction,
                "bbox_bev": [x, y, x + bw, y + bh],
                "position_3d": [],
                "sign_type": "ground_arrow",
                "speed_value": None,
            })

        return arrows

    # ─── Internal ───────────────────────────────────────────────────────────

    def _load_ocr(self):
        """Load OCR: easyocr (primary) → pytesseract (fallback) → disabled."""
        # Try easyocr first — proven working in ph2 on Turing
        try:
            import os
            model_dir = str(Path.home() / ".EasyOCR" / "model")
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            # Turing's module system sets MODULE_PATH to /cm/shared/...
            # which easyocr picks up and tries to mkdir → PermissionError.
            # Override temporarily for easyocr init.
            old_module_path = os.environ.get("MODULE_PATH")
            os.environ["MODULE_PATH"] = model_dir
            try:
                import easyocr
                reader = easyocr.Reader(
                    cfg.OCR_LANGUAGES, gpu=True, verbose=False,
                    model_storage_directory=model_dir,
                )
            finally:
                # Restore MODULE_PATH for Turing's module system
                if old_module_path is not None:
                    os.environ["MODULE_PATH"] = old_module_path
                else:
                    os.environ.pop("MODULE_PATH", None)
            self._ocr = ("easyocr", reader)
            print("[RoadSignDetector] EasyOCR loaded for speed sign reading.")
            return
        except ImportError:
            print("[RoadSignDetector] easyocr not installed.")
        except Exception as e:
            print(f"[RoadSignDetector] easyocr init failed: {e}")

        # Fallback: pytesseract (needs tesseract binary on system)
        try:
            import pytesseract
            # Quick check that binary exists
            pytesseract.get_tesseract_version()
            self._ocr = ("tesseract", pytesseract)
            print("[RoadSignDetector] pytesseract loaded for speed sign reading.")
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"[RoadSignDetector] pytesseract unusable ({e}).")

        print("[RoadSignDetector] No OCR available — speed verification via Qwen only.")
        self._ocr = None

    def _read_speed(self, frame: np.ndarray, bbox: list) -> Optional[int]:
        """Crop the sign, run OCR, return the first found integer."""
        if self._ocr is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, cfg.SPEED_SIGN_SIZE)

        ocr_type, ocr_engine = self._ocr

        if ocr_type == "easyocr":
            results = ocr_engine.readtext(crop, detail=0)
            all_text = " ".join(results)
        elif ocr_type == "tesseract":
            ocr_cfg = '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'
            all_text = ocr_engine.image_to_string(crop, config=ocr_cfg).strip()
        else:
            return None

        # Valid US speed limits in mph
        VALID_SPEED_LIMITS = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80}

        nums = re.findall(r"\d+", all_text)
        for n in nums:
            val = int(n)
            if val in VALID_SPEED_LIMITS:
                return val
        return None

    def _arrow_direction_from_contour(self, cnt: np.ndarray) -> Optional[str]:
        """Coarse arrow direction from contour centroid vs bounding rect."""
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        x, y, w, h = cv2.boundingRect(cnt)
        rect_cx = x + w / 2
        rect_cy = y + h / 2
        dx = cx - rect_cx
        dy = cy - rect_cy

        if abs(dx) < w * 0.1 and abs(dy) < h * 0.1:
            return "straight"
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        return "straight"


# ─── Utility ────────────────────────────────────────────────────────────────

def _compute_iou(box1: list, box2: list) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / max(union, 1e-6)
