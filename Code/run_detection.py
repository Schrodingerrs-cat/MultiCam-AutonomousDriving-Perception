"""
EinsteinVision Phase 3 — Full Detection Pipeline.

Pipeline per frame:
  PER CAMERA (front, back, left, right):
    1. YOLO12x detection
    2. UniDepth metric depth
    3. BoT-SORT tracking
    4. Camera offset on track IDs
    5. 3D reconstruction (ObjectReconstructor: mask+depth→3D)
    6. 3D-MOOD → orientation + dimensions (vehicles only)

  FRONT CAMERA ONLY:
    7. TwinLiteNet+ drivable mask
    8. Lane detection
    9. RoadMarkingDetector (ground arrows, mask mandatory)

  AFTER ALL CAMERAS:
   10. VehicleOrientationEstimator (velocity-based)
   11. Motion detection (RAFT optional)
   12. BrakeLightDetector
   13. TrafficLightClassifier (front + FOV filter, max 3)
   14. YOLOE small objects — ALL 4 CAMERAS with depth + 3D
   15. SpeedBumpDetector
   16. CollisionPredictor
   17. Stability filtering
   18. JSON write

NO SigLIP2. NO FastSAM. NO PCA orientation. NO DeepBox.
"""

import argparse
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# --- GPU guard: NEVER silently fall back to CPU ---
if not torch.cuda.is_available():
    print("=" * 60)
    print("FATAL: CUDA not available. Refusing to run on CPU.")
    print(f"  torch version: {torch.__version__}")
    print(f"  CUDA built for: {torch.version.cuda}")
    print("  Fix: pip install torch matching your driver's CUDA version")
    print("  Check driver: nvidia-smi on compute node")
    print("=" * 60)
    sys.exit(1)
else:
    _GPU_NAME = torch.cuda.get_device_name(0)
    _GPU_MEM = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {_GPU_NAME} ({_GPU_MEM:.1f} GB)")
    DEVICE = "cuda"
from utils.calibration import CameraCalibration

# --- Phase 3 new modules (already in Code_p3) ---
from detection.traffic_light_classifier import TrafficLightClassifier as P3TrafficLightClassifier
from detection.road_marking_detector import RoadMarkingDetector
from detection.vehicle_orientation import VehicleOrientationEstimator
from detection.brake_indicator_detector import BrakeLightDetector
from detection.speed_bump_detector import SpeedBumpDetector
from detection.collision_predictor import CollisionPredictor
from detection.motion_detector import MotionDetector
from detection.small_object_detector import SmallObjectDetector

# --- Ported from ph2 ---
from detection.object_detector import ObjectDetector, make_detection
from detection.tracker import MultiClassTracker, GlobalIDManager
from detection.vehicle_classifier import VehicleClassifier
from detection.pose_estimator import PoseEstimator
from detection.object_reconstructor import ObjectReconstructor
from detection.segmentation import Mask2FormerSegmenter, match_mask_to_detection
from detection.road_sign_detector import RoadSignDetector
from detection.track_state import TrackStateManager
from detection.classification_cache import ClassificationCache
from detection.mood3d_estimator import Mood3DEstimator
from utils.transforms import fuse_multicam_detections
from utils.scene_builder import build_drivable_mesh

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_CAMERA_KEYS = ["front", "back", "left", "right"]
_CAM_KEYWORDS = {
    "front": "front",
    "back":  "back",
    "left":  "left_repeater",
    "right": "right_repeater",
}
_CAM_OFFSETS = {"front": 0, "back": 20000, "left": 30000, "right": 40000}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return str(obj)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def load_all_camera_videos(seq_dir: Path) -> dict[str, Path | None]:
    result: dict[str, Path | None] = {k: None for k in _CAMERA_KEYS}
    for folder_name in ("Undist", "Raw"):
        folder = seq_dir / folder_name
        if not folder.exists():
            continue
        videos = [p for p in sorted(folder.glob("*.mp4")) if not p.name.startswith("._")]
        for cam_id, keyword in _CAM_KEYWORDS.items():
            if result[cam_id] is not None:
                continue
            matches = [v for v in videos if keyword in v.name.lower()]
            if matches:
                result[cam_id] = matches[0]
    return result


def save_detection_json(data: dict, seq_id: int, frame_id: int):
    out_dir = cfg.DETECTIONS_DIR / f"seq{seq_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = cfg.DETECTION_JSON_FMT.format(seq_id=seq_id, frame_id=frame_id)
    with open(out_dir / fname, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _load_calib(seq_id: int) -> CameraCalibration:
    npz = cfg.CALIB_DIR / f"seq{seq_id:02d}_calib.npz"
    if npz.exists():
        return CameraCalibration.from_file(npz)
    return CameraCalibration()


def backproject_center(bbox, depth_map, K_inv, R_cam2ego, t_cam2ego):
    """Back-project bbox center to 3D ego frame using depth map."""
    x1, y1, x2, y2 = bbox
    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)
    H, W = depth_map.shape[:2]
    u = max(0, min(W - 1, u))
    v = max(0, min(H - 1, v))
    d = float(depth_map[v, u])
    if d < 0.5:
        return None
    pt_cam = d * (K_inv @ np.array([u, v, 1.0]))
    pt_ego = R_cam2ego @ pt_cam + t_cam2ego
    return pt_ego.tolist()


# --------------------------------------------------------------------------- #
# Stability Filters
# --------------------------------------------------------------------------- #

def _update_track_stability(track_id, frame_id, counts, last_seen):
    if track_id in last_seen and frame_id - last_seen[track_id] <= 2:
        counts[track_id] = counts.get(track_id, 0) + 1
    else:
        counts[track_id] = 1
    last_seen[track_id] = frame_id
    return counts[track_id]


def _filter_detections(output, frame_id, track_counts, track_last,
                       obj_track_counts, obj_track_last):
    # --- VEHICLES: 3 consecutive frames, conf>0.40, 2<depth<60, max 12 ---
    stable_vehicles = []
    for v in output["vehicles"]:
        tid = v.get("track_id")
        if tid is None:
            continue
        count = _update_track_stability(tid, frame_id, track_counts, track_last)
        if count < 3:
            continue
        pos = v.get("position_3d", [0, 0, 0])
        if pos == [0, 0, 0]:
            continue
        depth = pos[1]
        if depth < 2.0 or depth > 60.0:
            continue
        if v.get("confidence", 0) < 0.40:
            continue
        stable_vehicles.append(v)
    output["vehicles"] = stable_vehicles[:12]

    # --- PEDESTRIANS: conf>0.45, depth<50 ---
    output["pedestrians"] = [
        p for p in output["pedestrians"]
        if p.get("confidence", 0) > 0.45
        and p.get("position_3d", [0, 0, 0]) != [0, 0, 0]
        and p.get("position_3d", [0, 0, 100])[1] < 50.0
    ]

    # --- TRAFFIC LIGHTS: conf>0.35, top 3 ---
    tls = [t for t in output["traffic_lights"] if t.get("confidence", 0) > 0.35]
    tls.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    output["traffic_lights"] = tls[:3]

    # --- GROUND ARROWS: max 3 ---
    arrows = [s for s in output["road_signs"] if s.get("sign_type") == "ground_arrow"]
    other_signs = [s for s in output["road_signs"] if s.get("sign_type") != "ground_arrow"]
    output["road_signs"] = other_signs + arrows[:3]

    # Small objects: no stability filter, just cap at 12 per frame
    output["objects"] = output.get("objects", [])[:12]

    # --- SPEED BUMPS: depth 3-25m (skip filter for ego-motion detections) ---
    output["speed_bumps"] = [
        sb for sb in output.get("speed_bumps", [])
        if sb.get("source") == "ego_motion"
        or 3.0 <= sb.get("position_3d", [0, 0, 0])[1] <= 25.0
    ]

    return output


# --------------------------------------------------------------------------- #
# Model Loaders
# --------------------------------------------------------------------------- #

def _load_depth():
    device = DEVICE
    print(f"[Init] Loading UniDepth ({cfg.UNIDEPTH_BACKBONE}) on {device}")
    try:
        from unidepth.models import UniDepthV2
        model = UniDepthV2.from_pretrained(
            f"lpiccinelli/unidepth-v2-{cfg.UNIDEPTH_BACKBONE}")
        model.to(device)
        model.eval()
        print("[Init] UniDepth loaded")
        return model, device
    except Exception as e:
        print(f"[Init] UniDepth failed: {e} — dummy depth active")
        return None, device


def _load_lane_detector():
    from detection.lane_detector import LaneDetector
    lane_det = LaneDetector()

    # P3 fix: disable lane count persistence that traps old all-solid types.
    # Setting confirm threshold to 1 means new lane counts are accepted
    # immediately, allowing fresh dashed classifications through.
    lane_det._LANE_COUNT_CONFIRM = 1

    return lane_det


def _load_yoloe():
    """Load YOLOE model for small object detection."""
    # Diagnostic: check if YOLOE class exists
    try:
        from ultralytics import YOLOE
        print("[Init] YOLOE import OK")
    except ImportError:
        print("[Init] YOLOE not available in ultralytics — using COCO fallback for small objects")
        return "coco_fallback"

    try:
        device = DEVICE
        model = YOLOE(cfg.YOLOE_SO_MODEL)
        model.set_classes(cfg.YOLOE_SO_CLASSES)
        model.to(device)
        print(f"[Init] YOLOE small-object loaded ({len(cfg.YOLOE_SO_CLASSES)} classes)")
        return model
    except Exception as e:
        print(f"[Init] YOLOE model load failed: {e} — using COCO fallback")
        return "coco_fallback"


# --------------------------------------------------------------------------- #
# Depth inference helper
# --------------------------------------------------------------------------- #

def run_depth(model, device, frame_bgr, K=None):
    """Run UniDepth on a frame. Returns (H,W) float32 depth in metres."""
    H, W = frame_bgr.shape[:2]
    if model is None:
        return np.full((H, W), 10.0, dtype=np.float32)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)

    intrinsics = None
    if K is not None:
        intrinsics = torch.from_numpy(K).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model.infer(tensor, intrinsics)

    depth = pred["depth"].squeeze().cpu().numpy()
    if depth.shape != (H, W):
        depth = cv2.resize(depth.astype(np.float32), (W, H),
                           interpolation=cv2.INTER_LINEAR)
    depth = np.clip(depth, 0.0, cfg.DEPTH_MAX_DIST).astype(np.float32)
    return depth


# --------------------------------------------------------------------------- #
# YOLOE small object detection (with depth + 3D)
# --------------------------------------------------------------------------- #

_YOLOE_LABEL_MAP = {
    "dark green or black cylindrical garbage can or trash bin on sidewalk": "dustbin",
    "orange traffic cone on the road":                       "traffic_cone",
    "thin tall gray metal street light pole or sign pole":   "traffic_pole",
    "orange and white striped traffic cylinder or delineator post": "traffic_cylinder",
    "short red metallic fire hydrant on the sidewalk":       "fire_hydrant",
    "short metal bollard on the road":                       "bollard",
    "concrete road barrier or jersey barrier":               "barrier",
    "speed bump or road hump on the road surface":           "speed_bump",
}

_YOLOE_ARROW_MAP = {
    "road arrow marking pointing straight ahead": "straight",
    "road arrow marking pointing left":           "left",
    "road arrow marking pointing right":          "right",
    "road arrow marking for U-turn":              "u_turn",
}


def detect_small_objects(yoloe_model, frame_bgr, existing_bboxes,
                         depth_map, K_inv, R_cam2ego, t_cam2ego):
    """Detect small objects with YOLOE on full frame + backproject to 3D.

    Same pattern as vehicles/pedestrians: run detector on whole scene,
    get bboxes, backproject center to world coords via depth + extrinsics.
    """
    if yoloe_model is None or yoloe_model == "coco_fallback":
        return []

    try:
        results = yoloe_model(frame_bgr, conf=cfg.YOLOE_SO_CONF,
                              iou=cfg.YOLOE_SO_IOU,
                              imgsz=cfg.YOLOE_SO_IMGSIZE, verbose=False)[0]
    except Exception as e:
        print(f"[YOLOE] inference error: {e}")
        return []

    if results.boxes is None or len(results.boxes) == 0:
        return []

    detections = []
    for box in results.boxes:
        conf = float(box.conf.item())
        cls_idx = int(box.cls.item())
        cls_name = yoloe_model.names.get(cls_idx, "")
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = [float(x1), float(y1), float(x2), float(y2)]

        if any(_iou(bbox, eb) > 0.5 for eb in existing_bboxes):
            continue

        label = _YOLOE_LABEL_MAP.get(cls_name)
        if label is None:
            continue

        det = {
            "label": label,
            "type": label,
            "bbox": bbox,
            "confidence": conf,
            "position_3d": [0.0, 0.0, 0.0],
        }

        # Backproject to 3D — same as vehicles/TLs/signs
        if depth_map is not None and K_inv is not None:
            pos = backproject_center(bbox, depth_map, K_inv, R_cam2ego, t_cam2ego)
            if pos is not None:
                det["position_3d"] = pos

        detections.append(det)

    return detections


# --------------------------------------------------------------------------- #
# Main process_sequence
# --------------------------------------------------------------------------- #

def process_sequence(seq_id, models, max_frames=None):
    """Process one sequence through the full pipeline."""

    # Unpack models
    obj_detector   = models["obj_detector"]
    depth_model    = models["depth_model"]
    depth_device   = models["depth_device"]
    lane_det       = models["lane_det"]
    mask2former    = models["mask2former"]
    reconstructor  = models["reconstructor"]
    mood3d         = models["mood3d"]
    sign_detector  = models["sign_detector"]
    tl_clf_p3      = models["tl_clf_p3"]
    tl_clf_kastel  = models["tl_clf_kastel"]
    vehicle_clf    = models["vehicle_clf"]
    pose_est       = models["pose_est"]
    small_obj_det  = models["small_obj_det"]
    road_marking   = models["road_marking"]
    clf_cache      = models["clf_cache"]
    track_state    = models["track_state"]
    ori_est        = models["ori_est"]
    brake_det      = models["brake_det"]
    bump_det       = models["bump_det"]
    collision_pred = models["collision_pred"]
    motion_det     = models["motion_det"]
    frame_skip     = models.get("frame_skip", 1)

    # --- Find sequence directory ---
    seq_dir = cfg.SEQUENCES_DIR / f"scene{seq_id}"
    if not seq_dir.exists():
        candidates = list(cfg.SEQUENCES_DIR.glob(f"*{seq_id}*"))
        if candidates:
            seq_dir = candidates[0]
        else:
            print(f"[Seq {seq_id}] Not found — skipping.")
            return

    print(f"\n[Seq {seq_id}] Processing {seq_dir.name}")

    calib = _load_calib(seq_id)
    cam_extrinsics = {cam_id: calib.get_extrinsics(cam_id) for cam_id in _CAMERA_KEYS}

    # Per-camera trackers
    cam_trackers = {cam_id: MultiClassTracker(device=DEVICE) for cam_id in _CAMERA_KEYS}

    cam_paths = load_all_camera_videos(seq_dir)
    caps = {}
    for cam_id, path in cam_paths.items():
        if path is not None:
            caps[cam_id] = cv2.VideoCapture(str(path))
            print(f"  [{cam_id}] {path.name}")
        else:
            caps[cam_id] = None
            print(f"  [{cam_id}] NOT FOUND")

    if caps.get("front") is None or not caps["front"].isOpened():
        print(f"[Seq {seq_id}] Front camera missing — skipping.")
        return

    total_frames = int(caps["front"].get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    fps = caps["front"].get(cv2.CAP_PROP_FPS)
    print(f"[Seq {seq_id}] {total_frames} frames @ {fps:.1f} fps")

    # Per-sequence stability state
    track_counts: dict[int, int] = {}
    track_last: dict[int, int] = {}
    obj_track_counts: dict[int, int] = {}
    obj_track_last: dict[int, int] = {}
    small_obj_id_counter = 50000

    # Reset stateful modules
    motion_det.reset()
    bump_det.reset()
    collision_pred.reset()
    ori_est.reset()
    brake_det.reset()
    track_state.reset()
    clf_cache.reset()
    vehicle_clf.reset_votes()
    reconstructor.prune_smoother(set())
    for trk in cam_trackers.values():
        trk.reset()

    prev_tl_red = False  # previous frame's traffic light red state for motion detector

    for frame_id in tqdm(range(total_frames), desc=f"Seq {seq_id}", unit="frame"):
        # Read all cameras
        frames: dict[str, np.ndarray | None] = {}
        for cam_id, cap in caps.items():
            if cap is None or not cap.isOpened():
                frames[cam_id] = None
                continue
            ret, frm = cap.read()
            frames[cam_id] = frm if ret else None

        front_frame = frames.get("front")
        if front_frame is None:
            break

        # Frame skip: read all frames (to advance caps) but only process every Nth
        if frame_skip > 1 and frame_id % frame_skip != 0:
            continue

        # Accumulators
        all_vehicles = []
        all_pedestrians = []
        all_traffic_lights = []
        all_road_signs = []
        all_objects = []
        depth_maps: dict[str, np.ndarray] = {}
        raw_yolo_results: dict = {}
        lanes = []
        drivable_mask = None
        drivable_pts = []
        masks_by_cam: dict[str, list] = {}

        # ===============================================================
        # PER CAMERA: Steps 1-6
        # ===============================================================
        import time as _time
        for cam_id in _CAMERA_KEYS:
            frame = frames.get(cam_id)
            if frame is None:
                continue

            cam_K = calib.get_K(cam_id)
            cam_K_inv = calib.get_K_inv(cam_id)
            R_cam2ego, t_cam2ego = cam_extrinsics[cam_id]
            cam_offset = _CAM_OFFSETS[cam_id]

            # --- Step 1: YOLO detection ---
            _t0 = _time.time()
            dets = obj_detector.detect(frame)
            # Store raw results for COCO fallback (small object detection)
            raw_yolo_results[cam_id] = obj_detector.last_raw_results
            print(f"[TIMING] YOLO_{cam_id}: {_time.time()-_t0:.2f}s", flush=True)

            # --- Step 2: UniDepth ---
            _t0 = _time.time()
            depth_map = run_depth(depth_model, depth_device, frame, K=cam_K)
            depth_maps[cam_id] = depth_map
            print(f"[TIMING] UniDepth_{cam_id}: {_time.time()-_t0:.2f}s", flush=True)

            # --- Step 3: BoT-SORT tracking ---
            _t0 = _time.time()
            dets = cam_trackers[cam_id].update(dets, frame)
            print(f"[TIMING] BoTSORT_{cam_id}: {_time.time()-_t0:.2f}s", flush=True)

            # --- Step 4: Camera offset on track IDs ---
            for category in ("vehicles", "pedestrians", "traffic_lights", "road_signs"):
                for det in dets.get(category, []):
                    det["camera"] = cam_id
                    if "track_id" in det:
                        det["track_id"] = int(det["track_id"]) + cam_offset

            # --- Step 5: 3D reconstruction (ObjectReconstructor) ---
            # Run Mask2Former on front camera for better masks
            cam_masks = []
            _t0 = _time.time()
            all_front_dets = []
            if cam_id == "front" and mask2former is not None and mask2former.is_available():
                all_front_dets = dets.get("vehicles", []) + dets.get("pedestrians", [])
                cam_masks = mask2former.segment_cropped(frame, all_front_dets)
                masks_by_cam["front"] = cam_masks
            print(f"[TIMING] Mask2Former_{cam_id}: {_time.time()-_t0:.2f}s (dets={len(all_front_dets) if cam_id=='front' else 0})", flush=True)

            _t0 = _time.time()
            for category in ("vehicles", "pedestrians"):
                for det in dets.get(category, []):
                    # Find matching mask
                    mask = match_mask_to_detection(det, cam_masks)
                    recon = reconstructor.reconstruct(
                        det, mask, depth_map, cam_K_inv, R_cam2ego, t_cam2ego)
                    if recon is not None:
                        det["reconstruction"] = recon
                        det["position_3d"] = recon["centroid_3d"]
                        det["size_3d"] = recon["size_3d"]
                    else:
                        # Fallback: simple backprojection
                        pos = backproject_center(
                            det["bbox"], depth_map, cam_K_inv, R_cam2ego, t_cam2ego)
                        det["position_3d"] = pos if pos is not None else [0.0, 0.0, 0.0]
            print(f"[TIMING] Reconstruct_{cam_id}: {_time.time()-_t0:.2f}s", flush=True)

            # 3D for TLs and signs via simple backprojection
            for category in ("traffic_lights", "road_signs"):
                for det in dets.get(category, []):
                    pos = backproject_center(
                        det["bbox"], depth_map, cam_K_inv, R_cam2ego, t_cam2ego)
                    det["position_3d"] = pos if pos is not None else [0.0, 0.0, 0.0]

            # --- Step 6: 3D-MOOD → orientation + dimensions (vehicles only) ---
            _t0 = _time.time()
            if mood3d is not None and mood3d.is_available():
                mood_results = mood3d.estimate_batch(frame, dets.get("vehicles", []), K=cam_K)
                for det in dets.get("vehicles", []):
                    tid = det.get("track_id")
                    if tid is not None and tid in mood_results:
                        mr = mood_results[tid]
                        det["mood3d"] = mr
                        # Store yaw for later (velocity vs stationary decision)
                        det["mood3d_yaw_deg"] = mr.get("yaw_deg", 0.0)
                        # Override dimensions if 3D-MOOD provides them
                        dims = mr.get("dimensions")
                        if dims is not None:
                            det["size_3d"] = dims
            print(f"[TIMING] MOOD3D_{cam_id}: {_time.time()-_t0:.2f}s", flush=True)

            all_vehicles.extend(dets.get("vehicles", []))
            all_pedestrians.extend(dets.get("pedestrians", []))
            all_traffic_lights.extend(dets.get("traffic_lights", []))
            all_road_signs.extend(dets.get("road_signs", []))

            # --- Steps 7-8-9: Front camera only ---
            if cam_id == "front":
                # Step 7: TwinLiteNet+ drivable mask
                _t0 = _time.time()
                try:
                    drivable_mask = lane_det.get_drivable_mask(frame)
                except Exception:
                    drivable_mask = None
                print(f"[TIMING] TwinLiteNet_drivable: {_time.time()-_t0:.2f}s", flush=True)

                # Step 8: Lane detection
                _t0 = _time.time()
                try:
                    try:
                        lanes_raw = lane_det.detect(
                            frame, K=cam_K, K_inv=cam_K_inv,
                            R_cam2ego=R_cam2ego, t_cam2ego=t_cam2ego,
                            depth_map=depth_map,
                        )
                    except TypeError:
                        # ph1 LaneDetector doesn't accept K=
                        lanes_raw = lane_det.detect(
                            frame, K_inv=cam_K_inv,
                            R_cam2ego=R_cam2ego, t_cam2ego=t_cam2ego,
                            depth_map=depth_map,
                        )
                    for lane in lanes_raw:
                        lane_dict = {
                            "lane_id": lane.get("lane_id", -1),
                            "points": lane.get("points", []),
                            "points_3d": lane.get("points_ego", []),
                            "type": lane.get("type", "solid"),
                            "color": lane.get("color", "white"),
                        }
                        if isinstance(lane_dict["points_3d"], np.ndarray):
                            lane_dict["points_3d"] = lane_dict["points_3d"].tolist()
                        if isinstance(lane_dict["points"], np.ndarray):
                            lane_dict["points"] = lane_dict["points"].tolist()
                        lanes.append(lane_dict)
                except Exception as e:
                    print(f"[Seq {seq_id}] Lane detection error: {e}")
                print(f"[TIMING] LaneDetection: {_time.time()-_t0:.2f}s", flush=True)

                # Step 9: Ground arrow detection — YOLOE only, constrained to drivable area
                _t0 = _time.time()
                yoloe_arrow_model = models.get("yoloe_arrow")
                if yoloe_arrow_model is not None:
                    try:
                        arrow_results = yoloe_arrow_model(
                            frame, conf=cfg.YOLOE_ARROW_CONF,
                            iou=cfg.YOLOE_SO_IOU,
                            imgsz=cfg.YOLOE_ARROW_IMGSIZE, verbose=False)[0]
                        if arrow_results.boxes is not None:
                            H_f, W_f = frame.shape[:2]
                            arrow_count = 0
                            for box in arrow_results.boxes:
                                if arrow_count >= 3:
                                    break
                                cls_idx = int(box.cls.item())
                                cls_name = yoloe_arrow_model.names.get(cls_idx, "")
                                direction = _YOLOE_ARROW_MAP.get(cls_name)
                                if direction is None:
                                    continue
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                conf_val = float(box.conf.item())
                                # Must be in lower 60% of frame (road level)
                                cy = (y1 + y2) / 2
                                if cy < H_f * 0.4:
                                    continue
                                # Must overlap drivable mask if available
                                if drivable_mask is not None:
                                    cx_i = max(0, min(W_f - 1, int((x1 + x2) / 2)))
                                    cy_i = max(0, min(H_f - 1, int(cy)))
                                    mv = drivable_mask[cy_i, cx_i]
                                    if isinstance(mv, np.ndarray):
                                        mv = mv.max()
                                    if mv < 0.1:
                                        continue
                                all_road_signs.append({
                                    "label": "ground_arrow",
                                    "sign_type": "ground_arrow",
                                    "direction": direction,
                                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                    "confidence": conf_val,
                                    "camera": "front",
                                })
                                arrow_count += 1
                    except Exception as e:
                        print(f"[Seq {seq_id}] YOLOE arrow error: {e}")

                print(f"[TIMING] GroundArrows: {_time.time()-_t0:.2f}s", flush=True)

                # Drivable mesh
                if drivable_mask is not None:
                    try:
                        drivable_pts = build_drivable_mesh(
                            drivable_mask, depth_map, cam_K_inv, R_cam2ego, t_cam2ego
                        ).tolist()
                    except Exception:
                        drivable_pts = []

        # ===============================================================
        # AFTER ALL CAMERAS: Steps 10-18
        # ===============================================================

        # --- Step 10: VehicleOrientationEstimator ---
        _t0 = _time.time()
        for v in all_vehicles:
            tid = v.get("track_id")
            pos = v.get("position_3d", [0, 0, 0])
            if tid is None or pos == [0, 0, 0]:
                v["orientation_deg"] = 0.0
                v["orientation_source"] = "unknown"
                v["is_moving"] = False
                continue
            ori = ori_est.update(tid, frame_id, pos)
            v["orientation_deg"] = ori["orientation_deg"]
            v["orientation_source"] = ori["orientation_source"]
            v["is_moving"] = ori["is_moving"]

            # RULE: 3D-MOOD yaw is primary for ALL vehicles (better than velocity)
            # Velocity heading is fallback only when mood3d unavailable
            if "mood3d_yaw_deg" in v:
                v["orientation_deg"] = v["mood3d_yaw_deg"]
                v["orientation_source"] = "mood3d"

        print(f"[TIMING] VehicleOrientation: {_time.time()-_t0:.2f}s", flush=True)

        # --- Step 11: RAFT optical flow motion detection ---
        _t0 = _time.time()
        if motion_det.is_available():
            motion_det.set_traffic_light_red(prev_tl_red)
            motion_det.update_vehicles(all_vehicles, frames)

        # Fallback: for non-front vehicles that RAFT didn't process,
        # derive moving_direction_deg from velocity-based orientation_deg
        for v in all_vehicles:
            if v.get("is_moving") and v.get("moving_direction_deg") is None:
                if v.get("orientation_source") != "unknown":
                    v["moving_direction_deg"] = v.get("orientation_deg", 0.0)

        print(f"[TIMING] RAFT_motion: {_time.time()-_t0:.2f}s", flush=True)

        # --- Step 12: BrakeLightDetector (front camera only — we follow vehicles) ---
        _t0 = _time.time()
        for v in all_vehicles:
            cam_id = v.get("camera", "front")
            # Only detect brake lights on front camera vehicles
            # We're following them, so front cam sees their rears
            if cam_id != "front":
                v["brake_light"] = False
                v["indicator"] = None
                continue
            frame = frames.get("front")
            if frame is None:
                v["brake_light"] = False
                v["indicator"] = None
                continue
            bi = brake_det.detect(frame, v)
            v["brake_light"] = bi["brake_light"]
            v["indicator"] = bi["indicator"]

        print(f"[TIMING] BrakeLightDet: {_time.time()-_t0:.2f}s", flush=True)

        # --- Vehicle sub-classification (DINOv2 + Qwen) with cache ---
        _t0 = _time.time()
        vehicles_needing_clf = []
        for v in all_vehicles:
            tid = v.get("track_id")
            if tid is None:
                vehicles_needing_clf.append(v)
                continue
            cached = clf_cache.get_cached(tid, "vehicle_subclass")
            if cached is not None:
                v["subclass"] = cached.get("subclass", v.get("subclass"))
            else:
                vehicles_needing_clf.append(v)

        if vehicles_needing_clf:
            vehicle_clf.classify_batch(vehicles_needing_clf, frames, frame_idx=frame_id)
            for v in vehicles_needing_clf:
                tid = v.get("track_id")
                if tid is not None and v.get("subclass") is not None:
                    clf_cache.put(tid, "vehicle_subclass",
                                  {"subclass": v["subclass"]}, frame_id)

        print(f"[TIMING] VehicleClassify: {_time.time()-_t0:.2f}s", flush=True)

        # --- TrackStateManager update ---
        for v in all_vehicles:
            tid = v.get("track_id")
            pos = v.get("position_3d")
            if tid is None or pos is None or pos == [0, 0, 0]:
                continue
            raw_ori = v.get("orientation_deg", 0.0)
            raw_sub = v.get("subclass")
            state = track_state.update(tid, frame_id,
                                       np.array(pos, dtype=np.float64),
                                       raw_ori, raw_sub)
            # Use smoothed position/orientation only — don't overwrite subclass
            # assigned by DINO/QWEN classifier
            v["position_3d"] = state.pos.tolist()
            v["orientation_deg"] = round(state.orientation, 1)

        # --- Step 13: Traffic light classification (front + FOV filter, max 3) ---
        _t0 = _time.time()
        # First try KASTEL on full front frame
        kastel_tls = []
        if tl_clf_kastel is not None:
            kastel_tls = tl_clf_kastel.detect_kastel(front_frame)

        # Merge KASTEL results with YOLO TLs
        front_w = front_frame.shape[1]
        classified_tls = []

        # Add KASTEL detections first (they have color+arrow)
        for ktl in kastel_tls:
            # Only front camera, in-FOV
            if tl_clf_p3.is_in_fov(ktl["bbox"], front_w):
                # Clean schema
                ktl.pop("arrow_direction", None)
                ktl.pop("arrow_confidence", None)
                ktl.pop("source_model", None)
                ktl["camera"] = "front"
                ktl["track_id"] = 90000 + len(classified_tls)
                # Backproject
                front_K_inv = calib.get_K_inv("front")
                R_f, t_f = cam_extrinsics["front"]
                pos = backproject_center(ktl["bbox"], depth_maps.get("front",
                    np.full(front_frame.shape[:2], 10.0, np.float32)),
                    front_K_inv, R_f, t_f)
                ktl["position_3d"] = pos if pos is not None else [0.0, 0.0, 0.0]
                classified_tls.append(ktl)

        # Then classify YOLO-detected TLs not covered by KASTEL
        for tl in all_traffic_lights:
            if tl.get("camera") != "front":
                continue
            if not tl_clf_p3.is_in_fov(tl["bbox"], front_w):
                continue
            # Skip if overlaps a KASTEL detection
            if any(_iou(tl["bbox"], k["bbox"]) > 0.4 for k in classified_tls):
                continue
            result = tl_clf_p3.classify(front_frame, tl["bbox"])
            tl["tl_color"] = result["tl_color"]
            tl["tl_arrow"] = result["tl_arrow"]
            tl["confidence"] = max(tl.get("confidence", 0), result["confidence"])
            tl.pop("sign_type", None)
            tl.pop("direction", None)
            tl.pop("arrow_direction", None)
            tl.pop("arrow_confidence", None)
            classified_tls.append(tl)

        all_traffic_lights = classified_tls
        # Update prev_tl_red for next frame's motion detector
        prev_tl_red = any(tl.get("tl_color") == "red" for tl in all_traffic_lights)

        print(f"[TIMING] TrafficLightClf: {_time.time()-_t0:.2f}s", flush=True)

        # --- Road sign detection (bhaskrr on front frame) ---
        _t0 = _time.time()
        if sign_detector is not None:
            sign_result = sign_detector.detect_signs_and_lights(front_frame)
            for rs in sign_result.get("road_signs", []):
                rs["camera"] = "front"
                rs["track_id"] = 95000 + len(all_road_signs)
                front_K_inv = calib.get_K_inv("front")
                R_f, t_f = cam_extrinsics["front"]
                pos = backproject_center(rs["bbox"], depth_maps.get("front",
                    np.full(front_frame.shape[:2], 10.0, np.float32)),
                    front_K_inv, R_f, t_f)
                rs["position_3d"] = pos if pos is not None else [0.0, 0.0, 0.0]
                # Deduplicate against existing
                if not any(_iou(rs["bbox"], ex["bbox"]) > 0.4
                           for ex in all_road_signs):
                    all_road_signs.append(rs)

        print(f"[TIMING] RoadSignDet: {_time.time()-_t0:.2f}s", flush=True)

        # --- Pedestrian pose estimation (YOLO Pose, batched per camera) ---
        _t0 = _time.time()
        if pose_est is not None:
            pose_est.estimate_batch(frames, all_pedestrians)

        print(f"[TIMING] PoseEstimation: {_time.time()-_t0:.2f}s", flush=True)

        # --- Step 14: Small objects — ALL 4 CAMERAS via YOLOE full-frame ---
        _t0 = _time.time()
        yoloe_model = models.get("yoloe_model")
        for cam_id in _CAMERA_KEYS:
            cam_frame = frames.get(cam_id)
            if cam_frame is None:
                continue

            cam_K_inv = calib.get_K_inv(cam_id)
            R_cam2ego, t_cam2ego = cam_extrinsics[cam_id]

            existing = [d["bbox"] for d in (all_vehicles + all_pedestrians +
                                             all_traffic_lights + all_road_signs +
                                             all_objects)
                        if d.get("camera") == cam_id]

            # YOLOE on full frame → backproject to 3D (same as vehicles)
            objs = detect_small_objects(
                yoloe_model, cam_frame, existing,
                depth_map=depth_maps.get(cam_id),
                K_inv=cam_K_inv,
                R_cam2ego=R_cam2ego,
                t_cam2ego=t_cam2ego,
            )

            # COCO fallback from YOLO12x results (appearance-based, no re-inference)
            yolo_res = raw_yolo_results.get(cam_id)
            if small_obj_det is not None and yolo_res is not None:
                coco_objs = small_obj_det.detect_with_coco_fallback(
                    cam_frame, yolo_res,
                    existing + [o["bbox"] for o in objs])
                # Backproject COCO fallback detections to 3D
                for co in coco_objs:
                    if depth_maps.get(cam_id) is not None and cam_K_inv is not None:
                        pos = backproject_center(
                            co["bbox"], depth_maps[cam_id], cam_K_inv,
                            R_cam2ego, t_cam2ego)
                        if pos is not None:
                            co["position_3d"] = pos
                    objs.append(co)

            for o in objs:
                o["camera"] = cam_id
                small_obj_id_counter += 1
                o["track_id"] = small_obj_id_counter
            all_objects.extend(objs)

        print(f"[TIMING] YOLOE_SmallObj_4cam: {_time.time()-_t0:.2f}s", flush=True)

        # --- Step 15: SpeedBumpDetector (front only, gated by sequence) ---
        _t0 = _time.time()
        speed_bumps = []

        # Always strip speed_bump from all_objects (avoid polluting small objects)
        remaining_objects = []
        for o in all_objects:
            if o.get("label") == "speed_bump":
                if seq_id in cfg.SPEED_BUMP_SEQUENCES and o.get("camera") == "front":
                    speed_bumps.append({
                        "type": "speed_bump",
                        "bbox": o["bbox"],
                        "position_3d": o.get("position_3d", [0.0, 0.0, 0.0]),
                        "confidence": o["confidence"],
                        "source": "yoloe",
                    })
                # else: discard (non-bump sequence or non-front camera)
            else:
                remaining_objects.append(o)
        all_objects = remaining_objects

        if seq_id in cfg.SPEED_BUMP_SEQUENCES:
            front_K = calib.get_K("front")
            R_front, t_front = cam_extrinsics["front"]

            # SBP-YOLO fallback if YOLOE found nothing
            if not speed_bumps:
                speed_bumps = bump_det.detect(
                    frames.get("front"), drivable_mask,
                    front_K, R_front, t_front, frame_id,
                    depth_map=depth_maps.get("front"),
                )

            # Ego-motion fallback when neither found anything
            if not speed_bumps:
                speed_bumps = bump_det.detect_from_ego_motion(
                    frame_id, depth_maps.get("front"))

        print(f"[TIMING] SpeedBumpDet: {_time.time()-_t0:.2f}s", flush=True)

        # --- Step 16: CollisionPredictor (front camera only — ego path) ---
        _t0 = _time.time()
        for det in all_vehicles + all_pedestrians:
            tid = det.get("track_id")
            pos = det.get("position_3d")
            # Only predict collision for front-camera detections (ego driving path)
            if det.get("camera") != "front":
                det["collision_risk"] = "none"
                det["ttc_seconds"] = None
                continue
            if tid is None or pos is None or pos == [0, 0, 0]:
                det["collision_risk"] = "none"
                det["ttc_seconds"] = None
                continue
            size = det.get("size_3d")
            risk = collision_pred.update_and_assess(tid, pos, size, frame_id)
            det["collision_risk"] = risk["level"]
            det["ttc_seconds"] = risk["ttc_seconds"]

        print(f"[TIMING] CollisionPred: {_time.time()-_t0:.2f}s", flush=True)

        # ===============================================================
        # BUILD CANONICAL OUTPUT
        # ===============================================================
        _t0 = _time.time()

        final_vehicles = []
        for v in all_vehicles:
            final_vehicles.append({
                "label": v.get("label", "car"),
                "subclass": v.get("subclass") or {"car": "sedan", "truck": "truck", "bus": "truck"}.get(v.get("label", "car"), v.get("label", "sedan")),
                "bbox": v["bbox"],
                "confidence": v["confidence"],
                "track_id": v.get("track_id", 0),
                "camera": v.get("camera", "front"),
                "sources": [{"camera": v.get("camera", "front"),
                             "confidence": v["confidence"],
                             "bbox": v["bbox"],
                             "track_id": v.get("track_id", 0)}],
                "position_3d": v.get("position_3d", [0, 0, 0]),
                "size_3d": v.get("size_3d"),
                "orientation_deg": v.get("orientation_deg", 0.0),
                "orientation_source": v.get("orientation_source", "unknown"),
                "is_moving": v.get("is_moving", False),
                "moving_direction_deg": v.get("moving_direction_deg"),
                "flow_magnitude": v.get("flow_magnitude"),
                "sampson_distance": v.get("sampson_distance"),
                "brake_light": v.get("brake_light", False),
                "indicator": v.get("indicator"),
                "collision_risk": v.get("collision_risk", "none"),
                "ttc_seconds": v.get("ttc_seconds"),
            })

        final_peds = []
        for p in all_pedestrians:
            final_peds.append({
                "label": "person",
                "bbox": p["bbox"],
                "confidence": p["confidence"],
                "track_id": p.get("track_id", 0),
                "camera": p.get("camera", "front"),
                "position_3d": p.get("position_3d", [0, 0, 0]),
                "pose_label": p.get("pose_label", "unknown"),
                "walking_direction": p.get("walking_direction"),
                "heading_3d": p.get("heading_3d"),
                "collision_risk": p.get("collision_risk", "none"),
                "ttc_seconds": p.get("ttc_seconds"),
            })

        final_tls = []
        for tl in all_traffic_lights:
            final_tls.append({
                "label": "traffic_light",
                "bbox": tl["bbox"],
                "confidence": tl.get("confidence", 0.5),
                "track_id": tl.get("track_id", 0),
                "camera": "front",
                "position_3d": tl.get("position_3d", [0, 0, 0]),
                "tl_color": tl.get("tl_color", "unknown"),
                "tl_arrow": tl.get("tl_arrow"),  # Python None -> JSON null
            })

        final_signs = []
        for rs in all_road_signs:
            d = {
                "label": rs.get("label", rs.get("sign_type", "road_sign")),
                "bbox": rs["bbox"],
                "confidence": rs.get("confidence", 0.5),
                "camera": rs.get("camera", "front"),
            }
            if rs.get("sign_type") == "ground_arrow":
                d["sign_type"] = "ground_arrow"
                d["direction"] = rs.get("direction", "straight")
            else:
                d["sign_type"] = rs.get("sign_type", "stop_sign")
                d["track_id"] = rs.get("track_id", 0)
                d["position_3d"] = rs.get("position_3d", [0, 0, 0])
                d["speed_value"] = rs.get("speed_value")
            final_signs.append(d)

        final_objs = []
        for o in all_objects:
            final_objs.append({
                "label": o.get("label", "unknown"),
                "type": o.get("type", o.get("label", "unknown")),
                "bbox": o["bbox"],
                "confidence": o.get("confidence", 0.5),
                "camera": o.get("camera", "front"),
                "track_id": o.get("track_id", 0),
                "position_3d": o.get("position_3d", [0, 0, 0]),
            })

        output = {
            "vehicles": final_vehicles,
            "pedestrians": final_peds,
            "traffic_lights": final_tls,
            "road_signs": final_signs,
            "objects": final_objs,
            "speed_bumps": speed_bumps,
            "lanes": lanes,
            "drivable_pts": drivable_pts if drivable_pts else [],
        }

        print(f"[TIMING] BuildOutput: {_time.time()-_t0:.2f}s", flush=True)

        # --- Step 17: Stability filters ---
        _t0 = _time.time()
        output = _filter_detections(output, frame_id, track_counts, track_last,
                                    obj_track_counts, obj_track_last)

        # HARD RULE: no position_3d = [0,0,0]
        for cat in ("vehicles", "pedestrians", "traffic_lights"):
            output[cat] = [d for d in output[cat] if d.get("position_3d") != [0, 0, 0]]
        # Keep all objects regardless of position_3d
        pass  # no filter on objects

        # HARD RULE: no depth > 60m (vehicles + pedestrians)
        for cat in ("vehicles", "pedestrians"):
            output[cat] = [d for d in output[cat]
                           if d.get("position_3d", [0, 0, 100])[1] <= 60.0]
        # Traffic lights: max 80m (can be far overhead)
        output["traffic_lights"] = [
            t for t in output["traffic_lights"]
            if t.get("position_3d", [0, 0, 200])[1] <= 80.0
        ]
        # Road signs: max 60m (ground arrows have no depth field)
        output["road_signs"] = [
            s for s in output["road_signs"]
            if s.get("sign_type") == "ground_arrow"
            or s.get("position_3d", [0, 0, 200])[1] <= 60.0
        ]

        print(f"[TIMING] StabilityFilter: {_time.time()-_t0:.2f}s", flush=True)

        # --- Step 18: Write JSON ---
        _t0 = _time.time()
        payload = {
            "seq_id": seq_id,
            "frame_id": frame_id,
            "phase": 3,
            **output,
        }
        save_detection_json(payload, seq_id, frame_id)
        print(f"[TIMING] JSONWrite: {_time.time()-_t0:.2f}s", flush=True)

        # GPU memory check
        print(f"[TIMING] GPU_alloc: {torch.cuda.memory_allocated()/1e9:.2f}GB  GPU_reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
        print(f"[TIMING] === END FRAME {frame_id} ===", flush=True)

    # End of sequence cleanup
    active_ids = set()
    for v in all_vehicles + all_pedestrians:
        tid = v.get("track_id")
        if tid is not None:
            active_ids.add(tid)
    ori_est.prune(active_ids)
    brake_det.prune(active_ids)
    motion_det.prune(active_ids)
    collision_pred.prune(active_ids)
    track_state.prune(active_ids)
    vehicle_clf.prune(active_ids)
    reconstructor.prune_smoother(active_ids)
    if mood3d is not None:
        mood3d.prune(active_ids)
    if pose_est is not None:
        pose_est.prune_ankle_history(active_ids)

    for cap in caps.values():
        if cap is not None:
            cap.release()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="EinsteinVision Phase 3 Detection")
    ap.add_argument("--phase", type=int, default=3)
    ap.add_argument("--seq", nargs="+", default=["all"])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--frame-skip", type=int, default=1,
                    help="Process 1 in every N frames (default: 1 = all frames)")
    args = ap.parse_args()

    if args.seq == ["all"]:
        seq_ids = list(range(1, cfg.NUM_SEQUENCES + 1))
    else:
        seq_ids = [int(s) for s in args.seq]

    print("EinsteinVision Phase 3 — Full Pipeline")
    print(f"Sequences: {seq_ids}")
    print("Pipeline: YOLO12x + UniDepth + BoT-SORT + Mask2Former + 3D-MOOD + "
          "TwinLiteNet+ + YOLOE(4cam) + KASTEL + bhaskrr + DINOv2 + Qwen + RTMPose")

    # ===== Load all models ONCE =====
    print("\n--- Loading models ---")

    # 1. YOLO12x (ObjectDetector wraps it)
    obj_detector = ObjectDetector(phase=3, device=DEVICE)

    # 2. UniDepth
    depth_model, depth_device = _load_depth()

    # 3. Lane detector (TwinLiteNet+)
    try:
        lane_det = _load_lane_detector()
    except Exception as e:
        print(f"[Init] Lane detector failed: {e}")
        lane_det = type('Dummy', (), {
            'detect': lambda *a, **kw: [],
            'get_drivable_mask': lambda *a, **kw: None,
        })()

    # 4. Mask2Former
    try:
        mask2former = Mask2FormerSegmenter(device=DEVICE)
    except Exception as e:
        print(f"[Init] Mask2Former failed: {e}")
        mask2former = None

    # 5. ObjectReconstructor
    reconstructor = ObjectReconstructor()

    # 6. 3D-MOOD (replaces DeepBox)
    mood3d = Mood3DEstimator()
    if not mood3d.is_available():
        print("[Init] 3D-MOOD unavailable — orientation from velocity only")

    # 7. RoadSignDetector (bhaskrr)
    try:
        sign_detector = RoadSignDetector(phase=3)
    except Exception as e:
        print(f"[Init] RoadSignDetector failed: {e}")
        sign_detector = None

    # 8. Traffic light classifiers
    tl_clf_p3 = P3TrafficLightClassifier()
    # KASTEL classifier from ph2 (shares the KASTEL model)
    try:
        from detection.traffic_light_classifier import TrafficLightClassifier as _P2TL
        # The P3 TrafficLightClassifier is HSV-based.
        # We use ph2's TrafficLightClassifier for KASTEL full-frame detection.
        # But our p3 module doesn't have detect_kastel. Use ph2 TL classifier.
        ph2_dir = Path(__file__).resolve().parent.parent / "Code_ph2_VLM+Dino"
        tl_spec = importlib.util.spec_from_file_location(
            "ph2_tl_clf", ph2_dir / "detection" / "traffic_light_classifier.py")
        tl_mod = importlib.util.module_from_spec(tl_spec)
        saved_cfg = sys.modules.get("config")
        # ph2 TL classifier needs ph2 config for TL_COLOR_THRESHOLDS etc.
        # But we now have those in our config, so just load with our config
        tl_spec.loader.exec_module(tl_mod)
        if saved_cfg:
            sys.modules["config"] = saved_cfg
        sign_model = sign_detector.get_sign_model() if sign_detector else None
        tl_clf_kastel = tl_mod.TrafficLightClassifier(phase=3, sign_model=sign_model)
    except Exception as e:
        print(f"[Init] KASTEL TL classifier failed: {e}")
        tl_clf_kastel = None

    # 9. VehicleClassifier (DINOv2 + Qwen)
    vehicle_clf = VehicleClassifier()

    # 10. PoseEstimator
    try:
        pose_est = PoseEstimator()
    except Exception as e:
        print(f"[Init] PoseEstimator failed: {e}")
        pose_est = None

    # 11. YOLOE for small objects (full-frame, same 3D pipeline as vehicles)
    yoloe_model = _load_yoloe()
    # SmallObjectDetector kept for COCO appearance-based fallback
    small_obj_det = SmallObjectDetector()
    print("[Init] SmallObjectDetector ready (YOLOE full-frame + COCO fallback).")

    # 11b. YOLOE for ground arrow detection (separate model instance, different classes)
    yoloe_arrow = None
    if yoloe_model != "coco_fallback" and yoloe_model is not None:
        try:
            from ultralytics import YOLOE
            yoloe_arrow = YOLOE(cfg.YOLOE_SO_MODEL)
            yoloe_arrow.set_classes(cfg.YOLOE_ARROW_CLASSES)
            yoloe_arrow.to(DEVICE)
            print(f"[Init] YOLOE arrow model loaded ({len(cfg.YOLOE_ARROW_CLASSES)} classes)")
        except Exception as e:
            print(f"[Init] YOLOE arrow model failed: {e} — contour fallback only")
            yoloe_arrow = None

    # 12. Motion detector (RAFT optical flow)
    if cfg.RAFT_ENABLED:
        motion_det = MotionDetector(cameras=cfg.RAFT_CAMERAS)
    else:
        motion_det = MotionDetector(cameras=[])  # disabled — is_available() → False

    # 13. Phase 3 modules
    # Pass Qwen model to RoadMarkingDetector for arrow direction classification
    try:
        from detection.vehicle_classifier import get_qwen_model
        qwen_m, qwen_p = get_qwen_model()
    except Exception:
        qwen_m, qwen_p = None, None
    road_marking = RoadMarkingDetector(qwen_model=qwen_m, qwen_processor=qwen_p)
    clf_cache = ClassificationCache()
    track_state_mgr = TrackStateManager()
    ori_est = VehicleOrientationEstimator()
    brake_det = BrakeLightDetector()
    bump_det = SpeedBumpDetector(weights_path="weights/bump/sbp-yolo.pt")
    collision_pred = CollisionPredictor(fps=cfg.VIDEO_FPS)

    print("[Init] All models loaded.\n")

    # ===== Device Check =====
    print(f"\n[Device Check]")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    try:
        yolo_device = next(obj_detector.model.parameters()).device
        print(f"  YOLO: {yolo_device}")
    except Exception:
        pass
    if depth_model is not None:
        try:
            depth_device_actual = next(depth_model.parameters()).device
            print(f"  UniDepth: {depth_device_actual}")
        except Exception:
            pass
    print()

    # Pack into dict for process_sequence
    models = {
        "obj_detector":   obj_detector,
        "depth_model":    depth_model,
        "depth_device":   depth_device,
        "lane_det":       lane_det,
        "mask2former":    mask2former,
        "reconstructor":  reconstructor,
        "mood3d":         mood3d,
        "sign_detector":  sign_detector,
        "tl_clf_p3":      tl_clf_p3,
        "tl_clf_kastel":  tl_clf_kastel,
        "vehicle_clf":    vehicle_clf,
        "pose_est":       pose_est,
        "small_obj_det":  small_obj_det,
        "yoloe_model":    yoloe_model,
        "yoloe_arrow":    yoloe_arrow,
        "road_marking":   road_marking,
        "clf_cache":      clf_cache,
        "track_state":    track_state_mgr,
        "ori_est":        ori_est,
        "brake_det":      brake_det,
        "bump_det":       bump_det,
        "collision_pred": collision_pred,
        "motion_det":     motion_det,
        "frame_skip":     args.frame_skip,
    }

    for seq_id in seq_ids:
        process_sequence(seq_id, models, max_frames=args.max_frames)

    print("\n[Done] Phase 3 detection complete.")


if __name__ == "__main__":
    main()
