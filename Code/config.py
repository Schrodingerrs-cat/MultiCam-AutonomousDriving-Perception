"""
EinsteinVision Phase 3 Configuration — clean rewrite.
"""

from pathlib import Path

# --- Directories ---
BASE_DIR       = Path(__file__).parent.parent
DATA_DIR       = BASE_DIR / "Data"
SEQUENCES_DIR  = DATA_DIR / "Sequences"
ASSETS_DIR     = DATA_DIR / "Assets"
CALIB_DIR      = DATA_DIR / "Calib"
OUTPUT_DIR     = BASE_DIR / "Output"
DETECTIONS_DIR = OUTPUT_DIR / "detections_p3"
RENDERS_DIR    = OUTPUT_DIR / "renders"
VIDEOS_DIR     = OUTPUT_DIR / "videos"

# --- Phase ---
PHASE = 3
NUM_SEQUENCES = 13

# --- YOLO12x ---
YOLO_MODEL    = "yolo12x.pt"
YOLO_CONF     = 0.35
YOLO_IOU      = 0.45
YOLO_IMG_SIZE = 1280

COCO_IDS = {
    "person":        0,
    "bicycle":       1,
    "car":           2,
    "motorcycle":    3,
    "bus":           5,
    "truck":         7,
    "traffic_light": 9,
    "stop_sign":    11,
}

VEHICLE_IDS    = {1, 2, 3, 5, 7}   # 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
PEDESTRIAN_IDS = {0}
SIGN_IDS       = {11}
LIGHT_IDS      = {9}

VEHICLE_SUBCLASSES = {
    "car": ["sedan", "suv", "hatchback", "pickup"],
    "truck": ["truck"],
    "bus": ["bus"],
    "motorcycle": ["motorcycle"],
    "bicycle": ["bicycle"],
}

# --- UniDepth ---
UNIDEPTH_BACKBONE = "vitl14"
DEPTH_MAX_DIST    = 80.0

# --- Camera defaults (overridden by per-seq calibration) ---
DEFAULT_FX = 1594.6574
DEFAULT_FY = 1607.6942
DEFAULT_CX = 655.2961
DEFAULT_CY = 414.3627
CAMERA_HEIGHT = 1.20

# --- Multi-camera extrinsics ---
CAMERA_EXTRINSICS = {
    "front": {"position": [ 0.0,  1.5, 1.0], "yaw_deg":    0.0, "pitch_deg":  7.0},
    "back":  {"position": [ 0.0, -1.8, 0.9], "yaw_deg":  180.0, "pitch_deg":  5.0},
    "left":  {"position": [-0.9,  0.1, 1.3], "yaw_deg":  -90.0, "pitch_deg":  8.0},
    "right": {"position": [ 0.9,  0.1, 1.3], "yaw_deg":   90.0, "pitch_deg":  8.0},
}

# --- KASTEL TL ---
KASTEL_TL_MODEL   = "Code/weights/traffic_lights_yolov8x.pt"
KASTEL_TL_CONF    = 0.30
KASTEL_TL_IOU     = 0.45
KASTEL_TL_IMGSIZE = 1280

# --- YOLOE small objects ---
YOLOE_SO_MODEL   = "yoloe-11s-seg.pt"
YOLOE_SO_CONF    = 0.10
YOLOE_SO_IOU     = 0.45
YOLOE_SO_IMGSIZE = 640
YOLOE_SO_CLASSES = [
    "dark green or black cylindrical garbage can or trash bin on sidewalk",
    "orange traffic cone on the road",
    "thin tall gray metal street light pole or sign pole",
    "orange and white striped traffic cylinder or delineator post",
    "short red metallic fire hydrant on the sidewalk",
    "short metal bollard on the road",
    "concrete road barrier or jersey barrier",
    "speed bump or road hump on the road surface",
]

# --- YOLOE ground arrow detection ---
YOLOE_ARROW_CLASSES = [
    "road arrow marking pointing straight ahead",
    "road arrow marking pointing left",
    "road arrow marking pointing right",
    "road arrow marking for U-turn",
]
YOLOE_ARROW_CONF    = 0.15
YOLOE_ARROW_IMGSIZE = 640

# --- Classical BEV lane pipeline (from ph2 lane_detector) ---
CLASSICAL_LANE_ENABLED      = True
CLASSICAL_SOBEL_KSIZE       = 3
CLASSICAL_EDGE_THRESH       = 50
CLASSICAL_HIST_SMOOTH_SIG   = 20
CLASSICAL_SW_N_WINDOWS      = 12
CLASSICAL_SW_MARGIN         = 80
CLASSICAL_SW_MIN_PIX        = 40
CLASSICAL_POLY_DEGREE       = 2
CLASSICAL_FUSION_NN_W       = 0.6
CLASSICAL_FUSION_MIN_FRAMES = 3
CLASSICAL_CLAHE_CLIP        = 2.0
CLASSICAL_CLAHE_GRID        = 8

# --- Output format ---
DETECTION_JSON_FMT = "seq{seq_id:02d}_frame{frame_id:06d}.json"
VIDEO_FPS          = 30

# --- Collision ---
COLLISION_CRITICAL_TTC  = 1.0   # critical if TTC ≤ 1.0s AND within 3m
COLLISION_WARN_TTC      = 2.0   # warning if TTC ≤ 2.0s AND within 5m
COLLISION_MAX_DIST      = 5.0   # only predict for objects within 5m forward

# --- Speed bump detection — only sequences with confirmed bumps ---
SPEED_BUMP_SEQUENCES = {5, 9}   # from contents.md: scenes 5 & 9 have speed humps

# --- Segmentation (Mask2Former) ---
MASK2FORMER_HF_MODEL = "facebook/mask2former-swin-large-cityscapes-panoptic"
MASK2FORMER_CAMERAS  = ["front"]
MASK2FORMER_MODE     = "cropped"
MASK2FORMER_EXPAND   = 0.25

# --- Traffic Sign Detection (bhaskrr) ---
SIGN_CONF_THRESHOLD = 0.30
SIGN_IOU_THRESHOLD  = 0.45
SIGN_IMGSIZE        = 640

# --- Traffic Light HSV thresholds ---
TL_COLOR_THRESHOLDS = {
    "red":    [([0,  120, 70],  [10, 255, 255]),
               ([170,120, 70],  [180,255, 255])],
    "yellow": [([20, 100, 100], [35, 255, 255])],
    "green":  [([40,  50,  50], [90, 255, 255])],
}

# --- OCR ---
OCR_LANGUAGES   = ["en"]
SPEED_SIGN_SIZE = (128, 128)

# --- Pose ---
RTMPOSE_MODE          = "performance"      # "performance"=rtmpose-x, "balanced"=rtmpose-m, "lightweight"=rtmpose-s
RTMPOSE_BACKEND       = "onnxruntime"      # "onnxruntime" (GPU-fast) or "opencv" (CPU fallback)
POSE_CONF_THRESHOLD   = 0.3               # per-keypoint confidence threshold
POSE_MIN_DETECTION    = 0.5               # mediapipe fallback min detection confidence
POSE_MAX_PEDS_PER_CAM = 8                 # cap per-crop RTMPose calls per camera for speed

# --- Vehicle Classification (Qwen) ---
QWEN_VL_MODEL         = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_VL_DTYPE         = "float16"
QWEN_VL_DEVICE_MAP    = "auto"
QWEN_VL_3B_MODEL      = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_VL_3B_DTYPE      = "float16"
QWEN_VL_3B_DEVICE_MAP = "auto"
SUBCLASS_EMA_ALPHA    = 0.3

# --- RAFT Optical Flow (motion detection) ---
RAFT_ENABLED       = True
RAFT_MODEL         = "small"         # "small" (fast, CPU-ok) or "large" (accurate, needs GPU)
RAFT_CAMERAS       = ["front"]       # cameras to run flow on
RAFT_RESIZE        = (960, 520)      # (W, H) — must be divisible by 8
RAFT_NUM_UPDATES   = 6               # flow refinement iterations (fewer = faster)

# --- 3D-MOOD (replaces DeepBox) ---
MOOD3D_ENABLED  = True
MOOD3D_BACKBONE = "swin_t"
MOOD3D_CKPT_URL = "https://huggingface.co/RoyYang0714/3D-MOOD/resolve/main/gdino3d_swin-t_120e_omni3d_699f69.pt"
