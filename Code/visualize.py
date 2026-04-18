#!/usr/bin/env python3
"""
EinsteinVision — Detection Visualization Overlay.

Draws all detections on original video frames with clear, verifiable annotations.
Designed for manual verification of detection quality.

Usage:
  python3 Code_p3/visualize.py --seq 10 --camera front
  python3 Code_p3/visualize.py --seq 2  --camera front --max-frames 500
  python3 Code_p3/visualize.py --seq 3  --camera front --save-frames Output/viz_frames/seq03/
"""

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
SEQUENCES_DIR = DATA_DIR / "Sequences"

# Camera keyword mapping for video file lookup
_CAM_KEYWORDS = {
    "front": "front",
    "back": "back",
    "left": "left_repeater",
    "right": "right_repeater",
}

# ─── Color Palette (BGR) ─────────────────────────────────────────────────

# Vehicle state colors
C_MOVING      = (0, 220, 0)      # bright green
C_PARKED      = (160, 160, 160)  # gray
C_COLLISION_C = (0, 0, 255)      # red — critical
C_COLLISION_W = (0, 180, 255)    # orange — warning

# Detection category colors
C_PED         = (255, 220, 0)    # cyan-ish
C_TL_RED      = (0, 0, 220)
C_TL_GREEN    = (0, 200, 0)
C_TL_YELLOW   = (0, 220, 255)
C_TL_OFF      = (128, 128, 128)
C_STOP        = (0, 0, 220)     # red
C_SPEED       = (0, 140, 255)   # orange
C_GROUND_ARR  = (255, 80, 0)    # blue
C_OBJECT      = (255, 0, 200)   # purple
C_BUMP        = (0, 230, 255)   # yellow
C_BRAKE       = (60, 60, 255)   # deep red
C_INDICATOR   = (0, 165, 255)   # orange
C_LANE_SOLID  = (180, 180, 180)
C_LANE_DASHED = (0, 200, 200)
C_LANE_YELLOW = (0, 200, 255)
C_WHITE       = (255, 255, 255)
C_BLACK       = (0, 0, 0)

TL_COLORS = {
    "red": C_TL_RED, "green": C_TL_GREEN,
    "yellow": C_TL_YELLOW, "off": C_TL_OFF, "unknown": C_TL_OFF,
}

# Font
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_S = cv2.FONT_HERSHEY_PLAIN


def find_video(seq_id: int, camera: str) -> Path | None:
    seq_dir = SEQUENCES_DIR / f"scene{seq_id}"
    keyword = _CAM_KEYWORDS.get(camera, camera)
    for folder in ("Undist", "Raw"):
        folder_path = seq_dir / folder
        if not folder_path.exists():
            continue
        for mp4 in sorted(folder_path.glob("*.mp4")):
            if keyword in mp4.name.lower():
                return mp4
    return None


def find_detections_dir(seq_id: int) -> Path | None:
    for suffix in ("detections_final", "detections"):
        d = BASE_DIR / "Output" / suffix / f"seq{seq_id:02d}"
        if d.exists() and len(list(d.glob("*.json"))) > 0:
            return d
    return None


def load_detection(det_dir: Path, seq_id: int, frame_id: int) -> dict | None:
    for fmt in (f"seq{seq_id:02d}_frame{frame_id:06d}.json",
                f"seq{seq_id:02d}_frame{frame_id:04d}.json"):
        p = det_dir / fmt
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


# ─── Drawing Helpers ─────────────────────────────────────────────────────

def _text_bg(frame, text, org, font_scale, color, thickness=1, bg_alpha=0.6):
    """Draw text with semi-transparent background for readability."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = org
    # Background rect
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 2, y - th - 4), (x + tw + 2, y + baseline + 2), C_BLACK, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0, frame)
    cv2.putText(frame, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def _corner_label(frame, text, bbox, color, position="top"):
    """Draw label anchored to bbox corner."""
    x1, y1, x2, y2 = bbox
    if position == "top":
        _text_bg(frame, text, (x1, y1 - 4), 0.42, color)
    elif position == "bottom":
        _text_bg(frame, text, (x1, y2 + 14), 0.42, color)
    elif position == "right":
        _text_bg(frame, text, (x2 + 4, (y1 + y2) // 2), 0.38, color)


def _dashed_rect(frame, pt1, pt2, color, thickness=2, dash_len=8):
    """Draw dashed rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    for i in range(x1, x2, dash_len * 2):
        cv2.line(frame, (i, y1), (min(i + dash_len, x2), y1), color, thickness)
        cv2.line(frame, (i, y2), (min(i + dash_len, x2), y2), color, thickness)
    for i in range(y1, y2, dash_len * 2):
        cv2.line(frame, (x1, i), (x1, min(i + dash_len, y2)), color, thickness)
        cv2.line(frame, (x2, i), (x2, min(i + dash_len, y2)), color, thickness)


# ─── Vehicle Drawing ─────────────────────────────────────────────────────

def draw_vehicles(frame, vehicles, camera):
    for v in vehicles:
        if v.get("camera") != camera and camera != "all":
            continue
        x1, y1, x2, y2 = [int(c) for c in v["bbox"]]
        is_moving = v.get("is_moving", True)
        risk = v.get("collision_risk", "none")
        if isinstance(risk, dict):
            risk = risk.get("level", "none")

        # Box color: collision > moving/parked
        if risk == "critical":
            box_color = C_COLLISION_C
            thickness = 3
        elif risk == "warning":
            box_color = C_COLLISION_W
            thickness = 3
        elif is_moving:
            box_color = C_MOVING
            thickness = 2
        else:
            box_color = C_PARKED
            thickness = 1
            _dashed_rect(frame, (x1, y1), (x2, y2), C_PARKED, 1)

        if is_moving or risk != "none":
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        # Top label: class subclass #id
        subclass = v.get("subclass", "")
        tid = v.get("global_id") or v.get("track_id", "")
        label_parts = [v.get("label", "car")]
        if subclass and subclass not in ("None", "unknown"):
            label_parts.append(subclass)
        label_parts.append(f"#{tid}")
        top_text = " ".join(label_parts)
        _corner_label(frame, top_text, (x1, y1, x2, y2), box_color, "top")

        # Parked indicator
        if not is_moving:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(frame, "P", (cx - 8, cy + 8), FONT, 0.7, C_PARKED, 2, cv2.LINE_AA)

        # Orientation arrow (direction vehicle is heading)
        ori_deg = v.get("orientation_deg")
        if ori_deg is not None and is_moving:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            rad = math.radians(float(ori_deg))
            length = min(35, (x2 - x1) // 2)
            ex = int(cx + length * math.cos(rad))
            ey = int(cy - length * math.sin(rad))
            cv2.arrowedLine(frame, (cx, cy), (ex, ey), box_color, 2, tipLength=0.35)

        # Moving direction arrow (separate from orientation — shows motion vector)
        mov_deg = v.get("moving_direction_deg")
        if mov_deg is not None and is_moving:
            cx = (x1 + x2) // 2
            cy = y2 + 5
            rad = math.radians(float(mov_deg))
            ex = int(cx + 20 * math.cos(rad))
            ey = int(cy - 20 * math.sin(rad))
            cv2.arrowedLine(frame, (cx, cy), (ex, ey), (255, 255, 0), 1, tipLength=0.4)

        # Brake light — red glow at bottom of bbox
        if v.get("brake_light"):
            # Two red circles at rear (left and right tail lights)
            lx = x1 + (x2 - x1) // 4
            rx = x1 + 3 * (x2 - x1) // 4
            by = y2 - 3
            cv2.circle(frame, (lx, by), 5, C_BRAKE, -1)
            cv2.circle(frame, (rx, by), 5, C_BRAKE, -1)
            _text_bg(frame, "BRAKE", (lx - 5, y2 + 14), 0.38, C_BRAKE)

        # Turn indicator — blinking triangle
        indicator = v.get("indicator")
        if indicator == "left":
            tri_cx = x1 - 8
            tri_cy = (y1 + y2) // 2
            pts = np.array([[tri_cx, tri_cy],
                            [tri_cx - 12, tri_cy - 8],
                            [tri_cx - 12, tri_cy + 8]], np.int32)
            cv2.fillPoly(frame, [pts], C_INDICATOR)
            cv2.putText(frame, "L", (tri_cx - 22, tri_cy + 4), FONT, 0.4, C_INDICATOR, 1)
        elif indicator == "right":
            tri_cx = x2 + 8
            tri_cy = (y1 + y2) // 2
            pts = np.array([[tri_cx, tri_cy],
                            [tri_cx + 12, tri_cy - 8],
                            [tri_cx + 12, tri_cy + 8]], np.int32)
            cv2.fillPoly(frame, [pts], C_INDICATOR)
            cv2.putText(frame, "R", (tri_cx + 14, tri_cy + 4), FONT, 0.4, C_INDICATOR, 1)

        # Collision risk — bottom label with TTC
        if risk in ("critical", "warning"):
            ttc = v.get("ttc_seconds")
            risk_text = risk.upper()
            if ttc is not None:
                risk_text += f" TTC={ttc:.1f}s"
            # Thick colored border for visibility
            _corner_label(frame, risk_text, (x1, y1, x2, y2),
                          C_COLLISION_C if risk == "critical" else C_COLLISION_W, "bottom")


# ─── Pedestrian Drawing ──────────────────────────────────────────────────

def draw_pedestrians(frame, peds, camera):
    for p in peds:
        if p.get("camera") != camera and camera != "all":
            continue
        x1, y1, x2, y2 = [int(c) for c in p["bbox"]]
        risk = p.get("collision_risk", "none")
        if isinstance(risk, dict):
            risk = risk.get("level", "none")
        if risk == "critical":
            color = C_COLLISION_C
        elif risk == "warning":
            color = C_COLLISION_W
        else:
            color = C_PED

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label: pose + track id
        pose = p.get("pose_label", "person")
        tid = p.get("global_id") or p.get("track_id", "")
        label = f"PED {pose} #{tid}"
        _corner_label(frame, label, (x1, y1, x2, y2), color, "top")

        # Collision
        if risk in ("critical", "warning"):
            ttc = p.get("ttc_seconds")
            risk_text = risk.upper()
            if ttc is not None:
                risk_text += f" TTC={ttc:.1f}s"
            _corner_label(frame, risk_text, (x1, y1, x2, y2), color, "bottom")


# ─── Traffic Light Drawing ───────────────────────────────────────────────

def draw_traffic_lights(frame, tls, camera):
    for tl in tls:
        if tl.get("camera") != camera and camera != "all":
            continue
        x1, y1, x2, y2 = [int(c) for c in tl["bbox"]]
        color_name = tl.get("tl_color", "off") or "off"
        color = TL_COLORS.get(color_name, C_TL_OFF)

        # Filled circle at center with white border
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        radius = max(6, min(15, (x2 - x1) // 3))
        cv2.circle(frame, (cx, cy), radius + 2, C_WHITE, 2)
        cv2.circle(frame, (cx, cy), radius, color, -1)

        # Thin bbox outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Label
        text = color_name.upper()
        arrow = tl.get("tl_arrow")
        if arrow and arrow not in ("none", "None", None):
            text += f" {arrow}"
        _corner_label(frame, text, (x1, y1, x2, y2), color, "right")


# ─── Lane Drawing ────────────────────────────────────────────────────────

def draw_lanes(frame, lanes):
    for lane in lanes:
        pts = lane.get("points", [])
        if len(pts) < 2:
            continue
        pts_arr = np.array(pts, dtype=np.int32)
        lane_type = lane.get("type", "solid")
        color_name = lane.get("color", "white")

        if color_name == "yellow":
            line_color = C_LANE_YELLOW
        elif lane_type == "dashed":
            line_color = C_LANE_DASHED
        else:
            line_color = C_LANE_SOLID

        if lane_type == "dashed":
            for i in range(0, len(pts_arr) - 1, 2):
                end = min(i + 1, len(pts_arr) - 1)
                cv2.line(frame, tuple(pts_arr[i]), tuple(pts_arr[end]), line_color, 2)
        else:
            cv2.polylines(frame, [pts_arr], False, line_color, 2)

        # Label at midpoint
        mid = len(pts) // 2
        label = f"{color_name} {lane_type}"
        mx, my = int(pts[mid][0]), int(pts[mid][1])
        _text_bg(frame, label, (mx, my - 5), 0.32, line_color, 1, 0.5)


# ─── Road Sign Drawing ──────────────────────────────────────────────────

def draw_road_signs(frame, signs, camera):
    for s in signs:
        if s.get("camera") != camera and camera != "all":
            continue
        if "bbox" not in s:
            continue
        x1, y1, x2, y2 = [int(c) for c in s["bbox"]]
        sign_type = s.get("sign_type", s.get("label", "sign"))

        if "stop" in str(sign_type).lower():
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_STOP, 2)
            _corner_label(frame, "STOP", (x1, y1, x2, y2), C_STOP, "top")

        elif sign_type == "speed_limit":
            speed = s.get("speed_value", "?")
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_SPEED, 2)
            _corner_label(frame, f"{speed} mph", (x1, y1, x2, y2), C_SPEED, "top")

        elif sign_type == "ground_arrow":
            direction = s.get("direction", "?")
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_GROUND_ARR, 2)
            # Draw direction arrow inside bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            arrow_len = max(15, (x2 - x1) // 3)
            if direction == "left":
                cv2.arrowedLine(frame, (cx + arrow_len, cy), (cx - arrow_len, cy),
                                C_GROUND_ARR, 2, tipLength=0.3)
            elif direction == "right":
                cv2.arrowedLine(frame, (cx - arrow_len, cy), (cx + arrow_len, cy),
                                C_GROUND_ARR, 2, tipLength=0.3)
            else:  # straight
                cv2.arrowedLine(frame, (cx, cy + arrow_len), (cx, cy - arrow_len),
                                C_GROUND_ARR, 2, tipLength=0.3)
            _corner_label(frame, direction.upper(), (x1, y1, x2, y2), C_GROUND_ARR, "top")

        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
            text = str(s.get("text", sign_type))
            _corner_label(frame, text, (x1, y1, x2, y2), (0, 200, 0), "top")


# ─── Small Object Drawing ───────────────────────────────────────────────

def draw_objects(frame, objects, camera):
    for o in objects:
        if o.get("camera") != camera and camera != "all":
            continue
        if "bbox" not in o:
            continue
        x1, y1, x2, y2 = [int(c) for c in o["bbox"]]
        label = o.get("label", o.get("type", "object"))
        conf = o.get("confidence", 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_OBJECT, 2)
        _corner_label(frame, f"{label} {conf:.2f}", (x1, y1, x2, y2), C_OBJECT, "top")


# ─── Speed Bump Drawing ─────────────────────────────────────────────────

def draw_speed_bumps(frame, bumps):
    for sb in bumps:
        if "bbox" in sb:
            x1, y1, x2, y2 = [int(c) for c in sb["bbox"]]
            # Semi-transparent yellow overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), C_BUMP, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_BUMP, 2)
            conf = sb.get("confidence", 0)
            source = sb.get("source", "model")
            _text_bg(frame, f"BUMP {conf:.2f} ({source})", (x1, y1 - 4), 0.42, C_BUMP)
        elif "position_ego" in sb:
            H, W = frame.shape[:2]
            depth = sb.get("depth_m", 10)
            y_pos = int(H * 0.5 + H * 0.3 * (1.0 - depth / 25.0))
            y_pos = max(0, min(H - 10, y_pos))
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, y_pos - 5), (W, y_pos + 5), C_BUMP, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            _text_bg(frame, f"BUMP ~{depth:.0f}m", (10, y_pos - 8), 0.4, C_BUMP)


# ─── HUD (Head-Up Display) ──────────────────────────────────────────────

def draw_hud(frame, frame_id, data, camera):
    """Draw comprehensive stats panel."""
    H, W = frame.shape[:2]

    vehs = [v for v in data.get("vehicles", []) if v.get("camera") == camera or camera == "all"]
    peds = [p for p in data.get("pedestrians", []) if p.get("camera") == camera or camera == "all"]
    tls = [t for t in data.get("traffic_lights", []) if t.get("camera") == camera or camera == "all"]
    signs = [s for s in data.get("road_signs", []) if s.get("camera") == camera or camera == "all"]
    objs = [o for o in data.get("objects", []) if o.get("camera") == camera or camera == "all"]
    bumps = data.get("speed_bumps", [])
    lanes = data.get("lanes", [])

    n_moving = sum(1 for v in vehs if v.get("is_moving", True))
    n_parked = len(vehs) - n_moving
    n_brake = sum(1 for v in vehs if v.get("brake_light"))
    n_col = sum(1 for v in vehs + peds if v.get("collision_risk") in ("critical", "warning"))
    n_stop = sum(1 for s in signs if "stop" in str(s.get("sign_type", "")).lower())
    n_speed = sum(1 for s in signs if s.get("sign_type") == "speed_limit")
    n_arrows = sum(1 for s in signs if s.get("sign_type") == "ground_arrow")

    # Build lines
    lines = [
        (f"F:{frame_id}", C_WHITE),
        (f"Veh: {n_moving}mov {n_parked}park", C_MOVING),
    ]
    if n_brake > 0:
        lines.append((f"Brake: {n_brake}", C_BRAKE))
    if peds:
        lines.append((f"Ped: {len(peds)}", C_PED))
    if tls:
        colors_str = " ".join((t.get("tl_color") or "?")[0].upper() for t in tls[:4])
        lines.append((f"TL: {len(tls)} [{colors_str}]", C_TL_GREEN))
    if n_col > 0:
        lines.append((f"COLLISION: {n_col}", C_COLLISION_C))
    if n_stop > 0:
        lines.append((f"STOP: {n_stop}", C_STOP))
    if n_speed > 0:
        speeds = [s.get("speed_value") for s in signs if s.get("speed_value")]
        lines.append((f"Speed: {speeds}", C_SPEED))
    if n_arrows > 0:
        dirs = [s.get("direction", "?") for s in signs if s.get("sign_type") == "ground_arrow"]
        lines.append((f"Arrow: {dirs}", C_GROUND_ARR))
    if objs:
        labels = [o.get("label", "?") for o in objs[:4]]
        lines.append((f"Obj: {labels}", C_OBJECT))
    if bumps:
        lines.append((f"BUMP: {len(bumps)}", C_BUMP))
    if lanes:
        lines.append((f"Lanes: {len(lanes)}", C_LANE_SOLID))

    # Draw panel
    panel_h = 18 * len(lines) + 12
    panel_w = 260
    overlay = frame.copy()
    cv2.rectangle(overlay, (W - panel_w - 5, 5), (W - 5, panel_h + 5), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 22
    for text, color in lines:
        cv2.putText(frame, text, (W - panel_w, y), FONT, 0.42, color, 1, cv2.LINE_AA)
        y += 18


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EinsteinVision detection overlay")
    parser.add_argument("--seq", type=int, required=True, help="Sequence number")
    parser.add_argument("--camera", default="front",
                        choices=["front", "back", "left", "right", "all"])
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--det-dir", default=None, help="Override detection dir")
    parser.add_argument("--save-frames", default=None,
                        help="Save individual frames as images to this directory")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Process every Nth frame (for faster preview)")
    args = parser.parse_args()

    # Find video
    video_path = find_video(args.seq, args.camera if args.camera != "all" else "front")
    if video_path is None:
        print(f"Video not found for seq {args.seq} camera {args.camera}")
        sys.exit(1)
    print(f"Video: {video_path}")

    # Find detections
    if args.det_dir:
        det_dir = Path(args.det_dir)
    else:
        det_dir = find_detections_dir(args.seq)
    if det_dir is None:
        print(f"No detection JSONs found for seq {args.seq}")
        sys.exit(1)
    print(f"Detections: {det_dir}")
    n_jsons = len(list(det_dir.glob("*.json")))
    print(f"  {n_jsons} JSON files")

    # Output paths
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to: {args.save_frames}")

    if args.output is None and args.save_frames is None:
        out_dir = BASE_DIR / "Output" / "viz"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"seq{args.seq:02d}_{args.camera}.mp4")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        print(f"Output video: {args.output}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Failed to open video")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {W}x{H}, {total} frames")

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    writer = None
    if args.output:
        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (W, H),
        )

    end_frame = total
    if args.max_frames is not None:
        end_frame = min(total, args.start + args.max_frames)

    frames_written = 0
    frames_with_det = 0

    for frame_id in range(args.start, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_id - args.start) % args.frame_skip != 0:
            continue

        data = load_detection(det_dir, args.seq, frame_id)
        if data is not None:
            frames_with_det += 1
            cam = args.camera
            # Draw order: lanes (bottom) → bumps → vehicles → peds → TL → signs → objects → HUD
            draw_lanes(frame, data.get("lanes", []))
            draw_speed_bumps(frame, data.get("speed_bumps", []))
            draw_road_signs(frame, data.get("road_signs", []), cam)
            draw_objects(frame, data.get("objects", []), cam)
            draw_vehicles(frame, data.get("vehicles", []), cam)
            draw_pedestrians(frame, data.get("pedestrians", []), cam)
            draw_traffic_lights(frame, data.get("traffic_lights", []), cam)
            draw_hud(frame, frame_id, data, cam)
        else:
            cv2.putText(frame, f"Frame {frame_id} (no det)", (10, 30),
                        FONT, 0.6, (128, 128, 128), 1)

        if writer:
            writer.write(frame)
        if args.save_frames:
            cv2.imwrite(os.path.join(args.save_frames, f"f{frame_id:06d}.jpg"), frame)

        frames_written += 1
        if frames_written % 200 == 0:
            print(f"  {frames_written} frames processed", flush=True)

    cap.release()
    if writer:
        writer.release()

    print(f"\nDone. {frames_written} frames, {frames_with_det} with detections")
    if args.output:
        print(f"Video: {args.output}")
    if args.save_frames:
        print(f"Frames: {args.save_frames}/")


if __name__ == "__main__":
    main()
