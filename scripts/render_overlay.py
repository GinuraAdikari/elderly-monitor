"""Render visual overlay: bounding boxes, 17-keypoint skeleton, track IDs,
bed polygon, active state timeline, and feature HUD onto the video.

Usage:
  python scripts/render_overlay.py --video data/raw/gmdcsa_01.mp4 --polygon config/bed_polygons/gmdcsa_01.json
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import cv2
import numpy as np
import pandas as pd

# COCO 17 keypoint connections (skeleton bones)
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),               # Head
    (5, 6), (5, 11), (6, 12), (11, 12),           # Torso
    (5, 7), (7, 9),                               # Left arm
    (6, 8), (8, 10),                              # Right arm
    (11, 13), (13, 15),                           # Left leg
    (12, 14), (14, 16),                           # Right leg
]

# Color palette (BGR)
COLOR_BED = (0, 230, 255)         # Yellow / Gold
COLOR_PATIENT = (50, 220, 50)     # Bright Green
COLOR_OTHER = (50, 150, 255)      # Orange
COLOR_BONE = (240, 240, 240)      # White
COLOR_JOINT = (0, 200, 255)       # Yellowish


def draw_hud(frame, text_lines, x=20, y=30, bg_color=(20, 20, 20), text_color=(255, 255, 255)):
    """Draw a semi-transparent HUD banner with multiple lines of text."""
    line_h = 24
    padding = 10
    max_w = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] for line in text_lines)
    box_w = max_w + 2 * padding
    box_h = len(text_lines) * line_h + 2 * padding

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - padding), (x + box_w, y + box_h - padding), bg_color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x, y - padding), (x + box_w, y + box_h - padding), (80, 80, 80), 1)

    for i, line in enumerate(text_lines):
        yy = y + (i + 1) * line_h - 4
        # Highlight state line with special color
        color = text_color
        if line.startswith("STATE:"):
            if "BED" in line:
                color = (100, 255, 100)
            elif "FLOOR" in line:
                color = (80, 80, 255)
            elif "STAND" in line or "WALK":
                color = (100, 240, 255)
        cv2.putText(frame, line, (x + padding, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def render(video_path, polygon_path, out_path=None, perception_path=None, timeline_path=None, bins_path=None, kp_conf=0.3):
    video_path = pathlib.Path(video_path)
    base_dir = video_path.parents[2] / "outputs" / video_path.stem
    if not base_dir.exists():
        base_dir = pathlib.Path("outputs") / video_path.stem

    if out_path is None:
        out_path = base_dir / "overlay.mp4"
    else:
        out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if perception_path is None:
        perception_path = base_dir / "perception.parquet"
    if timeline_path is None:
        cand = base_dir / "rules" / "timeline.json"
        timeline_path = cand if cand.exists() else base_dir / "viterbi" / "timeline.json"
    if bins_path is None:
        bins_path = base_dir / "bins.parquet"
    meta_path = base_dir / "perception_meta.json"

    # Load polygon
    poly_data = json.load(open(polygon_path))
    poly_norm = np.array(poly_data["polygon"], dtype=float)

    # Load meta and perception if available
    meta = json.load(open(meta_path)) if pathlib.Path(meta_path).exists() else {}
    df_det = pd.read_parquet(perception_path) if pathlib.Path(perception_path).exists() else pd.DataFrame()
    timeline = json.load(open(timeline_path)) if pathlib.Path(timeline_path).exists() else []
    bins = pd.read_parquet(bins_path) if pathlib.Path(bins_path).exists() else pd.DataFrame()

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = meta.get("width", w0)
    target_h = meta.get("height", h0)
    bed_px = (poly_norm * np.array([target_w, target_h], dtype=float)).astype(np.int32)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (target_w, target_h))

    # Index detections by frame_idx
    det_by_frame = {}
    if not df_det.empty:
        for fidx, grp in df_det.groupby("frame_idx"):
            det_by_frame[int(fidx)] = grp

    main_track = 1
    if not df_det.empty:
        main_track = df_det.track_id.value_counts().idxmax()

    print(f"Rendering overlay for {video_path.name} -> {out_path.name} ({target_w}x{target_h} @ {src_fps:.1f} fps)")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame.shape[1], frame.shape[0]) != (target_w, target_h):
            frame = cv2.resize(frame, (target_w, target_h))

        t = frame_idx / src_fps

        # 1. Draw Bed Polygon
        cv2.polylines(frame, [bed_px.reshape(-1, 1, 2)], isClosed=True, color=COLOR_BED, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(frame, "BED ZONE", (bed_px[0, 0] + 10, bed_px[0, 1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BED, 2, cv2.LINE_AA)

        # 2. Draw detections for this frame (or interpolate from nearest sampled perception frame)
        # Note: perception is sampled at stride ~6 (5 fps). We look for exact match or nearest prior sample.
        nearest_f = min(det_by_frame.keys(), key=lambda k: abs(k - frame_idx)) if det_by_frame else None
        if nearest_f is not None and abs(nearest_f - frame_idx) <= meta.get("stride", 6):
            dets = det_by_frame[nearest_f]
            for _, row in dets.iterrows():
                tid = int(row.track_id)
                x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
                is_patient = (tid == main_track)
                box_color = COLOR_PATIENT if is_patient else COLOR_OTHER

                # Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
                lbl = f"ID: {tid} {'[Patient]' if is_patient else ''} ({row.conf:.2f})"
                cv2.putText(frame, lbl, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)

                # 17 Keypoints
                kps = []
                for k in range(17):
                    kx, ky, kc = float(row[f"kp{k}_x"]), float(row[f"kp{k}_y"]), float(row[f"kp{k}_c"])
                    kps.append((int(kx), int(ky), kc))
                    if kc >= kp_conf:
                        cv2.circle(frame, (int(kx), int(ky)), 3, COLOR_JOINT, -1, cv2.LINE_AA)

                # Bones
                for j1, j2 in SKELETON_EDGES:
                    p1, p2 = kps[j1], kps[j2]
                    if p1[2] >= kp_conf and p2[2] >= kp_conf:
                        cv2.line(frame, (p1[0], p1[1]), (p2[0], p2[1]), COLOR_BONE, 2, cv2.LINE_AA)

        # 3. Lookup State from Timeline
        cur_state = "UNKNOWN"
        state_conf = 1.0
        for seg in timeline:
            if seg["start"] <= t <= seg["end"]:
                cur_state = seg["state"]
                state_conf = seg.get("confidence", 1.0)
                break

        # 4. Lookup Features from Bins
        hud_lines = [
            f"TIME: {t:05.2f}s  (frame {frame_idx}/{total_frames})",
            f"STATE: {cur_state} ({state_conf:.2f})",
        ]
        sec_idx = int(np.floor(t))
        if not bins.empty and sec_idx < len(bins):
            brow = bins.iloc[sec_idx]
            hud_lines.append(f"POSTURE: Lie={brow.lying:.2f} Sit={brow.sitting:.2f} Stand={brow.standing:.2f}")
            hud_lines.append(f"LOCATION: in_bed={brow.p_bed:.2f} dist={brow.bed_dist_bl:.2f} BL")
            hud_lines.append(f"MOTION: spd={brow.speed_bl_s:.2f} BL/s mv={brow.moving:.2f}")

        draw_hud(frame, hud_lines, x=20, y=30)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Successfully generated overlay: {out_path} ({out_path.stat().st_size} bytes, {frame_idx} frames)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Render perception & state tracking overlay")
    ap.add_argument("--video", required=True, help="Video path")
    ap.add_argument("--polygon", required=True, help="Bed polygon JSON")
    ap.add_argument("--out", default=None, help="Output MP4 path")
    ap.add_argument("--perception", default=None, help="perception.parquet path")
    ap.add_argument("--timeline", default=None, help="timeline.json path")
    ap.add_argument("--bins", default=None, help="bins.parquet path")
    ap.add_argument("--kp-conf", type=float, default=0.3, help="Keypoint conf threshold")
    args = ap.parse_args()

    render(args.video, args.polygon, args.out, args.perception, args.timeline, args.bins, args.kp_conf)


if __name__ == "__main__":
    main()
