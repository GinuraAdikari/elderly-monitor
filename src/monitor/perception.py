"""Stage 1: person detection + pose + tracking -> perception.parquet (one row per person per sampled frame).

Columns: frame_idx, t, track_id, x1,y1,x2,y2, conf, kp{0..16}_{x,y,c}  (pixels of the *processed* frame).
Frame size / duration go to perception_meta.json next to the parquet.
"""
from __future__ import annotations
import json
import pathlib
import numpy as np
import pandas as pd

KP_COLS = [f"kp{i}_{a}" for i in range(17) for a in "xyc"]
COLS = ["frame_idx", "t", "track_id", "x1", "y1", "x2", "y2", "conf"] + KP_COLS


def run(video, out_dir, cfg, force=False):
    out_dir = pathlib.Path(out_dir)
    pq, mp = out_dir / "perception.parquet", out_dir / "perception_meta.json"
    if pq.exists() and mp.exists() and not force:
        return pd.read_parquet(pq), json.load(open(mp))

    import cv2
    from ultralytics import YOLO
    out_dir.mkdir(parents=True, exist_ok=True)
    v, p = cfg["video"], cfg["perception"]
    model = YOLO(p["pose_model"])  # fresh model per video => fresh tracker state
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    stride = max(1, round(fps / v["perception_fps"]))
    rows, lumas, idx, W, H = [], [], -1, None, None
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if W is None:
            h0, w0 = frame.shape[:2]
            scale = min(1.0, v["max_width"] / w0)
            W, H = int(w0 * scale), int(h0 * scale)
        if idx % stride:
            continue
        if (W, H) != (frame.shape[1], frame.shape[0]):
            frame = cv2.resize(frame, (W, H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lumas.append(float(gray.mean()))
        if gray.mean() < v["clahe_below_luma"]:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        r = model.track(frame, persist=True, tracker=p["tracker"], conf=p["det_conf"], verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else -np.ones(len(xyxy), int)
        kps = r.keypoints.data.cpu().numpy()  # [n,17,3]
        for b, c, i, k in zip(xyxy, conf, ids, kps):
            rows.append([idx, idx / fps, int(i), *b.tolist(), float(c), *k.reshape(-1).tolist()])
    cap.release()
    df = pd.DataFrame(rows, columns=COLS)
    meta = {"fps": fps, "stride": stride, "n_frames": idx + 1, "width": W, "height": H,
            "duration": (idx + 1) / fps, "mean_luma": float(np.mean(lumas)) if lumas else None}
    df.to_parquet(pq)
    json.dump(meta, open(mp, "w"), indent=2)
    return df, meta
