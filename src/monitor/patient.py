"""Stage 2: pick the patient out of all detected people; everyone else is treated as a caregiver/visitor."""
from __future__ import annotations
import numpy as np
import pandas as pd


def select_patient(df: pd.DataFrame, bed_px: np.ndarray, cfg: dict):
    """Main track = highest (presence + bed_weight * time-in-bed). Frames where the main track is missing may be
    filled by a nearby different track ID (cheap re-association after tracker ID switches)."""
    if df.empty:
        return df, {"main_track_id": None}
    import cv2
    pf, pc = cfg["video"]["perception_fps"], cfg["patient"]
    cnt = bed_px.astype(np.float32).reshape(-1, 1, 2)
    cx, cy = (df.x1 + df.x2) / 2, (df.y1 + df.y2) / 2
    inside = np.array([cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0 for x, y in zip(cx, cy)])
    stats = df.assign(inside=inside).groupby("track_id").agg(n=("t", "size"), inb=("inside", "sum"))
    score = (stats.n + pc["bed_weight"] * stats.inb) / pf
    main = int(score.idxmax())

    rows, last, reacq = [], None, 0
    for _, grp in df.sort_values("frame_idx").groupby("frame_idx"):
        m = grp[grp.track_id == main]
        if len(m):
            r = m.iloc[0]
        elif last is not None and grp.t.iloc[0] - last[0] <= pc["reacquire_sec"]:
            dist = np.hypot((grp.x1 + grp.x2) / 2 - last[1], (grp.y1 + grp.y2) / 2 - last[2])
            j = dist.idxmin()
            if dist[j] > pc["reacquire_dist"] * last[3]:
                continue
            r, reacq = grp.loc[j], reacq + 1
        else:
            continue
        rows.append(r)
        last = (r.t, (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2, r.y2 - r.y1)
    
    coasted_rows = []
    prev_r = None
    for r in rows:
        if prev_r is not None:
            dt = r.t - prev_r.t
            if 0.25 < dt <= (pc.get("max_coast_sec", 1.0) + 0.05):
                n_miss = int(round(dt / (1.0 / pf))) - 1
                for step in range(1, min(n_miss + 1, 6)):
                    coasted = prev_r.copy()
                    coasted.t = prev_r.t + step * (1.0 / pf)
                    coasted.conf = prev_r.conf * (0.9 ** step)
                    coasted_rows.append(coasted)
        coasted_rows.append(r)
        prev_r = r
    pat = pd.DataFrame(coasted_rows).reset_index(drop=True)
    return pat, {"main_track_id": main, "track_ids": [int(i) for i in stats.index], "reacquired_frames": reacq}
