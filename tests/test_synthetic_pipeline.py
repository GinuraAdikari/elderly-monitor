"""Synthetic end-to-end check of stages 2-8 (no video, no YOLO): lying -> sitting -> standing -> walking away -> gone."""
import numpy as np
import pandas as pd
from monitor import patient, features, states, events, alerts, report
from monitor.perception import COLS

W, H, FPS = 960, 720, 5
BED = np.array([[100, 300], [500, 300], [500, 600], [100, 600]], float)


def person(kind, cx, cy):
    """17 COCO keypoints (x,y,c) for a crude person; returns (bbox, kps)."""
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    if kind == "stand":      # vertical, knees well below hips
        ys = {0: -100, 5: -60, 6: -60, 11: 0, 12: 0, 13: 50, 14: 50, 15: 100, 16: 100}
        for i, y in ys.items(): k[i, :2] = (cx + (-10 if i % 2 else 10), cy + y)
    elif kind == "sit":      # torso vertical, knees level with hips (side view)
        ys = {0: -100, 5: -60, 6: -60, 11: 0, 12: 0}
        for i, y in ys.items(): k[i, :2] = (cx, cy + y)
        for i in (13, 14, 15, 16): k[i, :2] = (cx + 60, cy + 5 + (50 if i > 14 else 0))
    else:                    # lying: torso horizontal
        xs = {0: -100, 5: -60, 6: -60, 11: 0, 12: 0, 13: 50, 14: 50, 15: 100, 16: 100}
        for i, x in xs.items(): k[i, :2] = (cx + x, cy + (-8 if i % 2 else 8))
    for i in range(17):
        if k[i, 0] == 0: k[i, :2] = (cx, cy)
    x1, y1, x2, y2 = k[:, 0].min() - 10, k[:, 1].min() - 10, k[:, 0].max() + 10, k[:, 1].max() + 10
    return (x1, y1, x2, y2), k


def build():
    rows, t = [], 0.0
    plan = [("lie", 20, 300, 450, 0), ("sit", 10, 300, 450, 0), ("stand", 5, 300, 450, 0),
            ("stand", 20, 560, 450, 30)]  # walking away: 30 px/frame = 150 px/s
    fi = 0
    for kind, secs, cx, cy, vx in plan:
        for n in range(secs * FPS):
            x = cx + vx * n
            bb, k = person(kind, x if kind != "lie" else cx, cy if kind != "sit" else cy)
            rows.append([fi, fi / FPS, 1, *bb, 0.9, *k.reshape(-1)])
            fi += 1
    df = pd.DataFrame(rows, columns=COLS)
    return df[df.x2 < W - 5].reset_index(drop=True), fi / FPS + 10  # last 10 s: nobody in view


def test_stages(cfg):
    raw, dur = build()
    meta = {"width": W, "height": H, "duration": dur}
    pat, info = patient.select_patient(raw, BED, cfg)
    bins = features.build_bins(pat, raw, meta, BED, cfg)
    E, _ = states.emission_scores(bins, cfg)
    lab = states.decode(E, cfg, "viterbi")
    segs = states.merge_short(states.to_segments(lab, E[np.arange(len(lab)), lab], np.zeros(len(lab), int), 1, dur), 2)
    names = [s.state.value for s in segs]
    assert names[0] == "LYING_IN_BED", names
    assert "SITTING_ON_BED" in names and "WALKING" in names, names
    ev = events.detect_events(segs, bins, cfg)
    assert [e.event for e in ev][:1] == ["bed_exit"], (names, ev)
    dec, overall = alerts.evaluate(segs, ev, cfg)
    s = report.build_summary(segs, ev, dur)
    assert abs(sum(s["activity_duration_sec"].values()) - dur) < 1.5
