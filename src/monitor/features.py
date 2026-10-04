"""Stage 3: geometric features per frame -> 1 Hz bins (one row per second)."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from .perception import KP_COLS


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def posture_probs(angle, aspect, hk, f):
    """Vectorised, NaN-safe. Returns (lying, sitting, standing); the three sum to 1."""
    lie_as = sig((aspect - f["lying_aspect"]) / f["lying_aspect_slope"])
    lie_an = sig((angle - f["lying_angle_deg"]) / f["lying_angle_slope"])
    lying = np.where(np.isnan(angle), lie_as, 0.6 * lie_an + 0.4 * lie_as)
    sit_hk = sig((f["sit_hk_ratio"] - hk) / f["sit_hk_slope"])
    sit_as = sig((aspect - f["sit_aspect_mid"]) / f["sit_aspect_slope"])
    sit = np.where(np.isnan(hk), sit_as, sit_hk)
    up = 1.0 - lying
    return lying, up * sit, up * (1.0 - sit)


def frame_features(df, bed_px, cfg):
    import cv2
    f, kc = cfg["features"], cfg["perception"]["kp_conf"]
    n = len(df)
    kp = df[KP_COLS].to_numpy(float).reshape(n, 17, 3)
    xy, cf = kp[:, :, :2].copy(), kp[:, :, 2]
    xy[cf < kc] = np.nan

    def mid(a, b):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(xy[:, [a, b], :], axis=1)

    sh, hip, knee = mid(5, 6), mid(11, 12), mid(13, 14)
    bw, bh = (df.x2 - df.x1).to_numpy(), (df.y2 - df.y1).to_numpy()
    body_len = np.maximum(np.maximum(bw, bh), 1.0)
    d = sh - hip
    torso = np.linalg.norm(d, axis=1)
    bad = ~(torso > 0.05 * body_len)
    angle = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1])))
    hk = (knee[:, 1] - hip[:, 1]) / np.where(bad, np.nan, torso)
    angle[bad] = np.nan
    aspect = bw / np.maximum(bh, 1.0)
    lying, sitting, standing = posture_probs(angle, aspect, hk, f)

    cxy = np.stack([(df.x1 + df.x2) / 2, (df.y1 + df.y2) / 2], 1)
    anchor = np.where(np.isnan(hip), cxy, hip)  # hip midpoint, else bbox centre
    cnt = bed_px.astype(np.float32).reshape(-1, 1, 2)
    signed = np.array([cv2.pointPolygonTest(cnt, (float(x), float(y)), True) for x, y in anchor])
    sbl = signed / body_len
    return pd.DataFrame({
        "t": df.t.to_numpy(), "lying": lying, "sitting": sitting, "standing": standing,
        "core_vis": cf[:, [5, 6, 11, 12]].mean(1), "signed_bl": sbl,
        "in_bed": (signed >= 0).astype(float), "near_bed": (sbl >= -f["edge_margin_bl"]).astype(float),
        "ax": anchor[:, 0], "ay": anchor[:, 1], "body_len": body_len, "aspect": aspect, "angle": angle, "hk": hk,
    })


def build_bins(patient, everyone, meta, bed_px, cfg) -> pd.DataFrame:
    hz, pfps, f = cfg["video"]["state_hz"], cfg["video"]["perception_fps"], cfg["features"]
    nb = int(np.ceil(meta["duration"] * hz))
    cols = ["present", "lying", "sitting", "standing", "core_vis", "p_bed", "p_near", "bed_dist_bl",
            "cx", "cy", "body_len", "n_persons"]
    b = pd.DataFrame(0.0, index=range(nb), columns=cols)
    if len(patient):
        fr = frame_features(patient, bed_px, cfg)
        fr["bin"] = np.clip(np.floor(fr.t * hz).astype(int), 0, nb - 1)
        g = fr.groupby("bin")
        agg = pd.DataFrame({
            "present": (g.size() / (pfps / hz)).clip(upper=1.0),
            "lying": g.lying.mean(), "sitting": g.sitting.mean(), "standing": g.standing.mean(),
            "core_vis": g.core_vis.mean(), "p_bed": g.in_bed.mean(), "p_near": g.near_bed.mean(),
            "bed_dist_bl": (-g.signed_bl.median()).clip(lower=0),
            "cx": g.ax.median(), "cy": g.ay.median(), "body_len": g.body_len.median()})
        b.loc[agg.index, agg.columns] = agg.fillna(0.0)
    if len(everyone):
        ev = everyone.assign(bin=np.clip(np.floor(everyone.t * hz).astype(int), 0, nb - 1))
        npers = ev.groupby("bin").track_id.nunique()
        b.loc[npers.index, "n_persons"] = npers
    pres = b.present.to_numpy() > 0.3
    cx, cy, bl = b.cx.to_numpy(), b.cy.to_numpy(), b.body_len.to_numpy()
    sp = np.zeros(nb)
    for k in range(1, nb):
        if pres[k] and pres[k - 1] and bl[k] > 0:
            sp[k] = np.hypot(cx[k] - cx[k - 1], cy[k] - cy[k - 1]) / bl[k] * hz
    sp = pd.Series(sp).rolling(3, center=True, min_periods=1).mean().to_numpy()
    b["speed_bl_s"] = sp
    b["moving"] = sig((sp - f["walk_speed_bl_s"]) / f["walk_speed_slope"])
    b.insert(0, "t", np.arange(nb) / hz)
    return b
