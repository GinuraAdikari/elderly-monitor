"""Stage 4: per-second emission scores -> Viterbi decoding -> segments."""
from __future__ import annotations
import dataclasses
import numpy as np
from .schemas import State, STATES, IDX, Segment

LIB, SOB, SOUT = IDX[State.LYING_IN_BED], IDX[State.SITTING_ON_BED], IDX[State.SITTING_OUTSIDE_BED]
STD, WLK, OUT = IDX[State.STANDING], IDX[State.WALKING], IDX[State.OUT_OF_BED]
LOF, UNK = IDX[State.LYING_ON_FLOOR], IDX[State.UNKNOWN]


def emission_scores(b, cfg):
    """b: bins DataFrame (features.build_bins). Returns (E [T,S] in [0,1], occluded[T] bool)."""
    c, hz = cfg["states"], cfg["video"]["state_hz"]
    g = lambda k: np.nan_to_num(b[k].to_numpy(float))
    T = len(b)
    E = np.zeros((T, len(STATES)))
    present = g("present") >= c["min_present_frac"]
    ly, si, st, pb, pn, mv, vis = g("lying"), g("sitting"), g("standing"), g("p_bed"), g("p_near"), g("moving"), g("core_vis")
    E[:, LIB], E[:, LOF] = ly * pb, ly * (1 - pb)
    E[:, SOB], E[:, SOUT] = si * pn, si * (1 - pn)
    E[:, STD], E[:, WLK] = st * (1 - mv), st * mv
    E[:, UNK] = np.maximum(c["unknown_floor"], 1 - np.clip(vis / c["vis_good"], 0, 1))
    occ = present & (vis < c["occ_vis"]) & (pn > 0.5)           # mostly hidden but in the bed zone
    E[occ, LIB] = np.maximum(E[occ, LIB], c["occ_hold"])

    last_out, run = True, 0.0                                     # absence handling (defaults to out-of-bed if vacant)
    for i in range(T):
        if present[i]:
            last_out, run = bool(pn[i] < 0.5), 0.0
            continue
        run += 1.0 / hz
        E[i, :] = 0.0
        if run < c["out_of_view_sec"]:
            E[i, :] = 0.3                                         # short dropout: flat, let Viterbi hold the state
        elif last_out:
            E[i, OUT], E[i, UNK] = 0.9, 0.1
        else:
            E[i, UNK] = 0.9
    return E, occ


def build_transition(cfg):
    c, n = cfg["states"], len(STATES)
    T = np.full((n, n), c["forbidden_prob"])
    for s in STATES:
        i = IDX[s]
        allowed = [IDX[State(x)] for x in c["transitions"].get(s.value, [])]
        T[i, i] = c["stay_prob"]
        for j in allowed:
            T[i, j] = (1 - c["stay_prob"]) / len(allowed)
    return np.log(T / T.sum(1, keepdims=True))


def viterbi(logE, logT):
    T, S = logE.shape
    if T == 0:
        return np.zeros(0, int)
    dp, bp = np.zeros((T, S)), np.zeros((T, S), int)
    dp[0] = logE[0] - np.log(S)
    for t in range(1, T):
        cand = dp[t - 1][:, None] + logT
        bp[t], dp[t] = cand.argmax(0), cand.max(0) + logE[t]
    path = np.zeros(T, int)
    path[-1] = dp[-1].argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def decode(E, cfg, mode):
    """mode 'rules' = per-bin argmax (ablation baseline); anything else = Viterbi."""
    if mode == "rules":
        return E.argmax(1)
    return viterbi(np.log(np.clip(E, 1e-3, 1.0)), build_transition(cfg))


def to_segments(labels, conf, src, hz, duration):
    segs, T, i = [], len(labels), 0
    while i < T:
        j = i
        while j + 1 < T and labels[j + 1] == labels[i]:
            j += 1
        segs.append(Segment(i / hz, min((j + 1) / hz, duration), STATES[labels[i]],
                            float(np.mean(conf[i:j + 1])), "agent" if (src[i:j + 1] == 1).any() else "rules"))
        i = j + 1
    return segs


def _coalesce(segs):
    out = []
    for s in segs:
        if out and out[-1].state == s.state:
            p = out[-1]
            w1, w2 = p.dur, s.dur
            p.conf = (p.conf * w1 + s.conf * w2) / max(w1 + w2, 1e-9)
            p.end = s.end
            p.source = "agent" if "agent" in (p.source, s.source) else "rules"
        else:
            out.append(s)
    return out


def merge_short(segs, min_sec):
    segs = [dataclasses.replace(s) for s in segs]
    while len(segs) > 1:
        i = next((k for k, s in enumerate(segs) if s.dur < min_sec), None)
        if i is None:
            break
        nb = [segs[k] for k in (i - 1, i + 1) if 0 <= k < len(segs)]
        tgt = max(nb, key=lambda x: x.dur)
        tgt.start, tgt.end = min(tgt.start, segs[i].start), max(tgt.end, segs[i].end)
        segs.pop(i)
        segs = _coalesce(segs)
    return _coalesce(segs)


def segments_to_labels(segs, nbins, hz):
    lab = np.full(nbins, IDX[State.UNKNOWN])
    for s in segs:
        lab[int(round(s.start * hz)):int(round(s.end * hz))] = IDX[s.state]
    return lab
