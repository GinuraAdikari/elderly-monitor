"""Stage 6: deterministic triage — which segments are ambiguous enough to hand to the agent."""
from __future__ import annotations
import numpy as np
from .schemas import State, AmbiguityItem, IN_BED


def find_items(segs, bins, cfg):
    t, hz = cfg["triage"], cfg["video"]["state_hz"]
    lo, hi = t["bedfloor_range"]
    st = cfg["states"]
    cand = {}  # seg_index -> (priority, [reasons])

    def add(i, prio, why):
        p, r = cand.get(i, (0, []))
        cand[i] = (max(p, prio), r + [why])

    for i, s in enumerate(segs):
        sub = bins[(bins.t >= s.start) & (bins.t < s.end)]
        if len(sub) == 0:
            continue
        if s.state == State.UNKNOWN and s.dur >= t["unknown_min_sec"]:
            add(i, 3, "unknown_segment")
        elif s.conf < t["conf_tau"] and s.dur >= t["low_conf_min_sec"]:
            add(i, 2, "low_confidence")
        if s.state in (State.LYING_IN_BED, State.LYING_ON_FLOOR):
            pb, pn = sub.p_bed.mean(), sub.p_near.mean()
            if lo <= pb <= hi or (pn - pb) > 0.3:
                add(i, 4, "bed_vs_floor")
        occ = ((sub.core_vis < st["occ_vis"]) & (sub.p_near > 0.5)).sum() / hz
        if occ >= t["occluded_min_sec"]:
            add(i, 2, "occluded_in_bed")
        if (sub.n_persons >= 2).sum() / hz >= t["caregiver_min_sec"]:
            add(i, 2, "caregiver_present")
        prev_in = i > 0 and segs[i - 1].state in IN_BED
        nxt_in = i + 1 >= len(segs) or segs[i + 1].state in IN_BED
        if s.state == State.STANDING and prev_in and nxt_in:
            add(i, 3, "exit_ambiguous")
    items = []
    for i, (p, why) in sorted(cand.items(), key=lambda kv: -kv[1][0])[: t["max_items"]]:
        items.append(AmbiguityItem(0, segs[i].start, segs[i].end, "; ".join(why), i))
    items.sort(key=lambda x: x.t0)
    for n, it in enumerate(items):
        it.id = n
    return items
