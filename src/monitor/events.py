"""Stage 5: bed-exit / bed-return / stand-attempt events from the segment sequence (not from single frames)."""
from __future__ import annotations
from .schemas import State, Event, Segment, IN_BED, OUT_STATES, CONFIRMING_OUT


def _confirming(run, bins, c):
    for s in run:
        if s.state in CONFIRMING_OUT and s.dur >= c["confirm_min_sec"]:
            return s, s.start
        if s.state == State.STANDING:
            if s.dur >= c["standing_exit_sec"]:
                return s, s.start + c["standing_exit_sec"]
            if bins is not None and "bed_dist_bl" in bins and len(bins):
                sub = bins[(bins.t >= s.start) & (bins.t < s.end) & (bins.bed_dist_bl >= c["exit_min_dist_bl"])]
                if len(sub):
                    return s, float(sub.t.iloc[0])
    return None


def detect_events(segs: list[Segment], bins, cfg) -> list[Event]:
    c, ev, n, k = cfg["events"], [], len(segs), 0
    first = next((s for s in segs if s.state != State.UNKNOWN), None)
    at_bed = None if first is None else first.state in IN_BED
    last_known = None
    while k < n:
        s = segs[k]
        if s.state == State.UNKNOWN:
            k += 1
            continue
        if at_bed and s.state in OUT_STATES:
            m = k
            while m < n and segs[m].state not in IN_BED:
                m += 1
            run = segs[k:m]
            hit = _confirming(run, bins, c)
            if hit:
                cs, t_conf = hit
                ev.append(Event("bed_exit", s.start, t_conf, last_known.state if last_known else State.UNKNOWN,
                                cs.state, round((last_known.conf if last_known else cs.conf) / 2 + cs.conf / 2, 3),
                                source="agent" if any(x.source == "agent" for x in run) else "rules"))
                at_bed = False
            elif m < n and segs[m].start - s.start <= c["stand_attempt_max_sec"]:
                ev.append(Event("stand_attempt", s.start, segs[m].start,
                                last_known.state if last_known else State.UNKNOWN, run[0].state,
                                round(sum(x.conf for x in run) / len(run), 3)))
            known = [x for x in run if x.state != State.UNKNOWN]
            last_known = known[-1] if known else last_known
            k = m
            continue
        if at_bed is False and s.state in IN_BED:
            m, total = k, 0.0
            while m < n and segs[m].state in IN_BED | {State.UNKNOWN}:
                total += segs[m].dur if segs[m].state in IN_BED else 0.0
                m += 1
            if total >= c["return_confirm_sec"]:
                appr = last_known if last_known and last_known.state in OUT_STATES else s
                ev.append(Event("bed_return", appr.start, s.start + c["return_confirm_sec"],
                                appr.state, s.state, round(s.conf, 3),
                                source="agent" if any(x.source == "agent" for x in segs[k:m]) else "rules"))
                at_bed = True
            last_known = segs[m - 1]
            k = m
            continue
        last_known = s
        k += 1
    return ev
