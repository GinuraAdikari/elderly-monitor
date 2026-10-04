"""Stage 7: NORMAL / MONITOR / ALERT from segments + events. Every decision carries a human-readable reason."""
from __future__ import annotations
from .schemas import State, Decision, LEVELS, IN_BED, OUT_STATES
from .util import ms


def out_runs(segs):
    """Maximal runs of non-in-bed segments that contain at least one out-of-bed state (UNKNOWN inside a run counts)."""
    runs, cur = [], []
    for s in segs + [None]:
        if s is not None and s.state not in IN_BED:
            cur.append(s)
            continue
        if cur and any(x.state in OUT_STATES for x in cur):
            runs.append((cur[0].start, cur[-1].end))
        cur = []
    return runs


def evaluate(segs, events, cfg):
    a, dec = cfg["alerts"], []

    def hit(level, rule, start, limit, why):
        dec.append(Decision(level, rule, why, start + limit, start))

    for i, s in enumerate(segs):
        if s.state == State.LYING_ON_FLOOR and s.dur >= a["floor_alert_sec"]:
            hit("ALERT", "floor", s.start, a["floor_alert_sec"], f"Lying on floor for {s.dur:.0f}s (limit {a['floor_alert_sec']}s) since {ms(s.start)}")
        if s.state == State.SITTING_ON_BED and s.dur >= a["sit_monitor_sec"]:
            hit("MONITOR", "sit_on_bed", s.start, a["sit_monitor_sec"], f"Sitting on bed {s.dur:.0f}s (limit {a['sit_monitor_sec']}s) since {ms(s.start)}")
        if s.state == State.STANDING and s.dur >= a["standing_monitor_sec"]:
            hit("MONITOR", "standing_still", s.start, a["standing_monitor_sec"], f"Standing {s.dur:.0f}s (limit {a['standing_monitor_sec']}s) since {ms(s.start)}")
        if s.state == State.UNKNOWN:
            prev = segs[i - 1].state if i else None
            if s.dur >= a["unknown_alert_sec"] and prev in (State.STANDING, State.WALKING, State.SITTING_OUTSIDE_BED, State.OUT_OF_BED):
                hit("ALERT", "unknown_after_upright", s.start, a["unknown_alert_sec"], f"Cannot determine state for {s.dur:.0f}s after last seen {prev.value} at {ms(s.start)}")
            elif s.dur >= a["unknown_monitor_sec"]:
                hit("MONITOR", "unknown", s.start, a["unknown_monitor_sec"], f"State undetermined for {s.dur:.0f}s since {ms(s.start)}")
    run_level = {}
    for r0, r1 in out_runs(segs):
        d = r1 - r0
        if d >= a["out_alert_sec"]:
            hit("ALERT", "out_of_bed", r0, a["out_alert_sec"], f"Out of bed {d:.0f}s (limit {a['out_alert_sec']}s) since {ms(r0)}")
            run_level[(r0, r1)] = "ALERT"
        elif d >= a["out_monitor_sec"]:
            hit("MONITOR", "out_of_bed", r0, a["out_monitor_sec"], f"Out of bed {d:.0f}s (monitor limit {a['out_monitor_sec']}s) since {ms(r0)}")
            run_level[(r0, r1)] = "MONITOR"
    exits = sorted(e.start_time for e in events if e.event == "bed_exit")
    for i in range(len(exits)):
        w = [x for x in exits[i:] if x - exits[i] <= a["exits_window_sec"]]
        if len(w) >= a["exits_monitor_count"]:
            hit("MONITOR", "repeated_exits", exits[i], w[a["exits_monitor_count"] - 1] - exits[i],
                f"{len(w)} bed exits within {a['exits_window_sec']}s starting {ms(exits[i])}")
            break
    for e in events:  # per-event decision
        lvl = "NORMAL"
        for (r0, r1), l in run_level.items():
            if r0 - 1 <= e.start_time <= r1 and e.event == "bed_exit" and LEVELS[l] > LEVELS[lvl]:
                lvl = l
        if e.event == "bed_exit" and e.confidence < a["event_conf_monitor"] and LEVELS[lvl] < 1:
            lvl = "MONITOR"
        e.decision = lvl
    dec.sort(key=lambda d: d.t_trigger)
    overall = max((d.level for d in dec), key=lambda l: LEVELS[l], default="NORMAL")
    return dec, overall
