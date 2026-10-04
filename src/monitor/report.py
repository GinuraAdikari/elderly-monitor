"""Stage 8: write timeline / events / summary / decisions."""
from __future__ import annotations
import json
import pathlib
from .schemas import State, IN_BED, OUT_STATES
from .util import hms, ms, dur
from .alerts import out_runs


def activity_seconds(segs):
    d = {s: 0.0 for s in State}
    for s in segs:
        d[s.state] += s.dur
    return d


def build_summary(segs, events, duration):
    a = activity_seconds(segs)
    in_bed = sum(a[s] for s in IN_BED)
    out = sum(a[s] for s in OUT_STATES)
    runs = out_runs(segs)
    longest = max((r1 - r0 for r0, r1 in runs), default=0.0)
    names = {s: s.value.lower() for s in State}
    return {
        "observation_duration_sec": round(duration),
        "activity_duration_sec": {names[s]: round(v, 1) for s, v in a.items()},
        "bed_exit_count": sum(e.event == "bed_exit" for e in events),
        "bed_return_count": sum(e.event == "bed_return" for e in events),
        "stand_attempt_count": sum(e.event == "stand_attempt" for e in events),
        "total_in_bed_sec": round(in_bed, 1),
        "total_out_of_bed_sec": round(out, 1),                       # excludes UNKNOWN (reported separately)
        "total_out_of_bed_incl_unknown_sec": round(out + a[State.UNKNOWN], 1),
        "longest_out_of_bed_period_sec": round(longest, 1),
        "final_state": segs[-1].state.value.lower() if segs else "unknown",
        "human_readable": {
            "total_observation_time": dur(duration),
            "activity_summary": {names[s]: dur(v) for s, v in a.items() if s != State.LYING_ON_FLOOR or v > 0},
            "bed_summary": {"time_in_bed": dur(in_bed), "time_out_of_bed": dur(out),
                            "bed_exit_count": sum(e.event == "bed_exit" for e in events)},
        },
    }


def write_all(out_dir, segs, events, decisions, overall, duration, extra=None):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "timeline.txt").write_text("\n".join(f"{ms(s.start)} – {ms(s.end)}    {s.state.value}" for s in segs) + "\n")
    json.dump([{"start": round(s.start, 2), "end": round(s.end, 2), "state": s.state.value,
                "confidence": round(s.conf, 3), "source": s.source} for s in segs],
              open(out / "timeline.json", "w"), indent=2)
    json.dump([{"event": e.event, "start_time": hms(e.start_time), "confirmed_time": hms(e.confirmed_time),
                "start_sec": round(e.start_time, 1), "confirmed_sec": round(e.confirmed_time, 1),
                "previous_state": e.previous_state.value.lower(), "current_state": e.current_state.value.lower(),
                "confidence": e.confidence, "decision": e.decision, "source": e.source} for e in events],
              open(out / "events.json", "w"), indent=2)
    json.dump({"overall": overall, "decisions": [{"level": d.level, "rule": d.rule, "reason": d.reason,
                                                 "t_trigger": hms(d.t_trigger)} for d in decisions]},
              open(out / "decisions.json", "w"), indent=2)
    summary = build_summary(segs, events, duration)
    summary["overall_decision"] = overall
    if extra:
        summary.update(extra)
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    return summary
