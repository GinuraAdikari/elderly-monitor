import pandas as pd
from monitor import events, alerts, report
from monitor.schemas import Segment, State

S = State
EMPTY = pd.DataFrame({"t": [], "bed_dist_bl": []})


def seg(a, b, s, c=0.9):
    return Segment(a, b, s, c)


def test_exit_and_return(cfg):
    segs = [seg(0, 60, S.LYING_IN_BED), seg(60, 70, S.SITTING_ON_BED), seg(70, 75, S.STANDING), seg(75, 140, S.WALKING),
            seg(140, 200, S.SITTING_OUTSIDE_BED), seg(200, 210, S.WALKING), seg(210, 220, S.SITTING_ON_BED), seg(220, 300, S.LYING_IN_BED)]
    ev = events.detect_events(segs, EMPTY, cfg)
    kinds = [e.event for e in ev]
    assert kinds == ["bed_exit", "bed_return"]
    assert ev[0].start_time == 70 and ev[0].confirmed_time == 75 and ev[0].previous_state == S.SITTING_ON_BED
    assert ev[1].start_time == 200 and ev[1].confirmed_time == 215


def test_stand_and_sit_back_is_not_exit(cfg):
    segs = [seg(0, 30, S.LYING_IN_BED), seg(30, 40, S.SITTING_ON_BED), seg(40, 45, S.STANDING), seg(45, 60, S.SITTING_ON_BED)]
    kinds = [e.event for e in events.detect_events(segs, EMPTY, cfg)]
    assert "bed_exit" not in kinds and kinds == ["stand_attempt"]


def test_turning_in_bed_no_events(cfg):
    assert events.detect_events([seg(0, 100, S.LYING_IN_BED)], EMPTY, cfg) == []


def test_floor_lying_alerts(cfg):
    segs = [seg(0, 50, S.LYING_IN_BED), seg(50, 60, S.STANDING), seg(60, 100, S.LYING_ON_FLOOR)]
    dec, overall = alerts.evaluate(segs, [], cfg)
    assert overall == "ALERT" and any(d.rule == "floor" for d in dec)


def test_long_edge_sitting_is_monitor(cfg):
    segs = [seg(0, 20, S.LYING_IN_BED), seg(20, 220, S.SITTING_ON_BED), seg(220, 300, S.LYING_IN_BED)]
    assert alerts.evaluate(segs, [], cfg)[1] == "MONITOR"


def test_normal_day(cfg):
    segs = [seg(0, 100, S.LYING_IN_BED)]
    assert alerts.evaluate(segs, [], cfg)[1] == "NORMAL"


def test_durations_sum_to_observation(cfg):
    segs = [seg(0, 70, S.LYING_IN_BED), seg(70, 100, S.WALKING), seg(100, 120, S.UNKNOWN)]
    s = report.build_summary(segs, [], 120)
    total = sum(s["activity_duration_sec"].values())
    assert abs(total - 120) < 1
    assert abs(s["total_in_bed_sec"] + s["total_out_of_bed_sec"] + s["activity_duration_sec"]["unknown"] - 120) < 1
