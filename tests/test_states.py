import numpy as np
from monitor import states
from monitor.features import posture_probs
from monitor.schemas import STATES, IDX, State


def _E(T):
    E = np.full((T, len(STATES)), 0.1)
    return E


def test_viterbi_ignores_one_second_blip(cfg):
    E = _E(30)
    E[:, IDX[State.LYING_IN_BED]] = 0.9
    E[10, IDX[State.WALKING]], E[10, IDX[State.LYING_IN_BED]] = 0.95, 0.2   # impossible jump lying -> walking
    path = states.decode(E, cfg, "viterbi")
    assert set(path) == {IDX[State.LYING_IN_BED]}


def test_viterbi_follows_sustained_change(cfg):
    E = _E(60)
    E[:30, IDX[State.LYING_IN_BED]] = 0.9
    E[30:40, IDX[State.SITTING_ON_BED]] = 0.9
    E[40:, IDX[State.STANDING]] = 0.9
    p = states.decode(E, cfg, "viterbi")
    assert STATES[p[5]] == State.LYING_IN_BED and STATES[p[35]] == State.SITTING_ON_BED and STATES[p[55]] == State.STANDING


def test_merge_short_and_coverage(cfg):
    lab = np.array([0] * 20 + [3] + [0] * 20)
    segs = states.to_segments(lab, np.ones(41), np.zeros(41, int), 1, 41)
    merged = states.merge_short(segs, 2)
    assert len(merged) == 1 and merged[0].start == 0 and merged[0].end == 41


def test_posture_probs(cfg):
    f = cfg["features"]
    ly, si, st = posture_probs(np.array([85.0]), np.array([2.0]), np.array([np.nan]), f)
    assert ly[0] > 0.9
    ly, si, st = posture_probs(np.array([5.0]), np.array([0.4]), np.array([1.1]), f)
    assert st[0] > 0.7 and abs(ly[0] + si[0] + st[0] - 1) < 1e-6
