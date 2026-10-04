"""Core data types shared by every stage."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    LYING_IN_BED = "LYING_IN_BED"
    SITTING_ON_BED = "SITTING_ON_BED"
    SITTING_OUTSIDE_BED = "SITTING_OUTSIDE_BED"
    STANDING = "STANDING"
    WALKING = "WALKING"
    OUT_OF_BED = "OUT_OF_BED"          # left the bed AND left camera view
    LYING_ON_FLOOR = "LYING_ON_FLOOR"  # added state: safety-critical (possible fall)
    UNKNOWN = "UNKNOWN"


STATES = list(State)
IDX = {s: i for i, s in enumerate(STATES)}
IN_BED = {State.LYING_IN_BED, State.SITTING_ON_BED}
OUT_STATES = {State.SITTING_OUTSIDE_BED, State.STANDING, State.WALKING,
              State.OUT_OF_BED, State.LYING_ON_FLOOR}
CONFIRMING_OUT = {State.WALKING, State.SITTING_OUTSIDE_BED, State.OUT_OF_BED, State.LYING_ON_FLOOR}
LEVELS = {"NORMAL": 0, "MONITOR": 1, "ALERT": 2}


@dataclass
class Segment:
    start: float
    end: float
    state: State
    conf: float = 1.0
    source: str = "rules"   # rules | agent

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class Event:
    event: str              # bed_exit | bed_return | stand_attempt
    start_time: float
    confirmed_time: float
    previous_state: State
    current_state: State
    confidence: float
    decision: str = "NORMAL"
    source: str = "rules"


@dataclass
class Decision:
    level: str              # NORMAL | MONITOR | ALERT
    rule: str
    reason: str
    t_trigger: float
    t_start: float = 0.0


@dataclass
class AmbiguityItem:
    id: int
    t0: float
    t1: float
    reason: str
    seg_index: int
