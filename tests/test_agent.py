"""Automated test suite for Stage 9 (Agent & VLM reasoning loop) with deterministic mock client.

Verifies:
- keep decision
- relabel decision
- budget exhaustion (max_tool_calls safeguard)
- invalid resolution schema / retry handling
- out-of-enum state protection
- VLM disk caching (second call makes zero client requests)
- bounded triage items (<= 12)
- bed exit confirmation via temporal context inspection
"""
import json
import types
import numpy as np
import pandas as pd
import pytest
from monitor.agent import Agent, Resolution, TOOLS, SYSTEM
from monitor.schemas import AmbiguityItem, Segment, State, STATES, IDX
from monitor.pipeline import apply_resolutions
from monitor.vlm import VLM
from monitor import triage


class MockContentBlock:
    def __init__(self, type_="tool_use", id_="call_1", name="submit_resolution", input_=None, text=""):
        self.type = type_
        self.id = id_
        self.name = name
        self.input = input_ or {}
        self.text = text


class MockMessageResponse:
    def __init__(self, content):
        self.content = content


class MockClient:
    def __init__(self, scripted_responses):
        self.scripted_responses = list(scripted_responses)
        self.call_history = []
        self.messages = types.SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.call_history.append(kwargs)
        if not self.scripted_responses:
            raise RuntimeError("No more scripted responses in MockClient")
        return self.scripted_responses.pop(0)


def create_test_context(n_sec=30):
    # Create simple synthetic bins and segments
    dates = pd.date_range("2026-01-01", periods=n_sec, freq="1s")
    df = pd.DataFrame({
        "t": np.arange(n_sec, dtype=float),
        "cx": np.full(n_sec, 300.0),
        "cy": np.full(n_sec, 400.0),
        "body_len": np.full(n_sec, 150.0),
        "present": np.full(n_sec, 1.0),
        "p_bed": np.array([1.0]*10 + [0.8]*5 + [0.0]*15),
        "p_near": np.array([1.0]*15 + [0.2]*15),
        "bed_dist_bl": np.array([0.0]*10 + [0.1]*5 + [2.0]*15),
        "lying": np.array([0.9]*10 + [0.1]*20),
        "sitting": np.array([0.1]*10 + [0.8]*5 + [0.0]*15),
        "standing": np.array([0.0]*15 + [0.9]*15),
        "moving": np.array([0.0]*15 + [0.8]*15),
        "core_vis": np.full(n_sec, 0.95),
        "n_persons": np.full(n_sec, 1)
    })
    segs = [
        Segment(0.0, 10.0, State.LYING_IN_BED, 0.95, "viterbi"),
        Segment(10.0, 15.0, State.SITTING_ON_BED, 0.50, "viterbi"),  # ambiguous
        Segment(15.0, 30.0, State.WALKING, 0.90, "viterbi")
    ]
    labels = np.array([IDX[State.LYING_IN_BED]]*10 + [IDX[State.SITTING_ON_BED]]*5 + [IDX[State.WALKING]]*15)
    return types.SimpleNamespace(bins=df, segs=segs, labels=labels)


def test_agent_keep_decision(cfg, tmp_path):
    ctx = create_test_context()
    trace_path = tmp_path / "traces.jsonl"
    item = AmbiguityItem(id=0, t0=10.0, t1=15.0, reason="low_confidence", seg_index=1)

    # Scripted: agent calls get_previous_context, then decides to keep
    script = [
        MockMessageResponse([MockContentBlock(
            type_="tool_use", id_="call_1", name="get_previous_context",
            input_={"t": 10.0, "sec": 10.0}
        )]),
        MockMessageResponse([MockContentBlock(
            type_="tool_use", id_="call_2", name="submit_resolution",
            input_={"decision": "keep", "t0": 10.0, "t1": 15.0, "reason": "Consistent sitting transition", "confidence": 0.85}
        )])
    ]
    client = MockClient(script)
    agent = Agent(client, cfg, ctx, vlm=None, trace_path=trace_path)

    resolutions = agent.resolve([item])
    assert len(resolutions) == 1
    assert resolutions[0].decision == "keep"
    assert resolutions[0].confidence == 0.85
    assert len(client.call_history) == 2
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text().strip())
    assert trace["resolution"]["decision"] == "keep"


def test_agent_relabel_and_apply(cfg, tmp_path):
    ctx = create_test_context()
    trace_path = tmp_path / "traces.jsonl"
    item = AmbiguityItem(id=0, t0=10.0, t1=15.0, reason="bed_floor_boundary", seg_index=1)

    # Scripted: agent checks bed overlap, relabels to SITTING_ON_BED with high confidence
    script = [
        MockMessageResponse([MockContentBlock(
            type_="tool_use", id_="call_1", name="check_bed_overlap",
            input_={"t": 12.0}
        )]),
        MockMessageResponse([MockContentBlock(
            type_="tool_use", id_="call_2", name="submit_resolution",
            input_={"decision": "relabel", "new_state": "SITTING_ON_BED", "t0": 10.0, "t1": 15.0,
                    "reason": "Clear bed overlap and sitting geometry", "confidence": 0.92}
        )])
    ]
    client = MockClient(script)
    agent = Agent(client, cfg, ctx, vlm=None, trace_path=trace_path)

    resolutions = agent.resolve([item])
    assert len(resolutions) == 1
    assert resolutions[0].decision == "relabel"
    assert resolutions[0].new_state == State.SITTING_ON_BED

    # Apply resolution and check label modification
    labels = np.zeros(30, dtype=int)
    conf = np.zeros(30, dtype=float)
    src = np.zeros(30, dtype=int)
    n_changed = apply_resolutions(resolutions, [item], labels, conf, src, cfg)
    assert n_changed == 1
    assert (labels[10:15] == IDX[State.SITTING_ON_BED]).all()
    assert (src[10:15] == 1).all()
    assert (conf[10:15] == 0.92).all()


def test_agent_budget_exhausted(cfg, tmp_path):
    ctx = create_test_context()
    trace_path = tmp_path / "traces.jsonl"
    item = AmbiguityItem(id=0, t0=10.0, t1=15.0, reason="low_confidence", seg_index=1)

    # Max tool calls is 4; script returns 5 consecutive tool calls without submitting
    script = [
        MockMessageResponse([MockContentBlock(type_="tool_use", id_=f"call_{i}", name="check_bed_overlap", input_={"t": 10.0})])
        for i in range(10)
    ]
    client = MockClient(script)
    agent = Agent(client, cfg, ctx, vlm=None, trace_path=trace_path)

    resolutions = agent.resolve([item])
    assert len(resolutions) == 1
    res = resolutions[0]
    assert res.decision == "keep"
    assert "budget exhausted" in res.reason


def test_agent_invalid_resolution_retry(cfg, tmp_path):
    ctx = create_test_context()
    trace_path = tmp_path / "traces.jsonl"
    item = AmbiguityItem(id=0, t0=10.0, t1=15.0, reason="low_confidence", seg_index=1)

    # First submit call is invalid (missing 'reason'), agent receives error tool_result and corrects
    script = [
        MockMessageResponse([MockContentBlock(
            type_="tool_use", id_="call_invalid", name="submit_resolution",
            input_={"decision": "keep", "t0": 10.0, "t1": 15.0, "confidence": 0.8}  # missing 'reason'
        )]),
        MockMessageResponse([MockContentBlock(
            type_="tool_use", id_="call_valid", name="submit_resolution",
            input_={"decision": "keep", "t0": 10.0, "t1": 15.0, "reason": "Corrected resolution", "confidence": 0.8}
        )])
    ]
    client = MockClient(script)
    agent = Agent(client, cfg, ctx, vlm=None, trace_path=trace_path)

    resolutions = agent.resolve([item])
    assert len(resolutions) == 1
    assert resolutions[0].decision == "keep"
    assert resolutions[0].reason == "Corrected resolution"


def test_out_of_enum_state_protection():
    # Resolution model must strictly enforce State enum values
    with pytest.raises(Exception):
        Resolution(item_id=0, decision="relabel", new_state="FLYING_IN_AIR", t0=0.0, t1=5.0, reason="fake", confidence=0.9)


def test_vlm_disk_caching(cfg, tmp_path):
    cache_path = tmp_path / "vlm_cache.json"

    # Mock VLM client
    script = [
        MockMessageResponse([MockContentBlock(
            type_="text", text='{"choice": "sitting_on_bed", "reason": "Person is clearly sitting on edge"}'
        )])
    ]
    client = MockClient(script)

    class DummyVLM(VLM):
        def frames(self, t0, t1):
            return [b"fake_jpg_1", b"fake_jpg_2"]

    vlm = DummyVLM(client, cfg, video="dummy.mp4", bins=pd.DataFrame(), bed_px=np.array([[0,0]]), cache_path=cache_path)

    # First call: makes API call and caches
    ans1 = vlm.ask(10.0, 15.0, "What is the posture?", ["sitting_on_bed", "standing"])
    assert ans1["choice"] == "sitting_on_bed"
    assert len(client.call_history) == 1
    assert cache_path.exists()

    # Second call with identical arguments: MUST hit cache and make NO client call
    ans2 = vlm.ask(10.0, 15.0, "What is the posture?", ["sitting_on_bed", "standing"])
    assert ans2["choice"] == "sitting_on_bed"
    assert len(client.call_history) == 1, "Second VLM call must use cache without API call"


def test_triage_item_bound(cfg):
    # Verify that max_items is bounded <= 12
    n_bins = 200
    df = pd.DataFrame({
        "t": np.arange(n_bins, dtype=float),
        "present": np.ones(n_bins),
        "core_vis": np.ones(n_bins),
        "p_bed": np.zeros(n_bins),
        "p_near": np.zeros(n_bins),
        "bed_dist_bl": np.ones(n_bins),
        "lying": np.zeros(n_bins),
        "sitting": np.zeros(n_bins),
        "standing": np.zeros(n_bins),
        "moving": np.zeros(n_bins),
        "n_persons": np.full(n_bins, 2),  # trigger caregiver for every segment
    })
    segs = [Segment(float(i*5), float((i+1)*5), State.WALKING, 0.3, "viterbi") for i in range(30)]
    items = triage.find_items(segs, df, cfg)
    assert len(items) <= cfg["triage"]["max_items"]
    assert len(items) <= 12


def test_brief_confirm_exit_scenario(cfg, tmp_path):
    """Traced case showing the brief's requirement: check previous/following context -> confirm exit."""
    ctx = create_test_context()
    trace_path = tmp_path / "traces.jsonl"
    item = AmbiguityItem(id=0, t0=10.0, t1=15.0, reason="bed_floor_boundary", seg_index=1)

    # Scripted scenario:
    # 1. Agent asks for previous context: receives [0-10 LYING_IN_BED]
    # 2. Agent asks for following context: receives [15-30 WALKING]
    # 3. Agent confirms transition is SITTING_ON_BED -> confirms exit
    script = [
        MockMessageResponse([MockContentBlock(
            type_="tool_use", id_="call_prev", name="get_previous_context",
            input_={"t": 10.0, "sec": 10.0}
        )]),
        MockMessageResponse([MockContentBlock(
            type_="tool_use", id_="call_next", name="get_following_context",
            input_={"t": 15.0, "sec": 10.0}
        )]),
        MockMessageResponse([MockContentBlock(
            type_="tool_use", id_="call_resolve", name="submit_resolution",
            input_={"decision": "keep", "t0": 10.0, "t1": 15.0,
                    "reason": "Previous is LYING_IN_BED, following is WALKING away; segment 10-15s is transitional SITTING_ON_BED confirming bed exit",
                    "confidence": 0.95}
        )])
    ]
    client = MockClient(script)
    agent = Agent(client, cfg, ctx, vlm=None, trace_path=trace_path)

    resolutions = agent.resolve([item])
    assert len(resolutions) == 1
    assert resolutions[0].decision == "keep"
    assert "confirming bed exit" in resolutions[0].reason

    # Verify tool sequence in trace
    trace = json.loads(trace_path.read_text().strip())
    tool_names = [s.get("tool") for s in trace["steps"] if "tool" in s]
    assert tool_names == ["get_previous_context", "get_following_context"]

