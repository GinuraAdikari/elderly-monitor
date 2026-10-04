"""Orchestrates all stages. Perception is cached once per video; each mode writes to its own folder."""
from __future__ import annotations
import pathlib
import types
import numpy as np
from . import ingest, perception, patient as patient_mod, features, states, events as events_mod, triage, alerts, report
from .agent import Agent, NullAgent, LocalAgent
from .geometry import load_polygon, to_px
from .schemas import IDX, State, STATES

MODES = ("rules", "viterbi", "agent")


def apply_resolutions(resolutions, items, labels, conf, src, cfg):
    a, hz = cfg["agent"], cfg["video"]["state_hz"]
    n_changed = 0
    for r in resolutions:
        if r.decision != "relabel" or r.new_state is None:
            continue
        it = items[r.item_id]
        t0, t1 = max(r.t0, it.t0 - a["context_pad_sec"]), min(r.t1, it.t1 + a["context_pad_sec"])
        if t1 - t0 <= 0 or t1 - t0 > a["max_relabel_sec"]:
            continue
        i0, i1 = int(round(t0 * hz)), min(int(round(t1 * hz)), len(labels))
        labels[i0:i1], src[i0:i1], conf[i0:i1] = IDX[r.new_state], 1, r.confidence
        n_changed += 1
    return n_changed


def _make_agent(cfg, ctx, video, bins, bed_px, out_dir):
    provider = cfg["agent"].get("provider", "anthropic")
    if provider == "none":
        return NullAgent()
    if provider == "local":
        return LocalAgent(cfg, ctx, out_dir / "agent_traces.jsonl")
    try:
        import anthropic
        from .vlm import VLM
        client = anthropic.Anthropic()
        vlm = VLM(client, cfg, video, bins, bed_px, out_dir.parent / ".vlm_cache.json")
        return Agent(client, cfg, ctx, vlm, out_dir / "agent_traces.jsonl")
    except Exception as e:
        return LocalAgent(cfg, ctx, out_dir / "agent_traces.jsonl")


def run(video, polygon_path, out_root, cfg, mode="agent", force_perception=False):
    assert mode in MODES
    video = pathlib.Path(video)
    base = pathlib.Path(out_root) / video.stem
    out_dir = base / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, meta = perception.run(video, base, cfg, force=force_perception)
    bed_px = to_px(load_polygon(polygon_path), meta["width"], meta["height"])
    pat, info = patient_mod.select_patient(raw, bed_px, cfg)
    bins = features.build_bins(pat, raw, meta, bed_px, cfg)
    bins.to_parquet(base / "bins.parquet")

    hz, dur = cfg["video"]["state_hz"], meta["duration"]
    E, occ = states.emission_scores(bins, cfg)
    labels = states.decode(E, cfg, mode)
    conf = E[np.arange(len(labels)), labels]
    src = np.zeros(len(labels), int)
    segs = states.merge_short(states.to_segments(labels, conf, src, hz, dur), cfg["states"]["min_segment_sec"])
    labels = states.segments_to_labels(segs, len(labels), hz)
    evs = events_mod.detect_events(segs, bins, cfg)

    extra = {"patient": info, "mode": mode, "agent_items": 0, "agent_relabels": 0}
    if mode == "agent":
        items = triage.find_items(segs, bins, cfg)
        ctx = types.SimpleNamespace(segs=segs, bins=bins, labels=labels)
        agent = _make_agent(cfg, ctx, video, bins, bed_px, out_dir)
        resolutions = agent.resolve(items)
        changed = apply_resolutions(resolutions, items, labels, conf, src, cfg)
        if changed:
            segs = states.merge_short(states.to_segments(labels, conf, src, hz, dur), cfg["states"]["min_segment_sec"])
            evs = events_mod.detect_events(segs, bins, cfg)
        extra.update(agent_items=len(items), agent_relabels=changed)

    decisions, overall = alerts.evaluate(segs, evs, cfg)
    summary = report.write_all(out_dir, segs, evs, decisions, overall, dur, extra)
    return summary
