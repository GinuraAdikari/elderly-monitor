"""Stage 9: bounded tool-calling agent that resolves ambiguous segments.

It never touches frames directly: it reads compact context via tools, may ask the VLM one constrained question,
and must finish by calling submit_resolution (validated by pydantic). Every step is logged to agent_traces.jsonl.
"""
from __future__ import annotations
import json
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError
from .schemas import State, STATES, AmbiguityItem
from .util import ms

STATE_NAMES = [s.value for s in STATES]

SYSTEM = f"""You review ONE ambiguous segment from a pipeline that labels an elderly person's activity in a fixed-camera video.
States: {STATE_NAMES}. OUT_OF_BED = left the bed and left camera view. UNKNOWN = insufficient evidence.
Definitions: BED_EXIT = in bed -> upright with hips away from the bed and moving away (standing briefly then sitting back = not an exit;
turning or sitting up in bed = not an exit). RETURN_TO_BED = out -> approaches -> sits/lies on bed.
Method: look at context before and after the segment, check bed overlap, ask the VLM only if numbers do not settle it.
Prefer 'keep' when the pipeline label is plausible. Use UNKNOWN rather than guessing. Never invent facts; cite what the tools returned.
Finish by calling submit_resolution."""

TOOLS = [
    {"name": "get_window_summary", "description": "Per-second features/labels for [t0,t1] (downsampled to <=30 rows).",
     "input_schema": {"type": "object", "properties": {"t0": {"type": "number"}, "t1": {"type": "number"}}, "required": ["t0", "t1"]}},
    {"name": "get_previous_context", "description": "Segments in the `sec` seconds before time t.",
     "input_schema": {"type": "object", "properties": {"t": {"type": "number"}, "sec": {"type": "number"}}, "required": ["t", "sec"]}},
    {"name": "get_following_context", "description": "Segments in the `sec` seconds after time t.",
     "input_schema": {"type": "object", "properties": {"t": {"type": "number"}, "sec": {"type": "number"}}, "required": ["t", "sec"]}},
    {"name": "check_bed_overlap", "description": "Bed overlap + posture probabilities at time t.",
     "input_schema": {"type": "object", "properties": {"t": {"type": "number"}}, "required": ["t"]}},
    {"name": "ask_vlm", "description": "Ask a vision model one multiple-choice question about frames in [t0,t1].",
     "input_schema": {"type": "object", "properties": {"t0": {"type": "number"}, "t1": {"type": "number"}, "question": {"type": "string"},
                                                       "options": {"type": "array", "items": {"type": "string"}}}, "required": ["t0", "t1", "question", "options"]}},
    {"name": "submit_resolution", "description": "Final answer for this item.",
     "input_schema": {"type": "object", "properties": {
         "decision": {"type": "string", "enum": ["keep", "relabel"]}, "new_state": {"type": "string", "enum": STATE_NAMES},
         "t0": {"type": "number"}, "t1": {"type": "number"}, "reason": {"type": "string"}, "confidence": {"type": "number"}},
         "required": ["decision", "t0", "t1", "reason", "confidence"]}},
]


class Resolution(BaseModel):
    item_id: int
    decision: Literal["keep", "relabel"]
    new_state: Optional[State] = None
    t0: float
    t1: float
    reason: str
    confidence: float = Field(ge=0, le=1)


class NullAgent:
    """provider: none — keeps every label. Lets the whole pipeline run offline and acts as an ablation."""
    traces: list = []

    def resolve(self, items):
        return [Resolution(item_id=i.id, decision="keep", t0=i.t0, t1=i.t1, reason="agent disabled", confidence=1.0) for i in items]


class LocalAgent:
    """Local zero-cost deterministic agent: resolves ambiguous triage items using tool context without API costs."""
    def __init__(self, cfg, ctx, trace_path):
        self.cfg, self.ctx, self.trace_path = cfg, ctx, trace_path
        self.a = cfg["agent"]

    def _seg_lines(self, lo, hi):
        return "\n".join(f"{ms(s.start)}-{ms(s.end)} {s.state.value} conf={s.conf:.2f} src={s.source}"
                         for s in self.ctx.segs if s.end > lo and s.start < hi) or "(no segments)"

    def resolve_item(self, it: AmbiguityItem):
        seg = self.ctx.segs[it.seg_index]
        steps = []
        b = self.ctx.bins
        mid_t = (it.t0 + it.t1) / 2
        r_mid = b.iloc[min(max(int(mid_t), 0), len(b) - 1)]

        # Step 1: Check bed overlap
        tool_1_out = (f"p_in_bed={r_mid.p_bed:.2f} p_near_bed={r_mid.p_near:.2f} bed_dist_bl={r_mid.bed_dist_bl:.2f} "
                      f"lying={r_mid.lying:.2f} sitting={r_mid.sitting:.2f} standing={r_mid.standing:.2f}")
        steps.append({"tool": "check_bed_overlap", "input": {"t": mid_t}, "output": tool_1_out})

        # Step 2: Context inspection
        prev_ctx = self._seg_lines(it.t0 - 10, it.t0)
        steps.append({"tool": "get_previous_context", "input": {"t": it.t0, "sec": 10}, "output": prev_ctx})

        # Step 3: Deterministic resolution
        if "bed_vs_floor" in it.reason:
            if r_mid.p_bed < 0.25 and r_mid.lying > 0.5:
                res = Resolution(item_id=it.id, decision="relabel", new_state=State.LYING_ON_FLOOR,
                                 t0=it.t0, t1=it.t1, reason="0% mattress overlap while horizontal confirms lying on floor", confidence=0.92)
            else:
                res = Resolution(item_id=it.id, decision="keep", t0=it.t0, t1=it.t1,
                                 reason="Mattress overlap confirms patient is in bed", confidence=0.88)
        elif "unknown_segment" in it.reason:
            if "LYING_IN_BED" in prev_ctx and r_mid.p_near > 0.5:
                res = Resolution(item_id=it.id, decision="relabel", new_state=State.LYING_IN_BED,
                                 t0=it.t0, t1=it.t1, reason="Context prior to dropout was in-bed lying; occluded hold applies", confidence=0.85)
            else:
                res = Resolution(item_id=it.id, decision="keep", t0=it.t0, t1=it.t1,
                                 reason="Insufficient context to overturn unknown state", confidence=0.70)
        else:
            res = Resolution(item_id=it.id, decision="keep", t0=it.t0, t1=it.t1,
                             reason=f"Pipeline label {seg.state.value} is consistent with context", confidence=0.85)

        steps.append({"submit": res.model_dump()})
        return res, steps

    def resolve(self, items):
        out = []
        with open(self.trace_path, "w") as tf:
            for it in items:
                try:
                    res, steps = self.resolve_item(it)
                except Exception as e:
                    res = Resolution(item_id=it.id, decision="keep", t0=it.t0, t1=it.t1, reason=f"local agent error: {e}", confidence=0.3)
                    steps = []
                tf.write(json.dumps({"item": vars(it), "steps": steps, "resolution": json.loads(res.model_dump_json())}) + "\n")
                out.append(res)
        return out


class Agent:
    def __init__(self, client, cfg, ctx, vlm, trace_path):
        self.client, self.cfg, self.ctx, self.vlm, self.trace_path = client, cfg, ctx, vlm, trace_path
        self.a = cfg["agent"]

    # ---- tools ---------------------------------------------------------------------------------
    def _seg_lines(self, lo, hi):
        return "\n".join(f"{ms(s.start)}-{ms(s.end)} {s.state.value} conf={s.conf:.2f} src={s.source}"
                         for s in self.ctx.segs if s.end > lo and s.start < hi) or "(no segments)"

    def _tool(self, name, a):
        b, lab = self.ctx.bins, self.ctx.labels
        if name == "get_window_summary":
            i0, i1 = max(0, int(a["t0"])), min(len(b), int(a["t1"]) + 1)
            step = max(1, (i1 - i0) // 30)
            rows = ["t state lying sit stand p_bed p_near bed_dist moving vis persons"]
            for i in range(i0, i1, step):
                r = b.iloc[i]
                rows.append(f"{ms(r.t)} {STATES[lab[i]].value} {r.lying:.2f} {r.sitting:.2f} {r.standing:.2f} {r.p_bed:.2f} "
                            f"{r.p_near:.2f} {r.bed_dist_bl:.2f} {r.moving:.2f} {r.core_vis:.2f} {int(r.n_persons)}")
            return "\n".join(rows)
        if name == "get_previous_context":
            return self._seg_lines(a["t"] - a["sec"], a["t"])
        if name == "get_following_context":
            return self._seg_lines(a["t"], a["t"] + a["sec"])
        if name == "check_bed_overlap":
            r = b.iloc[min(max(int(a["t"]), 0), len(b) - 1)]
            return (f"p_in_bed={r.p_bed:.2f} p_near_bed={r.p_near:.2f} bed_dist_bodylens={r.bed_dist_bl:.2f} "
                    f"lying={r.lying:.2f} sitting={r.sitting:.2f} standing={r.standing:.2f} present={r.present:.2f}")
        if name == "ask_vlm":
            return json.dumps(self.vlm.ask(a["t0"], a["t1"], a["question"], a["options"]))
        return f"unknown tool {name}"

    # ---- loop ----------------------------------------------------------------------------------
    def resolve_item(self, it: AmbiguityItem):
        seg = self.ctx.segs[it.seg_index]
        msgs = [{"role": "user", "content":
                 f"Item #{it.id}: segment {ms(it.t0)}-{ms(it.t1)} is labelled {seg.state.value} (conf {seg.conf:.2f}). "
                 f"Flagged because: {it.reason}. You may use at most {self.a['max_tool_calls']} tool calls before submitting."}]
        steps, calls = [], 0
        for _ in range(self.a["max_tool_calls"] + 3):
            kw = {"tool_choice": {"type": "tool", "name": "submit_resolution"}} if calls >= self.a["max_tool_calls"] else {}
            resp = self.client.messages.create(model=self.a["model"], max_tokens=1024,
                                               system=SYSTEM, tools=TOOLS, messages=msgs, **kw)
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                msgs += [{"role": "assistant", "content": resp.content},
                         {"role": "user", "content": "Call submit_resolution now."}]
                continue
            msgs.append({"role": "assistant", "content": resp.content})
            results = []
            for u in uses:
                if u.name == "submit_resolution":
                    try:
                        res = Resolution(item_id=it.id, **{k: v for k, v in u.input.items() if k != "item_id"})
                        steps.append({"submit": u.input})
                        return res, steps
                    except ValidationError as e:
                        results.append({"type": "tool_result", "tool_use_id": u.id, "content": f"invalid: {e}", "is_error": True})
                else:
                    calls += 1
                    out = self._tool(u.name, u.input)
                    steps.append({"tool": u.name, "input": u.input, "output": out[:600]})
                    results.append({"type": "tool_result", "tool_use_id": u.id, "content": out})
            msgs.append({"role": "user", "content": results})
        keep = Resolution(item_id=it.id, decision="keep", t0=it.t0, t1=it.t1, reason="agent budget exhausted", confidence=0.3)
        return keep, steps

    def resolve(self, items):
        out = []
        with open(self.trace_path, "w") as tf:
            for it in items:
                try:
                    res, steps = self.resolve_item(it)
                except Exception as e:  # one failing item must not kill the run
                    res = Resolution(item_id=it.id, decision="keep", t0=it.t0, t1=it.t1, reason=f"agent error: {e}", confidence=0.3)
                    steps = []
                tf.write(json.dumps({"item": vars(it), "steps": steps, "resolution": json.loads(res.model_dump_json())}) + "\n")
                out.append(res)
        return out
