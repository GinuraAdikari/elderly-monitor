"""Evaluation — a SEPARATE step; never called by the pipeline. Compares prediction folders with ground truth CSVs.

GT segments csv : start,end,state            (times: seconds, mm:ss or hh:mm:ss)
GT events csv   : time,event                 (event = bed_exit | bed_return)   [optional, recommended]
If no events csv is given, GT events are derived from GT segments with the same event rules (less independent).
"""
from __future__ import annotations
import json
import pathlib
import numpy as np
import pandas as pd
from .schemas import State, STATES, IDX, Segment
from .events import detect_events
from .util import parse_time, ms


def load_gt(path):
    df = pd.read_csv(path)
    return [Segment(parse_time(r.start), parse_time(r.end), State(str(r.state).strip().upper())) for r in df.itertuples()]


def validate_gt(segs):
    issues = []
    for a, b in zip(segs, segs[1:]):
        if b.start < a.end - 0.5:
            issues.append(f"overlap {ms(a.end)} > {ms(b.start)}")
        if b.start > a.end + 0.5:
            issues.append(f"gap {ms(a.end)}-{ms(b.start)}")
    issues += [f"non-positive segment at {ms(s.start)}" for s in segs if s.end <= s.start]
    return issues


def load_pred(folder):
    f = pathlib.Path(folder)
    segs = [Segment(d["start"], d["end"], State(d["state"]), d["confidence"], d["source"]) for d in json.load(open(f / "timeline.json"))]
    ev = json.load(open(f / "events.json"))
    summ = json.load(open(f / "summary.json"))
    return segs, ev, summ


def to_bins(segs, hz, n):
    lab = np.full(n, -1)
    for s in segs:
        lab[int(round(s.start * hz)):min(int(round(s.end * hz)), n)] = IDX[s.state]
    return lab


def frame_metrics(gt, pr):
    m = gt >= 0
    gt, pr = gt[m], pr[m]
    names = [s.value for s in STATES]
    conf = pd.DataFrame(0, index=names, columns=names)
    for g, p in zip(gt, pr):
        conf.iloc[g, max(p, 0)] += 1
    per = {}
    for i, n in enumerate(names):
        tp, fp, fn = conf.iloc[i, i], conf.iloc[:, i].sum() - conf.iloc[i, i], conf.iloc[i, :].sum() - conf.iloc[i, i]
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        per[n] = {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "support": int(conf.iloc[i, :].sum())}
    present = [n for n in names if per[n]["support"] > 0]
    return {"accuracy": float((gt == pr).mean()), "macro_f1": float(np.mean([per[n]["f1"] for n in present])),
            "per_state": per, "confusion": conf}


def match(gt_t, pr_t, tol):
    used, tp = set(), 0
    for g in sorted(gt_t):
        best = None
        for j, p in enumerate(sorted(pr_t)):
            if j not in used and abs(p - g) <= tol and (best is None or abs(p - g) < abs(sorted(pr_t)[best] - g)):
                best = j
        if best is not None:
            used.add(best)
            tp += 1
    fp, fn = len(pr_t) - tp, len(gt_t) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def duration_errors(gt_segs, pr_segs):
    rows = []
    for s in STATES:
        g = sum(x.dur for x in gt_segs if x.state == s)
        p = sum(x.dur for x in pr_segs if x.state == s)
        if g > 0 or p > 0:
            rows.append({"state": s.value, "gt_sec": round(g), "pred_sec": round(p), "abs_err_sec": round(abs(g - p)),
                         "rel_err_pct": round(100 * abs(g - p) / max(g, 1), 1)})
    return pd.DataFrame(rows)


def failure_runs(gt, pr, hz, min_sec=8, top=5):
    runs, i, n = [], 0, len(gt)
    while i < n:
        if gt[i] >= 0 and gt[i] != pr[i]:
            j = i
            while j + 1 < n and gt[j + 1] >= 0 and gt[j + 1] != pr[j + 1] and gt[j + 1] == gt[i] and pr[j + 1] == pr[i]:
                j += 1
            if (j - i + 1) / hz >= min_sec:
                runs.append({"t0": i / hz, "t1": (j + 1) / hz, "gt": STATES[gt[i]].value, "pred": STATES[pr[i]].value})
            i = j + 1
        else:
            i += 1
    return sorted(runs, key=lambda r: r["t0"] - r["t1"])[:top]


def md(df):
    df = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n"


def evaluate_many(gt_csv, events_csv, pred_dirs, tol, out_dir, video, cfg):
    hz = cfg["video"]["state_hz"]
    gt = load_gt(gt_csv)
    for issue in validate_gt(gt):
        print("[gt warning]", issue)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dur = max(s.end for s in gt)
    n = int(np.ceil(dur * hz))
    if events_csv:
        ev = pd.read_csv(events_csv)
        gt_ev = {k: [parse_time(t) for t in ev[ev.event == k].time] for k in ("bed_exit", "bed_return")}
    else:
        d = detect_events(gt, None, cfg)
        gt_ev = {k: [e.start_time for e in d if e.event == k] for k in ("bed_exit", "bed_return")}
    lines, table = [f"# Evaluation vs {pathlib.Path(gt_csv).name}\n", f"Event tolerance ±{tol}s; GT bed exits={len(gt_ev['bed_exit'])}, returns={len(gt_ev['bed_return'])}\n"], []
    for pdir in pred_dirs:
        name = pathlib.Path(pdir).name
        segs, ev_pred, summ = load_pred(pdir)
        g, p = to_bins(gt, hz, n), to_bins(segs, hz, n)
        fm = frame_metrics(g, p)
        exits = match(gt_ev["bed_exit"], [e["start_sec"] for e in ev_pred if e["event"] == "bed_exit"], tol)
        rets = match(gt_ev["bed_return"], [e["start_sec"] for e in ev_pred if e["event"] == "bed_return"], tol)
        de = duration_errors(gt, segs)
        table.append({"mode": name, "accuracy": round(fm["accuracy"], 3), "macro_f1": round(fm["macro_f1"], 3),
                      "exit_P": round(exits["precision"], 2), "exit_R": round(exits["recall"], 2), "false_exits": exits["fp"],
                      "return_P": round(rets["precision"], 2), "return_R": round(rets["recall"], 2),
                      "mean_abs_dur_err_s": round(de.abs_err_sec.mean(), 1)})
        lines += [f"\n## {name}\n", f"accuracy {fm['accuracy']:.3f}, macro-F1 {fm['macro_f1']:.3f}\n",
                  "### Confusion (rows=GT, cols=pred)\n", md(fm["confusion"].loc[(fm["confusion"].sum(1) > 0)]),
                  "### Bed events\n", f"exit: {exits}\n\nreturn: {rets}\n", "### Duration error\n", md(de)]
        json.dump({"frame": {k: v for k, v in fm.items() if k != "confusion"}, "exit": exits, "return": rets,
                   "durations": de.to_dict("records")}, open(out / f"metrics_{name}.json", "w"), indent=2, default=float)
        if video:
            fails = failure_runs(g, p, hz)
            json.dump(fails, open(out / f"failures_{name}.json", "w"), indent=2)
            import cv2
            cap = cv2.VideoCapture(str(video))
            for k, f in enumerate(fails):
                cap.set(cv2.CAP_PROP_POS_MSEC, (f["t0"] + f["t1"]) / 2 * 1000)
                ok, fr = cap.read()
                if ok:
                    cv2.imwrite(str(out / f"failure_{name}_{k}_{f['gt']}_as_{f['pred']}.jpg"), fr)
            cap.release()
    lines.insert(2, "## Ablation summary\n\n" + md(pd.DataFrame(table)))
    (out / "eval_report.md").write_text("\n".join(lines))
    print(md(pd.DataFrame(table)))
