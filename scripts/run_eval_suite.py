"""Run the full pipeline (rules, viterbi, agent) across all manifest videos and evaluate against ground truth.

Generates outputs/<video>/eval/eval_report.md with ablation tables and failure snapshots.
"""
import os
import pathlib
import pandas as pd
from monitor import config, pipeline
from monitor.eval import evaluate_many

cfg = config.load()
manifest = pd.read_csv("data/manifest.csv")
out_root = pathlib.Path("outputs")

print(f"Loaded manifest with {len(manifest)} videos.")

for idx, row in manifest.iterrows():
    vid_name = row["video"]
    vid_path = pathlib.Path(row["path"])
    polygon_path = pathlib.Path(f"config/bed_polygons/{vid_name}.json")
    if not polygon_path.exists():
        polygon_path = pathlib.Path(f"config/bed_polygons/{row['camera']}.json")
    
    print(f"\n=======================================================")
    print(f"[{idx+1}/{len(manifest)}] Processing {vid_name} ({row['split']})")
    print(f"Video: {vid_path}, Polygon: {polygon_path}")
    print(f"=======================================================")
    
    for mode in ["rules", "viterbi", "agent"]:
        print(f"  -> Running mode: {mode} ...", end=" ", flush=True)
        pipeline.run(vid_path, polygon_path, out_root, cfg, mode=mode)
        print("Done.")
    
    gt_csv = pathlib.Path(f"data/labels/{vid_name}.csv")
    events_csv = pathlib.Path(f"data/labels/{vid_name}_events.csv")
    if not events_csv.exists():
        events_csv = None
        
    eval_dir = out_root / vid_name / "eval"
    pred_dirs = [out_root / vid_name / m for m in ["rules", "viterbi", "agent"]]
    
    if gt_csv.exists():
        print(f"  -> Evaluating vs {gt_csv} ...")
        evaluate_many(
            gt_csv=str(gt_csv),
            events_csv=str(events_csv) if events_csv else None,
            pred_dirs=[str(p) for p in pred_dirs],
            tol=cfg["eval"]["event_tol_sec"] if "eval" in cfg and "event_tol_sec" in cfg["eval"] else 5.0,
            out_dir=str(eval_dir),
            video=str(vid_path),
            cfg=cfg
        )
        print(f"  -> Evaluation report written to {eval_dir / 'eval_report.md'}")
    else:
        print(f"  [WARN] No GT label found at {gt_csv}")

print("\nAll videos processed and evaluated successfully!")

