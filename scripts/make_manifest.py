"""Scan a folder of videos -> data/manifest.csv (fill in `camera`, `split`, `notes` by hand).
usage: python scripts/make_manifest.py data/raw --out data/manifest.csv
"""
import argparse
import csv
import pathlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--out", default="data/manifest.csv")
a = ap.parse_args()
rows = []
for p in sorted(pathlib.Path(a.root).rglob("*")):
    if p.suffix.lower() in {".avi", ".mp4", ".mkv", ".mov"}:
        cap = cv2.VideoCapture(str(p))
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT)
        rows.append({"video": p.stem, "path": str(p), "duration_s": round(n / fps, 1) if fps else "", "fps": round(fps, 2),
                     "width": int(cap.get(3)), "height": int(cap.get(4)), "camera": "", "split": "", "has_bed": "", "notes": ""})
        cap.release()
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["video"])
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} videos -> {a.out}")
