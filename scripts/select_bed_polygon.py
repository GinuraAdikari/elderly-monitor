"""Click the bed corners once per camera view -> config/bed_polygons/<name>.json (normalised coordinates).
usage:  python scripts/select_bed_polygon.py --video data/raw/x.avi --out config/bed_polygons/x.json [--frame-sec 5]
headless: add --points "x,y;x,y;x,y;x,y" (pixel coords of the full-resolution frame)
GUI keys: left-click add point | u undo | Enter/s save | q quit
"""
import argparse
import pathlib
import sys
import cv2
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from monitor.geometry import save_polygon

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frame-sec", type=float, default=2.0)
ap.add_argument("--points", default=None)
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
cap.set(cv2.CAP_PROP_POS_MSEC, a.frame_sec * 1000)
ok, frame = cap.read()
assert ok, "cannot read frame"
h, w = frame.shape[:2]
if a.points:
    pts = [tuple(map(float, p.split(","))) for p in a.points.split(";")]
else:
    pts = []
    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
    cv2.namedWindow("bed"); cv2.setMouseCallback("bed", on_mouse)
    while True:
        vis = frame.copy()
        for p in pts: cv2.circle(vis, tuple(map(int, p)), 4, (0, 255, 255), -1)
        if len(pts) > 1: cv2.polylines(vis, [np.array(pts, np.int32)], len(pts) > 2, (0, 255, 255), 2)
        cv2.imshow("bed", vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and pts: pts.pop()
        elif k in (13, ord("s")) and len(pts) >= 3: break
        elif k == ord("q"): sys.exit("aborted")
    cv2.destroyAllWindows()
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
save_polygon(a.out, [(x / w, y / h) for x, y in pts], a.video, [w, h])
print("saved", a.out)
