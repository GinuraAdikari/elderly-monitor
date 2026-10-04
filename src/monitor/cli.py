import argparse
import json
import pathlib
from . import config


def main():
    ap = argparse.ArgumentParser(prog="monitor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline on one video")
    r.add_argument("--video", required=True)
    r.add_argument("--polygon", required=True, help="bed polygon json (scripts/select_bed_polygon.py)")
    r.add_argument("--out", default="outputs")
    r.add_argument("--mode", default="agent", choices=["rules", "viterbi", "agent"])
    r.add_argument("--config", default=None)
    r.add_argument("--force-perception", action="store_true")

    e = sub.add_parser("eval", help="evaluate prediction folder(s) against ground truth (separate step)")
    e.add_argument("--gt", required=True, help="segments csv: start,end,state")
    e.add_argument("--events-gt", default=None, help="optional events csv: time,event")
    e.add_argument("--pred", nargs="+", required=True, help="one or more outputs/<video>/<mode> folders")
    e.add_argument("--tol", type=float, default=5.0)
    e.add_argument("--out", default=None)
    e.add_argument("--video", default=None, help="if given, export failure-case snapshots")
    e.add_argument("--config", default=None)

    a = ap.parse_args()
    cfg = config.load(a.config)
    if a.cmd == "run":
        from .pipeline import run
        print(json.dumps(run(a.video, a.polygon, a.out, cfg, a.mode, a.force_perception)["human_readable"], indent=2))
    else:
        from .eval import evaluate_many
        evaluate_many(a.gt, a.events_gt, a.pred, a.tol, a.out or str(pathlib.Path(a.pred[0]).parent / "eval"), a.video, cfg)


if __name__ == "__main__":
    main()
