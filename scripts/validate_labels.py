"""Check a ground-truth segments csv for gaps/overlaps/bad state names. usage: python scripts/validate_labels.py data/labels/x.csv"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from monitor.eval import load_gt, validate_gt

segs = load_gt(sys.argv[1])
issues = validate_gt(segs)
print(f"{len(segs)} segments, {segs[-1].end:.0f}s total")
print("\n".join(issues) if issues else "OK")
sys.exit(1 if issues else 0)
