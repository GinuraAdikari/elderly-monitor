import json
import numpy as np


def load_polygon(path) -> np.ndarray:
    """Bed polygon stored as normalised [0..1] (x, y) points, so it is resolution-independent."""
    return np.array(json.load(open(path))["polygon"], dtype=float)


def save_polygon(path, polygon_norm, video="", frame_wh=None):
    json.dump({"video": str(video), "frame_wh": frame_wh,
               "polygon": [[float(x), float(y)] for x, y in polygon_norm]}, open(path, "w"), indent=2)


def to_px(poly_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    return poly_norm * np.array([w, h], dtype=float)
