def hms(sec: float) -> str:
    s = int(round(sec))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def ms(sec: float) -> str:
    s = int(round(sec))
    return f"{s // 60:02d}:{s % 60:02d}"


def dur(sec: float) -> str:
    s = int(round(sec))
    return f"{s // 60}m {s % 60:02d}s"


def parse_time(x) -> float:
    """Accepts 12.5, '75', '01:15', '00:01:15'."""
    if isinstance(x, (int, float)):
        return float(x)
    parts = [float(p) for p in str(x).strip().split(":")]
    t = 0.0
    for p in parts:
        t = t * 60 + p
    return t
