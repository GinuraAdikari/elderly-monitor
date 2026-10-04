import pathlib
import yaml

DEFAULT = pathlib.Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load(path=None) -> dict:
    return yaml.safe_load(pathlib.Path(path or DEFAULT).read_text())
