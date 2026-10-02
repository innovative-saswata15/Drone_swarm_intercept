"""Load the YAML configuration file."""
from pathlib import Path
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path=None):
    """Return the configuration as a nested dict. Uses configs/default.yaml if no path is given."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, "r") as f:
        return yaml.safe_load(f)
