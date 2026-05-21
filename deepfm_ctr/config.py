"""Configuration loader from YAML files."""

import yaml
from pathlib import Path
from types import SimpleNamespace


def _dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace for dot-notation access."""
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = _dict_to_namespace(v)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [_dict_to_namespace(item) if isinstance(item, dict) else item for item in d]
    return d


def load_config(path="configs/default.yaml"):
    """Load YAML config and return as SimpleNamespace."""
    with open(path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = _dict_to_namespace(config_dict)

    # Ensure processed_dir is a Path
    config.data.processed_dir = Path(config.data.processed_dir)

    return config
