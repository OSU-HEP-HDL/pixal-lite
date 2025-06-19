import yaml
import os
import logging
from types import SimpleNamespace
from pathlib import Path

def resolve_path(obj):
    parts = []
    while isinstance(obj, dict) or hasattr(obj, '__dict__'):
        d = obj.__dict__ if hasattr(obj, '__dict__') else obj
        # Pull keys except 'base'
        keys = list(k for k in d if k != 'base')
        if keys:
            parts.insert(0, d[keys[0]])
        obj = d.get('base')
    parts.insert(0, obj)  # Add root base string
    return Path(*parts)

def resolve_parent_inserted_path(path_or_obj, folder_name, insert_depth=1):
    """
    Insert `folder_name` N levels above the leaf of the resolved path.

    Args:
        path_or_obj: Either a string path or a ConfigNamespace with `.path`
        folder_name: The folder name to insert
        insert_depth: How many parent levels up to insert (default: 1)

    Returns:
        Path object with folder_name inserted at desired location
    """
    if hasattr(path_or_obj, "path"):
        path_str = path_or_obj.path
    else:
        path_str = path_or_obj

    full_path = Path(resolve_path(path_str))
    parent = full_path
    for _ in range(insert_depth):
        parent = parent.parent

    relative_tail = full_path.relative_to(parent)
    return parent / folder_name / relative_tail

class ConfigNamespace(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        def convert(obj):
            if isinstance(obj, ConfigNamespace):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            return obj
        return convert(self)

def _dict_to_namespace(d):
    if isinstance(d, dict):
        return ConfigNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    return d

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config_data = yaml.safe_load(f)

    config = _dict_to_namespace(config_data)

    #print("\n📄 Loaded config:", path)
    #print("-------------------------")
    #or k in config.__dict__.keys():
    #   print(f"• {k}")

    return config

def configure_pixal_logger(log_file):
    logger = logging.getLogger("pixal")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    return logger

def load_and_merge_configs(config_dir):
    """Merge all YAML files in a directory into one config dictionary."""
    config_dir = Path(config_dir)
    merged = {}
    for file in sorted(config_dir.glob("*.yaml")):
        with open(file, "r") as f:
            cfg = yaml.safe_load(f) or {}
            merged.update(cfg)
    return merged