# validate_one_model.py
import argparse
from pixal.validate import detect
from pixal.modules.config_loader import load_config, resolve_path, configure_pixal_logger, load_and_merge_configs, ConfigNamespace, _dict_to_namespace
import tensorflow.keras.backend as K
from numba import cuda
import gc

from logging import FileHandler, StreamHandler, Formatter, getLogger
import logging
import pathlib as Path

parser = argparse.ArgumentParser()
parser.add_argument("--npz", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--metrics", required=True)
parser.add_argument("--config", default="configs/", help="Path to the configuration files")
parser.add_argument("--preprocess", action="store_true", help="Run preprocessing before validation")
parser.add_argument("--one_hot", action="store_true")
args = parser.parse_args()

config = load_and_merge_configs(args.config)
config = _dict_to_namespace(config)

path_config = load_config("configs/paths.yaml")

log_path = resolve_path(path_config.validate_log_path)
log_path.mkdir(parents=True, exist_ok=True)
if args.preprocess == "True":
    log_file = log_path / "validation.log"
else:
    log_file = log_path / "detect.log"

logger = configure_pixal_logger(log_file)

detect.run(args.npz, args.model, args.metrics, config=config, one_hot_encoding=args.one_hot)

# Clean up TensorFlow context
K.clear_session()
gc.collect()

# Reset GPU driver context
cuda.select_device(0)
cuda.close()