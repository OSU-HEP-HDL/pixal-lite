import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')

import numpy as np
import logging
from pathlib import Path
import pixal.modules.plotting as pltm
import pixal.architectures.autoencoder as autoencoder
from pixal.modules.config_loader import load_config, resolve_path, configure_pixal_logger

stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
sys.stderr.close()
sys.stderr = stderr_backup

path_config = load_config("configs/paths.yaml")
#loss_config = load_config("configs/parameters.yaml")

log_path = resolve_path(path_config.validate_log_path)
log_path.mkdir(parents=True, exist_ok=True)

log_file = log_path / "detect.log"

logger = configure_pixal_logger(log_file)

def run_detection(dataset, model_path, metric_dir, one_hot_encoding, config=None):
    X_test = dataset["data"]
    y_test = dataset.get("labels") if one_hot_encoding else None
    image_shape = dataset["shape"]

    X_test = X_test.reshape(X_test.shape[0], -1)

    if one_hot_encoding and y_test is not None:
        y_test = y_test.reshape(y_test.shape[0], -1)
   
  
    model = autoencoder.Autoencoder.load_model(model_path,config)
    inputs = [X_test, y_test] if one_hot_encoding else X_test
    predictions = model.predict(inputs)

    metric_dir = Path(metric_dir)
    metric_dir.mkdir(parents=True, exist_ok=True)

    if config.plotting.plot_distributions:
        pltm.plot_prediction_distribution(predictions, metric_dir)
        pltm.plot_truth_distribution(X_test, metric_dir)
        pltm.plot_combined_distribution(X_test, predictions, metric_dir)

    if config.plotting.plot_anomaly_heatmap:
        pltm.plot_mse_heatmap_overlay(
            X_test, predictions, image_shape, metric_dir, num_vars = len(config.preprocessing.preprocessor.channels),
            threshold=config.plotting.loss_cut, use_log_threshold=config.plotting.use_log_loss
        )

    if config.plotting.plot_roc_recall_curve:
        pltm.plot_anomaly_detection_curves(X_test, predictions, '', metric_dir)

    if config.plotting.plot_pixel_predictions:
        pltm.plot_pixel_predictions(X_test, predictions, "Pixel-wise Prediction Accuracy", metric_dir)

    if config.plotting.plot_confusion_matrix:
        pltm.plot_confusion_matrix(X_test, predictions, metric_dir)

    if config.plotting.plot_loss:
        pltm.plot_pixel_loss_and_log_loss(X_test, predictions, metric_dir, loss_threshold=config.plotting.loss_cut)
        pltm.plot_channelwise_pixel_loss(X_test, predictions, config, metric_dir, loss_threshold=config.plotting.loss_cut)
    
    pltm.plot_mse_heatmap(X_test, predictions, image_shape, output_dir=metric_dir.parent, threshold=config.plotting.loss_cut, use_log_threshold=False, channels=config.preprocessing.preprocessor.channels,weights=config.preprocessing.preprocessor.weights,max_images=1)

def run(npz_file, model_file, metric_dir, config=None, one_hot_encoding=False, quiet=False):
    npz_file = Path(npz_file) / config.preprocessing.preprocessor.file_name
    model_file = Path(model_file) / str(config.model_training.model_name + "." + config.model_training.model_file_extension)
    metric_dir = Path(metric_dir)

    if not npz_file.exists():
        logger.error(f"❌ .npz file not found: {npz_file}")
        return

    if not model_file.exists():
        logger.error(f"❌ Model file not found: {model_file}")
        return

    logger.info(f"🧪 Running detection on: {npz_file.name}")
    dataset = np.load(npz_file)
    run_detection(dataset, model_file, metric_dir, one_hot_encoding, config=config)
    logger.info(f"✅ Detection complete for: {npz_file.name}")
