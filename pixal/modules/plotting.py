import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pixal.modules.config_loader import load_config, resolve_path, configure_pixal_logger, deep_getattr
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import itertools
import cv2
import os
import logging 
import math

path_config = load_config("configs/paths.yaml")

log_path = resolve_path(path_config.validate_log_path)
log_path.mkdir(parents=True, exist_ok=True)

log_file = log_path / "detect.log"

logger = configure_pixal_logger(log_file)

import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def plot_mse_heatmap(
    X_test,
    predictions,
    image_shape,                  # (H, W)
    output_dir,                   # Path or str
    threshold,                    # float; your loss_cut
    use_log_threshold=False,
    channels=None,                # list[str] (e.g., ["R","G","B","LAB_a",...]) or None
    weights=None,                 # list[float] same length as channels, or None
    max_images=5,
):
    """
    Create weighted per-pixel MSE maps between X_test and predictions, then overlay anomalies
    (pixels where error >= threshold) on the original image when RGB is available. If RGB is not
    available, uses a normalized grayscale base from the available channels.

    Output: ONLY the overlay image (no side-by-side). Returns list of saved file paths.
    """

    H, W = image_shape
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N = min(max_images, len(X_test))
    eps = 1e-8

    # Infer channel count C
    flat_len = int(np.asarray(X_test[0]).size)
    if flat_len % (H * W) != 0:
        raise ValueError(
            f"Input length {flat_len} not divisible by H*W={H*W}. "
            "Check image_shape or feature vector size."
        )
    C = flat_len // (H * W)

    # Validate channels/weights
    if channels is not None and len(channels) != C:
        raise ValueError(f"len(channels)={len(channels)} must equal inferred C={C}.")
    if weights is None:
        w = np.ones((C,), dtype=np.float32)
    else:
        if len(weights) != C:
            raise ValueError(f"len(weights)={len(weights)} must equal C={C}.")
        w = np.asarray(weights, dtype=np.float32)
    w_sum = float(np.sum(w)) if np.any(w) else 1.0

    # Helper: try to build an RGB base from channels; otherwise fallback
    def make_base_bgr(ch_stack):
        """
        ch_stack: (H, W, C) float array (any scale)
        Returns uint8 BGR image for overlay base.
        """
        base = None
        if channels is not None:
            try:
                r_idx = channels.index("R")
                g_idx = channels.index("G")
                b_idx = channels.index("B")

                def to_u8(x):
                    x = np.asarray(x, dtype=np.float32)
                    x_min, x_max = float(x.min()), float(x.max())
                    if x_max - x_min < 1e-12:
                        return np.zeros_like(x, dtype=np.uint8)
                    return ((x - x_min) / (x_max - x_min) * 255.0).astype(np.uint8)

                R = to_u8(ch_stack[..., r_idx])
                G = to_u8(ch_stack[..., g_idx])
                B = to_u8(ch_stack[..., b_idx])
                rgb = np.stack([R, G, B], axis=-1)
                base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except ValueError:
                base = None

        if base is None:
            # Fallback: average the first up to 3 channels, normalize to 0..255
            use_c = min(C, 3)
            gray = np.mean(ch_stack[..., :use_c], axis=-1).astype(np.float32)
            gmin, gmax = float(gray.min()), float(gray.max())
            if gmax - gmin < 1e-12:
                gray_u8 = np.zeros_like(gray, dtype=np.uint8)
            else:
                gray_u8 = ((gray - gmin) / (gmax - gmin) * 255.0).astype(np.uint8)
            base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

        return base

    saved_paths = []

    for i in range(N):
        orig_flat = np.asarray(X_test[i], dtype=np.float32)
        pred_flat = np.asarray(predictions[i], dtype=np.float32)

        # Reshape to (H, W, C)
        orig = orig_flat.reshape(H, W, C)
        pred = pred_flat.reshape(H, W, C)

        # Weighted per-pixel MSE across channels
        diff = orig - pred                      # (H, W, C)
        sq = diff * diff                        # (H, W, C)
        error_map = np.tensordot(sq, w, axes=([2], [0])) / w_sum  # (H, W)

        # Threshold domain (raw or log)
        if use_log_threshold:
            vis_map = np.log10(error_map + eps)
            thr_value = np.log10(max(threshold, eps))
        else:
            vis_map = error_map
            thr_value = float(threshold)

        # Normalize error map to [0,1] for colormap
        vmin, vmax = float(vis_map.min()), float(vis_map.max())
        denom = (vmax - vmin) if (vmax > vmin) else 1.0
        norm = (vis_map - vmin) / (denom + eps)

        # Build overlay
        base_bgr = make_base_bgr(orig)
        heat_u8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(base_bgr, 0.60, heat_bgr, 0.40, 0.0)

        # Hard mark anomalies in RED (BGR)
        mask = (vis_map >= thr_value).astype(np.uint8)
        overlay[mask == 1] = [0, 0, 255]  # pure red in BGR

        # Save ONLY the overlay (no figure, no borders)
        out_path = output_dir / f"mse_overlay_{i}.png"
        ok = cv2.imwrite(str(out_path), overlay)
        if not ok:
            logger.warning(f"Failed to save overlay for image {i} at {out_path}")

        # Stats
        pct = 100.0 * mask.sum() / mask.size
        logger.info(
            f"[Image {i}] Anomalous pixels: {int(mask.sum()):,} ({pct:.2f}%), "
            f"C={C}, saved: {out_path}"
        )
        saved_paths.append(out_path)

    return saved_paths




def plot_mse_heatmap_overlay(X_test, predictions, image_shape, output_dir="analysis_plots",
                             num_vars=3, threshold=0.01, use_log_threshold=False):
    """
    Computes per-pixel MSE and overlays an anomaly heatmap on the original images.
    """
    os.makedirs(output_dir, exist_ok=True)

    height, width = image_shape

    for i in range(min(5, len(X_test))):  # Limit to first 5 examples
        original_flat = X_test[i]
        reconstructed_flat = predictions[i]

        # Infer channel count if not 3/unsure
        inferred_vars = original_flat.size // (height * width)
        if inferred_vars > 0 and inferred_vars != num_vars:
            num_vars = inferred_vars  # keep function resilient

        original_img = original_flat.reshape((height, width, num_vars))
        reconstructed_img = reconstructed_flat.reshape((height, width, num_vars))

        # 2D grayscale for display
        avg_original_img = np.mean(original_img, axis=-1)
        avg_reconstructed_img = np.mean(reconstructed_img, axis=-1)

        # Per-pixel MSE across channels
        error_map = np.mean((original_img - reconstructed_img) ** 2, axis=-1).astype(np.float32)

        # Optional log thresholding
        if use_log_threshold:
            error_map = np.log10(error_map + 1e-8)
            threshold_label = f"log10({threshold})"
            threshold_value = np.log10(threshold)
        else:
            threshold_label = f"{threshold}"
            threshold_value = threshold

        # Normalize error map to [0,1] for colormap
        emn, emx = float(error_map.min()), float(error_map.max())
        norm_error_map = (error_map - emn) / (emx - emn + 1e-8)

        # Anomaly mask and stats
        anomaly_mask = (error_map >= threshold_value).astype(np.uint8)
        num_pixels = anomaly_mask.size
        num_anomalous = int(anomaly_mask.sum())
        percent = (num_anomalous / num_pixels) * 100.0
        logger.info(f"[Image {i}] Anomalous Pixels: {num_anomalous:,}  Percentage: {percent:.2f}%")

        # Prepare overlay (base is grayscale -> BGR for OpenCV)
        original_bgr = cv2.cvtColor(
            np.clip(avg_original_img * 255.0, 0, 255).astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )
        heatmap_raw = np.clip(norm_error_map * 255.0, 0, 255).astype(np.uint8)
        heatmap_color_bgr = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_JET)

        overlay_bgr = cv2.addWeighted(original_bgr, 0.6, heatmap_color_bgr, 0.4, 0)
        # Paint anomalies in RED (BGR -> red is [0,0,255])
        overlay_bgr[anomaly_mask == 1] = [0, 0, 255]

        # Convert BGR -> RGB for matplotlib
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        axes[0].imshow(avg_original_img, cmap="gray")
        axes[0].set_title("Original (avg across channels)")
        axes[1].imshow(overlay_rgb)
        axes[1].set_title(f"Heatmap (Threshold: {threshold_label})")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"anomaly_overlay_{i}.png")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        logger.info(f"[✓] Heatmap Saved: {output_path}")



def analyze_mse_distribution(X_test, predictions, image_shape, output_dir="analysis_plots"):
    """
    Computes and plots the MSE distribution per image and visualizes per-pixel MSE for individual images.

    Parameters:
        model: Trained Autoencoder.
        X_test: Test images (flattened).
        y_test: Corresponding labels.
        image_shape: Tuple of the original image dimensions (height, width).
        output_dir: Directory to save results.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Compute per-image MSE
    mse_per_image = np.mean((X_test - predictions) ** 2, axis=1)
    logger.info(f"Shape of mse_per_image: {mse_per_image.shape}")

    # Plot histogram of MSE distribution
    plt.figure(figsize=(8, 5))
    plt.hist(mse_per_image, bins=50, alpha=0.7, color='blue', label="MSE Distribution")
    plt.axvline(np.mean(mse_per_image), color='r', linestyle='dashed', linewidth=2,
                label=f"Mean MSE: {np.mean(mse_per_image):.6f}")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.title("MSE Distribution Across Test Images")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "mse_distribution.png"))
    plt.close()

    # Compute per-pixel MSE for each image
    per_pixel_mse = (X_test - predictions) ** 2

    # Plot per-pixel MSE for the first few images
    num_images_to_plot = min(5, X_test.shape[0])
    for i in range(num_images_to_plot):
        # Recover the image shape with 3 channels
        height, width = image_shape
        mse_image = per_pixel_mse[i].reshape((height, width, 3))

        # Average across the HSV channels
        mse_image_avg = np.mean(mse_image, axis=-1)  # Shape becomes (height, width)

        plt.figure(figsize=(8, 5))
        plt.imshow(mse_image_avg, cmap='hot', interpolation='nearest')
        plt.colorbar(label='MSE per Pixel')
        plt.title(f"Per-Pixel MSE for Test Image {i}")
        plt.axis('off')

        plt.savefig(os.path.join(output_dir, f"mse_per_pixel_image_{i}.png"))
        plt.close()

    return mse_per_image


def analyze_pixel_validation_loss(X_test, predictions, image_shape, output_dir="analysis_plots"):
    """
    Computes and visualizes per-pixel absolute validation loss (difference) per image.

    Parameters:
        model: Trained Autoencoder.
        X_test: Test images (flattened).
        y_test: Corresponding labels.
        image_shape: Tuple of the pooled image dimensions (height, width).
        output_dir: Directory to save results.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Compute absolute validation loss per image
    abs_loss_per_image = np.mean(np.abs(X_test - predictions), axis=1)
    logger.info(f"Shape of abs_loss_per_image: {abs_loss_per_image.shape}")

    # Plot histogram of absolute validation loss distribution
    plt.figure(figsize=(8, 5))
    plt.hist(abs_loss_per_image, bins=50, alpha=0.7, color='green', label="Absolute Loss Distribution")
    plt.axvline(np.mean(abs_loss_per_image), color='r', linestyle='dashed', linewidth=2,
                label=f"Mean Absolute Loss: {np.mean(abs_loss_per_image):.6f}")
    plt.xlabel("Absolute Validation Loss")
    plt.ylabel("Frequency")
    plt.title("Absolute Validation Loss Distribution Across Test Images")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "abs_validation_loss_distribution.png"))
    plt.close()

    # Compute per-pixel absolute loss for each image
    per_pixel_abs_loss = np.abs(X_test - predictions)

    # Plot per-pixel absolute loss for the first few images
    num_images_to_plot = min(5, X_test.shape[0])
    for i in range(num_images_to_plot):
        height, width = image_shape
        loss_image = per_pixel_abs_loss[i].reshape((height, width, 3))

         # Average across the HSV channels
        loss_image_avg = np.mean(loss_image, axis=-1)  # Shape becomes (height, width)

        plt.figure(figsize=(8, 5))
        plt.imshow(loss_image_avg, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Absolute Loss per Pixel')
        plt.title(f"Per-Pixel Absolute Validation Loss for Test Image {i}")
        plt.axis('off')

        plt.savefig(os.path.join(output_dir, f"abs_loss_per_pixel_image_{i}.png"))
        plt.close()

    return abs_loss_per_image



def plot_anomaly_detection_curves(x_test, predictions, title_prefix='', output_dir="analysis_plots"):
    """
    Plot ROC and Precision-Recall curves per image based on pixel-level reconstruction error.

    Parameters:
        model: Trained model.
        x_test: Test input (n_images, n_pixels).
        y_test: Pixel-wise ground truth labels (n_images, n_pixels).
        title_prefix: Optional title prefix.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
   
    recon_error = (x_test - predictions) ** 2  # (n_images, n_pixels)
    x_flat = x_test.flatten()
    y_flat = predictions.flatten()

    n_images = x_test.shape[0]

    for i in range(n_images):
        x_true = x_test[i].flatten().astype(int)
        x_scores = recon_error[i].flatten()

        # ROC and AUC
        fpr, tpr, _ = roc_curve(x_true, x_scores)
        roc_auc = auc(fpr, tpr)

        # Precision-Recall and AUC
        precision, recall, _ = precision_recall_curve(x_true, x_scores)
        pr_auc = auc(recall, precision)
      
        # Plot
        plt.figure(figsize=(12, 5))

        # ROC
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='blue')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title_prefix}ROC Curve (Image {i})')
        plt.legend(loc='lower right')
        plt.grid(True)

        # PR
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f'AUC = {pr_auc:.2f}', color='green')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{title_prefix}Precision-Recall Curve (Image {i})')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"anomaly_detection_curves_image_{i}.png"))
        plt.close()


def plot_pixel_predictions(x_true, predictions, title="Pixel-wise Prediction Accuracy",output_dir="analysis_plots"):
    
    x_flat = x_true.flatten()
    y_flat = predictions.flatten()

    logger.info(f"x_true range: {x_flat.min()} → {x_flat.max()}")
    logger.info(f"predictions range: {y_flat.min():.4f} to {y_flat.max():.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(x_true.flatten(), predictions.flatten(), alpha=0.3, s=1)
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
    plt.xlabel("True Pixel Value")
    plt.ylabel("Predicted Pixel Value")
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, "pixel_predictions.png"))

def plot_prediction_distribution(predictions, output_dir="analysis_plots"):
    """
    Plot pixel-wise predictions of the model.

    Parameters:
        model: Trained model.
        x_test: Test input (n_images, n_pixels).
        y_test: Pixel-wise ground truth labels (n_images, n_pixels).
        title: Title for the plot.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.hist(predictions.flatten(), bins=100,log=True,label="Predictions")
    plt.title("Distribution of Model Predictions")
    plt.savefig(os.path.join(output_dir, "prediction_distribution_log.png"))
    plt.close()

def plot_truth_distribution(x_test, output_dir="analysis_plots"):
    """
    Plot pixel-wise predictions of the model.

    Parameters:
        model: Trained model.
        x_test: Test input (n_images, n_pixels).
        y_test: Pixel-wise ground truth labels (n_images, n_pixels).
        title: Title for the plot.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    #predictions = model.predict([x_test, y_test])


    plt.hist(x_test.flatten(), bins=100,log=True,label="Truth")
    plt.title("Distribution of Model Predictions")
    plt.savefig(os.path.join(output_dir, "truth_distribution_log.png"))
    plt.close()

def plot_combined_distribution(x_test, predictions, output_dir="analysis_plots"):
    """
    Plot pixel-wise predictions of the model.

    Parameters:
        model: Trained model.
        x_test: Test input (n_images, n_pixels).
        y_test: Pixel-wise ground truth labels (n_images, n_pixels).
        title: Title for the plot.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.hist(x_test.flatten(), bins=100,log=True,label="Truth",color='blue', alpha=0.5)
    plt.hist(predictions.flatten(), bins=100,log=True,label="Predictions",color='orange', alpha=0.5)
    plt.title("Overlayed Distribution of Model Predictions and Truth")
    plt.xlabel("Pixel Value")
    plt.ylabel("Log(Frequency)")
    plt.legend() 
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_distribution_log.png"))
    plt.close()

def plot_confusion_matrix(x_true, x_pred, output_dir="analysis_plots"):
    """
    Plot confusion matrix for pixel-wise predictions.

    Parameters:
        y_true: True labels.
        y_pred: Predicted labels.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    cm = tf.math.confusion_matrix(x_true.flatten(), x_pred.flatten(), num_classes=2)
    cm = cm.numpy()  # convert to NumPy array

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal", "Anomalous"])
    plt.yticks(tick_marks, ["Normal", "Anomalous"])
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
        plt.text(j, i, f"{cm_normalized[i, j]:.2f}", horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_pixel_loss_and_log_loss(x_true, x_pred, output_dir="analysis_plots", loss_threshold=0.01):
    """
    Plot histograms of per-pixel reconstruction losses in both linear and log scales,
    with red threshold lines at loss_threshold and log10(loss_threshold).

    Parameters:
        x_true (np.ndarray): Ground truth image data (flattened or reshaped)
        x_pred (np.ndarray): Predicted image data (same shape as x_true)
        output_dir (str): Directory to save plots
        loss_threshold (float): Threshold value for highlighting cutoff
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute per-pixel MSE
    pixel_losses = np.square(x_true - x_pred).flatten()

    # Plot: Standard loss histogram
    plt.figure(figsize=(8, 5))
    plt.hist(pixel_losses, bins=100, alpha=0.7, color='steelblue')
    plt.axvline(loss_threshold, color='red', linestyle='--', linewidth=2, label=f"Threshold = {loss_threshold:.4f}")
    plt.xlabel("Pixel-wise Loss (MSE)")
    plt.ylabel("Frequency")
    plt.title("Per-Pixel Loss Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pixel_loss_histogram.png"))
    plt.close()

    # Plot: Log-scaled histogram (Y-axis)
    log_threshold = np.log10(loss_threshold) if loss_threshold > 0 else 0
    plt.figure(figsize=(8, 5))
    plt.hist(pixel_losses, bins=100, alpha=0.7, color='darkorange', log=True)
    plt.axvline(loss_threshold, color='red', linestyle='--', linewidth=2, label=f"Threshold = {loss_threshold:.4f}")
    plt.xlabel("Pixel-wise Loss (MSE)")
    plt.ylabel("Frequency (log scale)")
    plt.title("Log-Scale Per-Pixel Loss Distribution")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pixel_loss_log_histogram.png"))
    plt.close()


def plot_channelwise_pixel_loss(x_true, x_pred, config, output_dir="analysis_plots", loss_threshold=0.01):
    """
    Plot per-channel histograms of per-pixel MSE losses on a single figure with subplots.

    Input shape should be (N, H*W*C) with C=3 assumed.
    The function reshapes to (N, H*W, C), then plots loss histograms per channel.

    Parameters:
        x_true (np.ndarray): Ground truth of shape (N, P*C), where P = H*W
        x_pred (np.ndarray): Predicted data of same shape
        config (object): Configuration object with config.preprocessor.channels
        output_dir (str): Directory to save plots
        loss_threshold (float): Threshold line to show in histograms
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract channels from config or default to RGB
    channels = deep_getattr(config, "preprocessing.preprocessor.channels", ["R", "G", "B"])

    channel_map = {'H': "Hue", 'S': "Saturation", 'V': "Value",
                   'R': "Red", 'G': "Green", 'B': "Blue"}
    color_defaults = {'R': 'red', 'G': 'green', 'B': 'blue',
                      'H': 'purple', 'S': 'orange', 'V': 'brown'}

    # Check input shape
    if x_true.ndim != 2 or x_pred.ndim != 2 or x_true.shape != x_pred.shape:
        raise ValueError(f"[✗] Expected shape (N, H*W*C). Got {x_true.shape} and {x_pred.shape}")

    num_samples, total_values = x_true.shape
    num_channels = len(channels)
    if total_values % num_channels != 0:
        raise ValueError(f"[✗] Total values {total_values} not divisible by channel count {num_channels}")

    num_pixels = total_values // num_channels

    # Reshape to (N, P, C)
    x_true = x_true.reshape((num_samples, num_pixels, num_channels))
    x_pred = x_pred.reshape((num_samples, num_pixels, num_channels))

    # Combine all samples: (N, P, C) → (N*P, C)
    x_true = x_true.reshape(-1, num_channels)
    x_pred = x_pred.reshape(-1, num_channels)

    # Plotting setup
    n_cols = min(num_channels, 3)
    n_rows = math.ceil(num_channels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for c_idx, ch in enumerate(channels):
        losses = np.square(x_true[:, c_idx] - x_pred[:, c_idx])
        channel_name = channel_map.get(ch, ch)
        color = color_defaults.get(ch, 'gray')

        ax = axes[c_idx]
        ax.hist(losses, bins=100, alpha=0.7, color=color,log=True)
        ax.axvline(loss_threshold, color='black', linestyle='--', linewidth=2,
                   label=f"Threshold = {loss_threshold:.4f}")
        ax.set_xlabel(f"{channel_name} Loss (MSE)")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_title(f"{channel_name} Channel")
        ax.legend()
        ax.grid(True)

    for idx in range(num_channels, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "combined_channel_loss_histogram.png")
    plt.savefig(out_path)
    plt.close()

    logger.info("[✓] Combined channel-wise histogram saved to %s", out_path)

