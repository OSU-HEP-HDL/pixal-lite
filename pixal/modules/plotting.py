import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pixal.modules.config_loader import load_config, resolve_path, configure_pixal_logger
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

def plot_mse_heatmap(X_test, predictions, output_dir="mse_plots"):
    """
    Computes per-pixel MSE and overlays an anomaly heatmap on the original images.
    
    Parameters:
        model: Trained Autoencoder model.
        X_test: Test images (flattened).
        y_test: Corresponding labels.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute per-pixel MSE
    mse = np.mean((X_test - predictions) ** 2, axis=1)  # Mean MSE per image
    pixel_mse = np.mean((X_test - predictions) ** 2, axis=0)  # Mean MSE per pixel across dataset

    logger.info(f"Mean MSE across test set: {np.mean(mse):.6f}")

    # Iterate over images
    for i in range(min(5, len(X_test))):  # Show first 5 images
        original = X_test[i].reshape(90, 90)  # Reshape to image size (adjust as needed)
        reconstructed = predictions[i].reshape(90, 90)

        # Compute per-pixel error (reshaped)
        error_map = ((X_test[i] - predictions[i]) ** 2).reshape(90, 90)

        # Normalize error map for visualization
        norm_error_map = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_error_map, cv2.COLORMAP_JET)

        # Overlay heatmap on the original image
        overlay = cv2.addWeighted(cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_GRAY2BGR), 0.5, heatmap, 0.5, 0)

        # Save and display results
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original Image")
        axes[1].imshow(overlay)
        axes[1].set_title("Anomaly Heatmap")
        axes[2].imshow(reconstructed, cmap="gray")
        axes[2].set_title("Reconstructed Image")

        for ax in axes:
            ax.axis("off")

        plt.savefig(os.path.join(output_dir, f"mse_heatmap_{i}.png"))
        plt.show()

    return mse


def plot_mse_heatmap_overlay(X_test, predictions, image_shape, output_dir="analysis_plots", threshold=0.01, use_log_threshold=False):
    """
    Computes per-pixel MSE and overlays an anomaly heatmap on the original images.

    Parameters:
        model: Trained Autoencoder model.
        X_test: Test images (flattened).
        y_test: Corresponding one-hot labels.
        image_shape: Tuple representing the (height, width) of the original image.
        output_dir: Directory to save plots.
        threshold: MSE value above which pixels are considered anomalous.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in range(min(5, len(X_test))):  # Limit to first 5 examples

        original_flat = X_test[i]
        reconstructed_flat = predictions[i]
        height, width = image_shape
        
        original_img = original_flat.reshape((height, width, 3))
        avg_original_img = np.mean(original_img, axis=-1)  # Shape becomes (height, width)
        reconstructed_img = reconstructed_flat.reshape((height, width, 3))
        avg_reconstructed_img = np.mean(reconstructed_img, axis=-1)  # Shape becomes (height, width)

        # Compute per-pixel squared error
        error_map = np.mean(np.square(original_img - reconstructed_img), axis=-1)


        # Apply log transformation if requested
        if use_log_threshold:
            error_map = np.log10(error_map + 1e-8)  # shift to avoid log(0)
            threshold_label = f"log10({threshold})"
            threshold_value = np.log10(threshold)
        else:
            threshold_label = f"{threshold}"
            threshold_value = threshold

        # Normalize the error map for heatmap visualization
        norm_error_map = (error_map - np.min(error_map)) / (np.max(error_map) - np.min(error_map) + 1e-8)

        # Create anomaly mask
        anomaly_mask = (error_map >= threshold_value).astype(np.uint8)
        num_pixels = anomaly_mask.size
        num_anomalous = np.sum(anomaly_mask)
        percent = (num_anomalous / num_pixels) * 100

        logger.info(f"[Image {i}] Anomalous Pixels: {num_anomalous:,}  Percentage: {percent:.2f}%")

        # Prepare image for overlay
        original_bgr = cv2.cvtColor((avg_original_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        heatmap_raw = (norm_error_map * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)
        overlay[anomaly_mask == 1] = [255, 0, 0]

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        axes[0].imshow(original_img, cmap="gray")
        axes[0].set_title("Original")
        axes[1].imshow(overlay)
        axes[1].set_title(f"Heatmap (Threshold: {threshold_label})")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"anomaly_overlay_{i}.png")
        plt.savefig(output_path)
        plt.close()

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
    channels = getattr(getattr(config, 'preprocessor', {}), 'channels', ['R', 'G', 'B'])
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

