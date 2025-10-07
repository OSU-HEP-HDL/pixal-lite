import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')
from pixal.modules.preprocessing import save_crop_preview
import numpy as np
import cv2
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm
import yaml
import logging

logger = logging.getLogger("pixal")

class ImageDataProcessor:
    def __init__(self, image_folders, pool_size=8, channels=("H", "S", "V"), file_name="out.npz", quiet=False, one_hot_encoding=False, zero_pruning=False, zero_pruning_padding=1, bg_threshold=2, crop_box=None, validation=False):
        self.image_folders = image_folders
        self.pool_size = pool_size
        self.image_shape = None 
        self.channels = channels
        self.quiet = quiet
        self.file_name = file_name
        self.one_hot_encoding = one_hot_encoding
        self.zero_pruning = zero_pruning
        self.zero_pruning_padding = zero_pruning_padding
        self.crop_box = None
        self.bg_threshold = bg_threshold
        self.crop_box = crop_box
        self.validation = validation

    def compute_feature_maps(self, bgr_image: np.ndarray) -> dict:
        """
        Returns a dict of single-channel float32 feature maps in [0,1] (except sin/cos which are [-1,1] then remapped).
        Keys are stable and documented below.
        """
        # Base conversions
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb).astype(np.float32)

        # Split raw channels
        R, G, B = [rgb[..., i] for i in range(3)]
        H, S, V = [hsv[..., i] for i in range(3)]
        L_lab, a_lab, b_lab = [lab[..., i] for i in range(3)]
        Y, Cr, Cb = [ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]]

        # Normalizations to [0,1] (OpenCV ranges: H[0..179], others [0..255], Lab L[0..100], a/b[0..255] with 128 center)
        def norm01(x, lo, hi):
            x = (x - lo) / (hi - lo + 1e-7)
            return np.clip(x, 0.0, 1.0).astype(np.float32)

        # RGB/HSV/YCrCb
        Rn, Gn, Bn = norm01(R, 0, 255), norm01(G, 0, 255), norm01(B, 0, 255)
        Hn, Sn, Vn = norm01(H, 0, 179), norm01(S, 0, 255), norm01(V, 0, 255)
        Yn, Crn, Cbn = norm01(Y, 0, 255), norm01(Cr, 0, 255), norm01(Cb, 0, 255)

        # Lab normalize (OpenCV Lab: L in [0,100], a/b shifted by +128 in [0,255])
        Ln = norm01(L_lab, 0, 100)
        an = norm01(a_lab, 0, 255)   # if you prefer centered: (a_lab-128)/127 → [-1,1] then map to [0,1]
        bn = norm01(b_lab, 0, 255)

        # Opponent & chromaticity
        intensity = (R + G + B + 1e-7)
        r_chroma = (R / intensity).astype(np.float32)  # already 0..1
        g_chroma = (G / intensity).astype(np.float32)
        # Opponent
        O1 = (R - G)                    # emphasize red-green
        O2 = ((R + G) * 0.5 - B)        # yellow-blue
        O3 = (R + G + B) / 3.0          # intensity
        # Normalize opponent roughly to [0,1] using image-local robust range
        def robust01(x):
            lo = np.percentile(x, 1.0)
            hi = np.percentile(x, 99.0)
            return norm01(np.clip(x, lo, hi), lo, hi)
        O1n, O2n, O3n = robust01(O1), robust01(O2), norm01(O3, 0, 255)

        # LCh from Lab
        # Convert OpenCV a,b (0..255) back to signed approx: a* ~ a-128, b* ~ b-128
        a_signed = (a_lab - 128.0).astype(np.float32)
        b_signed = (b_lab - 128.0).astype(np.float32)
        C = np.sqrt(a_signed**2 + b_signed**2)                  # chroma magnitude
        h_rad = np.arctan2(b_signed, a_signed)                  # hue angle in radians [-pi, pi]
        # Normalize C per-image (robust) and encode hue as sin/cos to avoid wrap
        Cn = robust01(C)
        sin_h = np.sin(h_rad).astype(np.float32)                # [-1,1]
        cos_h = np.cos(h_rad).astype(np.float32)                # [-1,1]
        # Map sin/cos to [0,1] for consistency
        sin_h01 = (sin_h * 0.5 + 0.5).astype(np.float32)
        cos_h01 = (cos_h * 0.5 + 0.5).astype(np.float32)

        # Gradients / edges on luminance (use L* or Y)
        L_for_edges = Ln
        # Sobel derivatives (float32)
        Gx = cv2.Sobel(L_for_edges, cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(L_for_edges, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(Gx * Gx + Gy * Gy)
        grad_mag = grad_mag / (grad_mag.max() + 1e-7)

        # Laplacian
        lap = cv2.Laplacian(L_for_edges, cv2.CV_32F, ksize=3)
        # Center to [0,1] using robust range
        lapn = robust01(lap)

        # Local stddev (3x3) as local contrast
        k = 3
        mu = cv2.blur(L_for_edges, (k, k))
        mu2 = cv2.blur(L_for_edges * L_for_edges, (k, k))
        local_var = np.maximum(mu2 - mu * mu, 0.0)
        local_std = np.sqrt(local_var)
        local_std = local_std / (local_std.max() + 1e-7)

        # Build the feature dict (single-channel maps)
        feats = {
            # Raw color spaces
            "R": Rn, "G": Gn, "B": Bn,
            "H": Hn, "S": Sn, "V": Vn,
            "Y": Yn, "Cr": Crn, "Cb": Cbn,

            # Lab
            "LAB_L": Ln, "LAB_a": an, "LAB_b": bn,

            # LCh style
            "LCh_C": Cn, "LCh_sinH": sin_h01, "LCh_cosH": cos_h01,

            # Chromaticity & Opponent
            "r_chroma": r_chroma, "g_chroma": g_chroma,
            "Opp_O1": O1n, "Opp_O2": O2n, "Opp_O3": O3n,

            # Edges / local stats
            "GradMag": grad_mag, "Laplacian": lapn, "LocalStd": local_std,
        }
        return feats

    def compute_global_crop_box(self, folder_path, padding=1, bg_threshold=2):
        """
        Scan all images to determine the largest bounding box that includes all non-zero pixels.
        """
        image_paths = glob(os.path.join(folder_path, "*"))
        y_min_global, x_min_global = float('inf'), float('inf')
        y_max_global, x_max_global = 0, 0

        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                continue

            mask = np.any(image > bg_threshold, axis=-1)
            coords = np.argwhere(mask)
            if coords.size == 0:
                continue
    
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            y_min_global = min(y_min_global, y_min)
            x_min_global = min(x_min_global, x_min)
            y_max_global = max(y_max_global, y_max)
            x_max_global = max(x_max_global, x_max)

        # Apply padding (with bounds check)
        y_min_global = max(y_min_global - padding, 0)
        x_min_global = max(x_min_global - padding, 0)
        y_max_global = y_max_global + padding
        x_max_global = x_max_global + padding

        crop_box = {
            "y_min": int(y_min_global),
            "y_max": int(y_max_global),
            "x_min": int(x_min_global),
            "x_max": int(x_max_global),
            "padding": int(padding)
        }

        return crop_box

    def find_divisible_size(self, h, w):
        new_h = h - (h % self.pool_size)
        new_w = w - (w % self.pool_size)
        return new_h, new_w

    def apply_average_pooling(self, v_channel):
        h, w = v_channel.shape
        new_h, new_w = self.find_divisible_size(h, w)
        self.image_shape = (new_h // self.pool_size, new_w // self.pool_size)
        v_channel = v_channel[:new_h, :new_w]
        pooled = v_channel.reshape(new_h // self.pool_size, self.pool_size,
                                   new_w // self.pool_size, self.pool_size).mean(axis=(1, 3))
        return pooled

    def _compute_adaptive_crop(self, image, base_crop_box):
        """Re-center the stored crop box around the current foreground mask."""
        if base_crop_box is None:
            return None

        height, width = image.shape[:2]
        desired_height = min(base_crop_box["y_max"] - base_crop_box["y_min"], height)
        desired_width = min(base_crop_box["x_max"] - base_crop_box["x_min"], width)
        padding = base_crop_box.get("padding", 0)

        mask = np.any(image > self.bg_threshold, axis=-1)
        coords = np.argwhere(mask)

        if coords.size == 0:
            return base_crop_box

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        y_min = max(y_min - padding, 0)
        x_min = max(x_min - padding, 0)
        y_max = min(y_max + padding, height)
        x_max = min(x_max + padding, width)

        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2

        def clamp(center, length, max_length):
            start = int(round(center - length / 2))
            end = start + length
            if start < 0:
                start = 0
                end = min(length, max_length)
            if end > max_length:
                end = max_length
                start = max(0, end - length)
            # Ensure the slice has the requested size when possible
            if end - start < length and max_length >= length:
                start = max(0, min(start, max_length - length))
                end = min(max_length, start + length)
            return start, end

        y_min, y_max = clamp(center_y, desired_height, height)
        x_min, x_max = clamp(center_x, desired_width, width)

        return {
            "y_min": int(y_min),
            "y_max": int(y_max),
            "x_min": int(x_min),
            "x_max": int(x_max),
            "padding": int(padding)
        }

    def process_image(self, image_path, crop_box=None):
        image = cv2.imread(image_path)
        if image is None:
            if not self.quiet:
                logger.warning(f"Error loading image: {image_path}")
            return None
        active_crop_box = crop_box
        if crop_box and self.validation:
            active_crop_box = self._compute_adaptive_crop(image, crop_box)

        if active_crop_box:
            image = image[
                active_crop_box["y_min"]:active_crop_box["y_max"],
                active_crop_box["x_min"]:active_crop_box["x_max"]
            ]

        feature_maps = self.compute_feature_maps(image)

        # Decide which channels to use
        if isinstance(self.channels, str) and self.channels.upper() == "ALL":
            selected_keys = list(feature_maps.keys())
        else:
            selected_keys = [k for k in self.channels if k in feature_maps]
            unknown = [k for k in self.channels if k not in feature_maps]
            if unknown and not self.quiet:
                logger.warning(f"Unknown channels requested (ignored): {unknown}")

        if not selected_keys:
            return None

        # Pool + stack
        pooled_channels = []
        for key in selected_keys:
            pooled = self.apply_average_pooling(feature_maps[key])
            pooled_channels.append(pooled.reshape(-1, 1))

        combined = np.concatenate(pooled_channels, axis=1).astype(np.float32)

        if not self.quiet:
            mn, mx = float(combined.min()), float(combined.max())
            logger.info(f"Min/Max of normalized combined image ({len(selected_keys)} ch): {mn:.4f} - {mx:.4f}")

        return combined

    def process_images_in_folder(self, folder_path):
        image_paths = glob(os.path.join(folder_path, "*"))
        all_images_data = []

        if hasattr(self, 'zero_pruning') and self.zero_pruning:
            if self.crop_box is None:
                self.crop_box = self.compute_global_crop_box(folder_path, self.zero_pruning_padding, self.bg_threshold)
            if not self.quiet:
                logger.info(f"Using crop box: {self.crop_box} for folder: {folder_path}")

        for image_path in tqdm(image_paths, desc=f"Processing {Path(folder_path).name}", disable=self.quiet):
            image_data = self.process_image(image_path, self.crop_box)
            if image_data is not None:
                all_images_data.append(image_data)

        if not all_images_data:
            if not self.quiet:
                logger.warning(f"No valid images processed in {folder_path}.")
            return np.array([])

        all_images_data = np.array(all_images_data)
        if not self.quiet:
            logger.info(f"Processed {len(all_images_data)} images from {folder_path}, shape: {all_images_data.shape}")
        return all_images_data

    def load_and_label_data(self):
        data = []
        labels = []

        for idx, folder_path in enumerate(self.image_folders):
            images = self.process_images_in_folder(folder_path)
            if images.size == 0:
                continue
            num_images = images.shape[0]
            data.append(images)
            labels.extend([idx] * num_images)

        if not data:
            if not self.quiet:
                logger.warning("No images processed from any folder.")
            return None, None

        data = np.vstack(data)
        labels = np.array(labels)

        if not self.quiet:
            logger.info(f"Labels shape before one-hot encoding: {labels.shape}")

        if self.one_hot_encoding:
            num_classes = len(self.image_folders)
            labels = np.eye(num_classes)[labels]
            if not self.quiet:
                logger.info(f"Final one-hot labels shape: {labels.shape}")
        else:
            if not self.quiet:
                logger.info(f"Final raw label vector shape: {labels.shape}")

        if not self.quiet:
            logger.info(f"Final data shape: {data.shape}")
            logger.info(f"Final labels shape: {labels.shape}")

        return data, labels

    def save_data(self, output_dir):
        data, labels = self.load_and_label_data()
        if data is not None and labels is not None:
            output_file = output_dir / self.file_name
            if not self.quiet:
                logger.info(f"Image shape after pooling: {self.image_shape}")
            if self.one_hot_encoding:
                if not self.quiet:
                    logger.info(f"Data and labels saved to {output_file}")
                np.savez(output_file, data=data, labels=labels, shape=self.image_shape)
            else:
                if not self.quiet:
                        logger.info(f"Data saved to {output_file}")
                np.savez(output_file, data=data, shape=self.image_shape)

            if not self.validation:
                # === Save crop metadata if zero pruning was applied ===
                if hasattr(self, 'zero_pruning') and self.zero_pruning:
                    save_crop_preview(
                        self.image_folders,
                        self.crop_box,
                        output_dir
                    )
                    
                    metadata_file = output_dir / 'metadata' / "preprocessing.yaml"
                    metadata_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

                    # Step 1–2: Load existing data if the file exists
                    if metadata_file.exists():
                        with open(metadata_file, "r") as f:
                            data = yaml.safe_load(f) or {}
                    else:
                        data = {}

                    # Step 3: Update with new crop_box
                    data["crop_box"] = self.crop_box

                    # Step 4: Write back the updated data
                    with open(metadata_file, "w") as f:
                        yaml.dump(data, f)

                    if not self.quiet:
                        logger.info(f"📎 Updated crop metadata in {metadata_file}")
           

def run(input_dir, output_dir=None, config=None, quiet=False, validation=False):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pool_size = config.preprocessing.preprocessor.pool_size if config and hasattr(config.preprocessing.preprocessor, 'pool_size') else 4
    channels = config.preprocessing.preprocessor.channels if config and hasattr(config.preprocessing.preprocessor, 'channels') else ("H", "S", "V", "R", "G", "B")
    file_name = config.preprocessing.preprocessor.file_name if config and hasattr(config.preprocessing.preprocessor, 'file_name') else "out.npz"
    one_hot_encoding = config.model_training.one_hot_encoding if config and hasattr(config.model_training, "one_hot_encoding") else False
    zero_pruning = config.preprocessing.preprocessor.zero_pruning if config and hasattr(config.preprocessing.preprocessor, 'zero_pruning') else False
    zero_pruning_padding = config.preprocessing.preprocessor.zero_pruning_padding if config and hasattr(config.preprocessing.preprocessor, 'zero_pruning_padding') else False
    bg_threshold = config.preprocessing.preprocessor.bg_threshold if config and hasattr(config.preprocessing.preprocessor, 'bg_threshold') else 2
    crop_box = config.crop_box if config and hasattr(config, 'crop_box') else None

    if one_hot_encoding:
        # ✅ Multiple folders to process and label
        image_folders = [f for f in input_path.iterdir() if f.is_dir()]
        if not image_folders:
            raise ValueError(f"[✗] No subfolders found in {input_path} for one-hot encoding mode.")
        
        if not quiet:
            logger.info(f"🔍 Processing images from {len(image_folders)} folders for one-hot encoding...")
        
        processor = ImageDataProcessor(
            image_folders,
            pool_size=pool_size,
            channels=channels,
            file_name=file_name,
            quiet=quiet,
            one_hot_encoding=True,
            zero_pruning=zero_pruning,
            zero_pruning_padding=zero_pruning_padding,
            bg_threshold=bg_threshold,
            crop_box=crop_box,
            validation=validation 
        )
        processor.save_data(output_dir)
    
    else:
        # ✅ Directly process this single folder of images
        if not quiet:
            logger.info(f"🔍 Processing images from folder: {input_path.name}")

        processor = ImageDataProcessor(
            [input_path],
            pool_size=pool_size,
            channels=channels,
            file_name=file_name,
            quiet=quiet,
            one_hot_encoding=False,
            zero_pruning=zero_pruning,
            zero_pruning_padding=zero_pruning_padding,
            bg_threshold=bg_threshold,
            crop_box=crop_box,
            validation=validation
        )
        processor.save_data(output_dir)
