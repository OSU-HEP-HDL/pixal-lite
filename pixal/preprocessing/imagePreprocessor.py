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

    def process_image(self, image_path, crop_box=None):
        image = cv2.imread(image_path)
        if image is None:
            if not self.quiet:
                logger.warning(f"Error loading image: {image_path}")
            return None
        if crop_box:
            image = image[
                crop_box["y_min"]:crop_box["y_max"],
                crop_box["x_min"]:crop_box["x_max"]
            ]


        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        channel_map = {
            "H": hsv_image[:, :, 0],
            "S": hsv_image[:, :, 1],
            "V": hsv_image[:, :, 2],
            "R": rgb_image[:, :, 0],
            "G": rgb_image[:, :, 1],
            "B": rgb_image[:, :, 2],
        }

        pooled_channels = []
        for ch in self.channels:
            if ch in channel_map:
                pooled = self.apply_average_pooling(channel_map[ch])
                # Normalize each channel properly
                if ch == "H":
                    pooled = pooled / 179.0  # H is in range [0,179]
                else:
                    pooled = pooled / 255.0  # S, V, R, G, B are in [0,255]
                pooled_channels.append(pooled.reshape(-1, 1))
            else:
                if not self.quiet:
                    logger.warning(f"Warning: Unknown channel '{ch}' requested.")

        if not pooled_channels:
            return None

        combined = np.concatenate(pooled_channels, axis=1)
        if not self.quiet:
            logger.info(f"Min/Max of normalized combined image: {combined.min()} - {combined.max()}")
            
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