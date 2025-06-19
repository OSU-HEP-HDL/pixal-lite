import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')
from pixal.modules import preprocessing as mod
from pathlib import Path
from rembg import remove
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

logger = logging.getLogger("pixal")

def process_image(img_file, output_path, target_size=None, rename_images=False, index=None, one_hot_encoding=False):
    try:
        with Image.open(img_file) as img:
            img = img.convert("RGBA")
            logger.info(f"Removing background for image: {img}")
            output = remove(img)

            if target_size is None:
                target_size = output.size

            output = output.resize(target_size, Image.LANCZOS)

            if rename_images and index is not None:
                parent_name = img_file.parent.name
                new_name = f"{parent_name}_{index:03}_no_bg.png"
            else:
                new_name = f"{img_file.stem}_no_bg.png"

            if one_hot_encoding:    
                output_path = output_path / img_file.parent.name
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / new_name
            else:
                output_file = output_path / new_name
            logger.info(f"Saving image with background removed as: {output_file}")
            output.save(output_file)

        return output_file.name, target_size

    except Exception as e:
        return f"Error: {img_file.name} ({e})", target_size

def remove_backgrounds(input_folder, output_folder, max_workers=4, quiet=False, rename_images=False,one_hot_encoding=False):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        logger.error(f"Input folder does not exist: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    supported_extensions = ('.jpg', '.jpeg', '.png', '.webp')

    if one_hot_encoding:
        subdirs = [p for p in input_path.iterdir() if p.is_dir()]
    else:
        subdirs = [input_path]  # single-type mode

    for folder in subdirs:
        image_files = [f for f in folder.iterdir() if f.suffix.lower() in supported_extensions]
        if not image_files:
            logger.warning(f"No images found in {folder}")
            continue
        
        if one_hot_encoding:    
            output_folder_this = output_path / folder.name
        else:
            output_folder_this = output_path
        output_folder_this.mkdir(parents=True, exist_ok=True)

        target_size = None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_image, f, output_path, target_size, rename_images, idx, one_hot_encoding): f
                for idx, f in enumerate(image_files)
            }

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"Removing backgrounds in '{folder.name}'", disable=quiet):
                result, size = future.result()
                if isinstance(result, str) and result.startswith("Error"):
                    logger.warning(result)
                elif target_size is None:
                    target_size = size  # Set once

                    
def run(input_folder, output_folder, config=None, quiet=False,rename_images=False):
    max_workers = config.preprocessing.remove_background.max_workers if config and hasattr(config, 'remove_background') else 4
    rename_images = config.preprocessing.rename_images if config and hasattr(config, 'rename_images') else False
    one_hot_encoding = config.model_training.one_hot_encoding if config and hasattr(config, 'one_hot_encoding') else False
    remove_backgrounds(input_folder, output_folder, max_workers,quiet=quiet,rename_images=rename_images,one_hot_encoding=one_hot_encoding)

