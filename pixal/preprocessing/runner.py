import logging
import yaml
from ruamel.yaml import YAML
from pathlib import Path
from pixal.preprocessing import remove_background, align_images, imagePreprocessor
from pixal.modules.config_loader import load_config, resolve_path, resolve_parent_inserted_path
def run_preprocessing(input_dir, config=None, quiet=False):
    ruyaml = YAML()
    with open("configs/paths.yaml", "r") as f:
        paths = ruyaml.load(f)
    path_config = load_config("configs/paths.yaml")
    input_path = Path(input_dir)
    

    # Load the original YAML
    with open("configs/parameters.yaml", "r") as infile:
        full_config = yaml.safe_load(infile)
     # Extract the 'preprocessing' section
    preprocessing_section = full_config.get("preprocessing", {})
    plotting_section = full_config.get("plotting", {})
    
    
    if config.model_training.one_hot_encoding:
        # 📦 Standard one-hot preprocessing — all folders together
        metric_dir = resolve_path(path_config.aligned_metrics_path)
        metric_dir.mkdir(parents=True, exist_ok=True)

        bg_removed_dir = resolve_path(path_config.remove_background_path)
        bg_removed_dir.mkdir(parents=True, exist_ok=True)

        aligned_dir = resolve_path(path_config.aligned_images_path)
        aligned_dir.mkdir(parents=True, exist_ok=True)

        npz_dir = resolve_path(path_config.component_model_path)
        log_path = resolve_path(path_config.log_path)
        log_path.mkdir(parents=True, exist_ok=True)

        metadata_path = resolve_path(path_config.metadata_path)
        metadata_path.mkdir(parents=True, exist_ok=True)
        
        # Save to new YAML file
        with open(metadata_path / "preprocessing.yaml", "w") as outfile:
            yaml.dump({"preprocessing": preprocessing_section}, outfile, default_flow_style=False)
        with open(metadata_path / "plotting.yaml", "w") as outfile:
            yaml.dump({"plotting": plotting_section}, outfile, default_flow_style=False)
        
        # Logging setup
        logging.basicConfig(
            filename=log_path / "preprocessing.log",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        logger = logging.getLogger("pixal")
        if not quiet:
            logger.info(f"📁 Logging all preprocessing steps to {log_path}")

        reference_dir = None
        remove_background.run(input_dir, bg_removed_dir, config=config, quiet=quiet)
        align_images.run(bg_removed_dir, aligned_dir, reference_dir, metric_dir, config=config, quiet=quiet)
        imagePreprocessor.run(aligned_dir, npz_dir, config=config, quiet=quiet)

    else:
        # 🔁 Independent preprocessing per folder
        for subfolder in input_path.iterdir():
            if not subfolder.is_dir():
                continue

            folder_name = subfolder.name
            output_root = resolve_path(path_config.component_model_path) / folder_name
            output_root.mkdir(parents=True, exist_ok=True)

            metric_dir = resolve_parent_inserted_path(path_config.aligned_metrics_path, folder_name,2)
            bg_removed_dir = resolve_parent_inserted_path(path_config.remove_background_path, folder_name, 2)
            aligned_dir = resolve_parent_inserted_path(path_config.aligned_images_path, folder_name, 2)
            npz_dir = resolve_parent_inserted_path(path_config.component_model_path, folder_name, 0)
            log_path = resolve_parent_inserted_path(path_config.log_path, folder_name, 1)
            metadata_path = resolve_parent_inserted_path(path_config.metadata_path, folder_name, 1)

            for d in [metric_dir, bg_removed_dir, aligned_dir, npz_dir, log_path,metadata_path]:
                d.mkdir(parents=True, exist_ok=True)
            
            # Save to new YAML file
            with open(metadata_path / "preprocessing.yaml", "w") as outfile:
                yaml.dump({"preprocessing": preprocessing_section}, outfile, default_flow_style=False)
            with open(metadata_path / "plotting.yaml", "w") as outfile:
                yaml.dump({"plotting": plotting_section}, outfile, default_flow_style=False)
            with open(metadata_path / "paths.yaml", "w") as outfile:
                ruyaml.dump(paths, outfile)
                
            # Reconfigure logger for each folder
            log_file = log_path / "preprocessing.log"
            logger = logging.getLogger("pixal")

            # Remove existing handlers
            if logger.hasHandlers():
                logger.handlers.clear()

            # Add new file handler
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)

            if not quiet:
                logger.info(f"📁 Logging preprocessing for {folder_name} to {log_path}")

            reference_dir = None
            preprocess_input = bg_removed_dir
            remove_background.run(subfolder, bg_removed_dir, config=config, quiet=quiet)
            if config.preprocessing.alignment.apply_alignment:
                align_images.run(bg_removed_dir, aligned_dir, reference_dir, metric_dir, config=config, quiet=quiet)
                preprocess_input = aligned_dir
            imagePreprocessor.run(preprocess_input, npz_dir, config=config, quiet=quiet)


def run_remove_background(input_dir, config=None, quiet=False):
    path_config = load_config("configs/paths.yaml")
    input_path = Path(input_dir)

    if config.one_hot_encoding:
        output_dir = resolve_path(path_config.remove_background_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = resolve_path(path_config.log_path)
        log_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=log_path / "preprocessing.log",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        logger = logging.getLogger("pixal")
        if not quiet:
            logger.info(f"📁 Logging background removal to {log_path}")

        remove_background.run(input_dir, output_dir, config=config, quiet=quiet)

    else:
        for subfolder in input_path.iterdir():
            if not subfolder.is_dir():
                continue
            folder_name = subfolder.name
            output_root = resolve_path(path_config.component_model_path) / folder_name
            output_root.mkdir(parents=True, exist_ok=True)

            bg_removed_dir = resolve_parent_inserted_path(path_config.remove_background_path, folder_name, 2)
            log_path = resolve_parent_inserted_path(path_config.log_path, folder_name, 1)

            bg_removed_dir.mkdir(parents=True, exist_ok=True)
            log_path.mkdir(parents=True, exist_ok=True)
            
            # Reconfigure logger for each folder
            log_file = log_path / "preprocessing.log"
            logger = logging.getLogger("pixal")

            # Remove existing handlers
            if logger.hasHandlers():
                logger.handlers.clear()

            # Add new file handler
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)

            if not quiet:
                logger.info(f"📁 Logging preprocessing for {folder_name} to {log_path}")

            remove_background.run(subfolder, bg_removed_dir, config=config, quiet=quiet)

def run_align_images(input_dir, config=None, quiet=False):
    path_config = load_config("configs/paths.yaml")
    input_path = Path(input_dir)

    if config.one_hot_encoding:
        output_dir = resolve_path(path_config.aligned_images_path)
        metric_dir = resolve_path(path_config.aligned_metrics_path)
        log_path = resolve_path(path_config.log_path)

        for d in [output_dir, metric_dir, log_path]:
            d.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=log_path / "alignment.log",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        logger = logging.getLogger("pixal")
        if not quiet:
            logger.info(f"📁 Logging image alignment steps to {log_path}")

        align_images.run(input_dir, output_dir, metric_dir, config=config, quiet=quiet)

    else:
        for subfolder in input_path.iterdir():
            if not subfolder.is_dir():
                continue
            folder_name = subfolder.name
            output_root = resolve_path(path_config.component_model_path) / folder_name
            aligned_dir = resolve_parent_inserted_path(path_config.aligned_images_path, folder_name, 2)
            metric_dir = resolve_parent_inserted_path(path_config.aligned_metrics_path, folder_name,2)
            log_path = resolve_parent_inserted_path(path_config.log_path, folder_name, 1)

            for d in [aligned_dir, metric_dir, log_path]:
                d.mkdir(parents=True, exist_ok=True)

            # Reconfigure logger for each folder
            log_file = log_path / "alignment.log"
            logger = logging.getLogger("pixal")

            # Remove existing handlers
            if logger.hasHandlers():
                logger.handlers.clear()

            # Add new file handler
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)

            if not quiet:
                logger.info(f"📁 Logging preprocessing for {folder_name} to {log_path}")

            align_images.run(subfolder, output_dir, metric_dir, config=config, quiet=quiet)


def run_imagePreprocessor(config=None, quiet=False):
    path_config = load_config("configs/paths.yaml")
    input_path = resolve_path(path_config.aligned_images_path)

    if config.one_hot_encoding:
        output_dir = resolve_path(path_config.component_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        log_path = resolve_path(path_config.log_path)
        log_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=log_path / "preprocessing.log",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        logger = logging.getLogger("pixal")
        if not quiet:
            logger.info(f"📁 Logging all imagePreprocessor steps to {log_path}")

        imagePreprocessor.run(input_path, output_dir, config=config, quiet=quiet)

    else:
        for folder in input_path.iterdir():
            if not folder.is_dir():
                continue
            folder_name = folder.name
            output_root = resolve_path(path_config.component_model_path) / folder_name
            output_dir = output_root / "component_model"
            log_path = output_root / "logs"

            for d in [output_dir, log_path]:
                d.mkdir(parents=True, exist_ok=True)

            logging.basicConfig(
                filename=log_path / "preprocessing.log",
                filemode="w",
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            logger = logging.getLogger("pixal")
            if not quiet:
                logger.info(f"📁 Logging preprocessing for {folder_name} to {log_path}")

            imagePreprocessor.run(folder, output_dir, config=config, quiet=quiet)
