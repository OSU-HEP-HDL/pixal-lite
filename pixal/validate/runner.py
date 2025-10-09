import logging
from pathlib import Path
from pixal.preprocessing import remove_background, align_images, imagePreprocessor
from pixal.modules.config_loader import load_config, resolve_path, load_and_merge_configs, _dict_to_namespace,extract_component_name, list_type_dirs
import subprocess
import sys
import os
from datetime import datetime

def run_validation(input_dir, output_dir, quiet=False):
    # Get component paths to retrieve model + metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("./logs")
    log_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
    filename=log_path / f"{timestamp}.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
            
    logger = logging.getLogger("pixal")
    logger.info(f"Logging all preprocessing and validation steps to {log_path / f'{timestamp}.log'}")
    input_path = Path(input_dir)
    component_model = extract_component_name(input_path)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Component model identified as: {component_model}")
    logger.info(f"Output directory: {output_dir}")

    #path_config = load_config("configs/paths.yaml")

    component_path = os.environ.get("MODEL_DIR", "/mount/machine_learning/models")
    component_path = Path(component_path) / component_model
    logger.info(f"Component path: {component_path}")

    # Step 1: choose a real type folder under component_path
    subdirs = list_type_dirs(component_path)
    if subdirs:
        type_folder = subdirs[0]   # first valid one
    else:
        type_folder = None  # or handle “no types found” case

    model_config = load_config(component_path / type_folder / "metadata" / "model_training.yaml").get("model_training", {})
    logger.info(f"Model configuration loaded: {model_config}")

    input_dir = Path(input_dir)

    if model_config.one_hot_encoding:
        print("One hot encoding model detected")
    else:
        # Loop over each type folder (image categories) in input_dir
        for type_folder in list_type_dirs(input_dir):
            logger.info(f"🔍 Running validation for {type_folder.name}")

            base_model_path = component_path / type_folder.name
            base_input_path = Path(input_path) / type_folder.name
            base_output_path = Path(output_dir) / component_model / type_folder.name
            base_output_path.mkdir(parents=True, exist_ok=True)

            output_preprocess = base_output_path / "preprocessing"
            output_preprocess.mkdir(parents=True, exist_ok=True)
            output_validate = base_output_path / "validation"
            output_validate.mkdir(parents=True, exist_ok=True)

            logger.info(f"Base Model path: {base_model_path}")
            logger.info(f"Base output path: {base_output_path}")
            logger.info(f"Output preprocess path: {output_preprocess}")
            logger.info(f"Output validate path: {output_validate}")

            config_path = base_model_path / "metadata"
            logger.info(f"Config path: {config_path}")
            config = load_and_merge_configs(config_path)
            config = _dict_to_namespace(config)
            path_config = load_config(config_path / "paths.yaml")
            reference_dir = base_model_path / resolve_path(config.general_aligned_images_path)
            logger.info(f"Reference dir: {reference_dir}")
            
            

            #model_dir = resolve_path(path_config.component_model_path) / type_folder.name / resolve_path(path_config.model_path.model) 
            model_dir = base_model_path / "model"
            logger.info(f"Model dir: {model_dir}")
            
            # Dynamically resolve per-type paths
            bg_removed_dir = output_preprocess / resolve_path(config.general_remove_background_path)
            logger.info(f"Background removed dir: {bg_removed_dir}")
            aligned_dir = base_model_path / resolve_path(config.general_aligned_images_path)
            logger.info(f"Aligned dir: {aligned_dir}")
            reference_dir = resolve_path(reference_dir)  # assume same reference
            #output_dir =  Path(output_dir) / "figures" #base_path / resolve_path(path_config.general_aligned_metrics_path)
            npz_dir = base_output_path 
            logger.info(f"NPZ dir: {npz_dir}")

            #for d in [bg_removed_dir, aligned_dir, output_dir, npz_dir]:
            #    d.mkdir(parents=True, exist_ok=True)

            preprocess_input = bg_removed_dir
            logger.info(f"Preprocess input dir: {preprocess_input}")
            logger.info(f"Starting preprocessing steps for {type_folder.name}...")
            logger.info(f"Running background removal step for {base_input_path}")
            remove_background.run(base_input_path, bg_removed_dir, config=config, quiet=quiet)
            if config.preprocessing.alignment.apply_alignment:
                logger.info(f"Running alignment step for {type_folder.name}...")
                align_images.run(bg_removed_dir, aligned_dir, reference_dir, output_dir, config=config, quiet=quiet, detect=True)
                preprocess_input = aligned_dir
            logger.info(f"Running image preprocessing step for {type_folder.name}...")
            imagePreprocessor.run(preprocess_input, npz_dir, config=config, quiet=quiet,validation=True)

            logger.info(f"Preprocessing complete for {type_folder.name}.")
            logger.info(f"Starting validation step for {type_folder.name}...")
            args = [
                sys.executable,
                "pixal/validate/validate_one_model.py",
                "--npz", str(npz_dir),
                "--model", str(model_dir),
                "--metrics", str(output_validate),
                "--config", str(config_path),
                "--preprocess"
            ]
            if model_config.one_hot_encoding:
                args.append("--one_hot")

            subprocess.run(args)


def run_detection(config=None, quiet=False):
    path_config = load_config("configs/paths.yaml")

    component_path = resolve_path(path_config.component_model_path)
    parameter_path = resolve_path(path_config.metadata_path)

    # Step 1: Check if component_path has subdirs other than "metadata"
    subdirs = [d for d in component_path.iterdir() if d.is_dir() and d.name != "metadata"]

    # Step 2: If such a subdir exists, use it to build parameter_path
    if subdirs:
        type_folder = subdirs[0]  
        parameter_path = type_folder 

    model_config = load_config(parameter_path / "metadata" / "model_training.yaml").get("model_training", {})

    input_dir = resolve_path(path_config.component_validate_path)

    if model_config.one_hot_encoding:
        base_path = resolve_path(path_config.component_validate_path)
        model_dir = resolve_path(path_config.model_path)
        metric_dir = base_path / resolve_path(path_config.general_aligned_metrics_path)
        npz_dir = base_path 

        for d in [metric_dir, npz_dir]:
            d.mkdir(parents=True, exist_ok=True)

        args = [
                sys.executable,
                "pixal/validate/validate_one_model.py",
                "--npz", str(npz_dir),
                "--model", str(model_dir),
                "--metrics", str(metric_dir),
                "--config", "configs/parameters.yaml",
                "--preprocess"
        ]
        if config.one_hot_encoding:
            args.append("--one_hot")

        subprocess.run(args)
    
    else:
        for type_folder in input_dir.iterdir():
            if type_folder.name == "logs":
                continue
            
            config_path = str(Path(resolve_path(path_config.component_model_path)) / type_folder.name / "metadata")
            config = load_and_merge_configs(config_path)
            config = _dict_to_namespace(config)

            if config.model_training.one_hot_encoding:
                base_path = resolve_path(path_config.component_validate_path)
                model_dir = resolve_path(path_config.model_path)
            else:
                base_path = resolve_path(path_config.component_validate_path) / type_folder.name
                model_dir = resolve_path(path_config.component_model_path) / type_folder.name / resolve_path(path_config.model_path.model) 
        
            metric_dir = base_path / resolve_path(path_config.general_aligned_metrics_path)
            metric_dir.mkdir(parents=True, exist_ok=True)
            
            npz_dir = base_path 

            # Set up logging
            log_path = resolve_path(path_config.validate_log_path)
            log_path.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                filename=log_path / "detect.log",
                filemode="w",
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            file_handler = logging.FileHandler(log_path / "detect.log", mode='a')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            file_handler.setFormatter(formatter)

            logger = logging.getLogger("pixal")
            logger.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            
            if not quiet:
                logger.info(f"📁 Logging all preprocessing and validation steps to {log_path}")

            args = [
                    sys.executable,
                    "pixal/validate/validate_one_model.py",
                    "--npz", str(npz_dir),
                    "--model", str(model_dir),
                    "--metrics", str(metric_dir),
                    "--config",  str(config_path)
                ]
            if config.one_hot_encoding:
                args.append("--one_hot")

            subprocess.run(args)