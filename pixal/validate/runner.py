import logging
from pathlib import Path
from pixal.preprocessing import remove_background, align_images, imagePreprocessor
from pixal.modules.config_loader import load_config, resolve_path, load_and_merge_configs, _dict_to_namespace, resolve_parent_inserted_path
import subprocess
import sys

def run_validation(input_dir, config=None, quiet=False):
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
    input_dir = Path(input_dir)
    
    log_path = resolve_path(path_config.validate_log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path / "validation.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger("pixal")

    if not quiet:
        logger.info(f"📁 Logging all preprocessing and validation steps to {log_path}")
    
    if model_config.one_hot_encoding:

        config_path = str(Path(resolve_path(path_config.component_model_path)) / type_folder.name / "metadata")
        config = load_and_merge_configs(config_path)
        config = _dict_to_namespace(config)
         
        reference_dir = resolve_path(path_config.component_model_path) / resolve_path(path_config.general_aligned_images_path)
        base_path = resolve_path(path_config.component_validate_path)
        model_dir = resolve_path(path_config.model_path)

         # Dynamically resolve per-type paths
        bg_removed_dir = base_path / resolve_path(path_config.general_remove_background_path)
        aligned_dir = base_path / resolve_path(path_config.general_aligned_images_path)
        reference_dir = resolve_path(reference_dir)  # assume same reference
        metric_dir = base_path / resolve_path(path_config.general_aligned_metrics_path)
        npz_dir = base_path 

        for d in [bg_removed_dir, aligned_dir, metric_dir, npz_dir]:
            d.mkdir(parents=True, exist_ok=True)

        remove_background.run(input_dir, bg_removed_dir, config=config, quiet=quiet)
        align_images.run(bg_removed_dir, aligned_dir, reference_dir, metric_dir, config=config, quiet=quiet, detect=True)
        imagePreprocessor.run(aligned_dir, npz_dir, config=config, quiet=quiet,validation=True)

        args = [
                sys.executable,
                "pixal/validate/validate_one_model.py",
                "--npz", str(npz_dir),
                "--model", str(model_dir),
                "--metrics", str(metric_dir),
                "--config", str(config_path),
                "--preprocess"
        ]
        if model_config.one_hot_encoding:
            args.append("--one_hot")

        subprocess.run(args)

    else:
        # Loop over each type folder (image categories)
        for type_folder in input_dir.iterdir():
            
            if not type_folder.is_dir():
                continue

            logger.info(f"🔍 Running validation for {type_folder.name}")

            config_path = resolve_parent_inserted_path(path_config.metadata_path, type_folder.name,1)

            config = load_and_merge_configs(config_path)
            config = _dict_to_namespace(config)

            reference_dir = resolve_parent_inserted_path(path_config.reference_path, type_folder.name,1) 
            print(f"Reference path: {reference_dir}")

            model_dir = resolve_parent_inserted_path(path_config.model_path, type_folder.name,1) 
            print(f"Model path: {model_dir}")

            base_path = resolve_path(path_config.component_validate_path) / type_folder.name
           
            
            # Dynamically resolve per-type paths
            bg_removed_dir = base_path / resolve_path(path_config.general_remove_background_path)
            aligned_dir = base_path / resolve_path(path_config.general_aligned_images_path)
            reference_dir = resolve_path(reference_dir)  # assume same reference
            metric_dir = base_path / resolve_path(path_config.general_aligned_metrics_path)
            npz_dir = base_path 

            for d in [bg_removed_dir, aligned_dir, metric_dir, npz_dir]:
                d.mkdir(parents=True, exist_ok=True)

            remove_background.run(type_folder, bg_removed_dir, config=config, quiet=quiet)
            align_images.run(bg_removed_dir, aligned_dir, reference_dir, metric_dir, config=config, quiet=quiet, detect=True)
            imagePreprocessor.run(aligned_dir, npz_dir, config=config, quiet=quiet,validation=True)

            args = [
                sys.executable,
                "pixal/validate/validate_one_model.py",
                "--npz", str(npz_dir),
                "--model", str(model_dir),
                "--metrics", str(metric_dir),
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
        config_path = str(Path(resolve_path(path_config.component_model_path)) / type_folder.name / "metadata")
        config = load_and_merge_configs(config_path)
        config = _dict_to_namespace(config)
        
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
                "--config", str(config_path),
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