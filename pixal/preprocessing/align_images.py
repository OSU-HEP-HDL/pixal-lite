import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pixal.modules import preprocessing as mod
import logging
import difflib

logger = logging.getLogger("pixal")

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)



def align_images(image_paths, save_dir, reference_dir, metric_dir, knn_ratio=0.55, npts=10, ransac_thresh=7.0, save_metrics=False, draw_matches=False, save_overlays=False, quiet=False, detect=False,MIN_SCORE_THRESHOLD=0.5, MAX_MSE_THRESHOLD=10.0, MIN_GOOD_MATCHES=20):
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if detect:
    
        images = [cv2.imread(str(p)) for p in image_paths]
        ref_images = [cv2.imread(str(q)) for q in reference_dir]
        if any(img is None for img in images):
            raise ValueError("One or more images could not be loaded.")
        prev_image = ref_images[0]
       
    else:
        images = [cv2.imread(str(p)) for p in image_paths[1:]]
        ref_images = [cv2.imread(image_paths[0])]
        if any(img is None for img in images):
            raise ValueError("One or more images could not be loaded.")
        
       # prev_image = images[0]
        
    #prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    ##prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)
    results = []

    # sets starting image. m = 1 if detect is False, m = 0 if detect is True
    m = 1 if not detect else 0

    if len(images) <= m:
        if not quiet:
            logger.warning(f"Not enough images to align (found {len(images)}). Skipping alignment.")
        return  # or continue, depending on context

    for i in tqdm(range(m, len(images)), desc=f"Aligning images in {save_dir.name}", disable=quiet):
        curr_image = images[i]
        if curr_image is None:
            if not quiet:
                logger.warning(f"Error loading image: {image_paths[i]}")
            continue

        best_alignment = None
        best_metrics = {"score": -1, "mse": float("inf")}
        best_path = None
 
        for j, ref_image in enumerate(ref_images):
            prev_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
            prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)

            result = mod.get_src_pts(bf, sift, knn_ratio, curr_image, prev_des, prev_kp, npts, logger, return_matches=True)
            #match_result = mod.get_matching_src_pts(bf, sift, knn_ratio, curr_image, prev_des, prev_kp, npts, logger, return_matches=True)

            if result is None:
                continue
            
            src_npts, dst_npts, matches, curr_kp = result

            if draw_matches:
                limited_matches = sorted(matches, key=lambda m: m.distance)[:10]  # take the best 10 matches

                match_img = cv2.drawMatches(
                    curr_image, curr_kp,
                    ref_image, prev_kp,
                    limited_matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                match_img_path = metric_dir / Path(save_dir.name) / "matches"
                match_img_path.mkdir(parents=True, exist_ok=True)
                match_file_name = f"match_{i}_{j}.png"
                cv2.imwrite(str(match_img_path / match_file_name), match_img)
                logger.info(f"🔍 Saved keypoint match diagnostic: {match_img_path}")

            homography_matrix, mask = cv2.findHomography(src_npts, dst_npts, cv2.RANSAC, ransac_thresh)
            if homography_matrix is not None:
                        inliers = np.sum(mask)
                        logger.info(f"   → Found {inliers} inliers for reference image")
                        if inliers < MIN_GOOD_MATCHES:
                            continue  # Skip this reference if too few inliers

                        height, width = ref_image.shape[:2]
                        transformed_image = cv2.warpPerspective(curr_image, homography_matrix, (width, height))
                        img_name = Path(image_paths[i]).name.replace("no_bg", "aligned")
                        transformed_path = save_dir / img_name
                        cv2.imwrite(str(transformed_path), transformed_image)

                        score, mse = mod.alignment_score(str(image_paths[0]), str(transformed_path))

                        if not quiet:
                            logger.info(f"   → Inliers: {inliers}, Score: {score:.3f}, MSE: {mse:.3f}")

                        # Early exit if a good-enough match is found
                        if score >= MIN_SCORE_THRESHOLD and mse <= MAX_MSE_THRESHOLD:
                            best_alignment = transformed_image
                            best_metrics = {"score": score, "mse": mse,"index": j}
                            best_path = transformed_path
                            break  # No need to test more references

                        # Track best fallback candidate
                        if score > best_metrics["score"] and mse < best_metrics["mse"]:
                            ind = j
                            best_alignment = transformed_image
                            best_metrics = {"score": score, "mse": mse, "index": ind}
                            best_path = transformed_path

        if best_alignment is not None and best_path is not None:
            cv2.imwrite(str(best_path), best_alignment)
            logger.info(f"✅ Saved: {best_path}")
            results.append({
                "image": best_path,
                "score": best_metrics["score"],
                "mse": best_metrics["mse"],
                "inliers": inliers,
                "inlier_ratio": inliers / len(mask),
                "index": best_metrics["index"]
            })
        else:
            logger.warning(f"❌ Could not find a suitable alignment for {image_paths[i]}. Exiting.")
            exit(1)
    if save_metrics:
        
        metric_dir = Path(metric_dir) / Path(save_dir.name)
        metric_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Starting metric plotting")
        mod.save_alignment_metrics_csv(results,metric_dir)
        mod.plot_alignment_metrics(results,metric_dir)
        if detect:
            mod.stack_intensity_heatmap(save_dir,metric_dir,reference_dir[results[0]["index"]])
        else:
            mod.stack_intensity_heatmap(save_dir,metric_dir)
        
        if save_overlays:
            metric_dir = metric_dir / "overlay_diagnostics"
            metric_dir.mkdir(parents=True, exist_ok=True)
            reference_images = None
            if reference_dir:
                reference_images = reference_dir[results[0]["index"]]
            mod.save_overlay_diagnostics(save_dir,metric_dir,reference_images,logger)

def run(input_dir, output_dir=None, reference_dir=None, metric_dir=None, config=None, quiet=False, detect=False):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    subdirs = [p for p in input_path.iterdir() if p.is_dir()]
    if not subdirs:
        subdirs = [input_path]
    
    if reference_dir:
        reference_subdirs = [p for p in reference_dir.iterdir() if p.is_dir()]
        if not reference_subdirs:
            reference_subdirs = [reference_dir]
    
    knn_ratio = config.preprocessing.alignment.knn_ratio if config and hasattr(config.preprocessing, 'alignment') else 0.55
    npts = config.preprocessing.alignment.number_of_points if config and hasattr(config.preprocessing, 'alignment') else 10
    ransac_thresh = config.preprocessing.alignment.ransac_threshold if config and hasattr(config.preprocessing, 'alignment') else 7.0
    save_metrics = config.preprocessing.save_metrics if config and hasattr(config.preprocessing, 'save_metrics') else False
    MIN_SCORE_THRESHOLD = config.preprocessing.alignment.MIN_SCORE_THRESHOLD if config and hasattr(config.preprocessing, 'alignment') else 0.5
    MAX_MSE_THRESHOLD = config.preprocessing.alignment.MAX_MSE_THRESHOLD if config and hasattr(config.preprocessing, 'alignment') else 10.0
    MIN_GOOD_MATCHES = config.preprocessing.alignment.MIN_GOOD_MATCHES if config and hasattr(config.preprocessing, 'alignment') else 20
    one_hot_encoding = config.model_training.one_hot_encoding if config and hasattr(config.model_training, 'one_hot_encoding') else False
    draw_matches = config.preprocessing.draw_matches if config and hasattr(config.preprocessing, 'draw_matches') else False
    save_overlays = config.preprocessing.save_overlays if config and hasattr(config.preprocessing, 'save_overlays') else False
    
    # Aligns images for preprocessing
    if not reference_dir:
        for folder in subdirs:
            image_files = sorted([f for f in folder.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            if not image_files:
                if not quiet:
                    logger.warning(f"No images found in {folder}")
                continue
            if one_hot_encoding:
                sub_output = output_path / folder.name
            else:
                sub_output = output_path
            align_images(image_files, sub_output, reference_dir, metric_dir, knn_ratio, npts, ransac_thresh, save_metrics, draw_matches, save_overlays, quiet, detect, MIN_SCORE_THRESHOLD, MAX_MSE_THRESHOLD, MIN_GOOD_MATCHES)
            if not quiet:
                logger.info(f"📁 Completed alignment for folder: {folder.name}")
    
    # Aligns images for validation and detection
    else:
        reference_subdirs = mod.rearrange_by_similarity(subdirs, reference_subdirs)
        for folder1, folder2 in zip(subdirs,reference_subdirs):
            image_files = sorted([f for f in folder1.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            reference_files = sorted([d for d in folder2.iterdir() if d.suffix.lower() in ['.png', '.jpg', '.jpeg']]) # how to pair input to refernce?

            if not image_files:
                if not quiet:
                    logger.warning(f"No images found in {folder1}")
                continue
            if one_hot_encoding:
                sub_output = output_path / folder1.name
            else:
                sub_output = output_path
            align_images(image_files, sub_output, reference_files, metric_dir, knn_ratio, npts, ransac_thresh, save_metrics, draw_matches, save_overlays, quiet, detect, MIN_SCORE_THRESHOLD, MAX_MSE_THRESHOLD, MIN_GOOD_MATCHES)
            if not quiet:
                logger.info(f"📁 Completed alignment for folder: {folder1.name}")