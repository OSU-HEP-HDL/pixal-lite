from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from glob import glob

import difflib

logger = logging.getLogger("pixal")

from pathlib import Path

def rename_file_to_parent_dir(file_path):
    """
    Renames a file to match the name of its parent directory.
    Keeps the original file extension.

    Parameters:
        file_path (str or Path): The full path to the file.
    
    Returns:
        Path: The new file path after renaming.
    """
    file_path = Path(file_path)
    parent_dir_name = file_path.parent.name
    new_file_name = parent_dir_name + file_path.suffix
    new_file_path = file_path.with_name(new_file_name)

    file_path.rename(new_file_path)  # Actually renames the file on disk
    return new_file_path


def similarity_score(a, b):
    """Return a similarity ratio between two strings."""
    return difflib.SequenceMatcher(None, a, b).ratio()

def rearrange_by_similarity(reference_list, target_list):
    """
    Rearranges target_list based on similarity to reference_list.
    
    Each item in target_list is scored by its best match in reference_list,
    then the list is sorted by descending similarity.
    """
    scored_targets = []
    target_list = [Path(p) for p in target_list]    
    reference_list = [Path(p) for p in reference_list]

    for target in target_list:
        max_score = max(similarity_score(target.name, ref.name) for ref in reference_list)
        scored_targets.append((target, max_score))
    
    # Sort by similarity score descending
    sorted_targets = sorted(scored_targets, key=lambda x: x[1], reverse=True)
    
    return [item[0] for item in sorted_targets]


def get_equally_spaced_indices(n, start, end):
    # Generate n equally spaced indices from start to end
    indices = np.linspace(start, end, num=n, dtype=int)
    return indices

def get_score(pix1,pix2):
    # We are comparing against the first image, so we'll take pix1 as the true value
    pix1 = np.array(pix1)
    pix2 = np.array(pix2)
    t_values = []
    c_values = []
    
    t_values = pix1*2
    c_values = pix1+pix2

    for i in range(len(t_values)):
        for p in range(len(t_values[0])):
            if t_values[i][p] == 0:
                t_values[i][p] = 1
            if c_values[i][p] == 0:
                c_values[i][p] = 1
            
    f_values = np.array([a / b for a, b in zip(c_values, t_values)])

    # Calculate the average
    mean = np.mean(np.mean(f_values, axis=1),axis=0)
    
    return mean

def retrieve_grid_pixels(image, grid_size=(6, 6),offset=100):

    image = cv2.imread(image)
    
    if image is None:
        raise ValueError(f"Error loading image at {image}")

    # Get the height and width of the image
    height, width, _ = image.shape

        # Calculate valid height and width after the offset
    valid_height = height - offset
    valid_width = width - offset

    if valid_height < grid_size[0] or valid_width < grid_size[1]:
        raise ValueError("Image is too small for the specified grid size and offset.")

    # Calculate the spacing between pixels
    row_indices = np.linspace(offset, valid_height - 1, grid_size[0], dtype=int)
    col_indices = np.linspace(offset, valid_width - 1, grid_size[1], dtype=int)

    # Retrieve the pixels from the image
    pixels = []
    for r in row_indices:
        for c in col_indices:
            pixels.append(image[r, c])  # Append the pixel value

    return row_indices, col_indices

def mse(imageA, imageB):
    # Ensure images are of the same shape and type
    assert imageA.shape == imageB.shape, "Images must have the same dimensions."
    return np.mean((imageA - imageB) ** 2)

'''
def get_src_pts(bf, sift, knn_ratio,curr_image,prev_des,prev_kp, npts,logger=None,return_matches=False):
    # Convert to grayscale
        curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and descriptors in the current image
        curr_kp, curr_des = sift.detectAndCompute(curr_gray, None)
        
        # Apply KNN matching between the previous image and the current image
        matches = bf.knnMatch(curr_des, prev_des, k=2) #knnMatch

        good_matches = []
        for m, n in matches:
            if m.distance < knn_ratio * n.distance:
                good_matches.append(m)
        #logger.info(f"Number of good matches found: {len(good_matches)}")
        # Extract the matched keypoints
        if len(good_matches) > npts:  # At least 4 matches are required to compute the homography
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            #src_len = len(src_pts)
            #print(src_len)
            #print(src_pts.shape)
            
            #ind = get_equally_spaced_indices(npts,0,src_len-1)

            #src_npts = []
            #dst_npts = []
            #for n in range(len(ind)):
            #    src_npts.append((src_pts[ind[n]][0][0],src_pts[ind[n]][0][1]))
            #    dst_npts.append((dst_pts[ind[n]][0][0],dst_pts[ind[n]][0][1]))
            
            src_npts = np.array(src_pts) #npts
            dst_npts = np.array(dst_pts) #npts
            return src_npts, dst_npts, good_matches, curr_kp
'''
    
def get_src_pts(bf, sift, knn_ratio, curr_image, prev_des, prev_kp, npts, logger=None, return_matches=False):
    curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
    curr_kp, curr_des = sift.detectAndCompute(curr_gray, None)

    # FIXED: Swap curr_des and prev_des
    matches = bf.knnMatch(curr_des, prev_des, k=2)

    good_matches = [m for m, n in matches if m.distance < knn_ratio * n.distance]

    if len(good_matches) > npts:
        src_pts = np.float32([curr_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([prev_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        return src_pts, dst_pts, good_matches, curr_kp

def get_matching_src_pts(bf, sift, knn_ratio, curr_image, prev_des, prev_kp, npts, logger=None, return_matches=False):
    curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
    curr_kp, curr_des = sift.detectAndCompute(curr_gray, None)

    # FIXED: Swap curr_des and prev_des
    matches = bf.knnMatch(curr_des, prev_des, k=2)

    good_matches = [m for m, n in matches if m.distance < knn_ratio * n.distance]

    if len(good_matches) > npts:
        src_pts = np.float32([curr_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([prev_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        return src_pts, dst_pts, good_matches, curr_kp
    
def alignment_score(image1,image2):
    # This takes a set of pixels from both images and compares their values. 
    # The ideal scenario has all pixel values matching each other
    # This score tells us how close all chosen pixel values are to each other 
    # The idea is if all pixel values are almost exact, the image is aligned well

    # Let's get a grid of 9 pixels equally separated
    #row, col = retrieve_grid_pixels(img1)
    
    if isinstance(image1, str):
        image1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    if isinstance(image2, str):    
        image2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # Compute SSIM and MSE between the two images
    mse_score = mse(image1,image2)
    score = ssim(image1, image2, win_size=3)
    
    return score, mse_score


def plot_line_metric(values, title, ylabel, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(values, marker='o')
    plt.title(title)
    plt.xlabel("Image Index")
    plt.ylabel(ylabel)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_bar_metric(values, title, ylabel, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(values)), values)
    plt.title(title)
    plt.xlabel("Image Index")
    plt.ylabel(ylabel)
    plt.grid(True, axis='y')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_alignment_metrics(metrics, output_dir):
    """
    metrics: list of dicts with keys 'score', 'mse', 'inlier_ratio'
    output_dir: Path or str
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = [m["score"] for m in metrics]
    mses = [m["mse"] for m in metrics]
    inlier_ratios = [m["inlier_ratio"] for m in metrics]
    logger.info(f"Saving {output_dir}/alignment_score.png")
    plot_line_metric(
        scores,
        title="Alignment Score per Image",
        ylabel="Alignment Score",
        save_path=output_dir / "alignment_score.png"
    )
    logger.info(f"Saving {output_dir}/mse.png")
    plot_line_metric(
        mses,
        title="MSE per Image",
        ylabel="Mean Squared Error",
        save_path=output_dir / "mse.png"
    )
    logger.info(f"Saving {output_dir}/inlier_ratio.png")
    plot_bar_metric(
        inlier_ratios,
        title="Inlier Ratio per Image",
        ylabel="Inlier Ratio",
        save_path=output_dir / "inlier_ratio.png"
    )

# ------------------------------
# 📁 Save CSV of Metrics
# ------------------------------

def save_alignment_metrics_csv(metrics, output_path):
    """
    metrics: list of dicts
    output_path: str or Path
    """
    df = pd.DataFrame(metrics)
    output_path = Path(output_path)
    output_file = output_path / "results.csv"
    #output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"✅ Metrics CSV saved to {output_path}")

# ------------------------------
# 🖼️ Save Visual Diagnostics
# ------------------------------


def save_overlay_diagnostics(image_dir, output_dir, reference_dir=None, logger=None, blend_alpha=0.5):
    """
    Creates side-by-side overlays of images from image_dir against a reference.

    Parameters:
    - image_dir: Path to aligned images
    - output_dir: Path to diagnostics output
    - reference_dir: Optional Path to reference images (1-to-1 comparison)
    - blend_alpha: Alpha blending factor (0.0 to 1.0)
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir) / "overlay_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    if reference_dir:
        reference_dir = Path(reference_dir)
        if reference_dir.is_file():
            # Single image file used for all comparisons
            reference_paths = [reference_dir] * len(image_paths)
        elif reference_dir.is_dir():
            reference_paths = sorted([p for p in reference_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        else:
            logger.error(f"❌ Invalid reference path: {reference_dir}")
            return

        if len(image_paths) != len(reference_paths):
            logger.warning(f"❌ Image count mismatch: {len(image_paths)} aligned vs {len(reference_paths)} reference")
            return

        for img_path, ref_path in tqdm(zip(image_paths, reference_paths), total=len(image_paths),
                                       desc=f"Overlaying images in {image_dir.name}"):
            img = cv2.imread(str(img_path))
            ref = cv2.imread(str(ref_path))

            if img is None or ref is None:
                logger.warning(f"⚠️ Skipping unreadable pair: {img_path.name}, {ref_path.name}")
                continue

            ref_resized = cv2.resize(ref, (img.shape[1], img.shape[0]))
            blended = cv2.addWeighted(ref_resized, blend_alpha, img, 1 - blend_alpha, 0)
            side_by_side = np.concatenate([ref_resized, img, blended], axis=1)

            out_path = output_dir / f"diag_{img_path.name}"
            cv2.imwrite(str(out_path), side_by_side)
            logger.info(f"✅ Saved diagnostic: {out_path.name} in {out_path}")

    else:
        if len(image_paths) < 2:
            logger.info(f"❌ Need at least 2 images in {image_dir}")
            return

        ref_path = image_paths[0]
        ref = cv2.imread(str(ref_path))
        if ref is None:
            logger.warning(f"❌ Failed to read reference image: {ref_path.name}")
            return

        logger.info(f"🔍 Using reference: {ref_path.name}")

        for aligned_path in tqdm(image_paths[1:], desc=f"Overlaying images in {image_dir.name}"):
            aligned = cv2.imread(str(aligned_path))
            if aligned is None:
                logger.warning(f"⚠️ Skipping unreadable image: {aligned_path.name}")
                continue

            aligned_resized = cv2.resize(aligned, (ref.shape[1], ref.shape[0]))
            blended = cv2.addWeighted(ref, blend_alpha, aligned_resized, 1 - blend_alpha, 0)
            side_by_side = np.concatenate([ref, aligned_resized, blended], axis=1)

            out_path = output_dir / f"diag_{aligned_path.name}"
            cv2.imwrite(str(out_path), side_by_side)
            logger.info(f"✅ Saved diagnostic: {out_path.name}")

        
        
# ------------------------------
# 🔥 Stack HSV V-Channel Heatmap
# ------------------------------

def stack_intensity_heatmap(image_dir, save_path, reference_path=None, normalize=True):
    """
    Stacks HSV V-channel values for visual inspection and creates a normalized heatmap.

    Parameters:
    - image_dir: str or Path to directory containing image files
    - save_path: str or Path to output heatmap image
    - reference_path: optional str or Path to subtract a reference V-channel image
    - normalize: bool, if True scales values to [0, 1] before plotting
    """
    image_dir = Path(image_dir)
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    num_images = len(image_paths)
    
    if not image_paths:
        print("❌ No images to process.")
        return

    img0 = Image.open(image_paths[0]).convert('HSV')
    width, height = img0.size
    stack = np.zeros((height, width), dtype=np.float32)

    for img_path in image_paths:
        img = Image.open(img_path).convert('HSV').resize((width, height))
        _, _, v = img.split()
        stack += np.array(v, dtype=np.float32)

    if reference_path:
        ref = Image.open(reference_path).convert('HSV').resize((width, height))
        _, _, v_ref = ref.split()
        stack -= np.array(v_ref, dtype=np.float32)
        num_images += 1

    if normalize:
        min_val = np.min(stack)
        max_val = np.max(stack)
        if max_val != min_val:
            stack = (stack - min_val) / (max_val - min_val)
        else:
            stack[:] = 0.0  # If constant image

    plt.figure(figsize=(10, 6))
    im = plt.imshow(stack, cmap='viridis')
    title = f"{'Normalized ' if normalize else ''}Stacked V-Channel Heatmap\n({num_images} image{'s' if num_images != 1 else ''} used)"
    plt.title(title)
    plt.colorbar(im, label='Normalized Intensity' if normalize else 'Accumulated Intensity')
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    save_path = Path(save_path)
    save_file = save_path / "heatmap.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Heatmap saved to {save_file}")



def save_crop_preview(image_folders, crop_box, output_dir, preview_name="crop_preview.png", image_index=0):
    """
    Pick one image from the first folder, apply the crop_box, and write out a PNG.

    Args:
      image_folders (List[Path or str]): list of folders used by ImageDataProcessor
      crop_box (dict): {'y_min','y_max','x_min','x_max','padding'}
      output_dir (Path or str): where to write the preview
      preview_name (str): filename (PNG) under output_dir
      image_index (int): which image in the folder to sample (0 = first)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # find folder and image
    folder = Path(image_folders[0])
    all_paths = sorted(glob(str(folder / "*")))
    if not all_paths:
        logger.warning(f"No images found in {folder}, skipping preview.")
        return

    idx = min(image_index, len(all_paths)-1)
    img = cv2.imread(all_paths[idx])
    if img is None:
        logger.warning(f"Could not read {all_paths[idx]}, skipping preview.")
        return

    cb = crop_box
    cropped = img[cb["y_min"]:cb["y_max"], cb["x_min"]:cb["x_max"]]
    preview_path = output_dir / 'preprocessed_images' / preview_name
    cv2.imwrite(str(preview_path), cropped)
    logger.info(f"🖼️  Saved crop preview to {preview_path}")
