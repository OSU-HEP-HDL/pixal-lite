import numpy as np

def zero_prune_image(img: np.ndarray, padding, target_size=None) -> np.ndarray:
    """
    Crops the image to the tightest bounding box containing all non-zero pixels.
    Optionally resizes to a fixed size.

    Args:
        img (np.ndarray): Input image (H, W) or (H, W, C).
        target_size (tuple): Optional (height, width) to resize the cropped image.

    Returns:
        np.ndarray: Cropped (and optionally resized) image.
    """
    if img.ndim == 3:
        mask = np.any(img != 0, axis=-1)
    else:
        mask = img != 0

    coords = np.argwhere(mask)
    if coords.size == 0:
        return img  # fallback if image is all zero

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    y_min = max(y_min - padding, 0)
    x_min = max(x_min - padding, 0)
    y_max = min(y_max + padding, img.shape[0])
    x_max = min(x_max + padding, img.shape[1])

    cropped = img[y_min:y_max, x_min:x_max]

    if target_size:
        from PIL import Image
        pil_img = Image.fromarray(cropped.astype(np.uint8))
        cropped = np.array(pil_img.resize(target_size, Image.LANCZOS))

    return cropped
