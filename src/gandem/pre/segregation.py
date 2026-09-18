"""
Segregation and cropping of particles from raw images.
This module has only been tested on Morphologi 4 data.
"""

import os
import warnings
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import exposure
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm


def _center_bbox(
    img: np.ndarray,
    bb_coords: tuple[float, float],
    bb_dimn: tuple[float, float],
    img_dimn: tuple[int, int],
    contour: Optional[np.ndarray] = None,
    padding_factor=1.1,
) -> Optional[np.ndarray]:
    """Centre a bounding box around a particle and create a binary mask.

    Returns ``None`` if the bounding box touches the image edge.

    Args:
        img: Source grayscale image (unused directly but kept for context).
        bb_coords: ``(x, y)`` top-left corner of the bounding box.
        bb_dimn: ``(width, height)`` of the bounding box.
        img_dimn: ``(height, width)`` of the source image.
        contour: OpenCV contour to draw as a filled mask.
        padding_factor: Multiplicative padding applied to the square crop
            size.

    Returns:
        Binary mask of shape ``(size, size)`` or ``None`` if the particle
        is on the image edge.
    """
    x, y = bb_coords
    w, h = bb_dimn
    img_height, img_width = img_dimn

    # check if on image edge
    if x <= 0 or y <= 0 or (x + w) >= img_width or (y + h) >= img_height:
        return None

    # new bbox center
    center_x = x + w // 2
    center_y = y + h // 2

    # square size - take max dimn
    size = int(max(w, h) * padding_factor)

    # bottom left corner of new bbox
    x1 = max(center_x - size // 2, 0)
    y1 = max(center_y - size // 2, 0)

    # top right corner of new bbox
    x2 = max(x1 + size, 0)
    y2 = max(y1 + size, 0)

    canvas_h = y2 - y1
    canvas_w = x2 - x1
    local_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    # Shift contour to local coordinates
    cnt_shifted = contour - [x1, y1]

    # Draw the FILLED contour directly onto the local mask.
    cv2.drawContours(local_mask, [cnt_shifted], -1, 255, -1)

    return local_mask  # Return binary mask immediately


def _get_image_areas(img_path: str) -> Tuple[List[float], int]:
    """Extract contour areas from a single grayscale image.

    Applies Gaussian blur and Otsu thresholding before contour detection.

    Args:
        img_path: Path to the image file.

    Returns:
        A tuple of ``(areas, count)`` where *areas* is a list of contour
        areas and *count* is the total number of contours found.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], 0

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(cnt) for cnt in contours]
    return areas, len(contours)


def find_best_max_size(
    img_dir: str,
    csv_path: str,
    plot: Optional[bool] = False,
    plot_savepath: Optional[str] = None,
    num_workers: Optional[int] = None,
) -> float:
    """
    Find the best maximum-size threshold for particle segregation.

    Analyses all JPG images in *img_dir* in parallel, builds a cumulative
    distribution function (CDF) of contour areas, and determines the area
    threshold below which the fraction of excluded particles matches the
    difference between contour count and the number of results in the CSV.

    Args:
        img_dir: Directory containing particle images.
        csv_path: Path to a tab-separated results CSV file.
        plot: Whether to display / save a CDF plot.
        plot_savepath: File path to save the plot (only used when *plot* is
            ``True``).
        num_workers: Number of parallel workers. Defaults to all CPUs.

    Returns:
        The area threshold below which particles are considered too small.

    Raises:
        FileNotFoundError: If *img_dir* or *csv_path* do not exist.
    """
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Directory {img_dir} does not exist.")
    elif not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} does not exist.")

    # get all jpg files in directory
    jpg_files = [
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")
    ]

    areas = []
    total_contours = 0

    print(
        f"Calculating areas for {len(jpg_files)} images using {num_workers or os.cpu_count()} workers..."
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(executor.map(_get_image_areas, jpg_files), total=len(jpg_files))
        )

    for img_areas, n_cnts in results:
        areas.extend(img_areas)
        total_contours += n_cnts

    # Read results CSV
    results_df = pd.read_csv(csv_path, encoding="ISO-8859-1", sep="\t")

    n_results = len(results_df)  # Number of particles with results

    # Generate CDF and find size threshold
    areas_sorted = np.sort(np.array(areas))
    p = np.linspace(0, 1, len(areas))
    excluded_frac = 1 - n_results / total_contours

    def cdf(a) -> float:
        return np.interp(a, p, areas_sorted).item()

    min_size = cdf(excluded_frac)

    if plot:
        # Plot CDF and exclusion line
        plt.plot(areas_sorted, p)
        plt.axhline(y=excluded_frac, color="r", linestyle="--")

        plt.xlabel("Particle Area", fontsize=14)
        plt.ylabel("CDF", fontsize=14)

        plt.savefig(plot_savepath) if plot_savepath else plt.show()
        plt.close()

    return min_size


def _crop_single_image_task(args) -> Tuple[int, int]:
    """Worker function that crops particles from a single image.

    Applies bilateral filtering, Otsu thresholding, and morphological
    closing before extracting contours. Each qualifying particle is
    normalised, smoothed, and saved as a binary PNG mask.

    Args:
        args: Tuple of ``(img_path, max_size, target_size, output_dir)``.

    Returns:
        A tuple of ``(particles_processed, particles_skipped)``.
    """
    img_path, max_size, target_size, output_dir = args

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0, 0

    blurred = cv2.bilateralFilter(img, 9, 75, 75)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    particles_processed = 0
    particles_skipped = 0

    basename = os.path.basename(os.path.dirname(img_path))
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_dir = os.path.join(output_dir, basename)
    os.makedirs(save_dir, exist_ok=True)

    for i, cnt in enumerate(contours):
        # filter small contours
        area = cv2.contourArea(cnt)
        if area <= max_size:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        max_dim = max(w, h)
        if max_dim == 0:
            continue

        scale = (target_size * 0.9) / max_dim

        cx, cy = x + w / 2, y + h / 2
        cnt_centered = cnt.astype(float) - [cx, cy]
        cnt_scaled = cnt_centered * scale

        scaled_arc = cv2.arcLength(cnt_scaled.astype(np.float32), True)
        epsilon = 0.005 * scaled_arc
        cnt_smooth = cv2.approxPolyDP(cnt_scaled.astype(np.float32), epsilon, True)

        # Draw high-res result
        cnt_final = cnt_smooth + [target_size / 2, target_size / 2]
        cnt_final = cnt_final.astype(np.int32)

        mask = np.zeros((target_size, target_size), dtype=np.uint8)
        cv2.drawContours(mask, [cnt_final], -1, 255, -1)

        mask_final = cv2.bitwise_not(mask)

        # Unique filename: particle_{original_image_name}_{index}.png
        save_path = os.path.join(save_dir, f"particle_{img_name}_{i}.png")
        cv2.imwrite(save_path, mask_final)

        particles_processed += 1

    return particles_processed, particles_skipped


def crop_particles(
    img_dir: str,
    max_size: float = 50,
    target_size: int = 256,
    num_workers: Optional[int] = None,
) -> None:
    """
    Crop and extract individual particles from images in *img_dir*.

    Processes all JPG images in parallel, filters contours by area, and
    saves each qualifying particle as a normalised binary PNG in a
    ``cropped_particles`` directory.

    Args:
        img_dir: Directory containing the source images.
        max_size: Minimum contour area; contours smaller than this are
            skipped.
        target_size: Side length (pixels) of the square output images.
        num_workers: Number of parallel workers. Defaults to all CPUs.

    Raises:
        FileNotFoundError: If *img_dir* does not exist.
    """
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Directory {img_dir} does not exist.")

    # get all jpg files in directory
    jpg_files = [
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")
    ]

    output_dir = "cropped_particles"

    # Prepare arguments for parallel execution
    tasks = [(f, max_size, target_size, output_dir) for f in jpg_files]

    total_processed = 0
    total_skipped = 0

    print(f"Starting particle cropping with {num_workers or os.cpu_count()} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(executor.map(_crop_single_image_task, tasks), total=len(tasks))
        )

    for processed, skipped in results:
        total_processed += processed
        total_skipped += skipped

    warnings.warn(f"Skipped {total_skipped} particles (on image edge).")
    print(f"Processed {total_processed} particles.")


def process_single_image(
    img_path: str, max_size: float, target_size: int = 256
) -> None:
    """
    Process a single image interactively, extracting and displaying particles.

    Works like :func:`crop_particles` but for a single image, showing the
    results in a matplotlib figure rather than saving to disk.

    Args:
        img_path: Path to the image file.
        max_size: Minimum contour area; contours smaller than this are
            skipped.
        target_size: Side length (pixels) of the square output images.

    Raises:
        FileNotFoundError: If *img_path* does not exist.
        ValueError: If the image cannot be read.
    """
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"File {img_path} does not exist.")

    # read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image {img_path}")

    blurred = cv2.bilateralFilter(img, 9, 75, 75)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    processed_particles = []
    particles_skipped = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= max_size:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        max_dim = max(w, h)
        if max_dim == 0:
            continue

        # Scale
        scale = (target_size * 0.9) / max_dim
        cx, cy = x + w / 2, y + h / 2
        cnt_centered = cnt.astype(float) - [cx, cy]
        cnt_scaled = cnt_centered * scale

        # Smooth Scaled Vectors
        scaled_arc = cv2.arcLength(cnt_scaled.astype(np.float32), True)
        epsilon = 0.005 * scaled_arc
        cnt_smooth = cv2.approxPolyDP(cnt_scaled.astype(np.float32), epsilon, True)

        # Draw
        cnt_final = cnt_smooth + [target_size / 2, target_size / 2]
        cnt_final = cnt_final.astype(np.int32)

        binarised_img = np.zeros((target_size, target_size), dtype=np.uint8)
        cv2.drawContours(binarised_img, [cnt_final], -1, 255, -1)

        binarised_img = cv2.bitwise_not(binarised_img)

        processed_particles.append(binarised_img)

    if particles_skipped > 0:
        warnings.warn(f"Skipped {particles_skipped} particles (on image edge).")

    print(f"Found {len(processed_particles)} particles.")

    if not processed_particles:
        print("No particles found matching criteria.")
        return

    # Display images
    n = len(processed_particles)
    cols = 5
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(15, 3 * rows))
    for i, p_img in enumerate(processed_particles):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(p_img, cmap="gray")
        plt.axis("off")
        plt.title(f"Particle {i + 1}")

    plt.tight_layout()
    plt.show()
