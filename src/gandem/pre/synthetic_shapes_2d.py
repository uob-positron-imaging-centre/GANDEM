"""
Synthetic 2D primitive shapes for GANular examples and tests.
"""

import numpy as np
import cv2
import os
import random
from tqdm import tqdm, trange


def generate_pentagons2d(
    n: int,
    output_dir: str,
    img_size: int = 256,
    random_skew: bool = False,
):
    """
    Generates binary PNG images of regular pentagons.

    Each pentagon is randomly rotated and optionally subjected to a random
    affine skew. Images are saved as black-on-white PNGs.
    Args:
        n: Number of images to generate.
        output_dir: Directory to save the generated images (created if absent).
        img_size: Height and width of the square output image in pixels.
        random_skew: Whether to apply a random affine transformation to
            each image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    center = (img_size // 2, img_size // 2)

    for i in trange(n, desc="Generating Pentagons"):
        canvas = np.zeros((img_size, img_size), dtype=np.uint8)
        radius = (img_size // 2) - 10
        rotation_angle = random.uniform(0, 2 * np.pi)

        vertices = []
        for k in range(5):
            theta = rotation_angle + k * (2 * np.pi / 5)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            vertices.append([x, y])

        vertices = np.array(vertices, dtype=np.int32)
        vertices = vertices.reshape((-1, 1, 2))

        cv2.fillPoly(canvas, [vertices], 255)

        if random_skew:
            pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
            pts2 = np.float32(
                [
                    [50 + random.randint(-10, 10), 50 + random.randint(-10, 10)],
                    [200 + random.randint(-10, 10), 50 + random.randint(-10, 10)],
                    [50 + random.randint(-10, 10), 200 + random.randint(-10, 10)],
                ]
            )
            M = cv2.getAffineTransform(pts1, pts2)
            canvas = cv2.warpAffine(canvas, M, (img_size, img_size))

        final_img = cv2.bitwise_not(canvas)
        filename = os.path.join(output_dir, f"pentagon_{i:04d}.png")
        cv2.imwrite(filename, final_img)

    print(f"Successfully generated {n} images in '{output_dir}'.")


def generate_diamonds2d(
    n: int,
    output_dir: str,
    img_size: int = 256,
    random_skew: bool = False,
):
    """
    Generates binary PNG images of diamond shapes (rhombus).

    Each diamond has random half-width and half-height, is randomly rotated,
    and optionally subjected to a random affine skew.

    Args:
        n: Number of images to generate.
        output_dir: Directory to save the generated images (created if absent).
        img_size: Height and width of the square output image in pixels.
        random_skew: Whether to apply a random affine transformation to
            each image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    center = (img_size // 2, img_size // 2)

    for i in trange(n, desc="Generating Diamonds"):
        canvas = np.zeros((img_size, img_size), dtype=np.uint8)

        half_w = random.randint(img_size // 6, (img_size // 2) - 10)
        half_h = random.randint(img_size // 6, (img_size // 2) - 10)

        pts = np.array(
            [[0, -half_h], [half_w, 0], [0, half_h], [-half_w, 0]], dtype=np.float32
        )

        rotation_angle = random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        )

        rotated_pts = np.dot(pts, rotation_matrix.T)
        rotated_pts[:, 0] += center[0]
        rotated_pts[:, 1] += center[1]

        vertices = rotated_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [vertices], 255)

        if random_skew:
            pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
            pts2 = np.float32(
                [
                    [50 + random.randint(-10, 10), 50 + random.randint(-10, 10)],
                    [200 + random.randint(-10, 10), 50 + random.randint(-10, 10)],
                    [50 + random.randint(-10, 10), 200 + random.randint(-10, 10)],
                ]
            )
            M = cv2.getAffineTransform(pts1, pts2)
            canvas = cv2.warpAffine(canvas, M, (img_size, img_size))

        final_img = cv2.bitwise_not(canvas)
        filename = os.path.join(output_dir, f"diamond_{i:04d}.png")
        cv2.imwrite(filename, final_img)

    print(f"Successfully generated {n} images in '{output_dir}'.")
