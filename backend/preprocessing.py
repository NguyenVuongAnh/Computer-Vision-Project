"""
File usage for: backend/preprocessing.py
This file contains functions and pipelines for preprocessing medical images,
specifically for tasks like diabetic retinopathy detection. The pipelines
combine various image enhancement techniques to improve the quality and
visibility of features in the images before they are fed into a model.
"""

import cv2
from PIL import Image
import numpy as np

def green_channel(image):
    """
    Extracts the green channel from an RGB image and converts it back to
    a 3-channel RGB image. This is often used to enhance the visibility
    of blood vessels in retinal images.
    """
    green = image[:, :, 1]
    return cv2.cvtColor(green, cv2.COLOR_GRAY2RGB)


def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an
    image in the LAB color space to enhance local contrast.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe_filter = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    l = clahe_filter.apply(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def illumination_correction(image):
    """
    Corrects for non-uniform illumination in an image using a Gaussian blur
    to estimate the background illumination.
    """
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=25)

    corrected = cv2.addWeighted(
        image,
        4,
        blur,
        -4,
        128,
    )

    return corrected


def denoise(image):
    """
    Applies a median filter to reduce noise in an image.
    """
    return cv2.medianBlur(image, 3)


def ben_graham(image):
    """
    Implements the Ben Graham preprocessing technique, a common method for
    enhancing retinal images.
    """
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=30)

    enhanced = cv2.addWeighted(
        image,
        4,
        blur,
        -4,
        128,
    )

    return enhanced


def enhancing_image_pipeline_1(image):
    """
    A preprocessing pipeline that converts a PIL image to a NumPy array,
    extracts the green channel, applies CLAHE, and converts it back to a PIL image.

    Pipeline 1:
    PIL -> NumPy -> Green Channel -> CLAHE -> PIL
    """

    image = np.asarray(image, dtype=np.uint8)

    image = green_channel(image)
    image = clahe(image)

    image = Image.fromarray(image)

    return image


def enhancing_image_pipeline_2(image):
    """
    A preprocessing pipeline that converts a PIL image to a NumPy array,
    applies illumination correction, median denoising, and CLAHE, then
    converts it back to a PIL image.

    Pipeline 2:
    PIL -> NumPy -> Illumination Correction -> Median Denoise -> CLAHE -> PIL
    """

    image = np.asarray(image, dtype=np.uint8)

    image = illumination_correction(image)
    image = denoise(image)
    image = clahe(image)

    image = Image.fromarray(image)

    return image


def enhancing_image_pipeline_3(image):
    """
    A preprocessing pipeline that converts a PIL image to a NumPy array,
    applies the Ben Graham preprocessing, and converts it back to a PIL image.

    Pipeline 3:
    PIL -> NumPy -> Ben Graham -> PIL
    """

    image = np.asarray(image, dtype=np.uint8)

    image = ben_graham(image)

    image = Image.fromarray(image)

    return image


pipelines = {
    "pipeline1": enhancing_image_pipeline_1,
    "pipeline2": enhancing_image_pipeline_2,
    "pipeline3": enhancing_image_pipeline_3,
}

class Preprocessing:
    """
    A class to encapsulate the preprocessing pipelines.
    """

    def __init__(self):
        """
        Initializes the Preprocessing class.
        """
        pass