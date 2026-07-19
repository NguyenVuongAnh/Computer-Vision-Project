"""
Prediction module for TensorFlow/Keras models.

This module provides the Predictor class for running inference on Keras models.
"""

import numpy as np
import tensorflow as tf
from typing import Union


class Predictor:
    """
    A class for making predictions with a Keras model.

    This class wraps a Keras model and provides a simple interface for
    running inference on single images or batches of images.

    Attributes:
        model (tf.keras.Model): The Keras model used for prediction.
    """

    def __init__(self, model: tf.keras.Model) -> None:
        """
        Initialize the Predictor with a Keras model.

        Args:
            model: A Keras model instance to use for predictions.

        Raises:
            TypeError: If the provided model is not a Keras Model instance.
        """
        if not isinstance(model, tf.keras.Model):
            raise TypeError(
                f"Expected model to be a tf.keras.Model instance, "
                f"got {type(model).__name__}"
            )
        self.model = model

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run prediction on a single image.

        This method automatically adds a batch dimension if the input
        is a single image (i.e., has shape [height, width, channels]),
        runs the forward pass, and returns the raw model output.

        Args:
            image: Input image as a NumPy array. Can be either:
                - A single image with shape (height, width, channels)
                - A batched image with shape (batch, height, width, channels)

        Returns:
            np.ndarray: Raw model output. For a single image input,
                returns an array with shape (1, num_classes) or
                model output shape. For batched input, returns the
                model output with batch dimension preserved.

        Raises:
            TypeError: If image is not a NumPy array.
            ValueError: If image has invalid shape or dimensions.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected image to be a np.ndarray, got {type(image).__name__}"
            )

        # Store original shape to determine if we need to add batch dimension
        original_shape = image.shape

        # Validate input dimensions
        if image.ndim not in [3, 4]:
            raise ValueError(
                f"Expected image to have 3 or 4 dimensions (HWC or BHWC), "
                f"got {image.ndim} dimensions with shape {image.shape}"
            )

        # Add batch dimension if necessary (single image case)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        # Run prediction
        output = self.model.predict(image, verbose=0)

        # If input was a single image, return output as-is (with batch dim)
        # The caller can squeeze if needed
        return output