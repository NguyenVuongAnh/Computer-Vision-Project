"""
Feature map extraction module for TensorFlow/Keras models.

This module provides the FeatureMapExtractor class for extracting
intermediate feature maps from Keras models using the Functional API.
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Optional


class FeatureMapExtractor:
    """
    A class for extracting feature maps from intermediate layers of a Keras model.

    This class creates intermediate models using the Keras Functional API
    to extract feature maps from specified layers.

    Attributes:
        model (tf.keras.Model): The Keras model to extract features from.
    """

    def __init__(self, model: tf.keras.Model) -> None:
        """
        Initialize the FeatureMapExtractor with a Keras model.

        Args:
            model: A Keras model instance to extract features from.

        Raises:
            TypeError: If the provided model is not a Keras Model instance.
        """
        if not isinstance(model, tf.keras.Model):
            raise TypeError(
                f"Expected model to be a tf.keras.Model instance, "
                f"got {type(model).__name__}"
            )
        self.model = model

    def list_layers(self) -> List[str]:
        """
        List all layer names in the model.

        Returns:
            List[str]: A list of layer names in the model, in order.
        """
        return [layer.name for layer in self.model.layers]

    def extract(self, image: np.ndarray, layer_name: str) -> np.ndarray:
        """
        Extract feature map from a specified layer.

        Args:
            image: Input image as a NumPy array with shape (height, width, channels)
                or (batch, height, width, channels).
            layer_name: Name of the layer to extract features from.

        Returns:
            np.ndarray: Feature map as a NumPy array with batch dimension removed.
                Shape will be (height, width, channels) or similar depending
                on the layer output shape.

        Raises:
            TypeError: If image is not a NumPy array.
            ValueError: If image has invalid shape or dimensions.
            ValueError: If the specified layer name does not exist in the model.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected image to be a np.ndarray, got {type(image).__name__}"
            )

        # Validate input dimensions
        if image.ndim not in [3, 4]:
            raise ValueError(
                f"Expected image to have 3 or 4 dimensions (HWC or BHWC), "
                f"got {image.ndim} dimensions with shape {image.shape}"
            )

        # Check if layer exists
        layer_names = self.list_layers()
        if layer_name not in layer_names:
            raise ValueError(
                f"Layer '{layer_name}' not found in model. "
                f"Available layers: {layer_names}"
            )

        # Add batch dimension if necessary
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        # Create intermediate model using Functional API
        try:
            intermediate_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer(layer_name).output
            )
        except ValueError as e:
            raise ValueError(
                f"Failed to create intermediate model for layer '{layer_name}': {e}"
            )

        # Extract feature map
        feature_map = intermediate_model.predict(image, verbose=0)

        # Remove batch dimension
        feature_map = feature_map[0]

        return feature_map