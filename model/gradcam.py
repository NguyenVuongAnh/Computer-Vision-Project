"""
Grad-CAM visualization module for TensorFlow/Keras models.

This module provides the GradCAM class for generating Class Activation Maps
using Gradient-weighted Class Activation Mapping (Grad-CAM).
"""

import numpy as np
import tensorflow as tf
from typing import Optional


class GradCAM:
    """
    A class for generating Grad-CAM heatmaps for Keras models.

    This class implements the Grad-CAM algorithm to visualize which parts
    of an image the model is focusing on for a particular class prediction.

    Attributes:
        model (tf.keras.Model): The Keras model to generate Grad-CAM for.
    """

    def __init__(self, model: tf.keras.Model) -> None:
        """
        Initialize the GradCAM with a Keras model.

        Args:
            model: A Keras model instance to generate Grad-CAM for.

        Raises:
            TypeError: If the provided model is not a Keras Model instance.
        """
        if not isinstance(model, tf.keras.Model):
            raise TypeError(
                f"Expected model to be a tf.keras.Model instance, "
                f"got {type(model).__name__}"
            )
        self.model = model

    def generate(
        self,
        image: np.ndarray,
        layer_name: str,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specified layer and class.

        Args:
            image: Input image as a NumPy array with shape (height, width, channels).
            layer_name: Name of the convolutional layer to use for Grad-CAM.
            target_class: Target class index for Grad-CAM. If None, uses the
                predicted class (argmax of model output).

        Returns:
            np.ndarray: Grad-CAM heatmap as a NumPy array with shape
                (height, width), normalized to [0, 1].

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

        # Store original image size for resizing
        if image.ndim == 3:
            original_height, original_width = image.shape[:2]
            image_batch = np.expand_dims(image, axis=0)
        else:
            original_height, original_width = image.shape[1:3]
            image_batch = image

        # Convert to TensorFlow tensor for GradientTape
        image_batch = tf.convert_to_tensor(image_batch, dtype=tf.float32)

        # Check if layer exists
        layer_names = [layer.name for layer in self.model.layers]
        if layer_name not in layer_names:
            raise ValueError(
                f"Layer '{layer_name}' not found in model. "
                f"Available layers: {layer_names}"
            )

        # Get the target layer
        target_layer = self.model.get_layer(layer_name)

        # Create a model that outputs both the target layer activations
        # and the final predictions
        grad_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[target_layer.output, self.model.output]
        )

        # Use GradientTape to compute gradients
        with tf.GradientTape() as tape:
            # Watch the input image
            tape.watch(image_batch)
            # Forward pass to get layer output and predictions
            conv_outputs, predictions = grad_model(image_batch)

            # Determine target class if not provided
            if target_class is None:
                target_class = int(tf.argmax(predictions[0]))

            # Get the score for the target class
            target_score = predictions[:, target_class]

        # Compute gradients of the target class score with respect to
        # the convolutional layer output
        grads = tape.gradient(target_score, conv_outputs)

        # Global average pooling of gradients to get weights
        # Shape: (batch_size, channels)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        # Multiply each channel in the feature map by its corresponding weight
        # conv_outputs shape: (batch_size, height, width, channels)
        # pooled_grads shape: (batch_size, channels)
        # We need to multiply each channel by its weight
        pooled_grads = tf.expand_dims(pooled_grads, axis=1)
        pooled_grads = tf.expand_dims(pooled_grads, axis=1)
        # pooled_grads shape: (batch_size, 1, 1, channels)

        # Weight the feature maps
        weighted_outputs = conv_outputs * pooled_grads

        # Sum across channels to get the heatmap
        heatmap = tf.reduce_sum(weighted_outputs, axis=-1)

        # Apply ReLU to focus on positive influences
        heatmap = tf.nn.relu(heatmap)

        # Normalize the heatmap to [0, 1]
        heatmap_min = tf.reduce_min(heatmap)
        heatmap_max = tf.reduce_max(heatmap)

        if heatmap_max > heatmap_min:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            # If all values are the same, set to zeros
            heatmap = tf.zeros_like(heatmap)

        # Remove batch dimension
        heatmap = heatmap[0]

        # Resize heatmap to original image size
        heatmap = tf.image.resize(
            tf.expand_dims(heatmap, axis=-1),
            [original_height, original_width],
            method='bilinear'
        )
        heatmap = tf.squeeze(heatmap, axis=-1)

        # Convert to NumPy array
        heatmap = heatmap.numpy()

        return heatmap