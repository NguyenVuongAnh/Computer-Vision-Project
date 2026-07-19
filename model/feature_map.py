"""
Feature map extraction module for TensorFlow/Keras models.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List


class FeatureMapExtractor:
    """
    Extract intermediate feature maps from TensorFlow/Keras models.
    """

    def __init__(self, model: tf.keras.Model):

        if not isinstance(model, tf.keras.Model):
            raise TypeError( f"Expected tf.keras.Model, got {type(model).__name__}" )

        self.model = model

        # Extract all layers at once
        self.feature_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[layer.output for layer in self.model.layers]
        )

        # Cache single layer models
        self.layer_models = {}

    def list_layers(self) -> List[str]:
        """
        Return all layer names.
        """
        return [layer.name for layer in self.model.layers]

    def layer_info(self) -> List[dict]:
        """
        Return layer information.
        """

        info = []
        for index, layer in enumerate(self.model.layers):
            info.append(
                {
                    "index": index,
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "output_shape": layer.output.shape
                }
            )
        return info

    def show_layers(self):
        """
        Print all available layers.
        """

        print(
            f"{'Index':<8}"
            f"{'Layer Name':<35}"
            f"{'Type':<25}"
            f"Output Shape"
        )
        print("-" * 100)

        for item in self.layer_info():
            print(
                f"{item['index']:<8}"
                f"{item['name']:<35}"
                f"{item['type']:<25}"
                f"{item['output_shape']}"
            )

    def _validate_image( self, image: np.ndarray ):
        if not isinstance(image, np.ndarray):
            raise TypeError( "image must be numpy array" )
        if image.ndim == 3:
            image = np.expand_dims( image, axis=0 )
        elif image.ndim != 4:
            raise ValueError( f"Invalid image shape {image.shape}" )
        return image

    def extract(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all layer feature maps.

        Returns:
            {
                layer_name: feature_map
            }
        """
        image = self._validate_image( image )
        outputs = self.feature_model.predict( image, verbose=0 )
        feature_maps = {}
        for layer, output in zip( self.model.layers, outputs ):
            feature_maps[layer.name] = output[0]
        return feature_maps



    def extract_layer( self, image: np.ndarray, layer_name: str ) -> np.ndarray:
        """
        Extract feature map from one layer.
        """
        image = self._validate_image( image )

        if layer_name not in self.list_layers():
            raise ValueError( f"Layer '{layer_name}' not found" )
        layer = self.model.get_layer( layer_name )
        # Check feature map dimension
        if len(layer.output.shape) != 4:
            raise ValueError(
                f"Layer '{layer_name}' is not feature map layer. "
                f"Output shape: {layer.output.shape}"
            )

        # Create model once
        if layer_name not in self.layer_models:
            self.layer_models[layer_name] = tf.keras.Model(
                inputs=self.model.input,
                outputs=layer.output
            )
        feature_map = self.layer_models[layer_name].predict( image, verbose=0 )
        return feature_map[0]