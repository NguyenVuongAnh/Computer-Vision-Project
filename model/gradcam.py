import numpy as np
import tensorflow as tf
from typing import Optional


class GradCAM:


    def __init__(self, model: tf.keras.Model):
        if not isinstance(model, tf.keras.Model):
            raise TypeError("model must be tf.keras.Model")
        self.model = model
        self.grad_models = {}

    def list_layers(self):
        return [
            layer.name
            for layer in self.model.layers
        ]

    def show_layers(self):
        print(
            f"{'Index':<8}"
            f"{'Layer Name':<35}"
            f"{'Type':<25}"
            f"Output Shape"
        )
        print("-"*100)

        for i, layer in enumerate(self.model.layers):
            print(
                f"{i:<8}"
                f"{layer.name:<35}"
                f"{layer.__class__.__name__:<25}"
                f"{layer.output.shape}"
            )

    def generate(self, image: np.ndarray, layer_name: str, target_class: Optional[int] = None):
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be numpy array")
        if image.ndim == 3:
            h,w = image.shape[:2]
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 4:
            h,w = image.shape[1:3]
        else:
            raise ValueError(f"Invalid image shape {image.shape}")

        layer = self.model.get_layer(layer_name)
        # check feature map
        if len(layer.output.shape) != 4:
            raise ValueError(
                f"Layer {layer_name} is not convolutional feature map: "
                f"{layer.output.shape}"
            )

        # cache grad model
        if layer_name not in self.grad_models:
            self.grad_models[layer_name] = tf.keras.Model(
                inputs=self.model.input,
                outputs=[
                    layer.output,
                    self.model.output
                ]
            )
        grad_model = self.grad_models[layer_name]
        image_tensor = tf.cast(image, tf.float32)

        with tf.GradientTape() as tape:
            conv_output, prediction = grad_model(image_tensor)
            if target_class is None:
                target_class = int(tf.argmax(prediction[0]))
            loss = prediction[:, target_class]
        gradients = tape.gradient(loss, conv_output)

        # channel importance
        weights = tf.reduce_mean(gradients, axis=(1,2))
        weights = weights[0]

        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(conv_output * weights, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)

        if max_value > 0:
            heatmap /= max_value
            
        heatmap = tf.image.resize(heatmap[...,None], (h,w))
        heatmap = tf.squeeze(heatmap)
        return heatmap.numpy()