import numpy as np
import tensorflow as tf
from typing import Dict


class Predictor:

    def __init__(self, model: tf.keras.Model):

        if not isinstance(model, tf.keras.Model):
            raise TypeError( "model must be tf.keras.Model")
        self.model = model

    def predict(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError( "image must be numpy array")
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        elif image.ndim != 4:
            raise ValueError(f"Invalid image shape {image.shape}")
        return self.model.predict(image, verbose=0)
    
    def predict_class(self, image: np.ndarray) -> Dict:
        probabilities = self.predict(image)
        probabilities = probabilities[0]
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        return {
            "class_id": class_id,
            "confidence": confidence,
            "probabilities": probabilities
        }