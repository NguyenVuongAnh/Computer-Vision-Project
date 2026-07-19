"""
Demo script for the Model Analysis module using ResNet50.

This script demonstrates the capabilities of the model analysis module
by analyzing a ResNet50 model and visualizing feature maps and Grad-CAM
on a sample image.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os

# Import the model analysis module
from model import Predictor, FeatureMapExtractor, GradCAM, ModelSummary


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for ResNet50.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (height, width)
    
    Returns:
        Preprocessed image as NumPy array
    """
    # Load image
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(target_size)
    
    # Convert to NumPy array
    img_array = np.array(img)
    
    # Preprocess for ResNet50
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    return img_array, img
