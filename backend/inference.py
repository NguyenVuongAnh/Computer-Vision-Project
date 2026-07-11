"""
File usage for: backend/inference.py
This file defines the `Inference` class, which is responsible for
making predictions using the trained CMACNet model and post-processing the results.
"""

from model.CMAC import CMACNet
import torch
import torch.nn.functional as F

class Inference:
    """
    The Inference class provides a simple interface for model prediction
    and post-processing.
    """

    def __init__(self):
        """
        Initializes the Inference class and the model.
        """
        self.model = CMACNet()
        self.model.eval()

    def predict(self, image_tensor):
        """
        Takes an image tensor as input, runs inference, and post-processes the output.
        Returns a processed segmentation mask and class probabilities.
        """
        with torch.no_grad():
            raw_mask, raw_classification = self.model(image_tensor)
        
        # Post-process the segmentation mask
        # Apply sigmoid to get probabilities and then threshold to get a binary mask
        processed_mask = torch.sigmoid(raw_mask)
        binary_mask = (processed_mask > 0.5).float()

        # Post-process the classification output
        # Apply softmax to get class probabilities
        class_probabilities = F.softmax(raw_classification, dim=1)

        return binary_mask, class_probabilities