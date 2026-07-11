"""
File usage for: main.py

This script serves as a clear, step-by-step example of how to use the
Inference class to get predictions from the CMACNet model.

It simulates a real-world scenario by:
1. Creating a random data tensor that mimics a preprocessed image.
2. Initializing the inference engine.
3. Running the prediction.
4. Clearly printing and explaining the model's outputs (a segmentation mask
   and classification probabilities for different lesion types).
"""
import torch
from backend.inference import Inference

def run_inference_example():
    """
    A detailed example function to demonstrate how to run inference with clear explanations.
    """
    print("--- CMACNet Inference Example ---")
    print("This script demonstrates how to use the model for prediction.")

    # --- 1. Model and Inference Engine Initialization ---
    print("\nStep 1: Initializing the inference engine...")
    # The Inference class from 'backend/inference.py' loads and prepares the model.
    try:
        inference_engine = Inference()
        print("Inference engine initialized successfully.")
    except Exception as e:
        print(f"Error initializing the model: {e}")
        print("Please ensure the 'model' and 'backend' directories are set up correctly.")
        return

    # --- 2. Input Data Simulation ---
    print("\nStep 2: Preparing a sample input tensor...")
    # In a real application, you would load and preprocess an image here.
    # For this example, we'll create a random tensor to simulate a single
    # preprocessed RGB image of size 256x256 pixels.
    # The shape is (batch_size, channels, height, width).
    batch_size = 1
    channels = 3
    height = 256
    width = 256
    random_image_tensor = torch.randn(batch_size, channels, height, width)
    print(f"Created a random image tensor with shape: {random_image_tensor.shape}")
    print("(This simulates a batch of 1 image, 3 color channels, 256x256 pixels).")

    # --- 3. Running Inference ---
    print("\nStep 3: Running prediction on the sample tensor...")
    # The predict method returns two outputs: a segmentation mask and classification probabilities.
    try:
        mask, classification_probs = inference_engine.predict(random_image_tensor)
        print("Prediction complete.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return

    # --- 4. Interpreting the Output ---
    print("\nStep 4: Analyzing the model's output...")

    # The model produces two distinct outputs for two different tasks.
    
    # Output 1: Classification
    print("\n--- Output 1: Classification Probabilities ---")
    # The model's classification head returns 4 scores. We'll assume they correspond
    # to the first four lesion types for this example.
    CLASSIFICATION_CLASSES = ["EX", "HE", "MA", "OD"]
    print(f"The model predicts probabilities for {len(CLASSIFICATION_CLASSES)} different classes.")
    print(f"Output tensor shape: {classification_probs.shape}")

    # We get one set of probabilities for each image in the batch.
    # Since our batch size is 1, we look at the first (and only) result.
    predicted_probs = classification_probs[0]
    
    print("\nPredicted Probabilities for each class:")
    for i, class_name in enumerate(CLASSIFICATION_CLASSES):
        prob = predicted_probs[i].item()
        print(f"  - Class '{class_name}': {prob:.4f} (or {prob*100:.2f}%)")
    
    # Determine the most likely class
    predicted_class_index = torch.argmax(predicted_probs).item()
    predicted_class_name = CLASSIFICATION_CLASSES[predicted_class_index]
    print(f"\nConclusion: The most likely image-level class is '{predicted_class_name}'.")

    # Output 2: Segmentation Masks
    print("\n--- Output 2: Segmentation Masks ---")
    # The 5 lesion classes for segmentation are:
    SEGMENTATION_CLASSES = ["EX", "HE", "MA", "OD", "SE"]
    print("The model also generates a separate pixel-wise mask for each of the 5 lesion types.")
    print(f"Output mask shape: {mask.shape}")
    print(f"(This corresponds to a batch of 1 image, with {mask.shape[1]} separate masks, one for each lesion type).")
    print("Each mask contains values (0 or 1) for each pixel, where '1' indicates the presence of that specific lesion.")
    
    # Let's inspect a small corner of the first mask to see what it looks like.
    print(f"\nExample: A 5x5 corner of the predicted mask for the first class ('{SEGMENTATION_CLASSES[0]}'):")
    print(mask[0, 0, :5, :5]) # Print the top-left 5x5 area of the first mask

    print("\n--- End of Example ---")


if __name__ == "__main__":
    run_inference_example()