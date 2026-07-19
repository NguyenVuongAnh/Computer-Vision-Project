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


def display_feature_maps(feature_maps, layer_names, max_features=8):
    """
    Display feature maps in a grid.
    
    Args:
        feature_maps: List of feature map arrays
        layer_names: List of layer names
        max_features: Maximum number of feature channels to display
    """
    n_layers = len(feature_maps)
    fig, axes = plt.subplots(n_layers, max_features, figsize=(max_features * 2, n_layers * 2))
    
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for i, (feature_map, layer_name) in enumerate(zip(feature_maps, layer_names)):
        # Get first max_features channels
        n_channels = min(max_features, feature_map.shape[-1])
        
        for j in range(n_channels):
            ax = axes[i, j] if n_layers > 1 else axes[j]
            
            # Get the j-th channel
            channel = feature_map[:, :, j]
            
            # Normalize for visualization
            channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            
            ax.imshow(channel, cmap='viridis')
            ax.set_title(f'Ch {j}', fontsize=8)
            ax.axis('off')
        
        # Fill remaining subplots with empty plots
        for j in range(n_channels, max_features):
            ax = axes[i, j] if n_layers > 1 else axes[j]
            ax.axis('off')
        
        # Set row label
        if n_layers > 1:
            axes[i, 0].set_ylabel(layer_name, fontsize=10, rotation=0, ha='right', va='center')
    
    plt.suptitle('Feature Maps from Different Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_maps.png', dpi=150, bbox_inches='tight')
    print("✓ Feature maps saved to 'feature_maps.png'")
    plt.close()


def display_gradcam_overlay(original_img, heatmap, alpha=0.4):
    """
    Create and save Grad-CAM overlay visualization.
    
    Args:
        original_img: Original PIL Image
        heatmap: Grad-CAM heatmap as NumPy array
        alpha: Transparency of heatmap overlay
    """
    # Resize heatmap to match original image size
    heatmap_resized = tf.image.resize(
        tf.expand_dims(heatmap, axis=-1),
        [original_img.size[1], original_img.size[0]],
        method='bilinear'
    ).numpy().squeeze()
    
    # Create colormap
    colormap = plt.cm.jet(heatmap_resized)[:, :, :3]
    
    # Convert original image to array
    img_array = np.array(original_img) / 255.0
    
    # Create overlay
    overlay = (1 - alpha) * img_array + alpha * colormap
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle('Grad-CAM Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Grad-CAM visualization saved to 'gradcam_visualization.png'")
    plt.close()


def main():
    """Main demo function."""
    print("\n" + "=" * 70)
    print("Model Analysis Module - ResNet50 Demo")
    print("=" * 70 + "\n")
    
    # Load pretrained ResNet50 model
    print("Loading ResNet50 model...")
    model = tf.keras.applications.ResNet50(weights="imagenet")
    print("✓ ResNet50 loaded successfully\n")
    
    # Initialize analysis tools
    predictor = Predictor(model)
    feature_extractor = FeatureMapExtractor(model)
    gradcam = GradCAM(model)
    model_summary = ModelSummary(model)
    
    # Display model information
    print("=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)
    
    # Model summary
    summary_str = model_summary.summary()
    print("\n--- Model Summary ---")
    print(summary_str[:500] + "...")  # Print first 500 chars
    
    # Parameter counts
    params = model_summary.parameter_count()
    print(f"\n--- Parameter Counts ---")
    print(f"Total Parameters: {params['formatted']} ({params['count']:,})")
    
    trainable = model_summary.trainable_parameter_count()
    print(f"Trainable Parameters: {trainable['formatted']} ({trainable['percentage']}%)")
    
    non_trainable = model_summary.non_trainable_parameter_count()
    print(f"Non-trainable Parameters: {non_trainable['formatted']} ({non_trainable['percentage']}%)")
    
    # List available layers
    layers = feature_extractor.list_layers()
    print(f"\n--- Available Layers ---")
    print(f"Total layers: {len(layers)}")
    print(f"Sample layers: {layers[:10]}")
    
    # Find convolutional layers for feature map extraction
    conv_layers = [name for name in layers if 'conv' in name.lower()]
    print(f"\nConvolutional layers: {len(conv_layers)}")
    print(f"Sample conv layers: {conv_layers[:5]}")
    
    # Load and process image
    print("\n" + "=" * 70)
    print("IMAGE ANALYSIS")
    print("=" * 70)
    
    image_path = 'diabetic-retinopathy.jpg'
    if not os.path.exists(image_path):
        print(f"⚠ Image '{image_path}' not found. Using random image.")
        image_array = np.random.randn(224, 224, 3).astype(np.float32)
        from PIL import Image as PILImage
        img = PILImage.fromarray(((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8))
    else:
        print(f"Loading image: {image_path}")
        image_array, img = load_and_preprocess_image(image_path)
    
    print(f"✓ Image loaded: {img.size}")
    
    # Run prediction
    print("\n--- Prediction ---")
    output = predictor.predict(image_array)
    print(f"Output shape: {output.shape}")
    
    # Decode predictions
    decoded = tf.keras.applications.resnet50.decode_predictions(output, top=5)[0]
    print("\nTop 5 predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded, 1):
        print(f"  {i}. {label}: {score:.4f}")
    
    predicted_class = int(tf.argmax(output[0]))
    print(f"\nPredicted class index: {predicted_class}")
    
    # Extract feature maps from multiple layers
    print("\n--- Feature Map Extraction ---")
    feature_map_layers = ['conv1_conv', 'conv2_block1_1_conv', 'conv3_block1_1_conv']
    feature_maps = []
    valid_layers = []
    
    for layer_name in feature_map_layers:
        if layer_name in layers:
            print(f"Extracting feature map from '{layer_name}'...")
            feature_map = feature_extractor.extract(image_array, layer_name)
            feature_maps.append(feature_map)
            valid_layers.append(layer_name)
            print(f"  Shape: {feature_map.shape}")
    
    if feature_maps:
        print("\nGenerating feature map visualization...")
        display_feature_maps(feature_maps, valid_layers, max_features=8)
    
    # Generate Grad-CAM
    print("\n--- Grad-CAM Generation ---")
    target_layer = 'conv5_block3_out'  # Last convolutional layer in ResNet50
    print(f"Generating Grad-CAM for layer '{target_layer}'...")
    print(f"Using predicted class: {predicted_class}")
    
    heatmap = gradcam.generate(image_array, layer_name=target_layer, target_class=predicted_class)
    print(f"✓ Heatmap generated: shape={heatmap.shape}, range=[{heatmap.min():.4f}, {heatmap.max():.4f}]")
    
    print("\nGenerating Grad-CAM visualization...")
    display_gradcam_overlay(img, heatmap)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ Model analysis completed successfully!")
    print("✓ Generated outputs:")
    print("  - feature_maps.png: Feature map visualizations")
    print("  - gradcam_visualization.png: Grad-CAM heatmap overlay")
    print("\nModel details:")
    print(f"  - Architecture: ResNet50")
    print(f"  - Total parameters: {params['formatted']}")
    print(f"  - Input shape: {model.input.shape}")
    print(f"  - Output shape: {model.output.shape}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()