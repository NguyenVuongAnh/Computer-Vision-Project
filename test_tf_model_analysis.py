"""
Test script for the Model Analysis module (TensorFlow/Keras version).

This script tests all components of the model analysis module to ensure
they work correctly with a pretrained Keras model.
"""

import numpy as np
import tensorflow as tf

# Import the model analysis module
from model import Predictor, FeatureMapExtractor, GradCAM, ModelSummary


def test_predictor():
    """Test the Predictor class."""
    print("=" * 60)
    print("Testing Predictor")
    print("=" * 60)

    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', input_shape=(224, 224, 3)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])

    # Initialize predictor
    predictor = Predictor(model)
    print(f"✓ Predictor initialized")

    # Create dummy input (single image)
    image = np.random.randn(224, 224, 3).astype(np.float32)

    # Test single prediction
    output = predictor.predict(image)
    print(f"✓ Single prediction completed")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output type: {type(output)}")

    # Validate output
    assert isinstance(output, np.ndarray), "Output should be a NumPy array"
    assert output.shape[0] == 1, "Output should have batch dimension"
    print("✓ Predictor validation passed")

    print()


def test_feature_map_extractor():
    """Test the FeatureMapExtractor class."""
    print("=" * 60)
    print("Testing FeatureMapExtractor")
    print("=" * 60)

    # Create a simple model with named layers
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', name='conv1')(inputs)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', name='conv2')(x)
    x = tf.keras.layers.ReLU(name='relu2')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='pool')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    outputs = tf.keras.layers.Dense(10, name='fc')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    extractor = FeatureMapExtractor(model)
    print(f"✓ FeatureMapExtractor initialized")

    # List layers
    layers = extractor.list_layers()
    print(f"✓ Listed {len(layers)} layers")
    print(f"  - Sample layers: {layers[:3]}")

    # Create dummy input
    image = np.random.randn(224, 224, 3).astype(np.float32)

    # Test single layer extraction
    feature_map = extractor.extract(image, "conv1")
    print(f"✓ Single layer extraction completed")
    print(f"  - Feature map shape: {feature_map.shape}")
    print(f"  - Feature map type: {type(feature_map)}")

    # Validate feature map
    assert isinstance(feature_map, np.ndarray), "Feature map should be a NumPy array"
    assert feature_map.ndim == 3, "Feature map should have 3 dimensions (H, W, C)"
    print("✓ FeatureMapExtractor validation passed")

    # Test with invalid layer name
    try:
        extractor.extract(image, "nonexistent_layer")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for invalid layer: {e}")

    print()


def test_gradcam():
    """Test the GradCAM class."""
    print("=" * 60)
    print("Testing GradCAM")
    print("=" * 60)

    # Create a simple model with named layers
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', name='conv1')(inputs)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', name='conv2')(x)
    x = tf.keras.layers.ReLU(name='relu2')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='pool')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    outputs = tf.keras.layers.Dense(10, name='fc')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    gradcam = GradCAM(model)
    print(f"✓ GradCAM initialized")

    # Create dummy input
    image = np.random.randn(224, 224, 3).astype(np.float32)

    # Test Grad-CAM generation
    heatmap = gradcam.generate(image, layer_name="conv2")
    print(f"✓ Grad-CAM generation completed")
    print(f"  - Heatmap shape: {heatmap.shape}")
    print(f"  - Heatmap type: {type(heatmap)}")
    print(f"  - Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")

    # Validate heatmap
    assert isinstance(heatmap, np.ndarray), "Heatmap should be a NumPy array"
    assert heatmap.ndim == 2, "Heatmap should have 2 dimensions (H, W)"
    assert heatmap.shape == (224, 224), f"Heatmap shape should be (224, 224), got {heatmap.shape}"
    assert heatmap.min() >= 0.0, "Heatmap values should be >= 0"
    assert heatmap.max() <= 1.0, "Heatmap values should be <= 1"
    print("✓ GradCAM validation passed")

    # Test with specific target class
    heatmap_class5 = gradcam.generate(image, layer_name="conv2", target_class=5)
    print(f"✓ Grad-CAM with target class completed")
    print(f"  - Heatmap shape: {heatmap_class5.shape}")

    # Test with invalid layer name
    try:
        gradcam.generate(image, layer_name="nonexistent_layer")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for invalid layer: {e}")

    print()


def test_model_summary():
    """Test the ModelSummary class."""
    print("=" * 60)
    print("Testing ModelSummary")
    print("=" * 60)

    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', input_shape=(224, 224, 3)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])

    summary = ModelSummary(model)
    print(f"✓ ModelSummary initialized")

    # Test summary string
    summary_str = summary.summary()
    print(f"✓ Summary retrieved")
    print(f"  - Summary type: {type(summary_str)}")
    print(f"  - Summary length: {len(summary_str)} characters")
    assert isinstance(summary_str, str), "Summary should be a string"
    assert len(summary_str) > 0, "Summary should not be empty"

    # Test parameter count
    params = summary.parameter_count()
    print(f"✓ Parameter count retrieved")
    print(f"  - Total: {params['count']} ({params['formatted']})")
    assert 'count' in params, "Parameter count should have 'count' key"
    assert 'formatted' in params, "Parameter count should have 'formatted' key"
    assert isinstance(params['count'], int), "Parameter count should be an integer"

    # Test trainable parameter count
    trainable = summary.trainable_parameter_count()
    print(f"✓ Trainable parameter count retrieved")
    print(f"  - Count: {trainable['count']} ({trainable['formatted']})")
    print(f"  - Percentage: {trainable['percentage']}%")
    assert 'count' in trainable, "Trainable count should have 'count' key"
    assert 'formatted' in trainable, "Trainable count should have 'formatted' key"
    assert 'percentage' in trainable, "Trainable count should have 'percentage' key"

    # Test non-trainable parameter count
    non_trainable = summary.non_trainable_parameter_count()
    print(f"✓ Non-trainable parameter count retrieved")
    print(f"  - Count: {non_trainable['count']} ({non_trainable['formatted']})")
    print(f"  - Percentage: {non_trainable['percentage']}%")
    assert 'count' in non_trainable, "Non-trainable count should have 'count' key"
    assert 'formatted' in non_trainable, "Non-trainable count should have 'formatted' key"
    assert 'percentage' in non_trainable, "Non-trainable count should have 'percentage' key"

    # Verify counts add up
    total = params['count']
    trainable_sum = trainable['count']
    non_trainable_sum = non_trainable['count']
    assert trainable_sum + non_trainable_sum == total, \
        f"Trainable ({trainable_sum}) + Non-trainable ({non_trainable_sum}) should equal total ({total})"

    print("✓ ModelSummary validation passed")

    print()


def test_with_pretrained_model():
    """Test with a pretrained model from TensorFlow."""
    print("=" * 60)
    print("Testing with Pretrained Model (ResNet50)")
    print("=" * 60)

    try:
        # Load pretrained model
        model = tf.keras.applications.ResNet50(weights="imagenet")
        print(f"✓ ResNet50 loaded")

        # Test Predictor
        predictor = Predictor(model)
        image = np.random.randn(224, 224, 3).astype(np.float32)
        output = predictor.predict(image)
        print(f"✓ Predictor works with ResNet50")
        print(f"  - Output shape: {output.shape}")

        # Test FeatureMapExtractor
        extractor = FeatureMapExtractor(model)
        layers = extractor.list_layers()
        print(f"✓ FeatureMapExtractor works with ResNet50")
        print(f"  - Available layers: {len(layers)}")

        if "conv3_block4_out" in layers:
            feature_map = extractor.extract(image, "conv3_block4_out")
            print(f"  - conv3_block4_out shape: {feature_map.shape}")

        # Test GradCAM
        gradcam = GradCAM(model)
        heatmap = gradcam.generate(image, layer_name="conv5_block3_out")
        print(f"✓ GradCAM works with ResNet50")
        print(f"  - Heatmap shape: {heatmap.shape}")
        print(f"  - Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")

        # Test ModelSummary
        summary = ModelSummary(model)
        params = summary.parameter_count()
        print(f"✓ ModelSummary works with ResNet50")
        print(f"  - Total parameters: {params['formatted']}")
        print(f"  - Model size: {summary.parameter_count()['formatted']}")

    except ImportError:
        print("⚠ TensorFlow not installed, skipping pretrained model test")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Model Analysis Module - Test Suite (TensorFlow/Keras)")
    print("=" * 60 + "\n")

    try:
        test_predictor()
        test_feature_map_extractor()
        test_gradcam()
        test_model_summary()
        test_with_pretrained_model()

        print("=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())