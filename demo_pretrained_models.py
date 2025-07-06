#!/usr/bin/env python3
"""
Demo script to test pretrained models from dog_emotion_classification package.

This script demonstrates how to use pretrained models for dog emotion classification
without training from scratch, significantly reducing computational costs.
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import time
import os
import sys

# Add package to path
sys.path.append('.')

# Import all classification modules
try:
    from dog_emotion_classification import (
        resnet, efficientnet, vit, convnext, densenet, mobilenet,
        vgg, inception, alexnet, squeezenet, shufflenet, swin,
        deit, nasnet, mlp_mixer
    )
    print("✅ Successfully imported all classification modules")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device}")

# Emotion classes
EMOTION_CLASSES = ['sad', 'angry', 'happy', 'relaxed']

def create_sample_image(size=(224, 224)):
    """Create a sample image for testing."""
    # Create a random RGB image
    image_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    return image

def test_pretrained_model(model_name, create_model_func, sample_image):
    """Test a pretrained model with sample image."""
    print(f"\n🧪 Testing {model_name}...")
    
    try:
        start_time = time.time()
        
        # Create pretrained model
        if 'timm' in model_name.lower():
            # For TIMM models, they handle pretrained internally
            model = create_model_func()
        else:
            # For torchvision models
            model = create_model_func(num_classes=4, pretrained=True)
        
        model.to(device)
        model.eval()
        
        # Get transforms
        if hasattr(model, 'transform'):
            transform = model.transform
        else:
            # Default transform
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # Preprocess image
        input_tensor = transform(sample_image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                # For models like DeiT with distillation
                outputs = outputs[0]
            
            probabilities = torch.softmax(outputs, dim=1)
            probs = probabilities.cpu().numpy()[0]
        
        # Get prediction
        predicted_idx = np.argmax(probs)
        predicted_emotion = EMOTION_CLASSES[predicted_idx]
        confidence = probs[predicted_idx]
        
        # Calculate model size
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
        
        load_time = time.time() - start_time
        
        print(f"  ✅ {model_name}")
        print(f"     Predicted: {predicted_emotion} (confidence: {confidence:.3f})")
        print(f"     Parameters: {param_count:,} ({model_size_mb:.1f} MB)")
        print(f"     Load time: {load_time:.2f}s")
        
        return {
            'name': model_name,
            'success': True,
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'parameters': param_count,
            'size_mb': model_size_mb,
            'load_time': load_time
        }
        
    except Exception as e:
        print(f"  ❌ {model_name} failed: {e}")
        return {
            'name': model_name,
            'success': False,
            'error': str(e)
        }

def main():
    """Main function to test all pretrained models."""
    print("🎯 Dog Emotion Classification - Pretrained Models Demo")
    print("=" * 60)
    
    # Create sample image
    sample_image = create_sample_image()
    print(f"📸 Created sample image: {sample_image.size}")
    
    # Define models to test
    models_to_test = [
        # Torchvision models (highly recommended)
        ("ResNet50", lambda: resnet.create_resnet_model('resnet50')),
        ("EfficientNet B0", lambda: efficientnet.create_efficientnet_model('efficientnet_b0')),
        ("EfficientNet B2", lambda: efficientnet.create_efficientnet_model('efficientnet_b2')),
        ("ConvNeXt Tiny", lambda: convnext.create_convnext_model('convnext_tiny')),
        ("ViT B/16", lambda: vit.create_vit_model('vit_b_16')),
        
        # Mobile optimized models
        ("MobileNet v2", lambda: mobilenet.create_mobilenet_model('mobilenet_v2')),
        ("MobileNet v3 Small", lambda: mobilenet.create_mobilenet_model('mobilenet_v3_small')),
        ("SqueezeNet 1.1", lambda: squeezenet.create_squeezenet_model('squeezenet1_1')),
        ("ShuffleNet v2 x1.0", lambda: shufflenet.create_shufflenet_model('shufflenet_v2_x1_0')),
        
        # Other torchvision models
        ("DenseNet121", lambda: densenet.create_densenet_model('densenet121')),
        ("VGG16", lambda: vgg.create_vgg_model('vgg16')),
        ("Inception v3", lambda: inception.create_inception_model('inception_v3')),
        ("AlexNet", lambda: alexnet.create_alexnet_model()),
        
        # TIMM models (require timm installation)
        ("Swin Transformer Tiny", lambda: swin.create_swin_model('swin_t')),
        ("DeiT Small (TIMM)", lambda: deit.load_deit_model(model_variant='small')),
        ("NASNet Mobile (TIMM)", lambda: nasnet.load_nasnet_model(model_variant='mobile')),
        ("MLP-Mixer Base (TIMM)", lambda: mlp_mixer.load_mlp_mixer_model(model_variant='base')),
    ]
    
    # Test all models
    results = []
    successful_models = []
    failed_models = []
    
    for model_name, create_func in models_to_test:
        result = test_pretrained_model(model_name, create_func, sample_image)
        results.append(result)
        
        if result['success']:
            successful_models.append(result)
        else:
            failed_models.append(result)
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Successful models: {len(successful_models)}/{len(models_to_test)}")
    print(f"❌ Failed models: {len(failed_models)}/{len(models_to_test)}")
    
    if successful_models:
        print(f"\n🏆 TOP PERFORMING MODELS:")
        # Sort by confidence
        successful_models.sort(key=lambda x: x['confidence'], reverse=True)
        
        for i, model in enumerate(successful_models[:5]):
            print(f"  {i+1}. {model['name']}")
            print(f"     Confidence: {model['confidence']:.3f}")
            print(f"     Size: {model['size_mb']:.1f} MB")
            print(f"     Load time: {model['load_time']:.2f}s")
    
    if failed_models:
        print(f"\n❌ FAILED MODELS:")
        for model in failed_models:
            print(f"  - {model['name']}: {model['error']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 60)
    
    # Find best models by category
    if successful_models:
        # Smallest model
        smallest = min(successful_models, key=lambda x: x['size_mb'])
        print(f"🔹 Smallest model: {smallest['name']} ({smallest['size_mb']:.1f} MB)")
        
        # Fastest loading
        fastest = min(successful_models, key=lambda x: x['load_time'])
        print(f"🔹 Fastest loading: {fastest['name']} ({fastest['load_time']:.2f}s)")
        
        # Most confident
        most_confident = max(successful_models, key=lambda x: x['confidence'])
        print(f"🔹 Most confident: {most_confident['name']} ({most_confident['confidence']:.3f})")
        
        # Balanced choice
        balanced_models = [m for m in successful_models if m['size_mb'] < 50 and m['load_time'] < 5]
        if balanced_models:
            balanced = max(balanced_models, key=lambda x: x['confidence'])
            print(f"🔹 Balanced choice: {balanced['name']} ({balanced['confidence']:.3f} confidence, {balanced['size_mb']:.1f} MB)")
    
    print(f"\n🎯 USAGE EXAMPLE:")
    print("=" * 60)
    print("""
# Quick start with EfficientNet B0
from dog_emotion_classification import efficientnet

# Create pretrained model
model = efficientnet.create_efficientnet_model('efficientnet_b0', num_classes=4, pretrained=True)

# Load image and predict
from PIL import Image
image = Image.open('dog_image.jpg')
transform = efficientnet.get_efficientnet_transforms(is_training=False)
prediction = efficientnet.predict_emotion_efficientnet(image, model, transform)

print(f"Predicted emotion: {prediction}")
""")
    
    print(f"\n✅ Demo completed! Check PRETRAINED_MODELS_SUMMARY.md for detailed information.")

if __name__ == "__main__":
    main() 