#!/usr/bin/env python3
"""
Test script for newly implemented algorithms: DeiT, NASNet, and MLP-Mixer
"""

import torch
import numpy as np
from PIL import Image
import os
import sys

# Add the package to path
sys.path.append('.')

def test_deit():
    """Test DeiT implementation"""
    print("🧪 Testing DeiT implementation...")
    
    try:
        from dog_emotion_classification.deit import load_deit_model, predict_emotion_deit
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Test model loading
        model = load_deit_model(model_variant='small', device=device)
        print("✅ DeiT model loaded successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, tuple):
                print(f"✅ DeiT output shape: {output[0].shape}, {output[1].shape}")
            else:
                print(f"✅ DeiT output shape: {output.shape}")
        
        # Test prediction function
        dummy_image = Image.new('RGB', (224, 224), color='red')
        result = predict_emotion_deit(model, dummy_image, device=device)
        print(f"✅ DeiT prediction: {result['predicted_emotion']} (confidence: {result['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ DeiT test failed: {e}")
        return False


def test_nasnet():
    """Test NASNet implementation"""
    print("\n🧪 Testing NASNet implementation...")
    
    try:
        from dog_emotion_classification.nasnet import load_nasnet_model, predict_emotion_nasnet
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Test model loading
        model = load_nasnet_model(model_variant='mobile', device=device)
        print("✅ NASNet model loaded successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ NASNet output shape: {output.shape}")
        
        # Test prediction function
        dummy_image = Image.new('RGB', (224, 224), color='blue')
        result = predict_emotion_nasnet(model, dummy_image, device=device)
        print(f"✅ NASNet prediction: {result['predicted_emotion']} (confidence: {result['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ NASNet test failed: {e}")
        return False


def test_mlp_mixer():
    """Test MLP-Mixer implementation"""
    print("\n🧪 Testing MLP-Mixer implementation...")
    
    try:
        from dog_emotion_classification.mlp_mixer import load_mlp_mixer_model, predict_emotion_mlp_mixer
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Test model loading
        model = load_mlp_mixer_model(model_variant='base', device=device)
        print("✅ MLP-Mixer model loaded successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ MLP-Mixer output shape: {output.shape}")
        
        # Test prediction function
        dummy_image = Image.new('RGB', (224, 224), color='green')
        result = predict_emotion_mlp_mixer(model, dummy_image, device=device)
        print(f"✅ MLP-Mixer prediction: {result['predicted_emotion']} (confidence: {result['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ MLP-Mixer test failed: {e}")
        return False


def test_package_import():
    """Test package import with new modules"""
    print("\n🧪 Testing package import...")
    
    try:
        from dog_emotion_classification import deit, nasnet, mlp_mixer
        print("✅ All new modules imported successfully")
        
        # Check version
        import dog_emotion_classification
        print(f"✅ Package version: {dog_emotion_classification.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Package import test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Testing newly implemented algorithms...")
    print("=" * 50)
    
    results = []
    
    # Test individual algorithms
    results.append(test_deit())
    results.append(test_nasnet())
    results.append(test_mlp_mixer())
    results.append(test_package_import())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    test_names = ["DeiT", "NASNet", "MLP-Mixer", "Package Import"]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n🎯 Overall Success Rate: {success_rate:.1f}% ({sum(results)}/{len(results)})")
    
    if all(results):
        print("\n🎉 All tests passed! New algorithms are ready for use.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 