"""
DenseNet models for dog emotion classification.

This module provides DenseNet121, DenseNet169, and DenseNet201 implementations optimized for 
dog emotion classification with 4 emotion classes: sad, angry, happy, relaxed.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os


def load_densenet_model(model_path, architecture='densenet121', num_classes=4, input_size=224, device='cuda'):
    """
    Load a pre-trained DenseNet model for dog emotion classification.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model checkpoint
    architecture : str
        DenseNet architecture ('densenet121', 'densenet169', 'densenet201')
    num_classes : int
        Number of emotion classes (default: 4)
    input_size : int
        Input image size (default: 224)
    device : str
        Device to load model on ('cuda' or 'cpu')
        
    Returns:
    --------
    tuple
        (model, transform) - loaded model and preprocessing transform
    """
    
    print(f"🔄 Loading {architecture.upper()} model from: {model_path}")
    
    # Create model based on architecture
    if architecture.lower() == 'densenet121':
        model = models.densenet121(pretrained=False)
        print(f"🏗️  Created DenseNet121 base model")
    elif architecture.lower() == 'densenet169':
        model = models.densenet169(pretrained=False)
        print(f"🏗️  Created DenseNet169 base model")
    elif architecture.lower() == 'densenet201':
        model = models.densenet201(pretrained=False)
        print(f"🏗️  Created DenseNet201 base model")
    else:
        raise ValueError(f"Unsupported DenseNet architecture: {architecture}")
    
    # Modify final classification layer for emotion classes
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    print(f"🔧 Modified classifier layer: Linear(in_features={in_features}, out_features={num_classes})")
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"📦 Loading from 'model_state_dict' key")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"📦 Loading from 'state_dict' key")
            else:
                state_dict = checkpoint
                print(f"📦 Using checkpoint directly as state_dict")
        else:
            state_dict = checkpoint
            print(f"📦 Using checkpoint directly as state_dict")
        
        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"✅ Loaded model with strict=True")
        except RuntimeError as e:
            print(f"⚠️  Strict loading failed, trying strict=False: {e}")
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded model with strict=False")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully on {device}")
    
    # Create preprocessing transform for DenseNet
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print(f"🎯 Model type: {architecture.upper()}")
    print(f"📏 Input size: {input_size}x{input_size}")
    
    return model, transform


def predict_emotion_densenet(image_path, model, transform, head_bbox=None, device='cuda',
                            emotion_classes=['angry', 'happy', 'relaxed', 'sad']):
    """
    Predict dog emotion using DenseNet model.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    model : torch.nn.Module
        Loaded DenseNet model
    transform : torchvision.transforms.Compose
        Preprocessing transform
    head_bbox : list, optional
        Bounding box [x1, y1, x2, y2] to crop head region
    device : str
        Device for inference
    emotion_classes : list
        List of emotion class names
        
    Returns:
    --------
    dict
        Emotion predictions with scores and predicted flag
    """
    
    try:
        # Load and preprocess image
        if isinstance(image_path, str):
            # Load from file path
            image = Image.open(image_path).convert('RGB')
        else:
            # Assume it's already a PIL Image
            image = image_path.convert('RGB')
        
        # Crop head region if bbox provided
        if head_bbox is not None:
            x1, y1, x2, y2 = head_bbox
            # Ensure coordinates are within image bounds
            width, height = image.size
            x1 = max(0, min(int(x1), width))
            y1 = max(0, min(int(y1), height))
            x2 = max(x1, min(int(x2), width))
            y2 = max(y1, min(int(y2), height))
            
            # Crop the head region
            image = image.crop((x1, y1, x2, y2))
        
        # Apply preprocessing transform
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probs = probabilities.cpu().numpy()[0]
        
        # Create result dictionary
        emotion_scores = {}
        for i, emotion in enumerate(emotion_classes):
            emotion_scores[emotion] = float(probs[i])
        
        emotion_scores['predicted'] = True
        
        return emotion_scores
        
    except Exception as e:
        print(f"❌ Error in DenseNet emotion prediction: {e}")
        # Return default scores on error
        emotion_scores = {emotion: 0.0 for emotion in emotion_classes}
        emotion_scores['predicted'] = False
        return emotion_scores


def get_densenet_transforms(input_size=224, is_training=True):
    """
    Get preprocessing transforms for DenseNet models.
    
    Parameters:
    -----------
    input_size : int
        Input image size (default: 224)
    is_training : bool
        Whether transforms are for training or inference
        
    Returns:
    --------
    torchvision.transforms.Compose
        Preprocessing transforms
    """
    
    if is_training:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Inference transforms
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform


def create_densenet_model(architecture='densenet121', num_classes=4, pretrained=True):
    """
    Create a DenseNet model for dog emotion classification.
    
    Parameters:
    -----------
    architecture : str
        DenseNet architecture ('densenet121', 'densenet169', 'densenet201')
    num_classes : int
        Number of emotion classes
    pretrained : bool
        Whether to use ImageNet pretrained weights
        
    Returns:
    --------
    torch.nn.Module
        DenseNet model
    """
    
    if architecture.lower() == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
    elif architecture.lower() == 'densenet169':
        model = models.densenet169(pretrained=pretrained)
    elif architecture.lower() == 'densenet201':
        model = models.densenet201(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported DenseNet architecture: {architecture}")
    
    # Modify final layer for emotion classification
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    
    return model


# Convenience functions for specific architectures
def load_densenet121_model(model_path, num_classes=4, input_size=224, device='cuda'):
    """Load DenseNet121 model for emotion classification."""
    return load_densenet_model(model_path, 'densenet121', num_classes, input_size, device)


def load_densenet169_model(model_path, num_classes=4, input_size=224, device='cuda'):
    """Load DenseNet169 model for emotion classification."""
    return load_densenet_model(model_path, 'densenet169', num_classes, input_size, device)


def load_densenet201_model(model_path, num_classes=4, input_size=224, device='cuda'):
    """Load DenseNet201 model for emotion classification."""
    return load_densenet_model(model_path, 'densenet201', num_classes, input_size, device)


def predict_emotion_densenet121(image_path, model, transform, head_bbox=None, device='cuda'):
    """Predict emotion using DenseNet121."""
    return predict_emotion_densenet(image_path, model, transform, head_bbox, device)


def predict_emotion_densenet169(image_path, model, transform, head_bbox=None, device='cuda'):
    """Predict emotion using DenseNet169."""
    return predict_emotion_densenet(image_path, model, transform, head_bbox, device)


def predict_emotion_densenet201(image_path, model, transform, head_bbox=None, device='cuda'):
    """Predict emotion using DenseNet201."""
    return predict_emotion_densenet(image_path, model, transform, head_bbox, device) 