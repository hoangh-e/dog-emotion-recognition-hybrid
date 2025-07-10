"""
DeiT (Data-efficient Image Transformers) models for dog emotion classification.

This module provides DeiT implementations optimized for dog emotion classification 
with 4 emotion classes: sad, angry, happy, relaxed.

Based on "Training data-efficient image transformers & distillation through attention"
by Hugo Touvron et al. (2021).
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import math
from functools import partial

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class DeiTModel(nn.Module):
    """DeiT model with distillation support for dog emotion classification."""
    
    def __init__(self, model_name='deit_small_patch16_224', num_classes=4, 
                 pretrained=True, distillation=True):
        super().__init__()
        self.distillation = distillation
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for DeiT. Install with: pip install timm>=0.6.0")
        
        # Load base DeiT model
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            distillation=distillation
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Distillation head (if enabled)
        if distillation:
            self.distillation_head = nn.Linear(self.feature_dim, num_classes)
        
        self.num_classes = num_classes
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        if self.distillation:
            # DeiT with distillation returns (cls_token, dist_token)
            if isinstance(features, tuple):
                cls_features, dist_features = features
                cls_logits = self.classifier(cls_features)
                dist_logits = self.distillation_head(dist_features)
                return cls_logits, dist_logits
            else:
                # Fallback if distillation not properly configured
                logits = self.classifier(features)
                return logits, logits
        else:
            logits = self.classifier(features)
            return logits


def load_deit_model(model_path=None, architecture='deit_small', num_classes=4, input_size=224, device='cuda'):
    """
    Load DeiT model for dog emotion classification.
    
    Args:
        model_path: Path to model checkpoint
        architecture: DeiT variant ('deit_tiny', 'deit_small', 'deit_base')
        num_classes: Number of emotion classes (default: 4)
        input_size: Input image size (default: 224)
        device: Device to load model on
    
    Returns:
        Loaded DeiT model
    """
    # Model name mapping
    model_names = {
        'deit_tiny': 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224', 
        'deit_base': 'deit_base_patch16_224'
    }
    
    model_name = model_names.get(architecture, 'deit_small_patch16_224')
    
    if not TIMM_AVAILABLE:
        raise ImportError("timm is required for DeiT. Install with: pip install timm>=0.6.0")
    
    # Create model
    model = DeiTModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=model_path is None,
        distillation=True
    )
    
    # Load checkpoint if provided
    if model_path and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict with strict=False to handle missing keys
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded DeiT checkpoint from {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Could not load checkpoint {model_path}: {e}")
    elif model_path and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    model = model.to(device)
    model.eval()
    
    return model


def predict_emotion_deit(image_path, model, transform=None, head_bbox=None, device='cuda',
                         emotion_classes=['angry', 'happy', 'relaxed', 'sad']):
    """
    Predict dog emotion using DeiT model.
    
    Args:
        image_path: Path to image or PIL Image
        model: Loaded DeiT model
        transform: Image preprocessing transforms
        head_bbox: Optional head bounding box for cropping
        device: Device to run inference on
        emotion_classes: List of emotion class names
    
    Returns:
        Dictionary with emotion predictions
    """
    
    # Load and preprocess image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
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
    
    if transform is None:
        transform = get_deit_transforms()
    
    # Apply transforms
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Handle distillation output
        if isinstance(outputs, tuple):
            cls_logits, dist_logits = outputs
            # Average predictions from both heads
            logits = (cls_logits + dist_logits) / 2
        else:
            logits = outputs
        
        # Get probabilities
        probabilities = torch.softmax(logits, dim=1)
        probs = probabilities.cpu().numpy()[0]
        
        # Get predicted class
        predicted_idx = np.argmax(probs)
        predicted_emotion = emotion_classes[predicted_idx]
        confidence = float(probs[predicted_idx])
        
        # Create result dictionary
        emotion_scores = {}
        for i, emotion in enumerate(emotion_classes):
            emotion_scores[emotion] = float(probs[i])
        
        emotion_scores['predicted'] = True
        
        return emotion_scores


def get_deit_transforms(input_size=224, is_training=False):
    """Get DeiT-specific image transforms."""
    
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def extract_head_bbox_deit(image_path, model, device='cuda'):
    """Extract head bounding box for improved DeiT prediction."""
    try:
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = np.array(image_path)
            
        if image is None:
            return None
            
        # Simple head detection using upper portion
        h, w = image.shape[:2]
        
        # Assume head is in upper 60% of image
        head_bbox = {
            'x': int(w * 0.1),
            'y': int(h * 0.05), 
            'width': int(w * 0.8),
            'height': int(h * 0.6)
        }
        
        return head_bbox
        
    except Exception as e:
        print(f"Error in head detection: {e}")
        return None


# Model variants configuration
DEIT_VARIANTS = {
    'deit_tiny': {
        'model_name': 'deit_tiny_patch16_224',
        'input_size': 224,
        'params': '5.7M'
    },
    'deit_small': {
        'model_name': 'deit_small_patch16_224', 
        'input_size': 224,
        'params': '22M'
    },
    'deit_base': {
        'model_name': 'deit_base_patch16_224',
        'input_size': 224,
        'params': '86M'
    }
}


def load_deit_model_standard(model_path, architecture='deit_small', num_classes=4, input_size=224, device='cuda'):
    """
    Standardized load function for DeiT model to match other modules.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model checkpoint
    architecture : str
        DeiT architecture ('deit_tiny', 'deit_small', 'deit_base')
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
    
    # Map architecture names
    variant_map = {
        'deit_tiny': 'tiny',
        'deit_small': 'small', 
        'deit_base': 'base'
    }
    variant = variant_map.get(architecture, 'small')
    
    # Load model using existing function
    model = load_deit_model(model_path, variant, num_classes, device)
    
    # Create transform
    transform = get_deit_transforms(input_size, is_training=False)
    
    print(f"✅ DeiT model loaded successfully on {device}")
    print(f"🎯 Model type: {architecture.upper()}")
    print(f"📏 Input size: {input_size}x{input_size}")
    
    return model, transform


def predict_emotion_deit_standard(image_path, model, transform, head_bbox=None, device='cuda',
                                 emotion_classes=['angry', 'happy', 'relaxed', 'sad']):
    """
    Standardized predict function for DeiT model to match other modules.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    model : torch.nn.Module
        Loaded DeiT model
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
    # Load and preprocess image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path.convert('RGB')
    
    # Crop head region if bbox provided
    if head_bbox is not None:
        x1, y1, x2, y2 = head_bbox
        width, height = image.size
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        x2 = max(x1, min(int(x2), width))
        y2 = max(y1, min(int(y2), height))
        image = image.crop((x1, y1, x2, y2))
    
    # Use existing prediction function
    result = predict_emotion_deit(image, model, transform, device=device)
    
    # Convert to standardized format if needed
    if 'probabilities' in result:
        emotion_scores = {}
        for emotion in emotion_classes:
            emotion_scores[emotion] = result['probabilities'].get(emotion, 0.0)
        emotion_scores['predicted'] = result.get('predicted_flag', False)
        return emotion_scores
    else:
        return result


# Alias functions for consistency with other modules
# Note: Renamed to avoid conflicts with existing functions

def load_deit_model_main(model_path, architecture='deit_small', num_classes=4, input_size=224, device='cuda'):
    """
    Main load function for DeiT model (standardized version).
    """
    return load_deit_model_standard(model_path, architecture, num_classes, input_size, device)


def predict_emotion_deit_main(image_path, model, transform, head_bbox=None, device='cuda',
                             emotion_classes=['angry', 'happy', 'relaxed', 'sad']):
    """
    Main predict function for DeiT model (standardized version).
    """
    return predict_emotion_deit_standard(image_path, model, transform, head_bbox, device, emotion_classes)


if __name__ == "__main__":
    # Test DeiT implementation
    print("Testing DeiT implementation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load model
        model = load_deit_model(model_variant='small', device=device)
        print("✅ DeiT model loaded successfully")
        
        # Test transforms
        transforms_fn = get_deit_transforms()
        print("✅ Transforms created successfully")
                
    except Exception as e:
        print(f"❌ Error testing DeiT: {e}") 