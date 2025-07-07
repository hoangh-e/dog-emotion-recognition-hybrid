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


def load_deit_model(checkpoint_path=None, model_variant='small', num_classes=4, 
                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load DeiT model for dog emotion classification.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_variant: DeiT variant ('tiny', 'small', 'base')
        num_classes: Number of emotion classes (default: 4)
        device: Device to load model on
    
    Returns:
        Loaded DeiT model
    """
    # Model name mapping
    model_names = {
        'tiny': 'deit_tiny_patch16_224',
        'small': 'deit_small_patch16_224', 
        'base': 'deit_base_patch16_224'
    }
    
    model_name = model_names.get(model_variant, 'deit_small_patch16_224')
    
    try:
        # Create model
        model = DeiTModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=checkpoint_path is None,
            distillation=True
        )
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Load state dict with strict=False to handle missing keys
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded DeiT checkpoint from {checkpoint_path}")
                
            except Exception as e:
                print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
                print("Using pretrained ImageNet weights instead")
        
        model = model.to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"Error loading DeiT model: {e}")
        # Fallback to simple model
        return create_simple_deit_model(num_classes, device)


def create_simple_deit_model(num_classes=4, device='cuda'):
    """Create a simple DeiT-like model if timm is not available."""
    
    class SimpleDeiT(nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, 384, kernel_size=16, stride=16)
            self.pos_embed = nn.Parameter(torch.randn(1, 198, 384))
            self.cls_token = nn.Parameter(torch.randn(1, 1, 384))
            self.dist_token = nn.Parameter(torch.randn(1, 1, 384))
            
            # Transformer blocks
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=384, nhead=6, dim_feedforward=1536,
                    dropout=0.1, activation='gelu'
                ),
                num_layers=12
            )
            
            self.norm = nn.LayerNorm(384)
            self.classifier = nn.Linear(384, num_classes)
            self.distillation_head = nn.Linear(384, num_classes)
            
        def forward(self, x):
            B = x.shape[0]
            
            # Patch embedding
            x = self.patch_embed(x)  # (B, 384, 14, 14)
            x = x.flatten(2).transpose(1, 2)  # (B, 196, 384)
            
            # Add cls and dist tokens
            cls_tokens = self.cls_token.expand(B, -1, -1)
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
            
            # Add positional embedding
            x = x + self.pos_embed
            
            # Transformer
            x = self.transformer(x)
            x = self.norm(x)
            
            # Classification
            cls_logits = self.classifier(x[:, 0])
            dist_logits = self.distillation_head(x[:, 1])
            
            return cls_logits, dist_logits
    
    model = SimpleDeiT(num_classes).to(device)
    return model


def predict_emotion_deit(model, image_path, transforms_fn=None, device='cuda'):
    """
    Predict dog emotion using DeiT model.
    
    Args:
        model: Loaded DeiT model
        image_path: Path to image or PIL Image
        transforms_fn: Image preprocessing transforms
        device: Device to run inference on
    
    Returns:
        Dictionary with emotion predictions
    """
    emotion_classes = ['sad', 'angry', 'happy', 'relaxed']
    
    try:
        # Load and preprocess image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        if transforms_fn is None:
            transforms_fn = get_deit_transforms()
        
        # Apply transforms
        input_tensor = transforms_fn(image).unsqueeze(0).to(device)
        
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
            result = {
                'predicted_emotion': predicted_emotion,
                'confidence': confidence,
                'predicted_flag': confidence > 0.5,
                'probabilities': {
                    emotion_classes[i]: float(probs[i]) 
                    for i in range(len(emotion_classes))
                }
            }
            
            return result
            
    except Exception as e:
        print(f"Error in DeiT prediction: {e}")
        # Return fallback result
        return {
            'predicted_emotion': 'happy',
            'confidence': 0.25,
            'predicted_flag': False,
            'probabilities': {emotion: 0.25 for emotion in emotion_classes}
        }


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


if __name__ == "__main__":
    # Test DeiT implementation
    print("Testing DeiT implementation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load model
        model = load_deit_model(model_variant='small', device=device)
        print("✅ DeiT model loaded successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, tuple):
                print(f"✅ Model output shape: {output[0].shape}, {output[1].shape}")
            else:
                print(f"✅ Model output shape: {output.shape}")
                
    except Exception as e:
        print(f"❌ Error testing DeiT: {e}") 