"""
ResMLP (Residual MLP) models for dog emotion classification.

This module provides ResMLP implementations optimized for 
dog emotion classification with 3 emotion classes: angry, happy, relaxed.

ResMLP is a pure MLP architecture that uses residual connections
and doesn't use convolutions or attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os


class Affine(nn.Module):
    """Affine transformation layer for ResMLP."""
    
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        return self.alpha * x + self.beta


class ResMLP_Block(nn.Module):
    """ResMLP block with residual connections."""
    
    def __init__(self, dim, seq_len, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        
        # Pre-norm
        self.norm1 = Affine(dim)
        self.norm2 = Affine(dim)
        
        # Token mixing MLP
        self.token_mix = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len),
            nn.Dropout(dropout)
        )
        
        # Channel mixing MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, dim)
        
        # Token mixing
        x = x + self.token_mix(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        
        # Channel mixing
        x = x + self.channel_mix(self.norm2(x))
        
        return x


class ResMLP_Model(nn.Module):
    """ResMLP model for emotion classification."""
    
    def __init__(self, num_classes=3, img_size=224, patch_size=16, 
                 embed_dim=384, depth=12, mlp_ratio=4, dropout=0.0):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # ResMLP blocks
        self.blocks = nn.ModuleList([
            ResMLP_Block(embed_dim, self.num_patches, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = Affine(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Apply ResMLP blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, embed_dim)
        
        # Classification head
        x = self.norm(x)
        x = self.head(x)
        
        return x


def load_resmlp_model(model_path, architecture='resmlp_12', num_classes=3, input_size=224, device='cuda'):
    """
    Load a pre-trained ResMLP model for dog emotion classification.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model checkpoint
    architecture : str
        ResMLP architecture ('resmlp_12', 'resmlp_24', 'resmlp_36')
    num_classes : int
        Number of emotion classes (default: 3)
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
    
    # Model configurations
    configs = {
        'resmlp_12': {'depth': 12, 'embed_dim': 384, 'mlp_ratio': 4},
        'resmlp_24': {'depth': 24, 'embed_dim': 384, 'mlp_ratio': 4},
        'resmlp_36': {'depth': 36, 'embed_dim': 384, 'mlp_ratio': 4}
    }
    
    if architecture not in configs:
        raise ValueError(f"Unsupported ResMLP architecture: {architecture}")
    
    config = configs[architecture]
    
    # Create model
    model = ResMLP_Model(
        num_classes=num_classes,
        img_size=input_size,
        patch_size=16,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        mlp_ratio=config['mlp_ratio']
    )
    
    print(f"🏗️  Created {architecture.upper()} model")
    
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
    
    # Create preprocessing transform
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


def predict_emotion_resmlp(image_path, model, transform, head_bbox=None, device='cuda',
                          emotion_classes=['angry', 'happy', 'relaxed']):
    """
    Predict dog emotion using ResMLP model.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    model : torch.nn.Module
        Loaded ResMLP model
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
        print(f"❌ Error in ResMLP emotion prediction: {e}")
        raise RuntimeError(f"ResMLP prediction failed: {e}")


def get_resmlp_transforms(input_size=224, is_training=True):
    """
    Get preprocessing transforms for ResMLP models.
    
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
        return transforms.Compose([
            transforms.Resize((int(input_size * 1.14), int(input_size * 1.14))),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_resmlp_model(architecture='resmlp_12', num_classes=3, pretrained=False):
    """
    Create a ResMLP model for emotion classification.
    
    Parameters:
    -----------
    architecture : str
        ResMLP architecture ('resmlp_12', 'resmlp_24', 'resmlp_36')
    num_classes : int
        Number of emotion classes (default: 3)
    pretrained : bool
        Whether to load pretrained weights (default: False)
        
    Returns:
    --------
    torch.nn.Module
        ResMLP model
    """
    
    # Model configurations
    configs = {
        'resmlp_12': {'depth': 12, 'embed_dim': 384, 'mlp_ratio': 4},
        'resmlp_24': {'depth': 24, 'embed_dim': 384, 'mlp_ratio': 4},
        'resmlp_36': {'depth': 36, 'embed_dim': 384, 'mlp_ratio': 4}
    }
    
    if architecture not in configs:
        raise ValueError(f"Unsupported ResMLP architecture: {architecture}")
    
    config = configs[architecture]
    
    model = ResMLP_Model(
        num_classes=num_classes,
        img_size=224,
        patch_size=16,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        mlp_ratio=config['mlp_ratio']
    )
    
    return model


# Convenience functions for specific architectures
def load_resmlp_12_model(model_path, num_classes=3, input_size=224, device='cuda'):
    return load_resmlp_model(model_path, 'resmlp_12', num_classes, input_size, device)

def load_resmlp_24_model(model_path, num_classes=3, input_size=224, device='cuda'):
    return load_resmlp_model(model_path, 'resmlp_24', num_classes, input_size, device)

def load_resmlp_36_model(model_path, num_classes=3, input_size=224, device='cuda'):
    return load_resmlp_model(model_path, 'resmlp_36', num_classes, input_size, device)

def predict_emotion_resmlp_12(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_resmlp(image_path, model, transform, head_bbox, device)

def predict_emotion_resmlp_24(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_resmlp(image_path, model, transform, head_bbox, device)

def predict_emotion_resmlp_36(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_resmlp(image_path, model, transform, head_bbox, device) 