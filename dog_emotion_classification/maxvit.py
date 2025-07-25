"""
MaxViT (Multi-Axis Vision Transformer) models for dog emotion classification.

This module provides MaxViT implementations optimized for 
dog emotion classification with 3 emotion classes: angry, happy, relaxed.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import math


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution block."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=4):
        super(MBConv, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Pointwise linear
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class WindowAttention(nn.Module):
    """Window-based multi-head self attention."""
    
    def __init__(self, dim, window_size=7, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, H, W, C = x.shape
        
        # Pad if needed
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        # Window partition
        x = x.view(B, Hp // self.window_size, self.window_size, 
                  Wp // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B * (Hp // self.window_size) * (Wp // self.window_size), 
                  self.window_size * self.window_size, C)
        
        # Multi-head attention
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], C)
        x = self.proj(x)
        
        # Window reverse
        x = x.view(B, Hp // self.window_size, Wp // self.window_size,
                  self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, C)
        
        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        return x


class GridAttention(nn.Module):
    """Grid-based multi-head self attention."""
    
    def __init__(self, dim, grid_size=7, num_heads=8):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, H, W, C = x.shape
        
        # Grid partition
        x = x.view(B, self.grid_size, H // self.grid_size, 
                  self.grid_size, W // self.grid_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B * self.grid_size * self.grid_size, 
                  (H // self.grid_size) * (W // self.grid_size), C)
        
        # Multi-head attention
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], C)
        x = self.proj(x)
        
        # Grid reverse
        x = x.view(B, self.grid_size, self.grid_size,
                  H // self.grid_size, W // self.grid_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, C)
        
        return x


class MaxViTBlock(nn.Module):
    """MaxViT block with MBConv and Multi-axis attention."""
    
    def __init__(self, dim, num_heads=8, window_size=7, grid_size=7):
        super().__init__()
        
        # MBConv block
        self.mbconv = MBConv(dim, dim, kernel_size=3, stride=1, expand_ratio=4)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-axis attention
        self.window_attn = WindowAttention(dim, window_size, num_heads)
        self.grid_attn = GridAttention(dim, grid_size, num_heads)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        # MBConv
        x = self.mbconv(x)
        
        # Convert to (B, H, W, C) for attention
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # Window attention
        x = x + self.window_attn(self.norm1(x))
        
        # Grid attention
        x = x + self.grid_attn(self.norm1(x))
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        # Convert back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        return x


class MaxViTModel(nn.Module):
    """MaxViT model for emotion classification."""
    
    def __init__(self, num_classes=3, depths=[2, 2, 5, 2], dims=[64, 128, 256, 512]):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[0], dims[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # Stages
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.Sequential(*[
                MaxViTBlock(dims[i]) for _ in range(depths[i])
            ])
            self.stages.append(stage)
            
            # Downsample between stages (except last)
            if i < len(depths) - 1:
                downsample = nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dims[i+1]),
                    nn.ReLU(inplace=True)
                )
                self.stages.append(downsample)
        
        # Head
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        # Global average pooling
        x = x.mean(dim=[2, 3])  # (B, C)
        
        # Normalize and classify
        x = self.norm(x)
        x = self.head(x)
        
        return x


def load_maxvit_model(model_path, architecture='maxvit_tiny', num_classes=3, input_size=224, device='cuda'):
    """
    Load a pre-trained MaxViT model for dog emotion classification.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model checkpoint
    architecture : str
        MaxViT architecture ('maxvit_tiny', 'maxvit_small', 'maxvit_base')
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
    
    # Create model based on architecture
    if architecture == 'maxvit_tiny':
        model = MaxViTModel(num_classes=num_classes, depths=[2, 2, 5, 2], dims=[64, 128, 256, 512])
    elif architecture == 'maxvit_small':
        model = MaxViTModel(num_classes=num_classes, depths=[2, 2, 5, 2], dims=[96, 192, 384, 768])
    elif architecture == 'maxvit_base':
        model = MaxViTModel(num_classes=num_classes, depths=[2, 6, 14, 2], dims=[96, 192, 384, 768])
    else:
        raise ValueError(f"Unsupported MaxViT architecture: {architecture}")
    
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
        
        # Load state dict with flexible handling
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"✅ Loaded model with strict=True")
        except RuntimeError as e:
            print(f"⚠️  Strict loading failed, trying strict=False: {e}")
            try:
                model.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded model with strict=False")
            except Exception as e2:
                print(f"⚠️  Could not load state dict: {e2}")
                print(f"⚠️  Creating model with checkpoint's architecture...")
                # Try to infer architecture from checkpoint
                if 'stem.0.weight' in state_dict:
                    stem_channels = state_dict['stem.0.weight'].shape[0]
                    if stem_channels == 64:
                        model = MaxViTModel(num_classes=num_classes, depths=[2, 2, 5, 2], dims=[64, 128, 256, 512])
                    elif stem_channels == 96:
                        model = MaxViTModel(num_classes=num_classes, depths=[2, 2, 5, 2], dims=[96, 192, 384, 768])
                    else:
                        model = MaxViTModel(num_classes=num_classes, depths=[2, 2, 5, 2], dims=[stem_channels, stem_channels*2, stem_channels*4, stem_channels*8])
                    
                    model.load_state_dict(state_dict, strict=False)
                    print(f"✅ Loaded model with inferred architecture (stem_channels={stem_channels})")
                else:
                    print(f"⚠️  Using model with default architecture")
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


def predict_emotion_maxvit(image_path, model, transform, head_bbox=None, device='cuda',
                          emotion_classes=['angry', 'happy', 'relaxed']):
    """
    Predict dog emotion using MaxViT model.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    model : torch.nn.Module
        Loaded MaxViT model
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
        print(f"❌ Error in MaxViT emotion prediction: {e}")
        raise RuntimeError(f"MaxViT prediction failed: {e}")


def get_maxvit_transforms(input_size=224, is_training=True):
    """
    Get preprocessing transforms for MaxViT models.
    
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


def create_maxvit_model(architecture='maxvit_tiny', num_classes=3, pretrained=False):
    """
    Create a MaxViT model for emotion classification.
    
    Parameters:
    -----------
    architecture : str
        MaxViT architecture ('maxvit_tiny', 'maxvit_small', 'maxvit_base')
    num_classes : int
        Number of emotion classes (default: 3)
    pretrained : bool
        Whether to use pretrained weights (default: False)
        
    Returns:
    --------
    torch.nn.Module
        MaxViT model
    """
    
    if architecture == 'maxvit_tiny':
        model = MaxViTModel(num_classes=num_classes, depths=[2, 2, 5, 2], dims=[64, 128, 256, 512])
    elif architecture == 'maxvit_small':
        model = MaxViTModel(num_classes=num_classes, depths=[2, 2, 5, 2], dims=[96, 192, 384, 768])
    elif architecture == 'maxvit_base':
        model = MaxViTModel(num_classes=num_classes, depths=[2, 6, 14, 2], dims=[96, 192, 384, 768])
    else:
        raise ValueError(f"Unsupported MaxViT architecture: {architecture}")
    
    return model


# Convenience functions for specific architectures
def load_maxvit_tiny_model(model_path, num_classes=3, input_size=224, device='cuda'):
    return load_maxvit_model(model_path, 'maxvit_tiny', num_classes, input_size, device)

def load_maxvit_small_model(model_path, num_classes=3, input_size=224, device='cuda'):
    return load_maxvit_model(model_path, 'maxvit_small', num_classes, input_size, device)

def load_maxvit_base_model(model_path, num_classes=3, input_size=224, device='cuda'):
    return load_maxvit_model(model_path, 'maxvit_base', num_classes, input_size, device)

def predict_emotion_maxvit_tiny(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_maxvit(image_path, model, transform, head_bbox, device)

def predict_emotion_maxvit_small(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_maxvit(image_path, model, transform, head_bbox, device)

def predict_emotion_maxvit_base(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_maxvit(image_path, model, transform, head_bbox, device) 