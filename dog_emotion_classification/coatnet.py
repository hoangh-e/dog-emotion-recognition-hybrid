"""
CoAtNet (Convolution and Attention Network) models for dog emotion classification.

This module provides CoAtNet implementations optimized for 
dog emotion classification with 4 emotion classes: sad, angry, happy, relaxed.
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
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=4, se_ratio=0.25):
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
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])
        
        # Squeeze-and-excitation
        if se_ratio > 0:
            se_dim = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, se_dim, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_dim, hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None
        
        # Pointwise linear
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        
        if self.se is not None:
            out = out * self.se(out)
        
        if self.use_residual:
            return x + out
        else:
            return out


class RelativePositionalBias(nn.Module):
    """Relative positional bias for attention."""
    
    def __init__(self, num_heads, max_distance=127):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_distance + 1) * (2 * max_distance + 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, height, width):
        coords_h = torch.arange(height)
        coords_w = torch.arange(width)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += height - 1
        relative_coords[:, :, 1] += width - 1
        relative_coords[:, :, 0] *= 2 * width - 1
        relative_position_index = relative_coords.sum(-1)
        
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            height * width, height * width, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        return relative_position_bias


class MultiHeadAttention(nn.Module):
    """Multi-head self attention with relative positional bias."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.relative_position_bias = RelativePositionalBias(num_heads)
    
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative positional bias
        relative_position_bias = self.relative_position_bias(H, W)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and MLP."""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CoAtNetStage(nn.Module):
    """CoAtNet stage with either convolution or transformer blocks."""
    
    def __init__(self, in_channels, out_channels, depth, stage_type='conv', 
                 num_heads=8, mlp_ratio=4., stride=1):
        super().__init__()
        self.stage_type = stage_type
        
        if stage_type == 'conv':
            # Convolution stage with MBConv blocks
            self.blocks = nn.ModuleList()
            for i in range(depth):
                block_stride = stride if i == 0 else 1
                block_in_channels = in_channels if i == 0 else out_channels
                self.blocks.append(
                    MBConv(block_in_channels, out_channels, stride=block_stride)
                )
        else:
            # Transformer stage
            if stride > 1:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.downsample = None
            
            self.blocks = nn.ModuleList([
                TransformerBlock(out_channels, num_heads, mlp_ratio)
                for _ in range(depth)
            ])
    
    def forward(self, x):
        if self.stage_type == 'conv':
            for block in self.blocks:
                x = block(x)
        else:
            # Transformer stage
            if self.downsample is not None:
                x = self.downsample(x)
            
            # Convert to sequence format
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # B, H*W, C
            
            for block in self.blocks:
                x = block(x)
            
            # Convert back to spatial format
            x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x


class CoAtNetModel(nn.Module):
    """CoAtNet model for emotion classification."""
    
    def __init__(self, num_classes=4, depths=[2, 2, 3, 5, 2], dims=[64, 96, 192, 384, 768]):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.SiLU(inplace=True)
        )
        
        # Stages (Conv-Conv-Conv-Transformer-Transformer)
        self.stages = nn.ModuleList()
        
        # Stage 0: Conv
        self.stages.append(
            CoAtNetStage(dims[0], dims[0], depths[0], 'conv', stride=1)
        )
        
        # Stage 1: Conv with downsample
        self.stages.append(
            CoAtNetStage(dims[0], dims[1], depths[1], 'conv', stride=2)
        )
        
        # Stage 2: Conv with downsample
        self.stages.append(
            CoAtNetStage(dims[1], dims[2], depths[2], 'conv', stride=2)
        )
        
        # Stage 3: Transformer with downsample
        self.stages.append(
            CoAtNetStage(dims[2], dims[3], depths[3], 'transformer', stride=2)
        )
        
        # Stage 4: Transformer
        self.stages.append(
            CoAtNetStage(dims[3], dims[4], depths[4], 'transformer', stride=1)
        )
        
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
        x = x.mean(dim=[2, 3])  # B, C
        
        # Normalize and classify
        x = self.norm(x)
        x = self.head(x)
        
        return x


def load_coatnet_model(model_path, architecture='coatnet_0', num_classes=4, input_size=224, device='cuda'):
    """
    Load a pre-trained CoAtNet model for dog emotion classification.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model checkpoint
    architecture : str
        CoAtNet architecture ('coatnet_0', 'coatnet_1', 'coatnet_2')
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
    if architecture == 'coatnet_0':
        model = CoAtNetModel(num_classes=num_classes, depths=[2, 2, 3, 5, 2], dims=[64, 96, 192, 384, 768])
    elif architecture == 'coatnet_1':
        model = CoAtNetModel(num_classes=num_classes, depths=[2, 2, 6, 14, 2], dims=[64, 96, 192, 384, 768])
    elif architecture == 'coatnet_2':
        model = CoAtNetModel(num_classes=num_classes, depths=[2, 2, 6, 14, 2], dims=[128, 128, 256, 512, 1026])
    else:
        raise ValueError(f"Unsupported CoAtNet architecture: {architecture}")
    
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


def predict_emotion_coatnet(image_path, model, transform, head_bbox=None, device='cuda',
                           emotion_classes=['angry', 'happy', 'relaxed', 'sad']):
    """
    Predict dog emotion using CoAtNet model.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    model : torch.nn.Module
        Loaded CoAtNet model
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
        print(f"❌ Error in CoAtNet emotion prediction: {e}")
        # Return default scores on error
        emotion_scores = {emotion: 0.0 for emotion in emotion_classes}
        emotion_scores['predicted'] = False
        return emotion_scores


def get_coatnet_transforms(input_size=224, is_training=True):
    """
    Get preprocessing transforms for CoAtNet models.
    
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


def create_coatnet_model(architecture='coatnet_0', num_classes=4, pretrained=False):
    """
    Create a CoAtNet model for emotion classification.
    
    Parameters:
    -----------
    architecture : str
        CoAtNet architecture ('coatnet_0', 'coatnet_1', 'coatnet_2')
    num_classes : int
        Number of emotion classes (default: 4)
    pretrained : bool
        Whether to use pretrained weights (default: False)
        
    Returns:
    --------
    torch.nn.Module
        CoAtNet model
    """
    
    if architecture == 'coatnet_0':
        model = CoAtNetModel(num_classes=num_classes, depths=[2, 2, 3, 5, 2], dims=[64, 96, 192, 384, 768])
    elif architecture == 'coatnet_1':
        model = CoAtNetModel(num_classes=num_classes, depths=[2, 2, 6, 14, 2], dims=[64, 96, 192, 384, 768])
    elif architecture == 'coatnet_2':
        model = CoAtNetModel(num_classes=num_classes, depths=[2, 2, 6, 14, 2], dims=[128, 128, 256, 512, 1026])
    else:
        raise ValueError(f"Unsupported CoAtNet architecture: {architecture}")
    
    return model


# Convenience functions for specific architectures
def load_coatnet_0_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_coatnet_model(model_path, 'coatnet_0', num_classes, input_size, device)

def load_coatnet_1_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_coatnet_model(model_path, 'coatnet_1', num_classes, input_size, device)

def load_coatnet_2_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_coatnet_model(model_path, 'coatnet_2', num_classes, input_size, device)

def predict_emotion_coatnet_0(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_coatnet(image_path, model, transform, head_bbox, device)

def predict_emotion_coatnet_1(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_coatnet(image_path, model, transform, head_bbox, device)

def predict_emotion_coatnet_2(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_coatnet(image_path, model, transform, head_bbox, device) 