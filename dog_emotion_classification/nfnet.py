"""
NFNet (Normalizer-Free Networks) models for dog emotion classification.

This module provides NFNet implementations optimized for 
dog emotion classification with 4 emotion classes: sad, angry, happy, relaxed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import math


class WSConv2d(nn.Conv2d):
    """Weight Standardized Convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-5):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.eps = eps
    
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation module."""
    
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, se_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class StochasticDepth(nn.Module):
    """Stochastic Depth (Drop Path) regularization."""
    
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class NFBlock(nn.Module):
    """NFNet Block with Normalizer-Free design."""
    
    def __init__(self, in_channels, out_channels, stride=1, expansion=0.5, 
                 se_ratio=0.25, drop_path=0.0, beta=1.0, alpha=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.beta = beta
        self.alpha = alpha
        
        width = int(out_channels * expansion)
        
        # Skip connection
        self.use_skip = stride == 1 and in_channels == out_channels
        
        # Main path
        self.conv1 = WSConv2d(in_channels, width, 1, bias=False)
        self.conv2 = WSConv2d(width, width, 3, stride=stride, padding=1, groups=width, bias=False)
        self.se = SqueezeExcite(width, se_ratio) if se_ratio > 0 else nn.Identity()
        self.conv3 = WSConv2d(width, out_channels, 1, bias=False)
        
        self.activation = nn.GELU()
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()
        
        # Skip path
        if not self.use_skip:
            self.skip_conv = WSConv2d(in_channels, out_channels, 1, stride=stride, bias=False)
    
    def forward(self, x):
        # Skip connection
        if self.use_skip:
            skip = x
        else:
            skip = self.skip_conv(x)
        
        # Main path
        out = self.conv1(x)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.activation(out)
        
        out = self.se(out)
        out = self.conv3(out)
        
        # Scaled residual connection
        out = self.drop_path(out)
        if self.use_skip:
            out = skip + self.alpha * out
        else:
            out = skip + out
        
        # Scale for next layer
        return out * self.beta


class NFNetStem(nn.Module):
    """NFNet stem with multiple convolutions."""
    
    def __init__(self, in_channels=3, out_channels=128):
        super().__init__()
        
        self.conv1 = WSConv2d(in_channels, out_channels // 2, 3, stride=2, padding=1, bias=False)
        self.conv2 = WSConv2d(out_channels // 2, out_channels // 2, 3, padding=1, bias=False)
        self.conv3 = WSConv2d(out_channels // 2, out_channels // 2, 3, padding=1, bias=False)
        self.conv4 = WSConv2d(out_channels // 2, out_channels, 3, stride=2, padding=1, bias=False)
        
        self.activation = nn.GELU()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        x = self.activation(x)
        
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.activation(x)
        
        return x


class NFNetModel(nn.Module):
    """NFNet model for emotion classification."""
    
    def __init__(self, num_classes=4, variant='F0', drop_path_rate=0.1):
        super().__init__()
        
        # Architecture configurations
        configs = {
            'F0': {'depths': [1, 2, 6, 3], 'widths': [256, 512, 1536, 1536], 'alpha': 0.2},
            'F1': {'depths': [2, 4, 12, 6], 'widths': [256, 512, 1536, 1536], 'alpha': 0.2},
            'F2': {'depths': [3, 6, 18, 9], 'widths': [256, 512, 1536, 1536], 'alpha': 0.2},
            'F3': {'depths': [4, 8, 24, 12], 'widths': [256, 512, 1536, 1536], 'alpha': 0.2},
            'F4': {'depths': [5, 10, 30, 15], 'widths': [256, 512, 1536, 1536], 'alpha': 0.2},
            'F5': {'depths': [6, 12, 36, 18], 'widths': [256, 512, 1536, 1536], 'alpha': 0.2},
        }
        
        if variant not in configs:
            raise ValueError(f"Unsupported NFNet variant: {variant}")
        
        config = configs[variant]
        depths = config['depths']
        widths = config['widths']
        alpha = config['alpha']
        
        # Stem
        self.stem = NFNetStem(3, widths[0])
        
        # Stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        current_width = widths[0]
        dp_idx = 0
        
        for stage_idx, (depth, width) in enumerate(zip(depths, widths)):
            stage = nn.ModuleList()
            
            for block_idx in range(depth):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                in_width = current_width if block_idx == 0 else width
                
                # Calculate beta for variance preservation
                if block_idx == 0 and stage_idx > 0:
                    beta = 1.0  # First block in stage
                else:
                    beta = 1.0 / math.sqrt(depth)  # Subsequent blocks
                
                block = NFBlock(
                    in_width, width, stride=stride,
                    drop_path=dp_rates[dp_idx],
                    beta=beta, alpha=alpha
                )
                stage.append(block)
                dp_idx += 1
            
            self.stages.append(stage)
            current_width = width
        
        # Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(widths[-1], num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (WSConv2d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            for block in stage:
                x = block(x)
        
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


def load_nfnet_model(model_path, architecture='nfnet_f0', num_classes=4, input_size=224, device='cuda'):
    """
    Load a pre-trained NFNet model for dog emotion classification.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model checkpoint
    architecture : str
        NFNet architecture ('nfnet_f0' to 'nfnet_f5')
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
    
    # Extract variant from architecture name
    variant = architecture.split('_')[-1].upper()  # F0, F1, etc.
    
    # Create model
    model = NFNetModel(num_classes=num_classes, variant=variant)
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


def predict_emotion_nfnet(image_path, model, transform, head_bbox=None, device='cuda',
                         emotion_classes=['sad', 'angry', 'happy', 'relaxed']):
    """
    Predict dog emotion using NFNet model.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    model : torch.nn.Module
        Loaded NFNet model
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
        print(f"❌ Error in NFNet emotion prediction: {e}")
        # Return default scores on error
        emotion_scores = {emotion: 0.0 for emotion in emotion_classes}
        emotion_scores['predicted'] = False
        return emotion_scores


def get_nfnet_transforms(input_size=224, is_training=True):
    """
    Get preprocessing transforms for NFNet models.
    
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


def create_nfnet_model(architecture='nfnet_f0', num_classes=4, pretrained=False):
    """
    Create an NFNet model for emotion classification.
    
    Parameters:
    -----------
    architecture : str
        NFNet architecture ('nfnet_f0' to 'nfnet_f5')
    num_classes : int
        Number of emotion classes (default: 4)
    pretrained : bool
        Whether to use pretrained weights (default: False)
        
    Returns:
    --------
    torch.nn.Module
        NFNet model
    """
    
    # Extract variant from architecture name
    variant = architecture.split('_')[-1].upper()  # F0, F1, etc.
    
    model = NFNetModel(num_classes=num_classes, variant=variant)
    return model


# Convenience functions for specific architectures
def load_nfnet_f0_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_nfnet_model(model_path, 'nfnet_f0', num_classes, input_size, device)

def load_nfnet_f1_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_nfnet_model(model_path, 'nfnet_f1', num_classes, input_size, device)

def load_nfnet_f2_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_nfnet_model(model_path, 'nfnet_f2', num_classes, input_size, device)

def load_nfnet_f3_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_nfnet_model(model_path, 'nfnet_f3', num_classes, input_size, device)

def load_nfnet_f4_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_nfnet_model(model_path, 'nfnet_f4', num_classes, input_size, device)

def load_nfnet_f5_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_nfnet_model(model_path, 'nfnet_f5', num_classes, input_size, device)

def predict_emotion_nfnet_f0(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_nfnet(image_path, model, transform, head_bbox, device)

def predict_emotion_nfnet_f1(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_nfnet(image_path, model, transform, head_bbox, device)

def predict_emotion_nfnet_f2(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_nfnet(image_path, model, transform, head_bbox, device)

def predict_emotion_nfnet_f3(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_nfnet(image_path, model, transform, head_bbox, device)

def predict_emotion_nfnet_f4(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_nfnet(image_path, model, transform, head_bbox, device)

def predict_emotion_nfnet_f5(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_nfnet(image_path, model, transform, head_bbox, device) 