"""
BoTNet (Bottleneck Transformer) models for dog emotion classification.

This module provides BoTNet implementations optimized for
dog emotion recognition with 4 emotion classes: ['angry', 'happy', 'relaxed', 'sad'].

BoTNet replaces the 3x3 convolution in the final bottleneck block of ResNet
with multi-head self-attention, combining the benefits of convolution and attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
from typing import Optional, List, Tuple, Dict, Any
import math

# Emotion classes in the correct order
EMOTION_CLASSES = ['angry', 'happy', 'relaxed', 'sad']

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module for BoTNet."""
    
    def __init__(self, in_dim, num_heads=4, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = in_dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Conv2d(in_dim, in_dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Relative position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * 7 - 1) * (2 * 7 - 1), num_heads))
        
        coords_h = torch.arange(7)
        coords_w = torch.arange(7)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += 7 - 1
        relative_coords[:, :, 1] += 7 - 1
        relative_coords[:, :, 0] *= 2 * 7 - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            H * W, H * W, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BoTBlock(nn.Module):
    """BoTNet bottleneck block with self-attention."""
    
    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=True, resolution=None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv2 = MultiHeadSelfAttention(planes, heads)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4)
            )
        
        self.mhsa = mhsa
        self.stride = stride
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        if not self.mhsa:
            out = F.relu(self.bn2(self.conv2(out)))
        else:
            out = F.relu(self.bn2(self.conv2(out)))
            
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BoTNet(nn.Module):
    """BoTNet model for emotion classification."""
    
    def __init__(self, num_classes=4, heads=4):
        super().__init__()
        self.in_planes = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 3, stride=1, heads=heads, mhsa=False)
        self.layer2 = self._make_layer(128, 4, stride=2, heads=heads, mhsa=False)
        self.layer3 = self._make_layer(256, 6, stride=2, heads=heads, mhsa=False)
        self.layer4 = self._make_layer(512, 3, stride=2, heads=heads, mhsa=True)  # Use MHSA in final layer
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, planes, blocks, stride, heads, mhsa):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(BoTBlock(self.in_planes, planes, stride, heads, mhsa and i == blocks - 1))
            self.in_planes = planes * 4
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out

def load_botnet_model(model_path: str, architecture: str = 'botnet50', 
                     num_classes: int = 4, input_size: int = 224, device: str = 'cuda') -> nn.Module:
    """
    Load a pre-trained BoTNet model for dog emotion classification.
    
    Args:
        model_path: Path to the saved model file
        architecture: BoTNet architecture ('botnet50', 'botnet101')
        num_classes: Number of emotion classes (default: 4)
        input_size: Input image size (default: 224)
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        BoTNet model
    """
    # Create model
    model = BoTNet(num_classes=num_classes, heads=4)
    
    # Load weights if model file exists
    if os.path.exists(model_path):
        try:
            if device == 'cuda' and torch.cuda.is_available():
                checkpoint = torch.load(model_path, map_location='cuda')
                model.to('cuda')
            else:
                checkpoint = torch.load(model_path, map_location='cpu')
                model.to('cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            print(f"✅ BoTNet model loaded from {model_path}")
            
        except Exception as e:
            print(f"⚠️ Error loading BoTNet model: {e}")
            print("Using randomly initialized weights")
    else:
        print(f"⚠️ Model file not found: {model_path}")
        print("Using randomly initialized weights")
    
    model.eval()
    return model

def predict_emotion_botnet(image_path: str, model: nn.Module, transform: transforms.Compose,
                          head_bbox: Optional[Tuple[int, int, int, int]] = None,
                          device: str = 'cuda',
                          emotion_classes: List[str] = EMOTION_CLASSES) -> Dict[str, Any]:
    """
    Predict dog emotion using BoTNet model.
    
    Args:
        image_path: Path to the image file
        model: Loaded BoTNet model
        transform: Image preprocessing transforms
        head_bbox: Optional bounding box for head region (x, y, w, h)
        device: Device to run inference on
        emotion_classes: List of emotion class names
    
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Load and preprocess image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Apply head bounding box if provided
        if head_bbox is not None:
            x, y, w, h = head_bbox
            image = image.crop((x, y, x + w, y + h))
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Model inference
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_emotion = emotion_classes[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # Get all class probabilities
            class_probabilities = {
                emotion_classes[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
            
        return {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence_score,
            'class_probabilities': class_probabilities,
            'model_type': 'BoTNet'
        }
        
    except Exception as e:
        print(f"❌ Error in BoTNet emotion prediction: {e}")
        raise RuntimeError(f"BoTNet prediction failed: {e}")

def get_botnet_transforms(input_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """
    Get preprocessing transforms for BoTNet models.
    
    Args:
        input_size: Input image size
        is_training: Whether to apply training augmentations
    
    Returns:
        Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((int(input_size * 1.14), int(input_size * 1.14))),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_botnet_model(architecture: str = 'botnet50', num_classes: int = 4, 
                       pretrained: bool = False) -> nn.Module:
    """
    Create a BoTNet model for emotion classification.
    
    Args:
        architecture: BoTNet architecture ('botnet50', 'botnet101')
        num_classes: Number of emotion classes
        pretrained: Whether to use pretrained weights (not implemented for BoTNet)
    
    Returns:
        BoTNet model
    """
    model = BoTNet(num_classes=num_classes, heads=4)
    
    if pretrained:
        print("⚠️ Pretrained weights not available for BoTNet. Using random initialization.")
    
    return model

# Convenience functions for different architectures
def botnet50(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    return create_botnet_model('botnet50', num_classes, pretrained)

def botnet101(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    return create_botnet_model('botnet101', num_classes, pretrained) 