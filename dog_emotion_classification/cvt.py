"""
CvT (Convolutional Vision Transformer) models for dog emotion classification.

This module provides CvT implementations optimized for
dog emotion recognition with 4 emotion classes: ['angry', 'happy', 'relaxed', 'sad'].

CvT introduces convolutions into vision transformers to improve efficiency
and performance by combining the benefits of CNNs and Transformers.
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

class ConvEmbed(nn.Module):
    """Convolutional embedding layer for CvT."""
    
    def __init__(self, patch_size=7, in_chans=3, embed_dim=64, stride=4, padding=2, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else None
        
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        if self.norm:
            x = self.norm(x)
        return x, (H, W)

class ConvAttention(nn.Module):
    """Convolutional attention mechanism for CvT."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0,
                 kernel_size=3, stride_kv=1, stride_q=1, padding_kv=1, padding_q=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.conv_proj_q = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride_q, padding=padding_q, bias=qkv_bias)
        self.conv_proj_k = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride_kv, padding=padding_kv, bias=qkv_bias)
        self.conv_proj_v = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride_kv, padding=padding_kv, bias=qkv_bias)
        
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        # Reshape to spatial format
        x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Apply convolutions
        q = self.conv_proj_q(x_spatial)
        k = self.conv_proj_k(x_spatial)
        v = self.conv_proj_v(x_spatial)
        
        # Flatten back to sequence format
        q = q.flatten(2).transpose(1, 2)  # B, H*W, C
        k = k.flatten(2).transpose(1, 2)  # B, H*W, C
        v = v.flatten(2).transpose(1, 2)  # B, H*W, C
        
        # Apply linear projections
        q = self.proj_q(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.proj_k(k).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.proj_v(v).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class ConvMLP(nn.Module):
    """Convolutional MLP for CvT."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CvTBlock(nn.Module):
    """CvT block with convolutional attention and MLP."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ConvAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class CvTStage(nn.Module):
    """CvT stage with multiple blocks."""
    
    def __init__(self, patch_size, patch_stride, patch_padding, embed_dim, depth, num_heads,
                 mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, in_chans=3):
        super().__init__()
        
        self.patch_embed = ConvEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            stride=patch_stride, padding=patch_padding, norm_layer=norm_layer)
        
        self.blocks = nn.ModuleList([
            CvTBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        
    def forward(self, x):
        x, (H, W) = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, H, W)
        return x, H, W

class CvTModel(nn.Module):
    """CvT model for emotion classification."""
    
    def __init__(self, img_size=224, in_chans=3, num_classes=4, 
                 patch_size=[7, 3, 3], patch_stride=[4, 2, 2], patch_padding=[2, 1, 1],
                 embed_dim=[64, 192, 384], depth=[1, 2, 10], num_heads=[1, 3, 6],
                 mlp_ratio=[4.0, 4.0, 4.0], qkv_bias=[True, True, True],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_stages = len(embed_dim)
        
        # Build stages
        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        
        for i in range(self.num_stages):
            stage = CvTStage(
                patch_size=patch_size[i],
                patch_stride=patch_stride[i],
                patch_padding=patch_padding[i],
                embed_dim=embed_dim[i],
                depth=depth[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias[i],
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur:cur + depth[i]],
                norm_layer=norm_layer,
                act_layer=act_layer,
                in_chans=in_chans if i == 0 else embed_dim[i-1]
            )
            self.stages.append(stage)
            cur += depth[i]
        
        # Classification head
        self.norm = norm_layer(embed_dim[-1])
        self.head = nn.Linear(embed_dim[-1], num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        for stage in self.stages:
            x, H, W = stage(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        return x

def load_cvt_model(model_path: str, architecture: str = 'cvt_13', 
                  num_classes: int = 4, input_size: int = 224, device: str = 'cuda') -> nn.Module:
    """
    Load a pre-trained CvT model for dog emotion classification.
    
    Args:
        model_path: Path to the saved model file
        architecture: CvT architecture ('cvt_13', 'cvt_21', 'cvt_w24')
        num_classes: Number of emotion classes (default: 4)
        input_size: Input image size (default: 224)
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        CvT model
    """
    # Architecture configurations
    configs = {
        'cvt_13': {
            'embed_dim': [64, 192, 384],
            'depth': [1, 2, 10],
            'num_heads': [1, 3, 6]
        },
        'cvt_21': {
            'embed_dim': [64, 192, 384],
            'depth': [1, 4, 16],
            'num_heads': [1, 3, 6]
        },
        'cvt_w24': {
            'embed_dim': [192, 768, 1024],
            'depth': [2, 2, 20],
            'num_heads': [3, 12, 16]
        }
    }
    
    if architecture not in configs:
        raise ValueError(f"Unsupported CvT architecture: {architecture}")
    
    config = configs[architecture]
    
    # Create model
    model = CvTModel(
        img_size=input_size,
        num_classes=num_classes,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads']
    )
    
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
            
            print(f"✅ CvT model loaded from {model_path}")
            
        except Exception as e:
            print(f"⚠️ Error loading CvT model: {e}")
            print("Using randomly initialized weights")
    else:
        print(f"⚠️ Model file not found: {model_path}")
        print("Using randomly initialized weights")
    
    model.eval()
    return model

def predict_emotion_cvt(image_path: str, model: nn.Module, transform: transforms.Compose,
                       head_bbox: Optional[Tuple[int, int, int, int]] = None,
                       device: str = 'cuda',
                       emotion_classes: List[str] = EMOTION_CLASSES) -> Dict[str, Any]:
    """
    Predict dog emotion using CvT model.
    
    Args:
        image_path: Path to the image file
        model: Loaded CvT model
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
            'model_type': 'CvT'
        }
        
    except Exception as e:
        print(f"❌ Error in CvT emotion prediction: {e}")
        return {
            'predicted_emotion': 'unknown',
            'confidence': 0.0,
            'class_probabilities': {emotion: 0.0 for emotion in emotion_classes},
            'model_type': 'CvT',
            'error': str(e)
        }

def get_cvt_transforms(input_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """
    Get preprocessing transforms for CvT models.
    
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

def create_cvt_model(architecture: str = 'cvt_13', num_classes: int = 4, 
                    pretrained: bool = False) -> nn.Module:
    """
    Create a CvT model for emotion classification.
    
    Args:
        architecture: CvT architecture ('cvt_13', 'cvt_21', 'cvt_w24')
        num_classes: Number of emotion classes
        pretrained: Whether to use pretrained weights (not implemented for CvT)
    
    Returns:
        CvT model
    """
    configs = {
        'cvt_13': {
            'embed_dim': [64, 192, 384],
            'depth': [1, 2, 10],
            'num_heads': [1, 3, 6]
        },
        'cvt_21': {
            'embed_dim': [64, 192, 384],
            'depth': [1, 4, 16],
            'num_heads': [1, 3, 6]
        },
        'cvt_w24': {
            'embed_dim': [192, 768, 1024],
            'depth': [2, 2, 20],
            'num_heads': [3, 12, 16]
        }
    }
    
    if architecture not in configs:
        raise ValueError(f"Unsupported CvT architecture: {architecture}")
    
    config = configs[architecture]
    
    model = CvTModel(
        num_classes=num_classes,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads']
    )
    
    if pretrained:
        print("⚠️ Pretrained weights not available for CvT. Using random initialization.")
    
    return model

# Convenience functions for different architectures
def cvt_13(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    return create_cvt_model('cvt_13', num_classes, pretrained)

def cvt_21(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    return create_cvt_model('cvt_21', num_classes, pretrained)

def cvt_w24(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    return create_cvt_model('cvt_w24', num_classes, pretrained) 