"""
CMT (Convolutional Multi-Head Transformer) models for dog emotion classification.

This module provides CMT implementations optimized for
dog emotion recognition with 4 emotion classes: ['angry', 'happy', 'relaxed', 'sad'].

CMT combines convolutional layers with multi-head transformers for efficient
vision tasks, providing both local and global feature extraction capabilities.
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

class ConvTokenizer(nn.Module):
    """Convolutional tokenizer for CMT."""
    
    def __init__(self, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None
        
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        if self.norm:
            x = self.norm(x)
        return x, H, W

class ConvStage(nn.Module):
    """Convolutional stage for CMT."""
    
    def __init__(self, in_chans, out_chans, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # x is in format (B, N, C), need to reshape to (B, C, H, W)
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        # Flatten back to sequence format
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        return x, H, W

class LocallyGroupedSelfAttention(nn.Module):
    """Locally grouped self-attention for CMT."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, ws=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.ws = ws
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        
        total_groups = h_group * w_group
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
        x = x.reshape(B, total_groups, self.ws * self.ws, C)
        
        qkv = self.qkv(x).reshape(B, total_groups, self.ws * self.ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(2, 3).reshape(B, total_groups, self.ws * self.ws, C)
        x = x.reshape(B, h_group, w_group, self.ws, self.ws, C).transpose(2, 3)
        x = x.reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CMTBlock(nn.Module):
    """CMT block with locally grouped self-attention."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, ws=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LocallyGroupedSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                               attn_drop=attn_drop, proj_drop=drop, ws=ws)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MLP(nn.Module):
    """MLP for CMT."""
    
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

class CMTModel(nn.Module):
    """CMT model for emotion classification."""
    
    def __init__(self, img_size=224, in_chans=3, num_classes=4, 
                 embed_dims=[46, 92, 184, 368], depths=[2, 2, 10, 2], num_heads=[1, 2, 4, 8],
                 mlp_ratios=[3.6, 3.6, 3.6, 3.6], qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_stages = len(embed_dims)
        
        # Patch embedding
        self.patch_embed = ConvTokenizer(in_chans=in_chans, embed_dim=embed_dims[0], norm_layer=norm_layer)
        
        # Build stages
        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(self.num_stages):
            if i > 0:
                # Convolutional downsampling
                downsample = ConvStage(embed_dims[i-1], embed_dims[i])
                self.stages.append(downsample)
                
            # Transformer blocks
            blocks = nn.ModuleList([
                CMTBlock(
                    dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j], norm_layer=norm_layer, act_layer=act_layer,
                    ws=7 if i == 0 else 7)
                for j in range(depths[i])
            ])
            self.stages.append(blocks)
            cur += depths[i]
        
        # Classification head
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        
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
        # Patch embedding
        x, H, W = self.patch_embed(x)
        
        # Process through stages
        for stage in self.stages:
            if isinstance(stage, ConvStage):
                x, H, W = stage(x)
            else:  # ModuleList of CMTBlocks
                for block in stage:
                    x = block(x, H, W)
        
        # Classification
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        return x

def load_cmt_model(model_path: str, architecture: str = 'cmt_ti', 
                  num_classes: int = 4, input_size: int = 224, device: str = 'cuda') -> nn.Module:
    """
    Load a pre-trained CMT model for dog emotion classification.
    
    Args:
        model_path: Path to the saved model file
        architecture: CMT architecture ('cmt_ti', 'cmt_xs', 'cmt_s', 'cmt_b')
        num_classes: Number of emotion classes (default: 4)
        input_size: Input image size (default: 224)
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        CMT model
    """
    # Architecture configurations
    configs = {
        'cmt_ti': {
            'embed_dims': [46, 92, 184, 368],
            'depths': [2, 2, 10, 2],
            'num_heads': [1, 2, 4, 8]
        },
        'cmt_xs': {
            'embed_dims': [52, 104, 208, 416],
            'depths': [3, 3, 12, 3],
            'num_heads': [1, 2, 4, 8]
        },
        'cmt_s': {
            'embed_dims': [64, 128, 256, 512],
            'depths': [3, 3, 16, 3],
            'num_heads': [1, 2, 4, 8]
        },
        'cmt_b': {
            'embed_dims': [76, 152, 304, 608],
            'depths': [4, 4, 20, 4],
            'num_heads': [1, 2, 4, 8]
        }
    }
    
    if architecture not in configs:
        raise ValueError(f"Unsupported CMT architecture: {architecture}")
    
    config = configs[architecture]
    
    # Create model
    model = CMTModel(
        img_size=input_size,
        num_classes=num_classes,
        embed_dims=config['embed_dims'],
        depths=config['depths'],
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
            
            print(f"✅ CMT model loaded from {model_path}")
            
        except Exception as e:
            print(f"⚠️ Error loading CMT model: {e}")
            print("Using randomly initialized weights")
    else:
        print(f"⚠️ Model file not found: {model_path}")
        print("Using randomly initialized weights")
    
    model.eval()
    return model

def predict_emotion_cmt(image_path: str, model: nn.Module, transform: transforms.Compose,
                       head_bbox: Optional[Tuple[int, int, int, int]] = None,
                       device: str = 'cuda',
                       emotion_classes: List[str] = EMOTION_CLASSES) -> Dict[str, Any]:
    """
    Predict dog emotion using CMT model.
    
    Args:
        image_path: Path to the image file
        model: Loaded CMT model
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
            'model_type': 'CMT'
        }
        
    except Exception as e:
        print(f"❌ Error in CMT emotion prediction: {e}")
        return {
            'predicted_emotion': 'unknown',
            'confidence': 0.0,
            'class_probabilities': {emotion: 0.0 for emotion in emotion_classes},
            'model_type': 'CMT',
            'error': str(e)
        }

def get_cmt_transforms(input_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """
    Get preprocessing transforms for CMT models.
    
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

def create_cmt_model(architecture: str = 'cmt_ti', num_classes: int = 4, 
                    pretrained: bool = False) -> nn.Module:
    """
    Create a CMT model for emotion classification.
    
    Args:
        architecture: CMT architecture ('cmt_ti', 'cmt_xs', 'cmt_s', 'cmt_b')
        num_classes: Number of emotion classes
        pretrained: Whether to use pretrained weights (not implemented for CMT)
    
    Returns:
        CMT model
    """
    configs = {
        'cmt_ti': {
            'embed_dims': [46, 92, 184, 368],
            'depths': [2, 2, 10, 2],
            'num_heads': [1, 2, 4, 8]
        },
        'cmt_xs': {
            'embed_dims': [52, 104, 208, 416],
            'depths': [3, 3, 12, 3],
            'num_heads': [1, 2, 4, 8]
        },
        'cmt_s': {
            'embed_dims': [64, 128, 256, 512],
            'depths': [3, 3, 16, 3],
            'num_heads': [1, 2, 4, 8]
        },
        'cmt_b': {
            'embed_dims': [76, 152, 304, 608],
            'depths': [4, 4, 20, 4],
            'num_heads': [1, 2, 4, 8]
        }
    }
    
    if architecture not in configs:
        raise ValueError(f"Unsupported CMT architecture: {architecture}")
    
    config = configs[architecture]
    
    model = CMTModel(
        num_classes=num_classes,
        embed_dims=config['embed_dims'],
        depths=config['depths'],
        num_heads=config['num_heads']
    )
    
    if pretrained:
        print("⚠️ Pretrained weights not available for CMT. Using random initialization.")
    
    return model

# Convenience functions for different architectures
def cmt_ti(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    return create_cmt_model('cmt_ti', num_classes, pretrained)

def cmt_xs(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    return create_cmt_model('cmt_xs', num_classes, pretrained)

def cmt_s(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    return create_cmt_model('cmt_s', num_classes, pretrained)

def cmt_b(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    return create_cmt_model('cmt_b', num_classes, pretrained) 