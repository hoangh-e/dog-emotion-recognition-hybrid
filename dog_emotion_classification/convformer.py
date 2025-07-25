"""
ConvFormer models for dog emotion classification.

This module provides ConvFormer implementations optimized for
dog emotion recognition with 3 emotion classes: ['angry', 'happy', 'relaxed', 'sad'].

ConvFormer combines convolution and transformer architectures for efficient
image classification with both local and global feature extraction.
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

# Emotion classes in the correct order
EMOTION_CLASSES = ['angry', 'happy', 'relaxed', 'sad']

class ConvFormerBlock(nn.Module):
    """ConvFormer block with convolution and attention mechanisms."""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=False, 
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                      attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMLP(in_features=dim, hidden_features=mlp_hidden_dim, 
                          act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head attention with convolution-based positional encoding."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ConvMLP(nn.Module):
    """Convolutional MLP for ConvFormer."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.0):
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

class ConvFormerModel(nn.Module):
    """ConvFormer model for emotion classification."""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=3,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        # Patch embedding with convolution
        self.patch_embed = ConvPatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            ConvFormerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        x = x[:, 0]  # Use CLS token
        x = self.head(x)
        return x

class ConvPatchEmbed(nn.Module):
    """Convolutional patch embedding."""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def load_convformer_model(model_path: str, architecture: str = 'convformer_small', 
                         num_classes: int = 4, input_size: int = 224, device: str = 'cuda') -> nn.Module:
    """
    Load a pre-trained ConvFormer model for dog emotion classification.
    
    Args:
        model_path: Path to the saved model file
        architecture: ConvFormer architecture ('convformer_tiny', 'convformer_small', 'convformer_base')
        num_classes: Number of emotion classes (default: 3)
        input_size: Input image size (default: 224)
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        ConvFormer model
    """
    # Architecture configurations
    configs = {
        'convformer_tiny': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
        'convformer_small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'convformer_base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12}
    }
    
    if architecture not in configs:
        raise ValueError(f"Unsupported ConvFormer architecture: {architecture}")
    
    config = configs[architecture]
    
    # Create model
    model = ConvFormerModel(
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
            
            print(f"✅ ConvFormer model loaded from {model_path}")
            
        except Exception as e:
            print(f"⚠️ Error loading ConvFormer model: {e}")
            print("Using randomly initialized weights")
    else:
        print(f"⚠️ Model file not found: {model_path}")
        print("Using randomly initialized weights")
    
    model.eval()
    return model

def predict_emotion_convformer(image_path: str, model: nn.Module, transform: transforms.Compose,
                              head_bbox: Optional[Tuple[int, int, int, int]] = None,
                              device: str = 'cuda',
                              emotion_classes: List[str] = EMOTION_CLASSES) -> Dict[str, Any]:
    """
    Predict dog emotion using ConvFormer model.
    
    Args:
        image_path: Path to the image file
        model: Loaded ConvFormer model
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
            'model_type': 'ConvFormer'
        }
        
    except Exception as e:
        print(f"❌ Error in ConvFormer emotion prediction: {e}")
        raise RuntimeError(f"ConvFormer prediction failed: {e}")

def get_convformer_transforms(input_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """
    Get preprocessing transforms for ConvFormer models.
    
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

def create_convformer_model(architecture: str = 'convformer_small', num_classes: int = 4, 
                           pretrained: bool = False) -> nn.Module:
    """
    Create a ConvFormer model for emotion classification.
    
    Args:
        architecture: ConvFormer architecture ('convformer_tiny', 'convformer_small', 'convformer_base')
        num_classes: Number of emotion classes
        pretrained: Whether to use pretrained weights (not implemented for ConvFormer)
    
    Returns:
        ConvFormer model
    """
    configs = {
        'convformer_tiny': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
        'convformer_small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'convformer_base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12}
    }
    
    if architecture not in configs:
        raise ValueError(f"Unsupported ConvFormer architecture: {architecture}")
    
    config = configs[architecture]
    
    model = ConvFormerModel(
        num_classes=num_classes,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads']
    )
    
    if pretrained:
        print("⚠️ Pretrained weights not available for ConvFormer. Using random initialization.")
    
    return model

# Convenience functions for different architectures
def convformer_tiny(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    return create_convformer_model('convformer_tiny', num_classes, pretrained)

def convformer_small(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    return create_convformer_model('convformer_small', num_classes, pretrained)

def convformer_base(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    return create_convformer_model('convformer_base', num_classes, pretrained) 