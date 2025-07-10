"""
MLP-Mixer (Multi-Layer Perceptron Mixer) models for dog emotion classification.

This module provides MLP-Mixer implementations optimized for dog emotion classification 
with 4 emotion classes: sad, angry, happy, relaxed.

Based on "MLP-Mixer: An all-MLP Architecture for Vision"
by Ilya Tolstikhin et al. (2021).
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

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class MlpBlock(nn.Module):
    """MLP block with GELU activation."""
    
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MixerBlock(nn.Module):
    """MLP-Mixer block with token mixing and channel mixing."""
    
    def __init__(self, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout=0.0):
        super().__init__()
        
        # Token mixing
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            MlpBlock(num_patches, tokens_mlp_dim, dropout)
        )
        
        # Channel mixing  
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            MlpBlock(hidden_dim, channels_mlp_dim, dropout)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_patches, hidden_dim)
        
        # Token mixing: mix information across patches
        # Transpose to (batch_size, hidden_dim, num_patches)
        token_mixed = x.transpose(1, 2)
        token_mixed = self.token_mixing[0](token_mixed)  # LayerNorm
        token_mixed = self.token_mixing[1](token_mixed)  # MLP
        token_mixed = token_mixed.transpose(1, 2)  # Back to original shape
        x = x + token_mixed
        
        # Channel mixing: mix information across channels
        channel_mixed = self.channel_mixing[0](x)  # LayerNorm
        channel_mixed = self.channel_mixing[1](channel_mixed)  # MLP
        x = x + channel_mixed
        
        return x


class MLPMixerModel(nn.Module):
    """MLP-Mixer model for dog emotion classification."""
    
    def __init__(self, image_size=224, patch_size=16, num_classes=4, 
                 hidden_dim=512, num_blocks=8, tokens_mlp_dim=256, 
                 channels_mlp_dim=2048, dropout=0.1):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden_dim = hidden_dim
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # MLP-Mixer blocks
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(self.num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Classification head
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, 3, image_size, image_size)
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch_size, hidden_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, hidden_dim)
        
        # MLP-Mixer blocks
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        
        # Global average pooling
        x = self.layer_norm(x)
        x = x.mean(dim=1)  # Average over patches
        
        # Classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


def load_mlp_mixer_model(checkpoint_path=None, model_variant='base', num_classes=4,
                        device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load MLP-Mixer model for dog emotion classification.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_variant: Model variant ('tiny', 'small', 'base', 'large')
        num_classes: Number of emotion classes (default: 4)
        device: Device to load model on
    
    Returns:
        Loaded MLP-Mixer model
    """
    
    print(f"🔄 Loading MLP-Mixer {model_variant} model")
    
    # Try to use timm if available
    if TIMM_AVAILABLE and checkpoint_path is None:
        try:
            # Create timm model
            model_name = f'mixer_{model_variant}_patch16_224'
            model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=num_classes
            )
            
            print(f"✅ Loaded {model_name} from timm")
            
        except Exception as e:
            print(f"Failed to load from timm: {e}")
            # Create custom implementation
            model = create_custom_mlp_mixer(model_variant, num_classes)
    else:
        # Use custom implementation
        model = create_custom_mlp_mixer(model_variant, num_classes)
    
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
            
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded MLP-Mixer checkpoint from {checkpoint_path}")
            
        except Exception as e:
            raise RuntimeError(f"Could not load checkpoint {checkpoint_path}: {e}")
    elif checkpoint_path and not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    return model


def create_custom_mlp_mixer(variant='base', num_classes=4):
    """Create custom MLP-Mixer model with different variants."""
    
    configs = {
        'tiny': {
            'hidden_dim': 256,
            'num_blocks': 8,
            'tokens_mlp_dim': 128,
            'channels_mlp_dim': 1024,
            'patch_size': 32
        },
        'small': {
            'hidden_dim': 512,
            'num_blocks': 8,
            'tokens_mlp_dim': 256,
            'channels_mlp_dim': 2048,
            'patch_size': 16
        },
        'base': {
            'hidden_dim': 768,
            'num_blocks': 12,
            'tokens_mlp_dim': 384,
            'channels_mlp_dim': 3072,
            'patch_size': 16
        },
        'large': {
            'hidden_dim': 1024,
            'num_blocks': 24,
            'tokens_mlp_dim': 512,
            'channels_mlp_dim': 4096,
            'patch_size': 16
        }
    }
    
    config = configs.get(variant, configs['base'])
    
    return MLPMixerModel(
        image_size=224,
        patch_size=config['patch_size'],
        num_classes=num_classes,
        hidden_dim=config['hidden_dim'],
        num_blocks=config['num_blocks'],
        tokens_mlp_dim=config['tokens_mlp_dim'],
        channels_mlp_dim=config['channels_mlp_dim'],
        dropout=0.1
    )


def create_simple_mlp_mixer(num_classes=4, device='cuda'):
    """Create a simple MLP-Mixer model."""
    
    class SimpleMixerBlock(nn.Module):
        def __init__(self, num_patches, hidden_dim):
            super().__init__()
            self.token_norm = nn.LayerNorm(hidden_dim)
            self.token_mlp = nn.Sequential(
                nn.Linear(num_patches, num_patches),
                nn.GELU(),
                nn.Linear(num_patches, num_patches)
            )
            
            self.channel_norm = nn.LayerNorm(hidden_dim)
            self.channel_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            
        def forward(self, x):
            # Token mixing
            residual = x
            x = self.token_norm(x)
            x = x.transpose(1, 2)
            x = self.token_mlp(x)
            x = x.transpose(1, 2)
            x = x + residual
            
            # Channel mixing
            residual = x
            x = self.channel_norm(x)
            x = self.channel_mlp(x)
            x = x + residual
            
            return x
    
    class SimpleMLPMixer(nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            self.patch_size = 16
            self.num_patches = (224 // 16) ** 2  # 196
            self.hidden_dim = 512
            
            # Patch embedding
            self.patch_embed = nn.Conv2d(3, self.hidden_dim, 16, stride=16)
            
            # Mixer blocks
            self.mixer_blocks = nn.ModuleList([
                SimpleMixerBlock(self.num_patches, self.hidden_dim)
                for _ in range(8)
            ])
            
            # Classification head
            self.norm = nn.LayerNorm(self.hidden_dim)
            self.classifier = nn.Linear(self.hidden_dim, num_classes)
            
        def forward(self, x):
            # Patch embedding
            x = self.patch_embed(x)  # (B, hidden_dim, 14, 14)
            x = x.flatten(2).transpose(1, 2)  # (B, 196, hidden_dim)
            
            # Mixer blocks
            for block in self.mixer_blocks:
                x = block(x)
            
            # Global average pooling
            x = self.norm(x)
            x = x.mean(dim=1)
            
            # Classification
            x = self.classifier(x)
            
            return x
    
    model = SimpleMLPMixer(num_classes).to(device)
    return model


def predict_emotion_mlp_mixer(model, image_path, transforms_fn=None, device='cuda'):
    """
    Predict dog emotion using MLP-Mixer model.
    
    Args:
        model: Loaded MLP-Mixer model
        image_path: Path to image or PIL Image
        transforms_fn: Image preprocessing transforms
        device: Device for inference
    
    Returns:
        Dictionary with emotion predictions
    """
    emotion_classes = ['angry', 'happy', 'relaxed', 'sad']
    
    # Load and preprocess image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    if transforms_fn is None:
        transforms_fn = get_mlp_mixer_transforms()
    
    # Apply transforms
    input_tensor = transforms_fn(image).unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
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


def get_mlp_mixer_transforms(input_size=224, is_training=False):
    """Get MLP-Mixer specific image transforms."""
    
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


# Model variants configuration
MLP_MIXER_VARIANTS = {
    'tiny': {
        'hidden_dim': 256,
        'num_blocks': 8,
        'patch_size': 32,
        'params': '17M',
        'description': 'MLP-Mixer Tiny for efficient inference'
    },
    'small': {
        'hidden_dim': 512,
        'num_blocks': 8,
        'patch_size': 16,
        'params': '19M',
        'description': 'MLP-Mixer Small balanced model'
    },
    'base': {
        'hidden_dim': 768,
        'num_blocks': 12,
        'patch_size': 16,
        'params': '59M',
        'description': 'MLP-Mixer Base standard model'
    },
    'large': {
        'hidden_dim': 1024,
        'num_blocks': 24,
        'patch_size': 16,
        'params': '208M',
        'description': 'MLP-Mixer Large for maximum accuracy'
    }
}


def load_mlp_mixer_model_standard(model_path, architecture='mlp_mixer_base', num_classes=4, input_size=224, device='cuda'):
    """
    Standardized load function for MLP-Mixer model to match other modules.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model checkpoint
    architecture : str
        MLP-Mixer architecture ('mlp_mixer_small', 'mlp_mixer_base', 'mlp_mixer_large')
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
        'mlp_mixer_small': 'small',
        'mlp_mixer_base': 'base',
        'mlp_mixer_large': 'large'
    }
    variant = variant_map.get(architecture, 'base')
    
    # Load model using existing function
    model = load_mlp_mixer_model(model_path, variant, num_classes, device)
    
    # Create transform
    transform = get_mlp_mixer_transforms(input_size, is_training=False)
    
    print(f"✅ MLP-Mixer model loaded successfully on {device}")
    print(f"🎯 Model type: {architecture.upper()}")
    print(f"📏 Input size: {input_size}x{input_size}")
    
    return model, transform


def predict_emotion_mlp_mixer_standard(image_path, model, transform, head_bbox=None, device='cuda',
                                      emotion_classes=['angry', 'happy', 'relaxed', 'sad']):
    """
    Standardized predict function for MLP-Mixer model to match other modules.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    model : torch.nn.Module
        Loaded MLP-Mixer model
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
    result = predict_emotion_mlp_mixer(model, image, transform, device)
    
    # Convert to standardized format
    emotion_scores = {}
    if 'probabilities' in result:
        for emotion in emotion_classes:
            emotion_scores[emotion] = result['probabilities'].get(emotion, 0.0)
        emotion_scores['predicted'] = result.get('predicted_flag', False)
        return emotion_scores
    else:
        # Return result as is if already in correct format
        return result


if __name__ == "__main__":
    # Test MLP-Mixer implementation
    print("Testing MLP-Mixer implementation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load model
        model = load_mlp_mixer_model(model_variant='base', device=device)
        print("✅ MLP-Mixer model loaded successfully")
        
        # Test transforms
        transforms_fn = get_mlp_mixer_transforms()
        print("✅ Transforms created successfully")
                
    except Exception as e:
        print(f"❌ Error testing MLP-Mixer: {e}") 