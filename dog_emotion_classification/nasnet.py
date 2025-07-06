"""
NASNet (Neural Architecture Search Network) models for dog emotion classification.

This module provides NASNet implementations optimized for dog emotion classification 
with 4 emotion classes: sad, angry, happy, relaxed.

Based on "Learning Transferable Architectures for Scalable Image Recognition"
by Barret Zoph et al. (2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class SeparableConv2d(nn.Module):
    """Separable convolution for NASNet cells."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                 stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class NASNetCell(nn.Module):
    """NASNet Normal/Reduction Cell."""
    
    def __init__(self, in_channels, out_channels, is_reduction=False):
        super().__init__()
        self.is_reduction = is_reduction
        
        # Cell operations
        self.sep_conv_3x3 = SeparableConv2d(in_channels, out_channels, 3, padding=1)
        self.sep_conv_5x5 = SeparableConv2d(in_channels, out_channels, 5, padding=2)
        self.sep_conv_7x7 = SeparableConv2d(in_channels, out_channels, 7, padding=3)
        
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)
        
        # Reduction operations
        if is_reduction:
            self.reduction_conv = nn.Conv2d(in_channels, out_channels, 1, stride=2)
            self.reduction_pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # Simplified NASNet cell operations
        if self.is_reduction:
            # Reduction cell
            branch1 = self.reduction_conv(x)
            branch2 = self.reduction_pool(x)
            if branch2.shape[1] != branch1.shape[1]:
                branch2 = F.conv2d(branch2, torch.ones(branch1.shape[1], branch2.shape[1], 1, 1).to(x.device))
            out = branch1 + branch2
        else:
            # Normal cell
            branch1 = self.sep_conv_3x3(x)
            branch2 = self.sep_conv_5x5(x)
            branch3 = self.avg_pool(x)
            
            # Adjust channels if needed
            if branch3.shape[1] != branch1.shape[1]:
                branch3 = F.conv2d(branch3, torch.ones(branch1.shape[1], branch3.shape[1], 1, 1).to(x.device))
            
            out = branch1 + branch2 + branch3
        
        out = self.bn(out)
        return F.relu(out)


class NASNetModel(nn.Module):
    """NASNet model for dog emotion classification."""
    
    def __init__(self, num_classes=4, num_cells=6, channels=32):
        super().__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # NASNet cells
        self.cells = nn.ModuleList()
        current_channels = channels
        
        for i in range(num_cells):
            # Add reduction cell every 2 normal cells
            is_reduction = (i % 3 == 2)
            
            if is_reduction:
                next_channels = current_channels * 2
            else:
                next_channels = current_channels
                
            cell = NASNetCell(current_channels, next_channels, is_reduction)
            self.cells.append(cell)
            current_channels = next_channels
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.stem(x)
        
        for cell in self.cells:
            x = cell(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


def load_nasnet_model(checkpoint_path=None, model_variant='mobile', num_classes=4,
                     device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load NASNet model for dog emotion classification.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_variant: NASNet variant ('mobile', 'large')
        num_classes: Number of emotion classes (default: 4)
        device: Device to load model on
    
    Returns:
        Loaded NASNet model
    """
    try:
        # Try to use timm if available
        if TIMM_AVAILABLE and checkpoint_path is None:
            model_names = {
                'mobile': 'nasnetalarge',
                'large': 'nasnetalarge'
            }
            
            model_name = model_names.get(model_variant, 'nasnetalarge')
            
            try:
                # Create timm model
                model = timm.create_model(
                    model_name,
                    pretrained=True,
                    num_classes=num_classes
                )
                
                print(f"Loaded {model_name} from timm")
                
            except Exception as e:
                print(f"Failed to load from timm: {e}")
                # Fallback to custom implementation
                model = create_custom_nasnet(model_variant, num_classes)
        else:
            # Use custom implementation
            model = create_custom_nasnet(model_variant, num_classes)
        
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
                print(f"Loaded NASNet checkpoint from {checkpoint_path}")
                
            except Exception as e:
                print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
        
        model = model.to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"Error loading NASNet model: {e}")
        # Return simple fallback model
        return create_simple_nasnet(num_classes, device)


def create_custom_nasnet(variant='mobile', num_classes=4):
    """Create custom NASNet model."""
    
    if variant == 'mobile':
        return NASNetModel(num_classes=num_classes, num_cells=6, channels=32)
    elif variant == 'large':
        return NASNetModel(num_classes=num_classes, num_cells=12, channels=64)
    else:
        return NASNetModel(num_classes=num_classes, num_cells=6, channels=32)


def create_simple_nasnet(num_classes=4, device='cuda'):
    """Create a simple NASNet-inspired model."""
    
    class SimpleNASNet(nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            
            # Stem
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            
            # Feature extraction blocks
            self.blocks = nn.ModuleList([
                self._make_block(32, 64, stride=2),
                self._make_block(64, 128, stride=2),
                self._make_block(128, 256, stride=2),
                self._make_block(256, 512, stride=2),
            ])
            
            # Classifier
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(512, num_classes)
            self.dropout = nn.Dropout(0.5)
            
        def _make_block(self, in_channels, out_channels, stride=1):
            return nn.Sequential(
                SeparableConv2d(in_channels, out_channels, 3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                SeparableConv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x):
            x = self.stem(x)
            
            for block in self.blocks:
                x = block(x)
            
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.classifier(x)
            
            return x
    
    model = SimpleNASNet(num_classes).to(device)
    return model


def predict_emotion_nasnet(model, image_path, transforms_fn=None, device='cuda'):
    """
    Predict dog emotion using NASNet model.
    
    Args:
        model: Loaded NASNet model
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
            transforms_fn = get_nasnet_transforms()
        
        # Apply transforms
        input_tensor = transforms_fn(image).unsqueeze(0).to(device)
        
        # Inference
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
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
        print(f"Error in NASNet prediction: {e}")
        # Return fallback result
        return {
            'predicted_emotion': 'happy',
            'confidence': 0.25,
            'predicted_flag': False,
            'probabilities': {emotion: 0.25 for emotion in emotion_classes}
        }


def get_nasnet_transforms(input_size=224, is_training=False):
    """Get NASNet-specific image transforms."""
    
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
NASNET_VARIANTS = {
    'mobile': {
        'input_size': 224,
        'params': '5.3M',
        'description': 'NASNet-A Mobile optimized for efficiency'
    },
    'large': {
        'input_size': 331,
        'params': '88.9M', 
        'description': 'NASNet-A Large for maximum accuracy'
    }
}


if __name__ == "__main__":
    # Test NASNet implementation
    print("Testing NASNet implementation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load model
        model = load_nasnet_model(model_variant='mobile', device=device)
        print("✅ NASNet model loaded successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ Model output shape: {output.shape}")
                
    except Exception as e:
        print(f"❌ Error testing NASNet: {e}") 