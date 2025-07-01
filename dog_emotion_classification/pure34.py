"""
PURe34 (Product-Unit Residual Network) implementation for dog emotion classification.

Based on the paper: "Deep residual learning with product units"
This implementation replaces standard convolutions with product units in residual blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np


class ProductUnit2D(nn.Module):
    """
    2D Product Unit layer for convolutional processing.
    
    This layer implements multiplicative feature interactions using:
    y(i,j) = exp(sum(w(m,n) * log(x(i+m,j+n))))
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ProductUnit2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Trainable weights for product unit
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Trainable threshold parameter to ensure positivity
        self.threshold = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=5.0)
        
    def forward(self, x):
        # Ensure inputs are positive by clamping with softplus threshold
        eps = 1e-7
        min_val = F.softplus(self.threshold) + eps
        x_clamped = torch.clamp(x, min=min_val)
        
        # Apply logarithm transformation
        log_x = torch.log(x_clamped)
        
        # Perform convolution in log space
        conv_output = F.conv2d(log_x, self.weight, self.bias, self.stride, self.padding)
        
        # Apply exponential to return to original space
        output = torch.exp(conv_output)
        
        return output


class ProductUnitResidualBlock(nn.Module):
    """
    Product-Unit Residual Block that replaces the second convolution with a product unit.
    """
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ProductUnitResidualBlock, self).__init__()
        
        # First convolution (standard)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution replaced with product unit
        self.prod_unit = ProductUnit2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        # First conv + BN (no ReLU in PURe)
        out = self.conv1(x)
        out = self.bn1(out)
        
        # Product unit + BN
        out = self.prod_unit(out)
        out = self.bn2(out)
        
        # Downsample identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)
            
        # Add residual connection
        out += identity
        
        return out


class PURe34(nn.Module):
    """
    PURe34 model for dog emotion classification.
    
    Architecture based on ResNet34 but with product units replacing 
    the second convolution in each residual block.
    """
    
    def __init__(self, num_classes=4):
        super(PURe34, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers with product units
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            
        layers = []
        layers.append(ProductUnitResidualBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(ProductUnitResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def load_resnet_model(model_path, num_classes=4, device='cuda'):
    """
    Load a ResNet34 model for emotion classification as fallback.
    
    Args:
        model_path (str): Path to the model checkpoint (.pth file)
        num_classes (int): Number of emotion classes (default: 4)
        device (str): Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        tuple: (model, transform) where model is the loaded ResNet model 
               and transform is the image preprocessing pipeline
    """
    import torchvision.models as models
    
    # Create ResNet34 model
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load state dict
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Define preprocessing transforms (ResNet typically uses 224x224, but we'll use 512x512 for consistency)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, transform


def load_pure34_model(model_path, num_classes=4, device='cuda'):
    """
    Load a trained PURe34 model from a checkpoint file.
    Falls back to ResNet if Pure34 loading fails.
    
    Args:
        model_path (str): Path to the model checkpoint (.pth file)
        num_classes (int): Number of emotion classes (default: 4)
        device (str): Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        tuple: (model, transform) where model is the loaded model 
               and transform is the image preprocessing pipeline
    """
    # Load state dict first to inspect
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Check if this is a Pure34 model or a ResNet model
    has_prod_unit = any('prod_unit' in key for key in state_dict.keys())
    has_conv2 = any('conv2' in key for key in state_dict.keys())
    
    if has_conv2 and not has_prod_unit:
        # This is a ResNet model - use ResNet directly
        print("⚠️  Detected ResNet model instead of Pure34. Loading as ResNet...")
        return load_resnet_model(model_path, num_classes, device)
    
    # Try to load as Pure34
    try:
        print("🔄 Attempting to load as Pure34 model...")
        model = PURe34(num_classes=num_classes)
        
        if has_prod_unit:
            print("✅ Loading native Pure34 model...")
            model.load_state_dict(state_dict)
        else:
            print("⚠️  No Pure34 or ResNet signature found. Using random initialization...")
            model._initialize_weights()
            
        model.to(device)
        model.eval()
        
        # Define preprocessing transforms (Pure34 uses 512x512 input)
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return model, transform
        
    except Exception as e:
        print(f"❌ Failed to load as Pure34: {e}")
        print("🔄 Falling back to ResNet model...")
        return load_resnet_model(model_path, num_classes, device)


def predict_emotion_pure34(image_path, model, transform, head_bbox=None, device='cuda'):
    """
    Predict dog emotion using PURe34 or ResNet model.
    
    Args:
        image_path (str): Path to the image file
        model: Loaded model (PURe34 or ResNet)
        transform: Image preprocessing transform
        head_bbox (list, optional): Bounding box [x1, y1, x2, y2] to crop head region
        device (str): Device to run inference on
        
    Returns:
        dict: Emotion probabilities and prediction status
              {'sad': float, 'angry': float, 'happy': float, 'relaxed': float, 'predicted': bool}
    """
    emotion_classes = ['sad', 'angry', 'happy', 'relaxed']
    
    try:
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Crop head region if bbox provided
        if head_bbox is not None:
            x1, y1, x2, y2 = map(int, head_bbox)
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:  # Valid crop region
                image = image[y1:y2, x1:x2]
        
        # Convert to PIL and apply transforms
        pil_image = Image.fromarray(image)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        
        # Map to emotion classes
        emotion_scores = {}
        for i, emotion in enumerate(emotion_classes):
            if i < len(probabilities):
                emotion_scores[emotion] = float(probabilities[i])
            else:
                emotion_scores[emotion] = 0.0
                
        emotion_scores['predicted'] = True
        return emotion_scores
        
    except Exception as e:
        print(f"❌ Error in Pure34 emotion classification for {image_path}: {e}")
        return {'sad': 0.0, 'angry': 0.0, 'happy': 0.0, 'relaxed': 0.0, 'predicted': False}


def download_pure34_model(output_path="pure34_best.pth"):
    """
    Download the pre-trained Pure34 model using gdown.
    
    Args:
        output_path (str): Path where to save the downloaded model
        
    Returns:
        str: Path to the downloaded model file
    """
    import subprocess
    import os
    
    # Google Drive file ID for the Pure34 model
    file_id = "11Oy8lqKF7MeMWV89SR-kN6sNLwNi-jjQ"
    
    try:
        # Download using gdown
        cmd = f"gdown {file_id} -O {output_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Pure34 model downloaded successfully: {output_path}")
            return output_path
        else:
            print(f"❌ Error downloading Pure34 model: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading Pure34 model: {e}")
        return None 