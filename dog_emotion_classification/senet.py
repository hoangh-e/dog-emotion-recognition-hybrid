"""
SE-Net (Squeeze-and-Excitation Networks) models for dog emotion classification.

This module provides SE-Net implementations optimized for 
dog emotion classification with 4 emotion classes: sad, angry, happy, relaxed.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import math


class SEModule(nn.Module):
    """Squeeze-and-Excitation module."""
    
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEBasicBlock(nn.Module):
    """Basic ResNet block with SE module."""
    
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.se = SEModule(planes, reduction)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class SEBottleneck(nn.Module):
    """Bottleneck ResNet block with SE module."""
    
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(planes * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class SEResNet(nn.Module):
    """SE-ResNet model for emotion classification."""
    
    def __init__(self, block, layers, num_classes=4, reduction=16):
        super(SEResNet, self).__init__()
        
        self.inplanes = 64
        self.reduction = reduction
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.reduction))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction=self.reduction))
        
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class SEResNeXtBottleneck(nn.Module):
    """ResNeXt bottleneck block with SE module."""
    
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 groups=32, base_width=4, reduction=16):
        super(SEResNeXtBottleneck, self).__init__()
        
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(planes * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


def se_resnet18(num_classes=4, **kwargs):
    """SE-ResNet-18 model."""
    return SEResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def se_resnet34(num_classes=4, **kwargs):
    """SE-ResNet-34 model."""
    return SEResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def se_resnet50(num_classes=4, **kwargs):
    """SE-ResNet-50 model."""
    return SEResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def se_resnet101(num_classes=4, **kwargs):
    """SE-ResNet-101 model."""
    return SEResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)


def se_resnet152(num_classes=4, **kwargs):
    """SE-ResNet-152 model."""
    return SEResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)


def se_resnext50_32x4d(num_classes=4, **kwargs):
    """SE-ResNeXt-50 32x4d model."""
    kwargs['groups'] = 32
    kwargs['base_width'] = 4
    return SEResNet(SEResNeXtBottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def se_resnext101_32x8d(num_classes=4, **kwargs):
    """SE-ResNeXt-101 32x8d model."""
    kwargs['groups'] = 32
    kwargs['base_width'] = 8
    return SEResNet(SEResNeXtBottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)


def load_senet_model(model_path, architecture='se_resnet50', num_classes=4, input_size=224, device='cuda'):
    """
    Load a pre-trained SE-Net model for dog emotion classification.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model checkpoint
    architecture : str
        SE-Net architecture ('se_resnet18', 'se_resnet34', 'se_resnet50', 'se_resnet101', 'se_resnet152', 
                           'se_resnext50_32x4d', 'se_resnext101_32x8d')
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
    if architecture == 'se_resnet18':
        model = se_resnet18(num_classes=num_classes)
    elif architecture == 'se_resnet34':
        model = se_resnet34(num_classes=num_classes)
    elif architecture == 'se_resnet50':
        model = se_resnet50(num_classes=num_classes)
    elif architecture == 'se_resnet101':
        model = se_resnet101(num_classes=num_classes)
    elif architecture == 'se_resnet152':
        model = se_resnet152(num_classes=num_classes)
    elif architecture == 'se_resnext50_32x4d':
        model = se_resnext50_32x4d(num_classes=num_classes)
    elif architecture == 'se_resnext101_32x8d':
        model = se_resnext101_32x8d(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported SE-Net architecture: {architecture}")
    
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


def predict_emotion_senet(image_path, model, transform, head_bbox=None, device='cuda',
                         emotion_classes=['angry', 'happy', 'relaxed', 'sad']):
    """
    Predict dog emotion using SE-Net model.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    model : torch.nn.Module
        Loaded SE-Net model
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
        print(f"❌ Error in SE-Net emotion prediction: {e}")
        # Return default scores on error
        emotion_scores = {emotion: 0.0 for emotion in emotion_classes}
        emotion_scores['predicted'] = False
        return emotion_scores


def get_senet_transforms(input_size=224, is_training=True):
    """
    Get preprocessing transforms for SE-Net models.
    
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


def create_senet_model(architecture='se_resnet50', num_classes=4, pretrained=False):
    """
    Create a SE-Net model for emotion classification.
    
    Parameters:
    -----------
    architecture : str
        SE-Net architecture ('se_resnet18', 'se_resnet34', 'se_resnet50', 'se_resnet101', 'se_resnet152',
                           'se_resnext50_32x4d', 'se_resnext101_32x8d')
    num_classes : int
        Number of emotion classes (default: 4)
    pretrained : bool
        Whether to use pretrained weights (default: False)
        
    Returns:
    --------
    torch.nn.Module
        SE-Net model
    """
    
    if architecture == 'se_resnet18':
        model = se_resnet18(num_classes=num_classes)
    elif architecture == 'se_resnet34':
        model = se_resnet34(num_classes=num_classes)
    elif architecture == 'se_resnet50':
        model = se_resnet50(num_classes=num_classes)
    elif architecture == 'se_resnet101':
        model = se_resnet101(num_classes=num_classes)
    elif architecture == 'se_resnet152':
        model = se_resnet152(num_classes=num_classes)
    elif architecture == 'se_resnext50_32x4d':
        model = se_resnext50_32x4d(num_classes=num_classes)
    elif architecture == 'se_resnext101_32x8d':
        model = se_resnext101_32x8d(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported SE-Net architecture: {architecture}")
    
    return model


# Convenience functions for specific architectures
def load_se_resnet18_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_senet_model(model_path, 'se_resnet18', num_classes, input_size, device)

def load_se_resnet34_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_senet_model(model_path, 'se_resnet34', num_classes, input_size, device)

def load_se_resnet50_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_senet_model(model_path, 'se_resnet50', num_classes, input_size, device)

def load_se_resnet101_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_senet_model(model_path, 'se_resnet101', num_classes, input_size, device)

def load_se_resnet152_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_senet_model(model_path, 'se_resnet152', num_classes, input_size, device)

def load_se_resnext50_32x4d_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_senet_model(model_path, 'se_resnext50_32x4d', num_classes, input_size, device)

def load_se_resnext101_32x8d_model(model_path, num_classes=4, input_size=224, device='cuda'):
    return load_senet_model(model_path, 'se_resnext101_32x8d', num_classes, input_size, device)

def predict_emotion_se_resnet18(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_senet(image_path, model, transform, head_bbox, device)

def predict_emotion_se_resnet34(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_senet(image_path, model, transform, head_bbox, device)

def predict_emotion_se_resnet50(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_senet(image_path, model, transform, head_bbox, device)

def predict_emotion_se_resnet101(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_senet(image_path, model, transform, head_bbox, device)

def predict_emotion_se_resnet152(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_senet(image_path, model, transform, head_bbox, device)

def predict_emotion_se_resnext50_32x4d(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_senet(image_path, model, transform, head_bbox, device)

def predict_emotion_se_resnext101_32x8d(image_path, model, transform, head_bbox=None, device='cuda'):
    return predict_emotion_senet(image_path, model, transform, head_bbox, device) 