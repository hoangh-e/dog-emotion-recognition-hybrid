"""
Dog Emotion Classification Package

This package provides comprehensive deep learning models for dog emotion classification
with 4 emotion classes: sad, angry, happy, relaxed.

Supported architectures:
- ResNet (ResNet50, ResNet101)
- PURe Networks (PURe34, PURe50)
- VGG (VGG16, VGG19)
- DenseNet (DenseNet121, DenseNet169, DenseNet201)
- Inception (Inception v3, GoogLeNet)
- MobileNet (MobileNet v2, v3 Large, v3 Small)
- EfficientNet (EfficientNet B0-B7)
- Vision Transformer (ViT-B/16, ViT-L/16, ViT-H/14)
- ConvNeXt (ConvNeXt Tiny, Small, Base, Large)
- AlexNet
- SqueezeNet (SqueezeNet 1.0, 1.1)
- ShuffleNet (ShuffleNet v2 x0.5, x1.0, x1.5, x2.0)
- Swin Transformer (Swin-T, Swin-S, Swin-B, Swin v2)
- DeiT (Data-efficient Image Transformers)
- NASNet (Neural Architecture Search)
- MLP-Mixer (Multi-Layer Perceptron Mixer)
"""

# Import all modules
from . import resnet
from . import pure
from . import pure34
from . import pure50
from . import vgg
from . import densenet
from . import inception
from . import mobilenet
from . import efficientnet
from . import vit
from . import convnext
from . import alexnet
from . import squeezenet
from . import shufflenet
from . import swin
from . import deit
from . import nasnet
from . import mlp_mixer

__version__ = "3.1.0"
__author__ = "Dog Emotion Recognition Team"
__description__ = "Comprehensive deep learning package for dog emotion classification with 19 algorithm families"

__all__ = [
    "resnet",
    "pure", 
    "pure34",
    "pure50",
    "vgg",
    "densenet", 
    "inception",
    "mobilenet",
    "efficientnet",
    "vit",
    "convnext",
    "alexnet",
    "squeezenet", 
    "shufflenet",
    "swin",
    "deit",
    "nasnet",
    "mlp_mixer"
] 