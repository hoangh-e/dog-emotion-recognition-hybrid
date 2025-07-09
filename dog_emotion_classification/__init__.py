"""
Dog Emotion Classification Package

This package provides comprehensive deep learning models for dog emotion classification
with 4 emotion classes in the correct order: ['angry', 'happy', 'relaxed', 'sad'].

✅ All modules have been standardized with:
- load_[model]_model() functions for loading trained models
- predict_emotion_[model]() functions for emotion prediction
- Correct emotion classes order: ['angry', 'happy', 'relaxed', 'sad']

Supported architectures:
- ResNet (ResNet50, ResNet101) ✅
- PURe Networks (PURe34, PURe50) ✅ 
- VGG (VGG16, VGG19) ✅
- DenseNet (DenseNet121, DenseNet169, DenseNet201) ✅
- Inception (Inception v3, GoogLeNet) ✅
- MobileNet (MobileNet v2, v3 Large, v3 Small) ✅
- EfficientNet (EfficientNet B0-B7) ✅
- Vision Transformer (ViT-B/16, ViT-L/16, ViT-H/14) ✅
- ConvNeXt (ConvNeXt Tiny, Small, Base, Large) ✅
- AlexNet ✅
- SqueezeNet (SqueezeNet 1.0, 1.1) ✅
- ShuffleNet (ShuffleNet v2 x0.5, x1.0, x1.5, x2.0) ✅
- Swin Transformer (Swin-T, Swin-S, Swin-B, Swin v2) ✅
- DeiT (Data-efficient Image Transformers) ✅
- NASNet (Neural Architecture Search) ✅
- MLP-Mixer (Multi-Layer Perceptron Mixer) ✅
- MaxViT (Multi-Axis Vision Transformer) ✅
- CoAtNet (Convolution and Attention Network) ✅
- NFNet (Normalizer-Free Networks) ✅
- ECA-Net (Efficient Channel Attention) ✅
- SE-Net (Squeeze-and-Excitation Networks) ✅
- ResMLP (Residual Multi-Layer Perceptron) ✅
- ConvFormer (Convolutional Transformer) ✅
- BoTNet (Bottleneck Transformer) ✅
- CvT (Convolutional Vision Transformer) ✅
- CMT (Convolutional Multi-Head Transformer) ✅

Each module provides:
1. load_[model]_model(model_path, num_classes=4, input_size, device='cuda')
2. predict_emotion_[model](image_path, model, transform, head_bbox=None, device='cuda', 
                           emotion_classes=['angry', 'happy', 'relaxed', 'sad'])
3. get_[model]_transforms(input_size, is_training=False)
4. create_[model]_model(num_classes=4, pretrained=True)

IMPORTANT: All models are trained with emotion classes in the order:
['angry', 'happy', 'relaxed', 'sad'] - Index 0=angry, 1=happy, 2=relaxed, 3=sad
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
from . import maxvit
from . import coatnet
from . import nfnet
from . import ecanet
from . import senet
from . import resmlp
from . import convformer
from . import botnet
from . import cvt
from . import cmt

__version__ = "3.3.0"
__author__ = "Dog Emotion Recognition Team"
__description__ = "Comprehensive deep learning package for dog emotion classification with 27 algorithm families"

# Emotion classes in correct order
EMOTION_CLASSES = ['angry', 'happy', 'relaxed', 'sad']

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
    "mlp_mixer",
    "maxvit",
    "coatnet", 
    "nfnet",
    "ecanet",
    "senet",
    "resmlp",
    "convformer",
    "botnet",
    "cvt",
    "cmt",
    "EMOTION_CLASSES"
] 