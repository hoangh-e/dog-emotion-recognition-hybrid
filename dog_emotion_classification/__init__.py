"""
Dog Emotion Classification Package

This package provides deep learning models for classifying dog emotions 
from images, including Pure34, Pure50, and ResNet model implementations.
"""

from .pure34 import PURe34, load_pure34_model, load_resnet_model, predict_emotion_pure34, debug_checkpoint_structure, create_resnet_model
from .pure import (
    Pure18, Pure34, Pure50, Pure101, Pure152, PUReNet, 
    get_pure_model, PureTrainer, get_pure_transforms, 
    predict_emotion_pure, load_pure_model, download_pure_model
)
from .pure50 import load_pure50_model, predict_emotion_pure50, get_pure50_transforms, create_pure50_model
from .resnet import (
    load_resnet_model as load_resnet_emotion_model, 
    predict_emotion_resnet, 
    get_resnet_transforms,
    create_resnet_model as create_resnet_emotion_model,
    load_resnet50_model, 
    load_resnet101_model,
    predict_emotion_resnet50,
    predict_emotion_resnet101
)

__version__ = "1.0.0"
__all__ = [
    # Pure34 legacy module
    "PURe34", "load_pure34_model", "load_resnet_model", "predict_emotion_pure34", 
    "debug_checkpoint_structure", "create_resnet_model",
    # Pure module (generic)
    "Pure18", "Pure34", "Pure50", "Pure101", "Pure152", "PUReNet",
    "get_pure_model", "PureTrainer", "get_pure_transforms", 
    "predict_emotion_pure", "load_pure_model", "download_pure_model",
    # Pure50 specific
    "load_pure50_model", "predict_emotion_pure50", "get_pure50_transforms", "create_pure50_model",
    # ResNet models
    "load_resnet_emotion_model", "predict_emotion_resnet", "get_resnet_transforms", "create_resnet_emotion_model",
    "load_resnet50_model", "load_resnet101_model", "predict_emotion_resnet50", "predict_emotion_resnet101"
] 