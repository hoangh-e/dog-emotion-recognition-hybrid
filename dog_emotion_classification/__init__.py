"""
Dog Emotion Classification Package

This package provides deep learning models for classifying dog emotions 
from images, including the Pure34 model implementation.
"""

from .pure34 import PURe34, load_pure34_model, load_resnet_model, predict_emotion_pure34, debug_checkpoint_structure, create_resnet_model
from .pure import (
    Pure18, Pure34, Pure50, Pure101, Pure152, PUReNet, 
    get_pure_model, PureTrainer, get_pure_transforms, 
    predict_emotion_pure, load_pure_model, download_pure_model
)

__version__ = "1.0.0"
__all__ = [
    # Pure34 legacy module
    "PURe34", "load_pure34_model", "load_resnet_model", "predict_emotion_pure34", 
    "debug_checkpoint_structure", "create_resnet_model",
    # New Pure module
    "Pure18", "Pure34", "Pure50", "Pure101", "Pure152", "PUReNet",
    "get_pure_model", "PureTrainer", "get_pure_transforms", 
    "predict_emotion_pure", "load_pure_model", "download_pure_model"
] 