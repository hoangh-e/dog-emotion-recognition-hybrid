"""
Dog Emotion Classification Package

This package provides deep learning models for classifying dog emotions 
from images, including the Pure34 model implementation.
"""

from .pure34 import PURe34, load_pure34_model, load_resnet_model, predict_emotion_pure34, debug_checkpoint_structure, create_resnet_model

__version__ = "1.0.0"
__all__ = ["PURe34", "load_pure34_model", "load_resnet_model", "predict_emotion_pure34", "debug_checkpoint_structure", "create_resnet_model"] 