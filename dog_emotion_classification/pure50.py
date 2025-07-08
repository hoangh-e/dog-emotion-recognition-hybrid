"""
Pure50 model wrapper for dog emotion classification.

This module provides Pure50-specific functions for compatibility with 
multi-model notebooks that expect separate pure50 loading functions.
"""

from .pure import Pure50, get_pure_transforms, predict_emotion_pure, load_pure_model


def load_pure50_model(model_path, num_classes=4, input_size=512, device='cuda'):
    """
    Load Pure50 model for dog emotion classification.
    
    This is a wrapper around the generic load_pure_model function
    specifically for Pure50 architecture.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved Pure50 model checkpoint
    num_classes : int
        Number of emotion classes (default: 4)
    input_size : int
        Input image size (default: 512)
    device : str
        Device to load model on ('cuda' or 'cpu')
        
    Returns:
    --------
    tuple
        (model, transform) - loaded Pure50 model and preprocessing transform
    """
    return load_pure_model(
        model_path=model_path,
        architecture='pure50',
        num_classes=num_classes,
        input_size=input_size,
        device=device
    )


def predict_emotion_pure50(image_path, model, transform, head_bbox=None, device='cuda',
                          emotion_classes=['angry', 'happy', 'relaxed', 'sad']):
    """
    Predict dog emotion using Pure50 model.
    
    This is a wrapper around the generic predict_emotion_pure function
    specifically for Pure50 architecture.
    
    Parameters:
    -----------
    image_path : str or PIL.Image
        Path to the input image or PIL Image object
    model : torch.nn.Module
        Loaded Pure50 model
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
    return predict_emotion_pure(
        image_path=image_path,
        model=model,
        transform=transform,
        head_bbox=head_bbox,
        device=device,
        emotion_classes=emotion_classes
    )


def get_pure50_transforms(input_size=512, is_training=True):
    """
    Get preprocessing transforms for Pure50 model.
    
    Parameters:
    -----------
    input_size : int
        Input image size (default: 512)
    is_training : bool
        Whether transforms are for training or inference
        
    Returns:
    --------
    torchvision.transforms.Compose
        Preprocessing transforms
    """
    return get_pure_transforms(input_size=input_size, is_training=is_training)


def create_pure50_model(num_classes=4, input_size=512):
    """
    Create a Pure50 model for dog emotion classification.
    
    Parameters:
    -----------
    num_classes : int
        Number of emotion classes
    input_size : int
        Input image size
        
    Returns:
    --------
    torch.nn.Module
        Pure50 model
    """
    return Pure50(num_classes=num_classes, input_size=input_size) 