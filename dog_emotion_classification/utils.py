"""
Utility functions for dog emotion classification package.

This module provides helper functions for dataset conversion and label management.
"""

import numpy as np
import pandas as pd
import torch


def convert_4class_to_3class_labels(labels):
    """
    Convert 4-class labels to 3-class by removing 'sad' class.
    
    Original 4-class mapping:
    0=angry → 0=angry (no change)
    1=happy → 1=happy (no change)  
    2=relaxed → 2=relaxed (no change)
    3=sad → REMOVE (filter out these samples)
    
    Parameters:
    -----------
    labels : array-like
        Array of labels with values 0, 1, 2, 3
        
    Returns:
    --------
    numpy.ndarray
        Filtered labels containing only samples with labels 0, 1, 2
    """
    labels = np.array(labels)
    
    # Create mask to filter out 'sad' class (label 3)
    mask = labels != 3
    filtered_labels = labels[mask]
    
    print(f"📊 Dataset conversion summary:")
    print(f"   Original samples: {len(labels)}")
    print(f"   Filtered samples: {len(filtered_labels)} (removed {np.sum(~mask)} 'sad' samples)")
    print(f"   Reduction: {(1 - len(filtered_labels)/len(labels))*100:.1f}%")
    
    return filtered_labels


def convert_4class_to_3class_dataset(data, labels):
    """
    Convert dataset from 4-class to 3-class by removing 'sad' samples.
    
    Parameters:
    -----------
    data : array-like
        Dataset samples (images, features, etc.)
    labels : array-like
        Corresponding labels
        
    Returns:
    --------
    tuple
        (filtered_data, filtered_labels) - dataset with 'sad' samples removed
    """
    labels = np.array(labels)
    
    # Create mask to filter out 'sad' class (label 3)
    mask = labels != 3
    
    # Filter both data and labels
    filtered_data = np.array(data)[mask] if hasattr(data, '__getitem__') else [data[i] for i in range(len(data)) if mask[i]]
    filtered_labels = labels[mask]
    
    print(f"📊 Dataset conversion summary:")
    print(f"   Original samples: {len(labels)}")
    print(f"   Filtered samples: {len(filtered_labels)} (removed {np.sum(~mask)} 'sad' samples)")
    print(f"   Reduction: {(1 - len(filtered_labels)/len(labels))*100:.1f}%")
    
    return filtered_data, filtered_labels


def convert_dataframe_4class_to_3class(df, label_column='label'):
    """
    Convert pandas DataFrame from 4-class to 3-class by removing 'sad' rows.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the dataset
    label_column : str
        Name of the column containing labels
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame with 'sad' samples removed
    """
    original_count = len(df)
    
    # Filter out 'sad' class
    if df[label_column].dtype == 'object':
        # String labels
        filtered_df = df[df[label_column] != 'sad'].copy()
    else:
        # Numeric labels (assuming 3 = sad)
        filtered_df = df[df[label_column] != 3].copy()
    
    filtered_count = len(filtered_df)
    
    print(f"📊 DataFrame conversion summary:")
    print(f"   Original rows: {original_count}")
    print(f"   Filtered rows: {filtered_count} (removed {original_count - filtered_count} 'sad' rows)")
    print(f"   Reduction: {(1 - filtered_count/original_count)*100:.1f}%")
    
    return filtered_df


def get_3class_emotion_mapping():
    """
    Get emotion class mapping for 3-class classification.
    
    Returns:
    --------
    dict
        Dictionary mapping class indices to emotion names
    """
    return {
        0: 'angry',
        1: 'happy', 
        2: 'relaxed'
    }


def get_3class_emotion_classes():
    """
    Get emotion class list for 3-class classification.
    
    Returns:
    --------
    list
        List of emotion class names in correct order
    """
    return ['angry', 'happy', 'relaxed']


def convert_4class_model_output_to_3class(output_tensor):
    """
    Convert 4-class model output to 3-class by removing 'sad' dimension.
    
    Note: This is for compatibility only. For optimal performance, 
    retrain models with 3-class targets.
    
    Parameters:
    -----------
    output_tensor : torch.Tensor
        Model output tensor with shape (..., 4)
        
    Returns:
    --------
    torch.Tensor
        Model output tensor with shape (..., 3) - 'sad' dimension removed
    """
    if output_tensor.shape[-1] != 4:
        raise ValueError(f"Expected last dimension to be 4, got {output_tensor.shape[-1]}")
    
    # Remove the last dimension (index 3 = 'sad')
    return output_tensor[..., :3]


# Emotion class constants
EMOTION_CLASSES_3CLASS = ['angry', 'happy', 'relaxed']
EMOTION_CLASSES_4CLASS = ['angry', 'happy', 'relaxed', 'sad']

# Class mappings
CLASS_MAPPING_3CLASS = {0: 'angry', 1: 'happy', 2: 'relaxed'}
CLASS_MAPPING_4CLASS = {0: 'angry', 1: 'happy', 2: 'relaxed', 3: 'sad'} 