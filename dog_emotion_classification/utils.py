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

# New constants for merge configuration
EMOTION_CLASSES_3CLASS_MERGE = ['angry', 'happy', 'sad']

# Class mappings
CLASS_MAPPING_3CLASS = {0: 'angry', 1: 'happy', 2: 'relaxed'}
CLASS_MAPPING_4CLASS = {0: 'angry', 1: 'happy', 2: 'relaxed', 3: 'sad'}
CLASS_MAPPING_3CLASS_MERGE = {0: 'angry', 1: 'happy', 2: 'sad'}


# =====================================
# NEW FUNCTIONS FOR MERGE CONFIGURATION
# =====================================

def convert_4class_to_3class_merge_relaxed_sad_labels(labels):
    """
    Convert 4-class labels to 3-class by merging 'relaxed' + 'sad' → 'sad'.
    
    Original 4-class mapping:
    0=angry → 0=angry (no change)
    1=happy → 1=happy (no change)  
    2=relaxed → 2=sad (merge to sad)
    3=sad → 2=sad (merge to sad)
    
    Parameters:
    -----------
    labels : array-like
        Array of labels with values 0, 1, 2, 3
        
    Returns:
    --------
    numpy.ndarray
        Converted labels with 'relaxed' and 'sad' merged to 'sad' (label 2)
    """
    labels = np.array(labels)
    converted_labels = labels.copy()
    
    # Merge 'relaxed' (2) and 'sad' (3) to 'sad' (2)
    converted_labels[labels == 3] = 2  # sad (3) → sad (2)
    # relaxed (2) stays as (2) but now represents 'sad'
    
    relaxed_count = np.sum(labels == 2)
    sad_count = np.sum(labels == 3)
    merged_count = np.sum(converted_labels == 2)
    
    print(f"📊 Dataset merge summary:")
    print(f"   Original 'relaxed' samples: {relaxed_count}")
    print(f"   Original 'sad' samples: {sad_count}")
    print(f"   Merged 'sad' samples: {merged_count}")
    print(f"   Total samples preserved: {len(converted_labels)}")
    
    return converted_labels


def convert_4class_to_3class_merge_relaxed_sad_dataset(data, labels):
    """
    Convert dataset from 4-class to 3-class by merging 'relaxed' + 'sad' → 'sad'.
    
    Parameters:
    -----------
    data : array-like
        Dataset samples (images, features, etc.)
    labels : array-like
        Corresponding labels
        
    Returns:
    --------
    tuple
        (data, converted_labels) - all data preserved with merged labels
    """
    converted_labels = convert_4class_to_3class_merge_relaxed_sad_labels(labels)
    
    # All data is preserved, only labels are converted
    return data, converted_labels


def convert_dataframe_4class_to_3class_merge_relaxed_sad(df, label_column='label'):
    """
    Convert pandas DataFrame from 4-class to 3-class by merging 'relaxed' + 'sad' → 'sad'.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the dataset
    label_column : str
        Name of the column containing labels
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with 'relaxed' and 'sad' merged to 'sad'
    """
    original_count = len(df)
    result_df = df.copy()
    
    # Convert labels based on type
    if df[label_column].dtype == 'object':
        # String labels: merge 'relaxed' → 'sad'
        result_df.loc[result_df[label_column] == 'relaxed', label_column] = 'sad'
        relaxed_count = len(df[df[label_column] == 'relaxed'])
        sad_count = len(df[df[label_column] == 'sad'])
    else:
        # Numeric labels: merge 3 → 2
        relaxed_count = len(df[df[label_column] == 2])
        sad_count = len(df[df[label_column] == 3])
        result_df.loc[result_df[label_column] == 3, label_column] = 2
    
    merged_count = len(result_df[result_df[label_column] == ('sad' if df[label_column].dtype == 'object' else 2)])
    
    print(f"📊 DataFrame merge summary:")
    print(f"   Original 'relaxed' rows: {relaxed_count}")
    print(f"   Original 'sad' rows: {sad_count}")
    print(f"   Merged 'sad' rows: {merged_count}")
    print(f"   Total rows preserved: {len(result_df)}")
    
    return result_df


def get_3class_emotion_mapping_merge():
    """
    Get emotion class mapping for 3-class classification (merge configuration).
    
    Returns:
    --------
    dict
        Dictionary mapping class indices to emotion names
    """
    return {
        0: 'angry',
        1: 'happy', 
        2: 'sad'
    }


def get_3class_emotion_classes_merge():
    """
    Get emotion class list for 3-class classification (merge configuration).
    
    Returns:
    --------
    list
        List of emotion class names in correct order
    """
    return ['angry', 'happy', 'sad']


def convert_4class_model_output_to_3class_merge(output_tensor):
    """
    Convert 4-class model output to 3-class by merging 'relaxed' + 'sad' → 'sad'.
    
    Parameters:
    -----------
    output_tensor : torch.Tensor
        Model output tensor with shape (..., 4)
        
    Returns:
    --------
    torch.Tensor
        Model output tensor with shape (..., 3) - 'relaxed' and 'sad' merged
    """
    if output_tensor.shape[-1] != 4:
        raise ValueError(f"Expected last dimension to be 4, got {output_tensor.shape[-1]}")
    
    # Merge 'relaxed' (index 2) and 'sad' (index 3) → 'sad' (index 2)
    merged_tensor = torch.zeros(output_tensor.shape[:-1] + (3,), dtype=output_tensor.dtype, device=output_tensor.device)
    merged_tensor[..., 0] = output_tensor[..., 0]  # angry
    merged_tensor[..., 1] = output_tensor[..., 1]  # happy
    merged_tensor[..., 2] = output_tensor[..., 2] + output_tensor[..., 3]  # relaxed + sad → sad
    
    return merged_tensor 