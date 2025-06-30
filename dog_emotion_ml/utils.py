"""
Các hàm tiện ích cho package Dog Emotion Recognition.

Module này cung cấp các hàm hỗ trợ cho việc kiểm tra dữ liệu, trực quan hóa,
và các thao tác chung được sử dụng trong package.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import json
import re
import logging
from typing import Dict, Any, Optional, Union
warnings.filterwarnings('ignore')


def validate_emotion_features(features, feature_names=None):
    """
    Kiểm tra tính hợp lệ của các giá trị đặc trưng cảm xúc.
    
    Parameters:
    -----------
    features : array-like
        Các giá trị đặc trưng cần kiểm tra
    feature_names : list, optional
        Tên các đặc trưng để báo lỗi
        
    Returns:
    --------
    bool
        True nếu hợp lệ, False nếu không
    """
    features = np.array(features)
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(features))]
    
    # Check for negative values
    if np.any(features < 0):
        negative_indices = np.where(features < 0)[0]
        print(f"Warning: Negative values found in features: {[feature_names[i] for i in negative_indices]}")
        return False
    
    # Check for values > 1
    if np.any(features > 1):
        over_one_indices = np.where(features > 1)[0]
        print(f"Warning: Values > 1 found in features: {[feature_names[i] for i in over_one_indices]}")
        return False
    
    # Check if sum is approximately 1 (for probability distributions)
    total = np.sum(features)
    if not (0.9 <= total <= 1.1):
        print(f"Warning: Feature sum ({total:.3f}) is not close to 1.0")
        return False
    
    return True


def normalize_probabilities(features):
    """
    Normalize features to sum to 1.0.
    
    Parameters:
    -----------
    features : array-like
        Feature values to normalize
        
    Returns:
    --------
    np.ndarray
        Normalized features
    """
    features = np.array(features)
    total = np.sum(features)
    
    if total == 0:
        return np.ones_like(features) / len(features)
    
    return features / total


def plot_feature_distributions(data, feature_cols, title="Feature Distributions"):
    """
    Plot distributions of features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing features
    feature_cols : list
        Names of feature columns to plot
    title : str
        Title for the plot
    """
    n_features = len(feature_cols)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        if i < len(axes):
            axes[i].hist(data[col], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        Label names
    title : str
        Title for the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_dict, metric='accuracy', title="Model Comparison"):
    """
    Plot comparison of model performance.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and scores as values
    metric : str
        Name of the metric being compared
    title : str
        Title for the plot
    """
    models = list(results_dict.keys())
    scores = list(results_dict.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, scores, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.title(title)
    plt.xlabel('Models')
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def analyze_class_distribution(data, label_col='label', title="Class Distribution"):
    """
    Analyze and plot class distribution.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing labels
    label_col : str
        Name of label column
    title : str
        Title for the plot
        
    Returns:
    --------
    dict
        Class distribution statistics
    """
    class_counts = data[label_col].value_counts()
    class_percentages = data[label_col].value_counts(normalize=True) * 100
    
    # Create distribution statistics
    stats = {
        'counts': class_counts.to_dict(),
        'percentages': class_percentages.to_dict(),
        'total_samples': len(data),
        'num_classes': len(class_counts),
        'balance_ratio': class_counts.min() / class_counts.max()
    }
    
    # Plot distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    class_counts.plot(kind='bar', ax=ax1, alpha=0.7, edgecolor='black')
    ax1.set_title('Class Counts')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Percentage plot
    class_percentages.plot(kind='bar', ax=ax2, alpha=0.7, edgecolor='black')
    ax2.set_title('Class Percentages')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Percentage (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Class Distribution Analysis:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Number of classes: {stats['num_classes']}")
    print(f"  Balance ratio: {stats['balance_ratio']:.3f}")
    print(f"  Class counts:")
    for class_name, count in stats['counts'].items():
        percentage = stats['percentages'][class_name]
        print(f"    {class_name}: {count} ({percentage:.1f}%)")
    
    return stats


def generate_sample_dataset(n_samples=1000, noise_level=0.1, random_state=42):
    """
    Generate a sample dataset for testing purposes.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise_level : float
        Amount of noise to add to probabilities
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Generated sample dataset
    """
    np.random.seed(random_state)
    
    data = []
    emotions = ['sad', 'angry', 'happy', 'relaxed']
    
    for i in range(n_samples):
        # Generate filename
        filename = f"sample_{i:04d}.jpg"
        
        # Generate emotion probabilities with one dominant emotion
        dominant_emotion = np.random.choice(4)
        emotion_probs = np.random.dirichlet([0.5, 0.5, 0.5, 0.5])
        emotion_probs[dominant_emotion] += 0.5
        emotion_probs = normalize_probabilities(emotion_probs)
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, 4)
            emotion_probs = np.clip(emotion_probs + noise, 0, 1)
            emotion_probs = normalize_probabilities(emotion_probs)
        
        # Generate tail probabilities
        tail_probs = np.random.dirichlet([1, 1, 1])
        
        # Add noise to tail probabilities
        if noise_level > 0:
            tail_noise = np.random.normal(0, noise_level, 3)
            tail_probs = np.clip(tail_probs + tail_noise, 0, 1)
            tail_probs = normalize_probabilities(tail_probs)
        
        # Determine true label
        true_emotion = emotions[dominant_emotion]
        
        row = [filename] + emotion_probs.tolist() + tail_probs.tolist() + [true_emotion]
        data.append(row)
    
    columns = ['filename', 'sad', 'angry', 'happy', 'relaxed', 'down', 'up', 'mid', 'label']
    df = pd.DataFrame(data, columns=columns)
    
    return df


def export_results_to_excel(results_dict, output_path, sheet_names=None):
    """
    Export results to Excel file with multiple sheets.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with sheet names as keys and DataFrames as values
    output_path : str
        Path to save Excel file
    sheet_names : list, optional
        Custom sheet names (uses dict keys if None)
    """
    if sheet_names is None:
        sheet_names = list(results_dict.keys())
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, data in results_dict.items():
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name=sheet_name, index=False)
            elif isinstance(data, dict):
                # Convert dict to DataFrame
                df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
                df.to_excel(writer, sheet_name=sheet_name)
            else:
                print(f"Warning: Unsupported data type for sheet '{sheet_name}'")
    
    print(f"Results exported to: {output_path}")


def calculate_ensemble_metrics(predictions_dict, y_true, labels=None):
    """
    Calculate metrics for multiple models.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary with model names as keys and predictions as values
    y_true : array-like
        True labels
    labels : list, optional
        Label names
        
    Returns:
    --------
    dict
        Metrics for each model
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    results = {}
    
    for model_name, y_pred in predictions_dict.items():
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    return results


def print_model_summary(model, model_name="Model"):
    """
    Print a summary of model information.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    model_name : str
        Name of the model
    """
    print(f"\n=== {model_name} Summary ===")
    print(f"Type: {type(model).__name__}")
    
    # Get parameters
    params = model.get_params()
    print(f"Parameters:")
    for param, value in params.items():
        print(f"  {param}: {value}")
    
    # Get feature importance if available
    if hasattr(model, 'feature_importances_'):
        print(f"Feature importances: {model.feature_importances_}")
    
    # Get coefficients if available
    if hasattr(model, 'coef_'):
        print(f"Coefficients shape: {model.coef_.shape}")
    
    # Get number of classes
    if hasattr(model, 'classes_'):
        print(f"Classes: {model.classes_}")


def create_data_report(data, output_path=None, title="Data Report"):
    """
    Create a comprehensive data report.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to analyze
    output_path : str, optional
        Path to save report (if None, prints to console)
    title : str
        Title for the report
        
    Returns:
    --------
    dict
        Report data
    """
    report = {
        'basic_info': {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum()
        },
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_summary': data.describe().to_dict(),
        'categorical_summary': {}
    }
    
    # Add categorical summaries
    for col in data.columns:
        if data[col].dtype == 'object':
            report['categorical_summary'][col] = data[col].value_counts().to_dict()
    
    # Format report
    report_text = f"{title}\n{'='*len(title)}\n\n"
    report_text += f"Dataset Shape: {report['basic_info']['shape']}\n"
    report_text += f"Memory Usage: {report['basic_info']['memory_usage'] / 1024:.2f} KB\n\n"
    
    report_text += "Column Information:\n"
    for col, dtype in report['basic_info']['dtypes'].items():
        missing = report['missing_values'][col]
        report_text += f"  {col}: {dtype}, {missing} missing values\n"
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_path}")
    else:
        print(report_text)
    
    return report


def safe_json_parse(json_string: str, fix_common_errors: bool = True) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON string with automatic error fixing for common issues
    
    Parameters:
    -----------
    json_string : str
        JSON string to parse
    fix_common_errors : bool
        Whether to attempt fixing common JSON syntax errors
        
    Returns:
    --------
    dict or None
        Parsed JSON dict if successful, None if failed
    """
    if not json_string or not isinstance(json_string, str):
        return None
    
    # First try direct parsing
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        if not fix_common_errors:
            logging.error(f"JSON parsing failed: {e}")
            return None
        
        print(f"⚠️  JSON parsing error: {e}")
        print("🔧 Attempting to fix common JSON syntax issues...")
        
        # Fix common JSON syntax errors
        fixed_json = _fix_json_syntax_errors(json_string, e)
        
        if fixed_json != json_string:
            try:
                result = json.loads(fixed_json)
                print("✅ Successfully fixed and parsed JSON")
                return result
            except json.JSONDecodeError as e2:
                logging.error(f"JSON parsing still failed after fixes: {e2}")
                return None
        else:
            logging.error(f"Could not fix JSON syntax error: {e}")
            return None


def _fix_json_syntax_errors(json_string: str, original_error: json.JSONDecodeError) -> str:
    """
    Attempt to fix common JSON syntax errors
    
    Parameters:
    -----------
    json_string : str
        Original JSON string
    original_error : json.JSONDecodeError
        Original parsing error
        
    Returns:
    --------
    str
        Fixed JSON string (may be unchanged if no fixes applied)
    """
    fixed = json_string
    error_msg = str(original_error).lower()
    
    # Fix trailing commas - this is the most common issue
    print("🔧 Fixing trailing commas...")
    # Remove trailing commas before closing brackets/braces
    fixed = re.sub(r',(\s*[\]\}])', r'\1', fixed)
    
    # Fix: Expected ',' or ']' after array element
    if "expected" in error_msg and ("after array element" in error_msg or "after object member" in error_msg):
        print("🔧 Fixing trailing comma syntax...")
        # More aggressive trailing comma removal
        fixed = re.sub(r',(\s*[\]\}])', r'\1', fixed)
    
    # Fix: Multiple consecutive commas
    if "," in fixed:
        print("🔧 Fixing multiple consecutive commas...")
        fixed = re.sub(r',+', ',', fixed)
        # Remove commas that are followed immediately by closing brackets
        fixed = re.sub(r',(\s*[\]\}])', r'\1', fixed)
    
    # Fix: Comma before opening brace/bracket
    fixed = re.sub(r',(\s*[\[\{])', r'\1', fixed)
    
    # Handle "Expecting value" errors which often indicate trailing commas
    if "expecting value" in error_msg:
        print("🔧 Fixing 'expecting value' error (likely trailing comma)...")
        # Remove trailing commas more aggressively
        fixed = re.sub(r',(\s*[\]\}])', r'\1', fixed)
        # Remove trailing comma at end of string
        fixed = re.sub(r',\s*$', '', fixed)
    
    # Handle "Expecting property name" errors in objects
    if "expecting property name" in error_msg:
        print("🔧 Fixing 'expecting property name' error...")
        fixed = re.sub(r',(\s*\})', r'\1', fixed)
    
    # Fix: Unexpected character after closing quote
    if "unexpected" in error_msg and "character" in error_msg:
        print("🔧 Fixing unexpected characters...")
        # Try to remove common problematic characters
        fixed = re.sub(r'(["\'])\s*[,;]\s*([}\]])', r'\1\2', fixed)
    
    # Fix: Missing quotes around keys
    if "expecting" in error_msg and "property name" in error_msg:
        print("🔧 Adding quotes around object keys...")
        # Add quotes around unquoted keys
        fixed = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
    
    # Fix: Single quotes instead of double quotes
    if "invalid" in error_msg:
        print("🔧 Converting single quotes to double quotes...")
        # Convert single quotes to double quotes (be careful with nested quotes)
        fixed = re.sub(r"'([^']*)'", r'"\1"', fixed)
    
    # Fix: Control characters
    print("🔧 Removing control characters...")
    fixed = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', fixed)
    
    return fixed


def validate_head_bbox_format(bbox_data: Union[str, list]) -> Optional[list]:
    """
    Validate and normalize head bounding box format
    
    Parameters:
    -----------
    bbox_data : str or list
        Bounding box data (JSON string or list)
        
    Returns:
    --------
    list or None
        Normalized bbox [x1, y1, x2, y2] or None if invalid
    """
    if bbox_data is None:
        return None
    
    # If it's a string, try to parse as JSON
    if isinstance(bbox_data, str):
        if not bbox_data.strip() or bbox_data.strip() == '[]':
            return None
        
        parsed_bbox = safe_json_parse(bbox_data)
        if parsed_bbox is None:
            return None
        bbox_data = parsed_bbox
    
    # Validate list format
    if not isinstance(bbox_data, list):
        return None
    
    if len(bbox_data) != 4:
        return None
    
    try:
        # Convert to float and validate
        bbox = [float(x) for x in bbox_data]
        
        # Basic sanity checks
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None
        
        if any(x < 0 for x in bbox):
            return None
        
        return bbox
    except (ValueError, TypeError):
        return None


def fix_json_file(file_path: str, backup: bool = True) -> bool:
    """
    Fix JSON syntax errors in a file
    
    Parameters:
    -----------
    file_path : str
        Path to JSON file to fix
    backup : bool
        Whether to create backup before fixing
        
    Returns:
    --------
    bool
        True if file was successfully fixed, False otherwise
    """
    try:
        # Read original file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup if requested
        if backup:
            backup_path = f"{file_path}.backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"📁 Created backup: {backup_path}")
        
        # Try to parse and fix
        parsed = safe_json_parse(content, fix_common_errors=True)
        
        if parsed is None:
            return False
        
        # Write fixed content back
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully fixed JSON file: {file_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error fixing JSON file {file_path}: {e}")
        return False


def validate_dataset_json_fields(df) -> Dict[str, Any]:
    """
    Validate JSON fields in dataset and report issues
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with potential JSON fields
        
    Returns:
    --------
    dict
        Validation report
    """
    report = {
        'total_rows': len(df),
        'json_fields': [],
        'issues_found': {},
        'fixed_count': 0
    }
    
    # Check for JSON-like fields
    json_like_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_values = df[col].dropna().head(5)
            for val in sample_values:
                if isinstance(val, str) and (val.strip().startswith('[') or val.strip().startswith('{')):
                    json_like_columns.append(col)
                    break
    
    report['json_fields'] = json_like_columns
    
    # Validate each JSON field
    for col in json_like_columns:
        issues = []
        fixed = 0
        
        for idx, value in df[col].items():
            if pd.isna(value) or not isinstance(value, str):
                continue
            
            parsed = safe_json_parse(value, fix_common_errors=False)
            if parsed is None:
                # Try to fix
                fixed_parsed = safe_json_parse(value, fix_common_errors=True)
                if fixed_parsed is not None:
                    fixed += 1
                else:
                    issues.append(f"Row {idx}: Could not parse '{value[:50]}...'")
        
        report['issues_found'][col] = issues
        report['fixed_count'] += fixed
    
    return report


# ==========================================
# BBOX VALIDATION FUNCTIONS
# ==========================================

def calculate_iou(box1, box2):
    """
    🎯 Tính Intersection over Union (IoU) giữa 2 bounding boxes
    
    Parameters:
    -----------
    box1, box2 : list or array-like
        Bounding box [x1, y1, x2, y2] format (corner coordinates)
    
    Returns:
    --------
    float
        IoU score (0.0 to 1.0)
    """
    # Ensure boxes are in [x1, y1, x2, y2] format
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def get_ground_truth_bbox(image_path):
    """
    🏷️ Đọc ground truth bounding box từ annotation file (.txt)
    
    Parameters:
    -----------
    image_path : str or Path
        Đường dẫn đến file ảnh
    
    Returns:
    --------
    list or None
        Ground truth bbox [x1, y1, x2, y2] trong pixel coordinates, 
        hoặc None nếu không tìm thấy
    """
    from pathlib import Path
    from PIL import Image
    
    image_path = Path(image_path)
    
    # Tìm annotation file tương ứng (.txt)
    dataset_dir = image_path.parent.parent  # từ /test/images lên /test hoặc từ /test lên /
    possible_annotation_dirs = [
        dataset_dir / 'labels',
        image_path.parent.parent / 'labels', 
        image_path.parent / 'labels',
        dataset_dir / 'test' / 'labels',
        dataset_dir
    ]
    
    annotation_file = None
    for ann_dir in possible_annotation_dirs:
        potential_file = ann_dir / f"{image_path.stem}.txt"
        if potential_file.exists():
            annotation_file = potential_file
            break
    
    if not annotation_file or not annotation_file.exists():
        return None
    
    try:
        # Đọc image dimensions để convert từ normalized coords
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    # YOLO format: class_id x_center y_center width height (normalized 0-1)
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert từ normalized coordinates sang pixel coordinates
                        x_center_px = x_center * img_width
                        y_center_px = y_center * img_height
                        width_px = width * img_width
                        height_px = height * img_height
                        
                        # Convert từ center format sang corner format [x1, y1, x2, y2]
                        x1 = x_center_px - width_px / 2
                        y1 = y_center_px - height_px / 2
                        x2 = x_center_px + width_px / 2
                        y2 = y_center_px + height_px / 2
                        
                        # Giả định rằng annotation đầu tiên là head bounding box
                        return [x1, y1, x2, y2]
                        
    except Exception as e:
        logging.warning(f"Error reading ground truth bbox from {annotation_file}: {e}")
        return None
    
    return None


def validate_head_detection_with_ground_truth(predicted_bbox, image_path, iou_threshold=0.3):
    """
    ✅ Validate predicted head bounding box với ground truth
    
    Parameters:
    -----------
    predicted_bbox : list
        Predicted bounding box [x1, y1, x2, y2]
    image_path : str or Path
        Đường dẫn đến file ảnh
    iou_threshold : float
        Ngưỡng IoU tối thiểu để chấp nhận (default: 0.3)
    
    Returns:
    --------
    dict
        {'valid': bool, 'iou': float, 'ground_truth_bbox': list or None, 'reason': str}
    """
    # Lấy ground truth bbox
    gt_bbox = get_ground_truth_bbox(image_path)
    
    if gt_bbox is None:
        # Không có ground truth, không thể validate
        return {
            'valid': False, 
            'iou': 0.0, 
            'ground_truth_bbox': None,
            'reason': 'No ground truth available'
        }
    
    # Tính IoU
    iou = calculate_iou(predicted_bbox, gt_bbox)
    
    # Kiểm tra ngưỡng
    is_valid = iou >= iou_threshold
    
    return {
        'valid': is_valid,
        'iou': iou,
        'ground_truth_bbox': gt_bbox,
        'reason': f'IoU {iou:.3f} {"≥" if is_valid else "<"} threshold {iou_threshold}'
    }


def validate_bbox_format(bbox, expected_format='xyxy'):
    """
    🔍 Validate bounding box format and convert if needed
    
    Parameters:
    -----------
    bbox : list or str
        Bounding box to validate
    expected_format : str
        Expected format ('xyxy' for [x1,y1,x2,y2] or 'xywh' for [x,y,w,h])
    
    Returns:
    --------
    dict
        {'valid': bool, 'bbox': list or None, 'format': str, 'reason': str}
    """
    if isinstance(bbox, str):
        try:
            bbox = eval(bbox)  # Convert string representation to list
        except:
            return {
                'valid': False,
                'bbox': None,
                'format': 'unknown',
                'reason': 'Could not parse bbox string'
            }
    
    if not isinstance(bbox, (list, tuple, np.ndarray)):
        return {
            'valid': False,
            'bbox': None,
            'format': 'unknown',
            'reason': 'Bbox must be list, tuple, or array'
        }
    
    bbox = list(bbox)
    
    if len(bbox) != 4:
        return {
            'valid': False,
            'bbox': None,
            'format': 'unknown',
            'reason': f'Bbox must have 4 values, got {len(bbox)}'
        }
    
    # Check if all values are numeric
    try:
        bbox = [float(x) for x in bbox]
    except (ValueError, TypeError):
        return {
            'valid': False,
            'bbox': None,
            'format': 'unknown',
            'reason': 'Bbox values must be numeric'
        }
    
    # Detect format based on values
    x1, y1, x2, y2 = bbox
    
    if expected_format == 'xyxy':
        # Check if x2 > x1 and y2 > y1 (valid corner format)
        if x2 <= x1 or y2 <= y1:
            return {
                'valid': False,
                'bbox': bbox,
                'format': 'xyxy',
                'reason': 'Invalid xyxy format: x2 <= x1 or y2 <= y1'
            }
        
        return {
            'valid': True,
            'bbox': bbox,
            'format': 'xyxy',
            'reason': 'Valid xyxy format'
        }
    
    elif expected_format == 'xywh':
        # Check if width and height are positive
        if x2 <= 0 or y2 <= 0:  # x2=width, y2=height in xywh format
            return {
                'valid': False,
                'bbox': bbox,
                'format': 'xywh',
                'reason': 'Invalid xywh format: width <= 0 or height <= 0'
            }
        
        return {
            'valid': True,
            'bbox': bbox,
            'format': 'xywh',
            'reason': 'Valid xywh format'
        }
    
    else:
        return {
            'valid': False,
            'bbox': None,
            'format': 'unknown',
            'reason': f'Unsupported expected format: {expected_format}'
        }