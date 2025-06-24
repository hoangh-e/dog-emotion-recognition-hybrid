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