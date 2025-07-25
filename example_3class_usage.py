#!/usr/bin/env python3
"""
Example script demonstrating how to use the updated 3-class dog emotion classification modules.

This script shows:
1. How to use the dataset conversion functions
2. How to create models with 3 classes
3. How to work with the new emotion classes

Note: This example doesn't require actual model files or PyTorch installation.
"""

import numpy as np
import pandas as pd

def example_dataset_conversion():
    """Example of converting 4-class dataset to 3-class."""
    print("📊 Dataset Conversion Examples")
    print("=" * 50)
    
    # Example 1: Convert label array
    print("\n1. Converting label array:")
    
    # Simulate 4-class labels (0=angry, 1=happy, 2=relaxed, 3=sad)
    original_labels = np.array([0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 0, 3, 1, 2])
    print(f"   Original labels (4-class): {original_labels}")
    print(f"   Distribution: angry={np.sum(original_labels==0)}, happy={np.sum(original_labels==1)}, relaxed={np.sum(original_labels==2)}, sad={np.sum(original_labels==3)}")
    
    # Import and use conversion function
    try:
        from dog_emotion_classification.utils import convert_4class_to_3class_labels
        
        converted_labels = convert_4class_to_3class_labels(original_labels)
        print(f"   Converted labels (3-class): {converted_labels}")
        print(f"   Distribution: angry={np.sum(converted_labels==0)}, happy={np.sum(converted_labels==1)}, relaxed={np.sum(converted_labels==2)}")
        
    except ImportError as e:
        print(f"   ❌ Could not import conversion function: {e}")
    
    # Example 2: Convert DataFrame
    print("\n2. Converting DataFrame:")
    
    # Create sample DataFrame
    df_data = {
        'filename': [f'dog_{i:03d}.jpg' for i in range(12)],
        'label': ['angry', 'happy', 'relaxed', 'sad'] * 3,
        'confidence': np.random.uniform(0.7, 0.95, 12)
    }
    original_df = pd.DataFrame(df_data)
    print(f"   Original DataFrame shape: {original_df.shape}")
    print(f"   Original classes: {original_df['label'].value_counts().to_dict()}")
    
    try:
        from dog_emotion_classification.utils import convert_dataframe_4class_to_3class
        
        converted_df = convert_dataframe_4class_to_3class(original_df, 'label')
        print(f"   Converted DataFrame shape: {converted_df.shape}")
        print(f"   Converted classes: {converted_df['label'].value_counts().to_dict()}")
        
    except ImportError as e:
        print(f"   ❌ Could not import DataFrame conversion function: {e}")

def example_emotion_classes():
    """Example of working with 3-class emotion constants."""
    print("\n\n😊 Emotion Classes Examples")
    print("=" * 50)
    
    try:
        # Import emotion classes
        from dog_emotion_classification import EMOTION_CLASSES
        from dog_emotion_classification.utils import (
            get_3class_emotion_classes, 
            get_3class_emotion_mapping,
            EMOTION_CLASSES_3CLASS
        )
        
        print(f"\n1. Main package emotion classes: {EMOTION_CLASSES}")
        print(f"2. Utils 3-class function: {get_3class_emotion_classes()}")
        print(f"3. Utils 3-class constant: {EMOTION_CLASSES_3CLASS}")
        print(f"4. Class mapping: {get_3class_emotion_mapping()}")
        
        # Example predictions
        print(f"\n5. Example model predictions:")
        sample_predictions = [0, 1, 2, 1, 0, 2]
        for pred in sample_predictions:
            emotion = EMOTION_CLASSES[pred]
            print(f"   Prediction {pred} = {emotion}")
            
    except ImportError as e:
        print(f"   ❌ Could not import emotion classes: {e}")

def example_model_usage():
    """Example of using updated model functions (without actually loading models)."""
    print("\n\n🏗️  Model Usage Examples")
    print("=" * 50)
    
    print("\n1. Model creation examples (theoretical):")
    
    models_to_show = ['resnet', 'alexnet', 'vit', 'efficientnet']
    
    for model_name in models_to_show:
        try:
            module = __import__(f'dog_emotion_classification.{model_name}', fromlist=[model_name])
            
            # Look for create functions
            create_functions = [func for func in dir(module) if func.startswith('create_') and func.endswith('_model')]
            
            if create_functions:
                print(f"   ✅ {model_name}: {create_functions[0]}(num_classes=3) - default 3 classes")
            else:
                print(f"   ⚠️  {model_name}: No create function found")
                
        except ImportError:
            print(f"   ❌ {model_name}: Module not available")
    
    print("\n2. Prediction function signatures:")
    for model_name in models_to_show:
        try:
            module = __import__(f'dog_emotion_classification.{model_name}', fromlist=[model_name])
            
            # Look for predict functions  
            predict_functions = [func for func in dir(module) if func.startswith('predict_emotion_')]
            
            if predict_functions:
                print(f"   ✅ {model_name}: {predict_functions[0]}(..., emotion_classes=['angry', 'happy', 'relaxed'])")
            else:
                print(f"   ⚠️  {model_name}: No predict function found")
                
        except ImportError:
            print(f"   ❌ {model_name}: Module not available")

def example_pipeline_configuration():
    """Example of how to update pipeline configuration for 3 classes."""
    print("\n\n🔧 Pipeline Configuration Example")
    print("=" * 50)
    
    # Example pipeline configuration
    pipeline_config = {
        'EMOTION_CLASSES': ['angry', 'happy', 'relaxed'],
        'NUM_CLASSES': 3,
        'ALGORITHMS': {
            'AlexNet': {
                'params': {'input_size': 224, 'num_classes': 3},
                'module': 'alexnet'
            },
            'DenseNet121': {
                'params': {'architecture': 'densenet121', 'input_size': 224, 'num_classes': 3},
                'module': 'densenet'
            },
            'ResNet50': {
                'params': {'architecture': 'resnet50', 'input_size': 224, 'num_classes': 3},
                'module': 'resnet'
            },
            'ViT-B/16': {
                'params': {'architecture': 'vit_b_16', 'input_size': 224, 'num_classes': 3},
                'module': 'vit'
            },
            'EfficientNet-B2': {
                'params': {'input_size': 260, 'num_classes': 3},
                'module': 'efficientnet'
            }
        }
    }
    
    print("\n1. Updated pipeline configuration:")
    print(f"   Emotion classes: {pipeline_config['EMOTION_CLASSES']}")
    print(f"   Number of classes: {pipeline_config['NUM_CLASSES']}")
    
    print("\n2. Algorithm configurations:")
    for algo_name, config in pipeline_config['ALGORITHMS'].items():
        num_classes = config['params']['num_classes']
        print(f"   {algo_name}: num_classes={num_classes} ✅")
    
    print("\n3. Expected performance improvements:")
    print("   • Easier 3-class problem vs 4-class")
    print("   • Better class separation")
    print("   • Reduced dataset size (~25% reduction)")
    print("   • Potentially higher accuracy")

def main():
    """Main demonstration function."""
    print("🎯 Dog Emotion Classification - 3-Class Configuration Examples")
    print("=" * 70)
    
    # Run examples
    example_dataset_conversion()
    example_emotion_classes()
    example_model_usage()
    example_pipeline_configuration()
    
    print("\n\n" + "=" * 70)
    print("✅ Summary of 3-Class Updates:")
    print("=" * 70)
    
    print("\n📦 Module Updates:")
    print("   • Updated 28 algorithm modules")
    print("   • Changed num_classes=4 → num_classes=3")
    print("   • Updated emotion_classes parameters")
    print("   • Updated docstrings and descriptions")
    
    print("\n🛠️  New Features:")
    print("   • Added utils module with conversion functions")
    print("   • convert_4class_to_3class_labels()")
    print("   • convert_4class_to_3class_dataset()")
    print("   • convert_dataframe_4class_to_3class()")
    print("   • get_3class_emotion_classes()")
    
    print("\n😊 Emotion Classes:")
    print("   • OLD: ['angry', 'happy', 'relaxed', 'sad']")
    print("   • NEW: ['angry', 'happy', 'relaxed']")
    print("   • Mapping: 0=angry, 1=happy, 2=relaxed")
    
    print("\n🔧 Next Steps for Implementation:")
    print("   1. Update your dataset by filtering out 'sad' samples")
    print("   2. Update pipeline scripts to use num_classes=3")
    print("   3. Retrain models for optimal 3-class performance")
    print("   4. Update inference scripts to use 3 classes")
    print("   5. Test with sample images")
    
    print("\n💡 Usage Tips:")
    print("   • Use utils.convert_4class_to_3class_dataset() for dataset conversion")
    print("   • Models will create with num_classes=3 by default")
    print("   • Old 4-class models need retraining for best performance")
    print("   • Expect better accuracy with 3-class problem")

if __name__ == "__main__":
    main() 