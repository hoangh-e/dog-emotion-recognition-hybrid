#!/usr/bin/env python3
"""
Example usage of the Dog Emotion Recognition Hybrid ML Package

This script demonstrates the complete workflow from data loading to meta-learning.
"""

import numpy as np
import pandas as pd
from dog_emotion_ml import EmotionMLClassifier, EnsembleMetaLearner

def create_sample_data():
    """Create sample data for demonstration purposes."""
    np.random.seed(42)
    
    # Create sample training data
    n_samples = 1000
    data = []
    
    emotions = ['sad', 'angry', 'happy', 'relaxed']
    
    for i in range(n_samples):
        # Generate filename
        filename = f"dog_{i:04d}.jpg"
        
        # Generate emotion probabilities (one dominant emotion)
        emotion_probs = np.random.dirichlet([1, 1, 1, 1])
        
        # Generate tail probabilities
        tail_probs = np.random.dirichlet([1, 1, 1])
        
        # Determine true label based on highest emotion probability
        true_emotion = emotions[np.argmax(emotion_probs)]
        
        row = [filename] + emotion_probs.tolist() + tail_probs.tolist() + [true_emotion]
        data.append(row)
    
    columns = ['filename', 'sad', 'angry', 'happy', 'relaxed', 'down', 'up', 'mid', 'label']
    df = pd.DataFrame(data, columns=columns)
    
    return df

def main():
    """Main demonstration function."""
    print("=== Dog Emotion Recognition Hybrid ML Package Demo ===\n")
    
    # Create sample datasets
    print("Creating sample datasets...")
    full_data = create_sample_data()
    train_data = full_data[:600]
    test_data = full_data[600:800]
    test_for_train_data = full_data[800:]
    
    # Save to CSV files
    train_data.to_csv('sample_train.csv', index=False)
    test_data.to_csv('sample_test.csv', index=False)
    test_for_train_data.to_csv('sample_test_for_train.csv', index=False)
    
    print(f"Created datasets:")
    print(f"  Training: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"  Test-for-train: {len(test_for_train_data)} samples")
    
    # Stage II: Train ML models
    print("\n=== Stage II: Training ML Models ===")
    classifier = EmotionMLClassifier(random_state=42)
    
    # Load datasets
    print("Loading datasets...")
    classifier.load_train_dataset('sample_train.csv')
    classifier.load_test_dataset('sample_test.csv')
    classifier.load_test_for_train_dataset('sample_test_for_train.csv')
    
    # Check data quality
    print("\nChecking data quality...")
    anomalies = classifier.display_anomalies_summary('train')
    
    # Train selected models (subset for demo)
    print("\nTraining selected ML models...")
    classifier.train_logistic_regression()
    classifier.train_svm(kernel='rbf')
    classifier.train_decision_tree()
    classifier.train_random_forest(n_estimators=50)
    classifier.train_xgboost(n_estimators=50)
    classifier.train_naive_bayes()
    
    # List trained models
    print("\nTrained models:")
    classifier.list_trained_models()
    
    # Evaluate some models
    print("\nModel evaluation:")
    for model_name in ['XGBoost', 'RandomForest', 'LogisticRegression_multinomial']:
        try:
            accuracy = classifier.evaluate_model(model_name)
            print(f"  {model_name}: {accuracy:.4f}")
        except Exception as e:
            print(f"  {model_name}: Error - {e}")
    
    # Generate meta-training data
    print("\nGenerating meta-training data...")
    meta_data = classifier.generate_meta_training_data()
    classifier.save_meta_training_data('sample_meta_train.csv')
    print(f"Meta-training data shape: {meta_data.shape}")
    
    # Stage III: Train meta-learner
    print("\n=== Stage III: Training Meta-Learner ===")
    meta_learner = EnsembleMetaLearner(random_state=42)
    
    # Load meta-training data
    print("Loading meta-training data...")
    meta_learner.load_meta_training_data('sample_meta_train.csv')
    
    # Analyze algorithm performance
    print("\nAnalyzing algorithm performance...")
    performance = meta_learner.analyze_algorithm_performance()
    distribution = meta_learner.analyze_algorithm_distribution()
    
    # Train meta-learner
    print("\nTraining meta-learner...")
    meta_learner.train_meta_learner(algorithm='DecisionTree', max_depth=5)
    
    # Get feature importance
    print("\nFeature importance:")
    importance = meta_learner.get_feature_importance()
    
    # Extract decision rules
    print("\nDecision rules:")
    rules = meta_learner.get_algorithm_selection_rules()
    
    # Save meta-model
    print("\nSaving meta-model...")
    meta_learner.save_meta_model('sample_meta_learner.joblib')
    
    # Make predictions
    print("\n=== Making Predictions ===")
    
    # Single prediction
    print("Single prediction example:")
    sample_emotion = [0.1, 0.05, 0.8, 0.05]  # sad, angry, happy, relaxed
    sample_tail = [0.1, 0.8, 0.1]  # down, up, mid
    
    print(f"Input features:")
    print(f"  Emotion: {sample_emotion}")
    print(f"  Tail: {sample_tail}")
    
    best_algo, confidence = meta_learner.predict_best_algorithm(sample_emotion, sample_tail)
    print(f"Recommended algorithm: {best_algo}")
    
    if confidence is not None:
        print("Algorithm confidence scores:")
        for i, algo in enumerate(meta_learner.algorithm_encoder.classes_):
            print(f"  {algo}: {confidence[i]:.4f}")
    
    # Batch prediction
    print("\nBatch prediction example:")
    batch_features = np.random.rand(5, 7)
    batch_algos, batch_confidence = meta_learner.predict_best_algorithms_batch(batch_features)
    
    print("Batch predictions:")
    for i, algo in enumerate(batch_algos):
        print(f"  Sample {i+1}: {algo}")
    
    # Demonstrate prediction
    print("\nDemonstration with random features:")
    meta_learner.demonstrate_prediction()
    
    print("\n=== Demo Complete ===")
    print("Generated files:")
    print("  - sample_train.csv")
    print("  - sample_test.csv")
    print("  - sample_test_for_train.csv")
    print("  - sample_meta_train.csv")
    print("  - sample_meta_learner.joblib")
    
    # Cleanup
    import os
    cleanup_files = [
        'sample_train.csv', 'sample_test.csv', 'sample_test_for_train.csv',
        'sample_meta_train.csv', 'sample_meta_learner.joblib'
    ]
    
    print("\nCleaning up temporary files...")
    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"  Removed {file}")

if __name__ == "__main__":
    main() 