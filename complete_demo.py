#!/usr/bin/env python3
"""
Complete Demo for Dog Emotion Recognition Hybrid ML Package v2.1.0

Demo hoàn chỉnh tất cả tính năng của package:
1. Roboflow Data Pipeline
2. Advanced Data Normalization 
3. 19+ ML Algorithms với 7 Ensemble Techniques
4. Meta-Learning cho Algorithm Selection
5. Evaluation và Visualization

Chạy script này để kiểm tra toàn bộ package hoạt động đúng.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test import tất cả modules"""
    print("🔧 Testing imports...")
    
    try:
        from dog_emotion_ml import (
            EmotionMLClassifier, 
            EnsembleMetaLearner,
            RoboflowDataProcessor,
            DataNormalizer,
            create_sample_roboflow_structure,
            demo_data_pipeline
        )
        from dog_emotion_ml.utils import generate_sample_dataset, analyze_class_distribution
        from dog_emotion_ml.ensemble_config import print_ensemble_summary
        
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_pipeline():
    """Test Roboflow data pipeline"""
    print("\n📊 Testing Roboflow Data Pipeline...")
    
    try:
        from dog_emotion_ml import RoboflowDataProcessor, DataNormalizer, create_sample_roboflow_structure
        
        # Test DataNormalizer
        normalizer = DataNormalizer()
        
        # Sample data
        emotion_data = np.random.dirichlet([1, 1, 1, 1], 50)
        tail_data = np.random.dirichlet([1, 1, 1], 50)
        
        emotion_norm, tail_norm = normalizer.fit_transform(emotion_data, tail_data)
        
        print(f"   Emotion normalization: {emotion_norm.shape}")
        print(f"   Emotion mean after Z-score: {emotion_norm.mean(axis=0)}")
        print(f"   Emotion std after Z-score: {emotion_norm.std(axis=0)}")
        print(f"   Tail pass-through: {tail_norm.shape}")
        
        # Test sample Roboflow structure
        sample_path = Path("./test_roboflow")
        if sample_path.exists():
            import shutil
            shutil.rmtree(sample_path)
        
        create_sample_roboflow_structure(sample_path)
        print(f"   Created sample Roboflow structure: {sample_path}")
        
        # Clean up
        if sample_path.exists():
            import shutil
            shutil.rmtree(sample_path)
        
        print("✅ Data pipeline test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        return False

def test_ml_pipeline():
    """Test ML pipeline với advanced normalization"""
    print("\n🤖 Testing ML Pipeline...")
    
    try:
        from dog_emotion_ml import EmotionMLClassifier
        from dog_emotion_ml.utils import generate_sample_dataset
        
        # Generate sample data
        train_data = generate_sample_dataset(n_samples=300, noise_level=0.1, random_state=42)
        test_data = generate_sample_dataset(n_samples=100, noise_level=0.1, random_state=123)
        test_for_train_data = generate_sample_dataset(n_samples=150, noise_level=0.1, random_state=456)
        
        # Save to temporary files
        train_file = "temp_train.csv"
        test_file = "temp_test.csv"
        test_for_train_file = "temp_test_for_train.csv"
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        test_for_train_data.to_csv(test_for_train_file, index=False)
        
        # Initialize classifier
        classifier = EmotionMLClassifier(random_state=42)
        
        # Load data
        classifier.load_train_dataset(train_file)
        classifier.load_test_dataset(test_file)
        classifier.load_test_for_train_dataset(test_for_train_file)
        
        # Prepare data with advanced normalization
        classifier.prepare_training_data(use_advanced_normalization=True)
        classifier.prepare_test_data(use_advanced_normalization=True)
        classifier.prepare_test_for_train_data(use_advanced_normalization=True)
        
        # Train key models (subset for speed)
        print("   Training key models...")
        classifier.train_logistic_regression()
        classifier.train_random_forest()
        classifier.train_xgboost()
        classifier.train_svm()
        classifier.train_voting_classifier()
        classifier.train_stacking_classifier()
        
        # Evaluate models
        print("   Evaluating models...")
        results = {}
        for model_name in classifier.trained_models:
            accuracy = classifier.evaluate_model(model_name)
            results[model_name] = accuracy
            print(f"      {model_name}: {accuracy:.4f}")
        
        # Generate meta-training data
        print("   Generating meta-training data...")
        meta_data = classifier.generate_meta_training_data()
        meta_file = "temp_meta_train.csv"
        classifier.save_meta_training_data(meta_file)
        
        print(f"   Meta-training data shape: {meta_data.shape}")
        
        # Clean up
        for file in [train_file, test_file, test_for_train_file, meta_file]:
            Path(file).unlink(missing_ok=True)
        
        print("✅ ML pipeline test successful!")
        return True, results
        
    except Exception as e:
        print(f"❌ ML pipeline test failed: {e}")
        return False, {}

def test_meta_learning():
    """Test meta-learning pipeline"""
    print("\n🧠 Testing Meta-Learning...")
    
    try:
        from dog_emotion_ml import EnsembleMetaLearner
        from dog_emotion_ml.utils import generate_sample_dataset
        
        # Create sample meta-training data
        np.random.seed(42)
        n_samples = 200
        
        # Create base features
        emotion_data = np.random.dirichlet([1, 1, 1, 1], n_samples)
        tail_data = np.random.dirichlet([1, 1, 1], n_samples)
        labels = np.random.choice(['sad', 'angry', 'happy', 'relaxed'], n_samples)
        
        # Create mock model predictions
        algorithms = ['LogisticRegression', 'RandomForest', 'XGBoost', 'SVM']
        
        meta_data = pd.DataFrame({
            'filename': [f'img_{i}.jpg' for i in range(n_samples)],
            'sad': emotion_data[:, 0],
            'angry': emotion_data[:, 1],
            'happy': emotion_data[:, 2],
            'relaxed': emotion_data[:, 3],
            'down': tail_data[:, 0],
            'up': tail_data[:, 1],
            'mid': tail_data[:, 2],
            'label': labels
        })
        
        # Add mock predictions
        for algo in algorithms:
            for emotion in ['sad', 'angry', 'happy', 'relaxed']:
                meta_data[f'{algo}_{emotion}'] = np.random.dirichlet([1, 1, 1, 1], n_samples)[:, ['sad', 'angry', 'happy', 'relaxed'].index(emotion)]
        
        # Save meta data
        meta_file = "temp_meta_data.csv"
        meta_data.to_csv(meta_file, index=False)
        
        # Initialize meta-learner
        meta_learner = EnsembleMetaLearner(random_state=42)
        
        # Load meta-training data
        meta_learner.load_meta_training_data(meta_file)
        
        # Analyze algorithm performance
        performance = meta_learner.analyze_algorithm_performance()
        print(f"   Algorithm performance analysis: {len(performance)} metrics")
        
        # Train meta-learner
        meta_learner.train_meta_learner(algorithm='DecisionTree', max_depth=5)
        
        # Test predictions
        test_emotion = [0.1, 0.2, 0.6, 0.1]  # happy dominant
        test_tail = [0.1, 0.8, 0.1]  # up dominant
        
        best_algo, confidence = meta_learner.predict_best_algorithm(test_emotion, test_tail)
        print(f"   Best algorithm prediction: {best_algo}")
        print(f"   Confidence shape: {confidence.shape if confidence is not None else 'None'}")
        
        # Test batch prediction
        batch_features = np.array([
            [0.1, 0.2, 0.6, 0.1, 0.1, 0.8, 0.1],  # happy + up
            [0.7, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1],  # sad + down
        ])
        
        batch_algos, batch_conf = meta_learner.predict_best_algorithms_batch(batch_features)
        print(f"   Batch predictions: {batch_algos}")
        
        # Get feature importance
        importance = meta_learner.get_feature_importance()
        print(f"   Feature importance available: {importance is not None}")
        
        # Clean up
        Path(meta_file).unlink(missing_ok=True)
        
        print("✅ Meta-learning test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Meta-learning test failed: {e}")
        return False

def test_ensemble_configs():
    """Test ensemble configurations"""
    print("\n⚙️  Testing Ensemble Configurations...")
    
    try:
        from dog_emotion_ml.ensemble_config import (
            get_ensemble_config, 
            get_base_estimator_config,
            get_feature_indices,
            print_ensemble_summary
        )
        
        # Test ensemble configs
        bagging_config = get_ensemble_config("bagging")
        print(f"   Bagging config loaded: {len(bagging_config)} parameters")
        
        xgboost_config = get_ensemble_config("boosting", "xgboost")
        print(f"   XGBoost config loaded: {len(xgboost_config)} parameters")
        
        # Test base estimator configs
        rf_config = get_base_estimator_config("RandomForest")
        print(f"   RandomForest config loaded: {len(rf_config)} parameters")
        
        # Test feature indices
        emotion_indices = get_feature_indices("emotion")
        tail_indices = get_feature_indices("tail")
        all_indices = get_feature_indices("all")
        
        print(f"   Emotion indices: {emotion_indices}")
        print(f"   Tail indices: {tail_indices}")
        print(f"   All indices: {all_indices}")
        
        print("✅ Ensemble configurations test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Ensemble configurations test failed: {e}")
        return False

def test_utilities():
    """Test utility functions"""
    print("\n🛠️  Testing Utilities...")
    
    try:
        from dog_emotion_ml.utils import (
            generate_sample_dataset,
            analyze_class_distribution,
            validate_emotion_features,
            normalize_probabilities
        )
        
        # Test sample dataset generation
        sample_data = generate_sample_dataset(n_samples=100, random_state=42)
        print(f"   Sample dataset generated: {sample_data.shape}")
        
        # Test class distribution analysis
        distribution = analyze_class_distribution(sample_data, label_col='label')
        print(f"   Class distribution analyzed: {len(distribution)} classes")
        
        # Test feature validation
        test_features = np.array([[0.2, 0.3, 0.4, 0.1], [0.1, 0.1, 0.7, 0.1]])
        is_valid = validate_emotion_features(test_features)
        print(f"   Feature validation: {is_valid}")
        
        # Test probability normalization
        unnormalized = np.array([[0.3, 0.4, 0.6, 0.2], [0.1, 0.2, 0.8, 0.3]])
        normalized = normalize_probabilities(unnormalized)
        print(f"   Probability normalization: {normalized.shape}")
        print(f"   Row sums after normalization: {normalized.sum(axis=1)}")
        
        print("✅ Utilities test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        return False

def run_complete_demo():
    """Run complete demo of all package features"""
    print("🚀 DOG EMOTION RECOGNITION HYBRID ML PACKAGE v2.1.0")
    print("=" * 70)
    print("Complete Demo - Testing All Features")
    print("=" * 70)
    
    results = {
        'imports': False,
        'data_pipeline': False,
        'ml_pipeline': False,
        'meta_learning': False,
        'ensemble_configs': False,
        'utilities': False
    }
    
    ml_results = {}
    
    # Test each component
    results['imports'] = test_imports()
    results['data_pipeline'] = test_data_pipeline()
    results['ml_pipeline'], ml_results = test_ml_pipeline()
    results['meta_learning'] = test_meta_learning()
    results['ensemble_configs'] = test_ensemble_configs()
    results['utilities'] = test_utilities()
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 DEMO RESULTS SUMMARY")
    print("=" * 70)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title():20s}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if ml_results:
        print(f"\n📊 ML Model Performance Summary:")
        best_model = max(ml_results, key=ml_results.get)
        best_score = ml_results[best_model]
        print(f"   Best Model: {best_model} ({best_score:.4f})")
        print(f"   Average Accuracy: {np.mean(list(ml_results.values())):.4f}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED! Package is working correctly.")
        print(f"Package dog_emotion_ml v2.1.0 is ready for production use!")
    else:
        print(f"\n⚠️  Some tests failed. Please check the error messages above.")
    
    print("=" * 70)
    
    return results, ml_results

if __name__ == "__main__":
    # Run complete demo
    test_results, model_results = run_complete_demo()
    
    # Optional: Show ensemble summary
    if test_results.get('imports', False):
        print(f"\n📚 Package Features Summary:")
        print("   - RoboflowDataProcessor: Xử lý dữ liệu từ Roboflow")
        print("   - DataNormalizer: Chuẩn hóa nâng cao (Z-score emotion, pass-through tail)")
        print("   - EmotionMLClassifier: 19+ thuật toán ML với 7 ensemble techniques")
        print("   - EnsembleMetaLearner: Meta-learning cho algorithm selection")
        print("   - Advanced normalization trong toàn bộ pipeline")
        print("   - Tích hợp YOLO + ResNet models")
        print("   - Support cho Roboflow dataset format") 