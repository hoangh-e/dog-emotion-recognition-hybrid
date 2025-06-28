"""
Dog Emotion Recognition Hybrid ML Package

Package chuyên nghiệp cho nhận diện cảm xúc chó sử dụng Machine Learning
với pipeline kết hợp ResNet emotion detection và YOLO tail detection.

Triển khai đầy đủ 7 kỹ thuật ensemble learning theo tài liệu nghiên cứu:
1. Bagging - Bootstrap Aggregating giảm phương sai
2. Boosting - XGBoost, AdaBoost, GradientBoosting, LightGBM
3. Stacking - Meta-model kết hợp heterogeneous base models
4. Voting - Soft/Hard voting ensemble
5. Negative Correlation - Giảm tương quan giữa learners
6. Heterogeneous - Kết hợp vision + classical models
7. Multi-level Deep - Implicit ensemble qua feature fusion
"""

from .emotion_ml import (
    EmotionMLClassifier,
    NegativeCorrelationEnsemble,
    HeterogeneousEnsemble,
    MultiLevelDeepEnsemble
)
from .ensemble_meta import EnsembleMetaLearner
from .ensemble_config import (
    ENSEMBLE_CONFIGS,
    BASE_ESTIMATOR_CONFIGS,
    FEATURE_MAPPING,
    ENSEMBLE_PERFORMANCE_INFO,
    get_ensemble_config,
    get_base_estimator_config,
    get_feature_indices,
    print_ensemble_summary
)
from .data_pipeline import (
    RoboflowDataProcessor,
    DataNormalizer,
    create_sample_roboflow_structure,
    demo_data_pipeline
)

__version__ = "2.1.0"
__author__ = "Dog Emotion Recognition Team"
__description__ = "Comprehensive ML package with 7 ensemble learning techniques for dog emotion recognition and data pipeline for Roboflow integration"

__all__ = [
    # Main classes
    "EmotionMLClassifier",
    "EnsembleMetaLearner",
    
    # Advanced ensemble classes
    "NegativeCorrelationEnsemble",
    "HeterogeneousEnsemble", 
    "MultiLevelDeepEnsemble",
    
    # Data pipeline classes
    "RoboflowDataProcessor",
    "DataNormalizer",
    
    # Configuration
    "ENSEMBLE_CONFIGS",
    "BASE_ESTIMATOR_CONFIGS",
    "FEATURE_MAPPING",
    "ENSEMBLE_PERFORMANCE_INFO",
    
    # Utility functions
    "get_ensemble_config",
    "get_base_estimator_config",
    "get_feature_indices",
    "print_ensemble_summary",
    "create_sample_roboflow_structure",
    "demo_data_pipeline"
] 