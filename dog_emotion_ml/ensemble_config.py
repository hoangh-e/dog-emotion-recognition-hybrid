"""
Cấu hình cho 7 kỹ thuật Ensemble Learning theo tài liệu nghiên cứu.

Module này định nghĩa các tham số và cấu hình mặc định cho từng kỹ thuật ensemble,
giúp dễ dàng điều chỉnh và tối ưu hóa hiệu suất của từng phương pháp.
"""

# Cấu hình cho các kỹ thuật Ensemble Learning
ENSEMBLE_CONFIGS = {
    # 1. Bagging - Bootstrap Aggregating
    "bagging": {
        "n_estimators": 10,
        "max_samples": 1.0,
        "max_features": 1.0,
        "bootstrap": True,
        "bootstrap_features": False,
        "description": "Bootstrap Aggregating - Giảm phương sai thông qua bootstrap sampling"
    },
    
    # 2. Boosting - Sequential ensemble
    "boosting": {
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "description": "XGBoost - Extreme Gradient Boosting với regularization"
        },
        "adaboost": {
            "n_estimators": 50,
            "learning_rate": 1.0,
            "algorithm": "SAMME.R",
            "description": "AdaBoost - Adaptive Boosting tập trung vào mẫu khó"
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.8,
            "description": "Gradient Boosting - Sequential learning với gradient descent"
        },
        "lightgbm": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "description": "LightGBM - Light Gradient Boosting với leaf-wise growth"
        }
    },
    
    # 3. Stacking - Meta-model combination
    "stacking": {
        "cv": 5,
        "stack_method": "predict_proba",
        "passthrough": False,
        "final_estimators": ["LogisticRegression", "XGBoost", "RandomForest"],
        "description": "Stacking - Meta-learner kết hợp predictions từ base models"
    },
    
    # 4. Voting - Simple ensemble voting
    "voting": {
        "voting_methods": ["soft", "hard"],
        "weights": None,  # Equal weights mặc định
        "description": "Voting - Kết hợp predictions bằng majority vote hoặc probability averaging"
    },
    
    # 5. Negative Correlation Ensemble
    "negative_correlation": {
        "n_estimators": 5,
        "correlation_penalty": 0.1,
        "base_estimator": "DecisionTree",
        "bootstrap_ratio": 0.8,
        "description": "Negative Correlation - Giảm tương quan giữa learners để tăng diversity"
    },
    
    # 6. Heterogeneous Ensemble
    "heterogeneous": {
        "vision_models": [
            {"type": "MLP", "hidden_layers": (100, 50), "activation": "relu"},
            {"type": "MLP", "hidden_layers": (200,), "activation": "tanh"}
        ],
        "classical_models": [
            {"type": "RandomForest", "n_estimators": 100},
            {"type": "SVM", "kernel": "rbf", "probability": True},
            {"type": "XGBoost", "n_estimators": 100}
        ],
        "combination_methods": ["weighted_vote", "stacking", "voting"],
        "vision_weight": 0.6,
        "classical_weight": 0.4,
        "description": "Heterogeneous - Kết hợp vision models với classical ML models"
    },
    
    # 7. Multi-level Deep Ensemble
    "multilevel_deep": {
        "emotion_features_idx": [0, 1, 2, 3],  # sad, angry, happy, relaxed
        "tail_features_idx": [4, 5, 6],        # down, up, mid
        "feature_weights": {
            "emotion": 0.7,
            "tail": 0.3
        },
        "meta_learner": {
            "type": "XGBoost",
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1
        },
        "feature_engineering": {
            "interaction_features": True,
            "statistical_features": True,
            "polynomial_features": False
        },
        "description": "Multi-level Deep - Implicit ensemble qua feature fusion và engineering"
    }
}

# Cấu hình mặc định cho từng loại base estimator
BASE_ESTIMATOR_CONFIGS = {
    "DecisionTree": {
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "criterion": "gini"
    },
    "RandomForest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "bootstrap": True
    },
    "SVM": {
        "kernel": "rbf",
        "C": 1.0,
        "probability": True,
        "gamma": "scale"
    },
    "LogisticRegression": {
        "multi_class": "multinomial",
        "solver": "lbfgs",
        "max_iter": 1000,
        "C": 1.0
    },
    "MLP": {
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 500,
        "alpha": 0.0001
    }
}

# Mapping giữa emotion và tail features với ResNet và YOLO outputs
FEATURE_MAPPING = {
    "emotion_features": {
        "sad": 0,
        "angry": 1, 
        "happy": 2,
        "relaxed": 3
    },
    "tail_features": {
        "down": 4,
        "up": 5,
        "mid": 6
    },
    "emotion_labels": ["sad", "angry", "happy", "relaxed"],
    "tail_labels": ["down", "up", "mid"]
}

# Thông tin về hiệu năng và đặc điểm của từng ensemble method
ENSEMBLE_PERFORMANCE_INFO = {
    "bagging": {
        "performance_rating": 8.0,
        "best_for": "High-variance models như Decision Trees",
        "pros": ["Giảm overfitting", "Parallel training", "Robust"],
        "cons": ["Không cải thiện bias", "Tốn memory"]
    },
    "boosting": {
        "performance_rating": 9.0,
        "best_for": "Unbalanced datasets, complex patterns",
        "pros": ["Cao accuracy", "Handle bias tốt", "Feature importance"],
        "cons": ["Dễ overfit", "Sequential training", "Sensitive to noise"]
    },
    "stacking": {
        "performance_rating": 9.0,
        "best_for": "Diverse base models với uncorrelated errors",
        "pros": ["Kết hợp ưu điểm nhiều models", "Flexible"],
        "cons": ["Phức tạp", "Risk of overfitting", "Computational cost"]
    },
    "voting": {
        "performance_rating": 7.5,
        "best_for": "Simple ensemble, interpretability",
        "pros": ["Đơn giản", "Fast", "Interpretable"],
        "cons": ["Không optimal weights", "Weak models ảnh hưởng"]
    },
    "negative_correlation": {
        "performance_rating": 9.0,
        "best_for": "Tăng diversity, giảm correlation",
        "pros": ["Tăng diversity", "Reduce correlation", "Better generalization"],
        "cons": ["Complex training", "Parameter tuning"]
    },
    "heterogeneous": {
        "performance_rating": 8.5,
        "best_for": "Multimodal features, complementary models",
        "pros": ["Exploit different model strengths", "Robust"],
        "cons": ["Complex tuning", "Synchronization issues"]
    },
    "multilevel_deep": {
        "performance_rating": 9.5,
        "best_for": "Multi-level features, implicit ensemble",
        "pros": ["No voting needed", "Feature engineering", "Scalable"],
        "cons": ["Feature engineering complexity", "Memory intensive"]
    }
}


def get_ensemble_config(ensemble_type, variant=None):
    """
    Lấy cấu hình cho một kỹ thuật ensemble cụ thể.
    
    Parameters:
    -----------
    ensemble_type : str
        Loại ensemble ('bagging', 'boosting', 'stacking', etc.)
    variant : str, optional
        Biến thể cụ thể (ví dụ: 'xgboost' cho boosting)
        
    Returns:
    --------
    dict
        Cấu hình cho ensemble method
    """
    if ensemble_type not in ENSEMBLE_CONFIGS:
        raise ValueError(f"Ensemble type '{ensemble_type}' không được hỗ trợ")
    
    config = ENSEMBLE_CONFIGS[ensemble_type].copy()
    
    if variant and isinstance(config, dict) and variant in config:
        return config[variant]
    
    return config


def get_base_estimator_config(estimator_type):
    """
    Lấy cấu hình mặc định cho base estimator.
    
    Parameters:
    -----------
    estimator_type : str
        Loại estimator ('DecisionTree', 'RandomForest', etc.)
        
    Returns:
    --------
    dict
        Cấu hình mặc định cho estimator
    """
    if estimator_type not in BASE_ESTIMATOR_CONFIGS:
        raise ValueError(f"Base estimator '{estimator_type}' không được hỗ trợ")
    
    return BASE_ESTIMATOR_CONFIGS[estimator_type].copy()


def get_feature_indices(feature_type):
    """
    Lấy indices của features theo loại.
    
    Parameters:
    -----------
    feature_type : str
        Loại features ('emotion', 'tail', 'all')
        
    Returns:
    --------
    list
        Danh sách indices
    """
    if feature_type == "emotion":
        return list(FEATURE_MAPPING["emotion_features"].values())
    elif feature_type == "tail":
        return list(FEATURE_MAPPING["tail_features"].values())
    elif feature_type == "all":
        emotion_idx = list(FEATURE_MAPPING["emotion_features"].values())
        tail_idx = list(FEATURE_MAPPING["tail_features"].values())
        return emotion_idx + tail_idx
    else:
        raise ValueError(f"Feature type '{feature_type}' không được hỗ trợ")


def print_ensemble_summary():
    """In tóm tắt về tất cả các kỹ thuật ensemble được hỗ trợ."""
    print("=== 7 KỸ THUẬT ENSEMBLE LEARNING ĐƯỢC HỖ TRỢ ===")
    print()
    
    for i, (ensemble_type, info) in enumerate(ENSEMBLE_PERFORMANCE_INFO.items(), 1):
        config = ENSEMBLE_CONFIGS[ensemble_type]
        description = config.get("description", "Không có mô tả")
        
        print(f"{i}. {ensemble_type.upper()}")
        print(f"   Mô tả: {description}")
        print(f"   Hiệu năng: {info['performance_rating']}/10")
        print(f"   Phù hợp cho: {info['best_for']}")
        print(f"   Ưu điểm: {', '.join(info['pros'])}")
        print(f"   Nhược điểm: {', '.join(info['cons'])}")
        print()


if __name__ == "__main__":
    # Demo cấu hình ensemble
    print_ensemble_summary()
    
    # Test lấy cấu hình
    print("=== DEMO CẤU HÌNH ===")
    bagging_config = get_ensemble_config("bagging")
    print(f"Bagging config: {bagging_config}")
    
    xgboost_config = get_ensemble_config("boosting", "xgboost")
    print(f"XGBoost config: {xgboost_config}")
    
    emotion_indices = get_feature_indices("emotion")
    print(f"Emotion feature indices: {emotion_indices}") 