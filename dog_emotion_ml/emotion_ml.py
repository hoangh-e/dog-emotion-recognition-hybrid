"""
Module cho Phần II: Tăng cường nhận diện cảm xúc bằng Machine Learning

Module này cung cấp chức năng toàn diện để huấn luyện nhiều thuật toán ML
trên dữ liệu nhận diện cảm xúc chó kết hợp ResNet và YOLO tail detection.

Module này triển khai đầy đủ 7 kỹ thuật ensemble learning theo tài liệu nghiên cứu:
1. Bagging (Bootstrap Aggregating) - Giảm phương sai
2. Boosting (XGBoost, AdaBoost, LightGBM) - Tập trung học mẫu khó  
3. Stacking (Stacked Generalization) - Meta-model kết hợp
4. Voting (Soft/Hard) - Bỏ phiếu đơn giản
5. Negative Correlation Ensemble - Giảm tương quan giữa learners
6. Heterogeneous Ensemble - Kết hợp vision + classical models
7. Multi-level Deep Ensembles - Implicit ensemble qua feature fusion
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                             BaggingClassifier, VotingClassifier, StackingClassifier,
                             GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')


class NegativeCorrelationEnsemble(BaseEstimator, ClassifierMixin):
    """
    Negative Correlation Ensemble - Kỹ thuật ensemble giảm tương quan
    
    Huấn luyện nhiều mô hình với penalty correlation để tăng đa dạng
    và giảm tương quan giữa các learners, nâng cao hiệu suất tổng thể.
    """
    
    def __init__(self, base_estimators=None, n_estimators=5, correlation_penalty=0.1, random_state=42):
        """
        Khởi tạo Negative Correlation Ensemble
        
        Parameters:
        -----------
        base_estimators : list, optional
            Danh sách base estimators. Nếu None sẽ dùng mặc định
        n_estimators : int, default=5
            Số lượng estimators
        correlation_penalty : float, default=0.1
            Hệ số penalty cho correlation
        random_state : int, default=42
            Seed ngẫu nhiên
        """
        self.base_estimators = base_estimators
        self.n_estimators = n_estimators
        self.correlation_penalty = correlation_penalty
        self.random_state = random_state
        self.estimators_ = []
        self.classes_ = None
        
    def fit(self, X, y):
        """Huấn luyện ensemble với negative correlation penalty"""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        # Tạo base estimators nếu chưa có
        if self.base_estimators is None:
            self.base_estimators = [
                DecisionTreeClassifier(random_state=self.random_state + i)
                for i in range(self.n_estimators)
            ]
        
        # Huấn luyện từng estimator với random sampling để giảm correlation
        self.estimators_ = []
        for i, base_est in enumerate(self.base_estimators[:self.n_estimators]):
            # Random sampling để tạo đa dạng
            np.random.seed(self.random_state + i)
            sample_indices = np.random.choice(len(X), size=int(0.8 * len(X)), replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            
            # Clone và huấn luyện
            estimator = clone(base_est)
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)
            
        return self
    
    def predict(self, X):
        """Dự đoán bằng cách kết hợp predictions với weight giảm correlation"""
        X = check_array(X)
        predictions = np.array([est.predict(X) for est in self.estimators_])
        
        # Weighted voting với penalty correlation
        weights = self._calculate_weights(predictions)
        weighted_preds = np.average(predictions, axis=0, weights=weights)
        
        return np.round(weighted_preds).astype(int)
    
    def predict_proba(self, X):
        """Dự đoán xác suất với negative correlation weighting"""
        X = check_array(X)
        probas = np.array([est.predict_proba(X) for est in self.estimators_])
        
        # Weighted average của probabilities
        weights = np.ones(len(self.estimators_)) / len(self.estimators_)
        weighted_probas = np.average(probas, axis=0, weights=weights)
        
        return weighted_probas
    
    def _calculate_weights(self, predictions):
        """Tính weights để giảm correlation giữa các predictions"""
        n_estimators = len(predictions)
        weights = np.ones(n_estimators)
        
        # Penalty cho correlation cao
        for i in range(n_estimators):
            for j in range(i + 1, n_estimators):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                if not np.isnan(corr):
                    penalty = abs(corr) * self.correlation_penalty
                    weights[i] -= penalty
                    weights[j] -= penalty
        
        # Normalize weights
        weights = np.maximum(weights, 0.1)  # Minimum weight
        weights = weights / np.sum(weights)
        
        return weights


class HeterogeneousEnsemble(BaseEstimator, ClassifierMixin):
    """
    Heterogeneous Ensemble - Kết hợp đa dạng các loại mô hình
    
    Kết hợp vision models (deep learning) với classical ML models
    để tận dụng ưu điểm của từng loại mô hình khác nhau.
    """
    
    def __init__(self, vision_models=None, classical_models=None, 
                 combination_method='weighted_vote', random_state=42):
        """
        Khởi tạo Heterogeneous Ensemble
        
        Parameters:
        -----------
        vision_models : list, optional
            Danh sách vision-based models (MLP, neural networks)
        classical_models : list, optional
            Danh sách classical ML models (SVM, RF, etc.)
        combination_method : str, default='weighted_vote'
            Phương pháp kết hợp: 'weighted_vote', 'stacking', 'voting'
        random_state : int, default=42
            Seed ngẫu nhiên
        """
        self.vision_models = vision_models
        self.classical_models = classical_models
        self.combination_method = combination_method
        self.random_state = random_state
        self.vision_estimators_ = []
        self.classical_estimators_ = []
        self.meta_learner_ = None
        self.classes_ = None
        
    def fit(self, X, y):
        """Huấn luyện heterogeneous ensemble"""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        # Tạo default models nếu chưa có
        if self.vision_models is None:
            self.vision_models = [
                MLPClassifier(hidden_layer_sizes=(100, 50), random_state=self.random_state),
                MLPClassifier(hidden_layer_sizes=(200,), random_state=self.random_state + 1)
            ]
            
        if self.classical_models is None:
            self.classical_models = [
                RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                SVC(probability=True, random_state=self.random_state),
                xgb.XGBClassifier(random_state=self.random_state, eval_metric='mlogloss')
            ]
        
        # Huấn luyện vision models
        self.vision_estimators_ = []
        for model in self.vision_models:
            estimator = clone(model)
            estimator.fit(X, y)
            self.vision_estimators_.append(estimator)
        
        # Huấn luyện classical models  
        self.classical_estimators_ = []
        for model in self.classical_models:
            estimator = clone(model)
            estimator.fit(X, y)
            self.classical_estimators_.append(estimator)
        
        # Huấn luyện meta-learner nếu dùng stacking
        if self.combination_method == 'stacking':
            self._fit_meta_learner(X, y)
            
        return self
    
    def _fit_meta_learner(self, X, y):
        """Huấn luyện meta-learner cho stacking method"""
        # Tạo meta-features từ base models
        meta_features = self._create_meta_features(X)
        
        # Huấn luyện meta-learner
        self.meta_learner_ = LogisticRegression(random_state=self.random_state)
        self.meta_learner_.fit(meta_features, y)
    
    def _create_meta_features(self, X):
        """Tạo meta-features từ predictions của base models"""
        meta_features = []
        
        # Features từ vision models
        for estimator in self.vision_estimators_:
            if hasattr(estimator, 'predict_proba'):
                probas = estimator.predict_proba(X)
                meta_features.append(probas)
            else:
                preds = estimator.predict(X).reshape(-1, 1)
                meta_features.append(preds)
        
        # Features từ classical models
        for estimator in self.classical_estimators_:
            if hasattr(estimator, 'predict_proba'):
                probas = estimator.predict_proba(X)
                meta_features.append(probas)
            else:
                preds = estimator.predict(X).reshape(-1, 1)
                meta_features.append(preds)
        
        return np.hstack(meta_features)
    
    def predict(self, X):
        """Dự đoán bằng heterogeneous ensemble"""
        X = check_array(X)
        
        if self.combination_method == 'stacking':
            meta_features = self._create_meta_features(X)
            return self.meta_learner_.predict(meta_features)
        
        elif self.combination_method == 'weighted_vote':
            # Weighted vote với trọng số khác nhau cho vision vs classical
            vision_preds = np.array([est.predict(X) for est in self.vision_estimators_])
            classical_preds = np.array([est.predict(X) for est in self.classical_estimators_])
            
            # Vision models có trọng số cao hơn cho features phức tạp
            vision_weight = 0.6
            classical_weight = 0.4
            
            vision_vote = np.mean(vision_preds, axis=0) * vision_weight
            classical_vote = np.mean(classical_preds, axis=0) * classical_weight
            
            final_preds = vision_vote + classical_vote
            return np.round(final_preds).astype(int)
        
        else:  # simple voting
            all_preds = []
            all_preds.extend([est.predict(X) for est in self.vision_estimators_])
            all_preds.extend([est.predict(X) for est in self.classical_estimators_])
            
            return np.round(np.mean(all_preds, axis=0)).astype(int)
    
    def predict_proba(self, X):
        """Dự đoán xác suất bằng heterogeneous ensemble"""
        X = check_array(X)
        
        if self.combination_method == 'stacking':
            meta_features = self._create_meta_features(X)
            return self.meta_learner_.predict_proba(meta_features)
        
        # Kết hợp probabilities từ tất cả models
        all_probas = []
        
        for estimator in self.vision_estimators_:
            if hasattr(estimator, 'predict_proba'):
                all_probas.append(estimator.predict_proba(X))
        
        for estimator in self.classical_estimators_:
            if hasattr(estimator, 'predict_proba'):
                all_probas.append(estimator.predict_proba(X))
        
        if all_probas:
            return np.mean(all_probas, axis=0)
        else:
            # Fallback nếu không có predict_proba
            preds = self.predict(X)
            n_classes = len(self.classes_)
            probas = np.zeros((len(X), n_classes))
            for i, pred in enumerate(preds):
                probas[i, pred] = 1.0
            return probas


class MultiLevelDeepEnsemble(BaseEstimator, ClassifierMixin):
    """
    Multi-level Deep Ensemble - Implicit ensemble qua feature fusion
    
    Kết hợp features từ nhiều cấp độ khác nhau (emotion + tail features)
    với meta-learner để tạo implicit ensemble không cần vote.
    """
    
    def __init__(self, emotion_features_idx=None, tail_features_idx=None, 
                 meta_learner=None, feature_weights=None, random_state=42):
        """
        Khởi tạo Multi-level Deep Ensemble
        
        Parameters:
        -----------
        emotion_features_idx : list, optional
            Indices của emotion features trong X
        tail_features_idx : list, optional  
            Indices của tail features trong X
        meta_learner : estimator, optional
            Meta-learner để kết hợp features. Mặc định dùng XGBoost
        feature_weights : dict, optional
            Trọng số cho từng loại features
        random_state : int, default=42
            Seed ngẫu nhiên
        """
        self.emotion_features_idx = emotion_features_idx or list(range(4))  # sad, angry, happy, relaxed
        self.tail_features_idx = tail_features_idx or list(range(4, 7))    # down, up, mid
        self.meta_learner = meta_learner
        self.feature_weights = feature_weights or {'emotion': 0.7, 'tail': 0.3}
        self.random_state = random_state
        self.meta_learner_ = None
        self.emotion_scaler_ = StandardScaler()
        self.tail_scaler_ = StandardScaler()
        self.classes_ = None
        
    def fit(self, X, y):
        """Huấn luyện multi-level deep ensemble"""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        # Tách features theo cấp độ
        emotion_features = X[:, self.emotion_features_idx]
        tail_features = X[:, self.tail_features_idx]
        
        # Scale từng loại features riêng biệt
        emotion_features_scaled = self.emotion_scaler_.fit_transform(emotion_features)
        tail_features_scaled = self.tail_scaler_.fit_transform(tail_features)
        
        # Tạo weighted features
        weighted_emotion = emotion_features_scaled * self.feature_weights['emotion']
        weighted_tail = tail_features_scaled * self.feature_weights['tail']
        
        # Kết hợp features với feature engineering
        combined_features = self._engineer_features(weighted_emotion, weighted_tail)
        
        # Khởi tạo meta-learner nếu chưa có
        if self.meta_learner is None:
            self.meta_learner = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='mlogloss'
            )
        
        # Huấn luyện meta-learner
        self.meta_learner_ = clone(self.meta_learner)
        self.meta_learner_.fit(combined_features, y)
        
        return self
    
    def _engineer_features(self, emotion_features, tail_features):
        """Feature engineering để tạo features cấp cao"""
        # Features gốc
        features = [emotion_features, tail_features]
        
        # Interaction features giữa emotion và tail
        interaction_features = []
        for i in range(emotion_features.shape[1]):
            for j in range(tail_features.shape[1]):
                interaction = emotion_features[:, i:i+1] * tail_features[:, j:j+1]
                interaction_features.append(interaction)
        
        if interaction_features:
            features.append(np.hstack(interaction_features))
        
        # Statistical features
        emotion_stats = np.column_stack([
            np.mean(emotion_features, axis=1),    # Mean emotion confidence
            np.std(emotion_features, axis=1),     # Emotion variance
            np.max(emotion_features, axis=1),     # Max emotion confidence
            np.argmax(emotion_features, axis=1)   # Dominant emotion
        ])
        
        tail_stats = np.column_stack([
            np.mean(tail_features, axis=1),       # Mean tail confidence  
            np.std(tail_features, axis=1),        # Tail variance
            np.max(tail_features, axis=1),        # Max tail confidence
            np.argmax(tail_features, axis=1)      # Dominant tail position
        ])
        
        features.extend([emotion_stats, tail_stats])
        
        # Kết hợp tất cả features
        return np.hstack(features)
    
    def predict(self, X):
        """Dự đoán bằng multi-level deep ensemble"""
        X = check_array(X)
        
        # Tách và xử lý features
        emotion_features = X[:, self.emotion_features_idx]
        tail_features = X[:, self.tail_features_idx]
        
        emotion_features_scaled = self.emotion_scaler_.transform(emotion_features)
        tail_features_scaled = self.tail_scaler_.transform(tail_features)
        
        weighted_emotion = emotion_features_scaled * self.feature_weights['emotion']
        weighted_tail = tail_features_scaled * self.feature_weights['tail']
        
        # Feature engineering
        combined_features = self._engineer_features(weighted_emotion, weighted_tail)
        
        # Dự đoán với meta-learner
        return self.meta_learner_.predict(combined_features)
    
    def predict_proba(self, X):
        """Dự đoán xác suất bằng multi-level deep ensemble"""
        X = check_array(X)
        
        # Tách và xử lý features
        emotion_features = X[:, self.emotion_features_idx]
        tail_features = X[:, self.tail_features_idx]
        
        emotion_features_scaled = self.emotion_scaler_.transform(emotion_features)
        tail_features_scaled = self.tail_scaler_.transform(tail_features)
        
        weighted_emotion = emotion_features_scaled * self.feature_weights['emotion']
        weighted_tail = tail_features_scaled * self.feature_weights['tail']
        
        # Feature engineering
        combined_features = self._engineer_features(weighted_emotion, weighted_tail)
        
        # Dự đoán xác suất với meta-learner
        return self.meta_learner_.predict_proba(combined_features)


class EmotionMLClassifier:
    """
    Lớp chính cho nhận diện cảm xúc chó sử dụng nhiều thuật toán ML.
    
    Xử lý tải dữ liệu, tiền xử lý, huấn luyện nhiều mô hình ML,
    và tạo dự đoán cho ensemble meta-learning.
    """
    
    def __init__(self, random_state=42):
        """
        Khởi tạo EmotionMLClassifier.
        
        Parameters:
        -----------
        random_state : int, default=42
            Seed ngẫu nhiên để có kết quả nhất quán
        """
        self.random_state = random_state
        self.emotion_features = ['sad', 'angry', 'happy', 'relaxed']
        self.tail_features = ['down', 'up', 'mid']
        self.feature_columns = self.emotion_features + self.tail_features
        
        # Lưu trữ dữ liệu
        self.train_data = None
        self.test_data = None
        self.test_for_train_data = None
        
        # Dữ liệu đã xử lý
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_test_for_train = None
        self.y_test_for_train = None
        
        # Lưu trữ mô hình
        self.trained_models = {}
        self.model_predictions = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_train_dataset(self, file_path, filename_col='filename', 
                          emotion_cols=None, tail_cols=None, label_col='label'):
        """
        Tải dữ liệu training từ file CSV, TXT, hoặc XLSX.
        
        Parameters:
        -----------
        file_path : str
            Đường dẫn tới file dữ liệu
        filename_col : str, default='filename'
            Tên cột chứa tên file/đường dẫn
        emotion_cols : list, optional
            Danh sách tên cột cảm xúc. Nếu None thì tự động phát hiện
        tail_cols : list, optional
            Danh sách tên cột đuôi. Nếu None thì tự động phát hiện
        label_col : str, default='label'
            Tên cột nhãn
            
        Returns:
        --------
        pd.DataFrame
            Dataset training đã tải và xử lý
        """
        self.train_data = self._load_dataset_file(file_path, filename_col, 
                                                 emotion_cols, tail_cols, label_col)
        return self.train_data
    
    def load_test_dataset(self, file_path, filename_col='filename',
                         emotion_cols=None, tail_cols=None, label_col='label'):
        """
        Tải dataset test từ file CSV, TXT, hoặc XLSX.
        
        Parameters:
        -----------
        file_path : str
            Đường dẫn tới file dataset
        filename_col : str, default='filename'
            Tên cột chứa tên file/đường dẫn
        emotion_cols : list, optional
            Danh sách tên cột cảm xúc. Nếu None thì tự động phát hiện
        tail_cols : list, optional
            Danh sách tên cột đuôi. Nếu None thì tự động phát hiện
        label_col : str, default='label'
            Tên cột nhãn
            
        Returns:
        --------
        pd.DataFrame
            Dataset test đã tải và xử lý
        """
        self.test_data = self._load_dataset_file(file_path, filename_col,
                                               emotion_cols, tail_cols, label_col)
        return self.test_data
    
    def load_test_for_train_dataset(self, file_path, filename_col='filename',
                                   emotion_cols=None, tail_cols=None, label_col='label'):
        """
        Tải dataset test-for-train từ file CSV, TXT, hoặc XLSX.
        
        Dataset này được sử dụng để tạo dự đoán sẽ được dùng
        để huấn luyện meta-learner trong Phần III.
        
        Parameters:
        -----------
        file_path : str
            Đường dẫn tới file dataset
        filename_col : str, default='filename'
            Tên cột chứa tên file/đường dẫn
        emotion_cols : list, optional
            Danh sách tên cột cảm xúc. Nếu None thì tự động phát hiện
        tail_cols : list, optional
            Danh sách tên cột đuôi. Nếu None thì tự động phát hiện
        label_col : str, default='label'
            Tên cột nhãn
            
        Returns:
        --------
        pd.DataFrame
            Dataset test-for-train đã tải và xử lý
        """
        self.test_for_train_data = self._load_dataset_file(file_path, filename_col,
                                                          emotion_cols, tail_cols, label_col)
        return self.test_for_train_data
    
    def _load_dataset_file(self, file_path, filename_col, emotion_cols, tail_cols, label_col):
        """
        Internal method to load dataset from various file formats.
        """
        # Determine file format and load
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep='\t')
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV, TXT, or XLSX.")
        
        # Auto-detect columns if not specified
        if emotion_cols is None:
            emotion_cols = self._detect_emotion_columns(df)
        if tail_cols is None:
            tail_cols = self._detect_tail_columns(df)
        
        # Validate required columns exist
        self._validate_columns(df, emotion_cols, tail_cols, filename_col, label_col)
        
        # Reorder columns consistently
        ordered_cols = [filename_col] + emotion_cols + tail_cols + [label_col]
        return df[ordered_cols]
    
    def _detect_emotion_columns(self, df):
        """Auto-detect emotion columns in dataset."""
        emotion_cols = []
        for emotion in self.emotion_features:
            found = False
            for col in df.columns:
                if emotion.lower() in col.lower() or col.lower() in emotion.lower():
                    emotion_cols.append(col)
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not find column for emotion: {emotion}")
        return emotion_cols
    
    def _detect_tail_columns(self, df):
        """Auto-detect tail columns in dataset."""
        tail_cols = []
        tail_patterns = {
            'down': ['down', 'tail_down', 'down_tail'],
            'up': ['up', 'tail_up', 'up_tail'],
            'mid': ['mid', 'middle', 'tail_mid', 'mid_tail']
        }
        
        for tail_state in self.tail_features:
            found = False
            patterns = tail_patterns[tail_state]
            for pattern in patterns:
                for col in df.columns:
                    if pattern.lower() in col.lower():
                        tail_cols.append(col)
                        found = True
                        break
                if found:
                    break
            if not found:
                raise ValueError(f"Could not find column for tail state: {tail_state}")
        return tail_cols
    
    def _validate_columns(self, df, emotion_cols, tail_cols, filename_col, label_col):
        """Validate that all required columns exist in the dataset."""
        required_cols = [filename_col] + emotion_cols + tail_cols + [label_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def check_data_anomalies(self, dataset_name='train'):
        """
        Check for data anomalies including missing values, outliers, and invalid data.
        
        Parameters:
        -----------
        dataset_name : str, default='train'
            Name of dataset to check ('train', 'test', 'test_for_train')
            
        Returns:
        --------
        dict
            Dictionary containing anomaly information and suggestions
        """
        if dataset_name == 'train':
            data = self.train_data
        elif dataset_name == 'test':
            data = self.test_data
        elif dataset_name == 'test_for_train':
            data = self.test_for_train_data
        else:
            raise ValueError("dataset_name must be 'train', 'test', or 'test_for_train'")
        
        if data is None:
            raise ValueError(f"No {dataset_name} dataset loaded")
        
        anomalies = {
            'missing_values': {},
            'invalid_probabilities': {},
            'outliers': {},
            'summary': {},
            'suggestions': []
        }
        
        # Check missing values
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                anomalies['missing_values'][col] = missing_count
        
        # Check probability columns (should be between 0 and 1)
        prob_cols = [col for col in data.columns if col not in [data.columns[0], data.columns[-1]]]
        for col in prob_cols:
            if data[col].dtype in ['float64', 'int64']:
                invalid_probs = data[(data[col] < 0) | (data[col] > 1)][col]
                if not invalid_probs.empty:
                    anomalies['invalid_probabilities'][col] = len(invalid_probs)
        
        # Check for outliers using IQR
        for col in prob_cols:
            if data[col].dtype in ['float64', 'int64']:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)]
                if not outliers.empty:
                    anomalies['outliers'][col] = len(outliers)
        
        # Generate summary
        total_rows = len(data)
        anomalies['summary'] = {
            'total_rows': total_rows,
            'total_missing': sum(anomalies['missing_values'].values()),
            'total_invalid_probs': sum(anomalies['invalid_probabilities'].values()),
            'total_outliers': sum(anomalies['outliers'].values())
        }
        
        # Generate suggestions
        if anomalies['missing_values']:
            anomalies['suggestions'].append("Use filter_missing_values() to handle missing data")
        if anomalies['invalid_probabilities']:
            anomalies['suggestions'].append("Use filter_invalid_probabilities() to fix probability values")
        if anomalies['outliers']:
            anomalies['suggestions'].append("Use filter_outliers() to handle outlier values")
        
        return anomalies
    
    def display_anomalies_summary(self, dataset_name='train'):
        """
        Display a formatted summary of data anomalies.
        
        Parameters:
        -----------
        dataset_name : str, default='train'
            Name of dataset to analyze
        """
        anomalies = self.check_data_anomalies(dataset_name)
        
        print(f"\n=== Data Anomalies Summary for {dataset_name.upper()} Dataset ===")
        print(f"Total rows: {anomalies['summary']['total_rows']}")
        print(f"Total missing values: {anomalies['summary']['total_missing']}")
        print(f"Total invalid probabilities: {anomalies['summary']['total_invalid_probs']}")
        print(f"Total outliers: {anomalies['summary']['total_outliers']}")
        
        if anomalies['missing_values']:
            print("\nMissing Values by Column:")
            for col, count in anomalies['missing_values'].items():
                print(f"  {col}: {count} missing")
        
        if anomalies['invalid_probabilities']:
            print("\nInvalid Probabilities by Column:")
            for col, count in anomalies['invalid_probabilities'].items():
                print(f"  {col}: {count} invalid values")
        
        if anomalies['outliers']:
            print("\nOutliers by Column:")
            for col, count in anomalies['outliers'].items():
                print(f"  {col}: {count} outliers")
        
        if anomalies['suggestions']:
            print("\nSuggested Actions:")
            for suggestion in anomalies['suggestions']:
                print(f"  - {suggestion}")
        
        return anomalies
    
    def filter_missing_values(self, dataset_name='train', method='drop', fill_value=0.0):
        """
        Filter missing values from the dataset.
        
        Parameters:
        -----------
        dataset_name : str, default='train'
            Name of dataset to filter
        method : str, default='drop'
            Method to handle missing values ('drop', 'fill')
        fill_value : float, default=0.0
            Value to use when method='fill'
        """
        data = self._get_dataset(dataset_name)
        
        if method == 'drop':
            filtered_data = data.dropna()
        elif method == 'fill':
            filtered_data = data.fillna(fill_value)
        else:
            raise ValueError("method must be 'drop' or 'fill'")
        
        self._set_dataset(dataset_name, filtered_data)
        print(f"Filtered {dataset_name} dataset: {len(data)} -> {len(filtered_data)} rows")
    
    def filter_invalid_probabilities(self, dataset_name='train', method='clip'):
        """
        Filter invalid probability values from the dataset.
        
        Parameters:
        -----------
        dataset_name : str, default='train'
            Name of dataset to filter
        method : str, default='clip'
            Method to handle invalid probabilities ('clip', 'drop')
        """
        data = self._get_dataset(dataset_name)
        prob_cols = [col for col in data.columns if col not in [data.columns[0], data.columns[-1]]]
        
        if method == 'clip':
            filtered_data = data.copy()
            for col in prob_cols:
                if data[col].dtype in ['float64', 'int64']:
                    filtered_data[col] = filtered_data[col].clip(0, 1)
        elif method == 'drop':
            mask = True
            for col in prob_cols:
                if data[col].dtype in ['float64', 'int64']:
                    mask &= (data[col] >= 0) & (data[col] <= 1)
            filtered_data = data[mask]
        else:
            raise ValueError("method must be 'clip' or 'drop'")
        
        self._set_dataset(dataset_name, filtered_data)
        print(f"Filtered {dataset_name} dataset: {len(data)} -> {len(filtered_data)} rows")
    
    def filter_outliers(self, dataset_name='train', method='iqr', factor=1.5):
        """
        Filter outliers from the dataset.
        
        Parameters:
        -----------
        dataset_name : str, default='train'
            Name of dataset to filter
        method : str, default='iqr'
            Method to detect outliers ('iqr', 'zscore')
        factor : float, default=1.5
            Factor for outlier detection (IQR factor or Z-score threshold)
        """
        data = self._get_dataset(dataset_name)
        prob_cols = [col for col in data.columns if col not in [data.columns[0], data.columns[-1]]]
        
        if method == 'iqr':
            mask = True
            for col in prob_cols:
                if data[col].dtype in ['float64', 'int64']:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    mask &= (data[col] >= Q1 - factor * IQR) & (data[col] <= Q3 + factor * IQR)
        elif method == 'zscore':
            from scipy import stats
            mask = True
            for col in prob_cols:
                if data[col].dtype in ['float64', 'int64']:
                    z_scores = np.abs(stats.zscore(data[col]))
                    mask &= z_scores < factor
        else:
            raise ValueError("method must be 'iqr' or 'zscore'")
        
        filtered_data = data[mask]
        self._set_dataset(dataset_name, filtered_data)
        print(f"Filtered {dataset_name} dataset: {len(data)} -> {len(filtered_data)} rows")
    
    def _get_dataset(self, dataset_name):
        """Internal method to get dataset by name."""
        if dataset_name == 'train':
            return self.train_data
        elif dataset_name == 'test':
            return self.test_data
        elif dataset_name == 'test_for_train':
            return self.test_for_train_data
        else:
            raise ValueError("dataset_name must be 'train', 'test', or 'test_for_train'")
    
    def _set_dataset(self, dataset_name, data):
        """Internal method to set dataset by name."""
        if dataset_name == 'train':
            self.train_data = data
        elif dataset_name == 'test':
            self.test_data = data
        elif dataset_name == 'test_for_train':
            self.test_for_train_data = data
        else:
            raise ValueError("dataset_name must be 'train', 'test', or 'test_for_train'")
    
    def prepare_training_data(self, use_advanced_normalization=True):
        """
        Prepare training data by extracting features and labels.
        
        Parameters:
        -----------
        use_advanced_normalization : bool, default=True
            Sử dụng chuẩn hóa nâng cao (Z-score cho emotion, pass-through cho tail)
        """
        if self.train_data is None:
            raise ValueError("No training data loaded. Use load_train_dataset() first.")
        
        # Extract features (columns 1-7: emotions + tail)
        self.X_train = self.train_data.iloc[:, 1:8].values
        self.y_train = self.train_data.iloc[:, -1].values
        
        # Encode labels
        self.y_train = self.label_encoder.fit_transform(self.y_train)
        
        # Advanced normalization theo yêu cầu
        if use_advanced_normalization:
            # Tách emotion features (4 cột đầu) và tail features (3 cột cuối)
            emotion_features = self.X_train[:, :4]  # sad, angry, happy, relaxed
            tail_features = self.X_train[:, 4:]     # down, up, mid
            
            # Z-score normalization cho emotion features
            from sklearn.preprocessing import StandardScaler
            emotion_scaler = StandardScaler()
            emotion_features_normalized = emotion_scaler.fit_transform(emotion_features)
            
            # Tail features pass through (đã là binary/probabilities)
            tail_features_normalized = tail_features
            
            # Kết hợp lại
            self.X_train = np.column_stack([emotion_features_normalized, tail_features_normalized])
            
            # Lưu scalers để sử dụng cho test data
            self.emotion_scaler = emotion_scaler
            self.tail_scaler = None  # Không cần scaler cho tail features
            
            print("Sử dụng chuẩn hóa nâng cao:")
            print(f"  - Emotion features (Z-score): mean={emotion_scaler.mean_}, std={emotion_scaler.scale_}")
            print(f"  - Tail features: pass-through (không chuẩn hóa)")
        else:
            # Chuẩn hóa truyền thống (tất cả features với StandardScaler)
            self.X_train = self.scaler.fit_transform(self.X_train)
            print("Sử dụng chuẩn hóa truyền thống (StandardScaler cho tất cả features)")
        
        print(f"Training data prepared: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
        print(f"Classes: {self.label_encoder.classes_}")
    
    def prepare_test_data(self, use_advanced_normalization=True):
        """
        Prepare test data by extracting features and labels.
        
        Parameters:
        -----------
        use_advanced_normalization : bool, default=True
            Sử dụng chuẩn hóa nâng cao (phải match với training data)
        """
        if self.test_data is None:
            raise ValueError("No test data loaded. Use load_test_dataset() first.")
        
        self.X_test = self.test_data.iloc[:, 1:8].values
        self.y_test = self.test_data.iloc[:, -1].values
        
        # Encode labels
        self.y_test = self.label_encoder.transform(self.y_test)
        
        # Apply same normalization as training data
        if use_advanced_normalization and hasattr(self, 'emotion_scaler'):
            # Tách emotion features và tail features
            emotion_features = self.X_test[:, :4]  # sad, angry, happy, relaxed
            tail_features = self.X_test[:, 4:]     # down, up, mid
            
            # Transform emotion features với fitted scaler
            emotion_features_normalized = self.emotion_scaler.transform(emotion_features)
            
            # Tail features pass through
            tail_features_normalized = tail_features
            
            # Kết hợp lại
            self.X_test = np.column_stack([emotion_features_normalized, tail_features_normalized])
            
            print("Test data: Sử dụng chuẩn hóa nâng cao (emotion Z-score, tail pass-through)")
        else:
            # Chuẩn hóa truyền thống
            self.X_test = self.scaler.transform(self.X_test)
            print("Test data: Sử dụng chuẩn hóa truyền thống")
        
        print(f"Test data prepared: {self.X_test.shape[0]} samples, {self.X_test.shape[1]} features")
    
    def prepare_test_for_train_data(self, use_advanced_normalization=True):
        """
        Prepare test-for-train data by extracting features and labels.
        
        Parameters:
        -----------
        use_advanced_normalization : bool, default=True
            Sử dụng chuẩn hóa nâng cao (phải match với training data)
        """
        if self.test_for_train_data is None:
            raise ValueError("No test-for-train data loaded. Use load_test_for_train_dataset() first.")
        
        self.X_test_for_train = self.test_for_train_data.iloc[:, 1:8].values
        self.y_test_for_train = self.test_for_train_data.iloc[:, -1].values
        
        # Encode labels
        self.y_test_for_train = self.label_encoder.transform(self.y_test_for_train)
        
        # Apply same normalization as training data
        if use_advanced_normalization and hasattr(self, 'emotion_scaler'):
            # Tách emotion features và tail features
            emotion_features = self.X_test_for_train[:, :4]  # sad, angry, happy, relaxed
            tail_features = self.X_test_for_train[:, 4:]     # down, up, mid
            
            # Transform emotion features với fitted scaler
            emotion_features_normalized = self.emotion_scaler.transform(emotion_features)
            
            # Tail features pass through
            tail_features_normalized = tail_features
            
            # Kết hợp lại
            self.X_test_for_train = np.column_stack([emotion_features_normalized, tail_features_normalized])
            
            print("Test-for-train data: Sử dụng chuẩn hóa nâng cao (emotion Z-score, tail pass-through)")
        else:
            # Chuẩn hóa truyền thống
            self.X_test_for_train = self.scaler.transform(self.X_test_for_train)
            print("Test-for-train data: Sử dụng chuẩn hóa truyền thống")
        
        print(f"Test-for-train data prepared: {self.X_test_for_train.shape[0]} samples")
    
    def create_dataset_from_roboflow(self, roboflow_path, yolo_model_path=None, 
                                   resnet_model_path=None, output_path=None, split='train'):
        """
        Tạo dataset từ Roboflow data sử dụng YOLO và ResNet models
        
        Parameters:
        -----------
        roboflow_path : str
            Đường dẫn đến thư mục Roboflow dataset
        yolo_model_path : str, optional
            Đường dẫn đến YOLO model cho tail detection
        resnet_model_path : str, optional
            Đường dẫn đến ResNet model cho emotion detection
        output_path : str, optional
            Đường dẫn lưu dataset CSV (nếu None sẽ auto-generate)
        split : str, default='train'
            Split để xử lý ('train', 'val', 'test')
            
        Returns:
        --------
        pd.DataFrame
            Dataset đã tạo
        """
        try:
            from .data_pipeline import RoboflowDataProcessor
        except ImportError:
            raise ImportError("RoboflowDataProcessor not available. Please check data_pipeline module.")
        
        # Initialize processor
        processor = RoboflowDataProcessor(
            dataset_path=roboflow_path,
            yolo_tail_model_path=yolo_model_path,
            resnet_emotion_model_path=resnet_model_path
        )
        
        # Auto-generate output path if not provided
        if output_path is None:
            output_path = f"{split}_dataset_from_roboflow.csv"
        
        # Create dataset
        dataset = processor.create_training_dataset(output_path, split=split)
        
        print(f"Dataset created successfully: {output_path}")
        print(f"Dataset shape: {dataset.shape}")
        print(f"Columns: {list(dataset.columns)}")
        
        return dataset
    
    def normalize_features_advanced(self, emotion_features, tail_features, fit=True):
        """
        Chuẩn hóa features theo phương pháp nâng cao
        
        Parameters:
        -----------
        emotion_features : array-like, shape (n_samples, 4)
            Emotion features [sad, angry, happy, relaxed]
        tail_features : array-like, shape (n_samples, 3)
            Tail features [down, up, mid]
        fit : bool, default=True
            Có fit scaler hay không (False cho test data)
            
        Returns:
        --------
        np.ndarray
            Combined normalized features
        """
        try:
            from .data_pipeline import DataNormalizer
        except ImportError:
            raise ImportError("DataNormalizer not available. Please check data_pipeline module.")
        
        if not hasattr(self, 'data_normalizer'):
            self.data_normalizer = DataNormalizer()
        
        if fit:
            emotion_norm, tail_norm = self.data_normalizer.fit_transform(emotion_features, tail_features)
        else:
            emotion_norm, tail_norm = self.data_normalizer.transform(emotion_features, tail_features)
        
        # Combine features
        if tail_norm is not None:
            return np.column_stack([emotion_norm, tail_norm])
        else:
            return emotion_norm
    
    def train_logistic_regression(self, multi_class='multinomial', solver='lbfgs', max_iter=1000):
        """Huấn luyện mô hình Logistic Regression."""
        if self.X_train is None:
            self.prepare_training_data()
        
        model = LogisticRegression(
            multi_class=multi_class, 
            solver=solver, 
            max_iter=max_iter,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        model_name = f"LogisticRegression_{multi_class}"
        self.trained_models[model_name] = model
        print(f"Huấn luyện {model_name}")
        return model
    
    def train_svm(self, kernel='rbf', decision_function_shape='ovr', C=1.0):
        """Huấn luyện mô hình SVM."""
        if self.X_train is None:
            self.prepare_training_data()
        
        model = SVC(
            kernel=kernel,
            decision_function_shape=decision_function_shape,
            C=C,
            probability=True,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        model_name = f"SVM_{kernel}_{decision_function_shape}"
        self.trained_models[model_name] = model
        print(f"Huấn luyện {model_name}")
        return model
    
    def train_decision_tree(self, max_depth=None, min_samples_split=2):
        """Huấn luyện mô hình Decision Tree."""
        if self.X_train is None:
            self.prepare_training_data()
        
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        self.trained_models["DecisionTree"] = model
        print("Huấn luyện DecisionTree")
        return model
    
    def train_random_forest(self, n_estimators=100, max_depth=None):
        """Huấn luyện mô hình Random Forest."""
        if self.X_train is None:
            self.prepare_training_data()
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        self.trained_models["RandomForest"] = model
        print("Huấn luyện RandomForest")
        return model
    
    def train_xgboost(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        """Huấn luyện mô hình XGBoost."""
        if self.X_train is None:
            self.prepare_training_data()
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='multi:softprob',
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        self.trained_models["XGBoost"] = model
        print("Huấn luyện XGBoost")
        return model
    
    def train_adaboost(self, n_estimators=50, learning_rate=1.0):
        """Huấn luyện mô hình AdaBoost."""
        if self.X_train is None:
            self.prepare_training_data()
        
        model = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        self.trained_models["AdaBoost"] = model
        print("Huấn luyện AdaBoost")
        return model
    
    def train_naive_bayes(self):
        """Huấn luyện mô hình Naive Bayes."""
        if self.X_train is None:
            self.prepare_training_data()
        
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        
        self.trained_models["NaiveBayes"] = model
        print("Huấn luyện NaiveBayes")
        return model
    
    def train_knn(self, n_neighbors=5):
        """Huấn luyện mô hình K-Nearest Neighbors."""
        if self.X_train is None:
            self.prepare_training_data()
        
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(self.X_train, self.y_train)
        
        self.trained_models["KNN"] = model
        print("Huấn luyện KNN")
        return model
    
    def train_lda(self):
        """Huấn luyện mô hình Linear Discriminant Analysis."""
        if self.X_train is None:
            self.prepare_training_data()
        
        model = LinearDiscriminantAnalysis()
        model.fit(self.X_train, self.y_train)
        
        self.trained_models["LDA"] = model
        print("Huấn luyện LDA")
        return model
    
    def train_qda(self):
        """Huấn luyện mô hình Quadratic Discriminant Analysis."""
        if self.X_train is None:
            self.prepare_training_data()
        
        model = QuadraticDiscriminantAnalysis()
        model.fit(self.X_train, self.y_train)
        
        self.trained_models["QDA"] = model
        print("Huấn luyện QDA")
        return model
    
    def train_mlp(self, hidden_layer_sizes=(100,), max_iter=500):
        """Huấn luyện mô hình Multi-layer Perceptron."""
        if self.X_train is None:
            self.prepare_training_data()
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        self.trained_models["MLP"] = model
        print("Huấn luyện MLP")
        return model
    
    def train_ovr_classifier(self, base_estimator_name='LogisticRegression'):
        """Huấn luyện One-vs-Rest classifier."""
        if self.X_train is None:
            self.prepare_training_data()
        
        if base_estimator_name == 'LogisticRegression':
            base_estimator = LogisticRegression(random_state=self.random_state)
        elif base_estimator_name == 'SVM':
            base_estimator = SVC(probability=True, random_state=self.random_state)
        else:
            raise ValueError("Base estimator không được hỗ trợ")
        
        model = OneVsRestClassifier(base_estimator)
        model.fit(self.X_train, self.y_train)
        
        model_name = f"OvR_{base_estimator_name}"
        self.trained_models[model_name] = model
        print(f"Huấn luyện {model_name}")
        return model
    
    def train_ovo_classifier(self, base_estimator_name='LogisticRegression'):
        """Huấn luyện One-vs-One classifier."""
        if self.X_train is None:
            self.prepare_training_data()
        
        if base_estimator_name == 'LogisticRegression':
            base_estimator = LogisticRegression(random_state=self.random_state)
        elif base_estimator_name == 'SVM':
            base_estimator = SVC(probability=True, random_state=self.random_state)
        else:
            raise ValueError("Base estimator không được hỗ trợ")
        
        model = OneVsOneClassifier(base_estimator)
        model.fit(self.X_train, self.y_train)
        
        model_name = f"OvO_{base_estimator_name}"
        self.trained_models[model_name] = model
        print(f"Huấn luyện {model_name}")
        return model
    
    def train_bagging_classifier(self, base_estimator_name='DecisionTree', n_estimators=10):
        """Huấn luyện Bagging classifier."""
        if self.X_train is None:
            self.prepare_training_data()
        
        if base_estimator_name == 'DecisionTree':
            base_estimator = DecisionTreeClassifier(random_state=self.random_state)
        else:
            raise ValueError("Base estimator không được hỗ trợ")
        
        model = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        model_name = f"Bagging_{base_estimator_name}"
        self.trained_models[model_name] = model
        print(f"Huấn luyện {model_name}")
        return model
    
    def train_voting_classifier(self, voting='soft'):
        """Huấn luyện Voting classifier."""
        if self.X_train is None:
            self.prepare_training_data()
        
        estimators = [
            ('lr', LogisticRegression(random_state=self.random_state)),
            ('rf', RandomForestClassifier(random_state=self.random_state)),
            ('svm', SVC(probability=True, random_state=self.random_state))
        ]
        
        model = VotingClassifier(estimators=estimators, voting=voting)
        model.fit(self.X_train, self.y_train)
        
        model_name = f"Voting_{voting}"
        self.trained_models[model_name] = model
        print(f"Huấn luyện {model_name}")
        return model
    
    def train_stacking_classifier(self, final_estimator_name='LogisticRegression'):
        """Huấn luyện Stacking classifier."""
        if self.X_train is None:
            self.prepare_training_data()
        
        estimators = [
            ('rf', RandomForestClassifier(random_state=self.random_state)),
            ('svm', SVC(probability=True, random_state=self.random_state)),
            ('nb', GaussianNB())
        ]
        
        if final_estimator_name == 'LogisticRegression':
            final_estimator = LogisticRegression(random_state=self.random_state)
        elif final_estimator_name == 'XGBoost':
            final_estimator = xgb.XGBClassifier(random_state=self.random_state)
        else:
            raise ValueError("Final estimator không được hỗ trợ")
        
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5
        )
        model.fit(self.X_train, self.y_train)
        
        model_name = f"Stacking_{final_estimator_name}"
        self.trained_models[model_name] = model
        print(f"Huấn luyện {model_name}")
        return model
    
    def train_logistic_regression_ovr(self):
        """Huấn luyện Logistic Regression với chiến lược One-vs-Rest."""
        if self.X_train is None:
            self.prepare_training_data()
        
        model = LogisticRegression(
            multi_class='ovr',
            solver='liblinear',
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        self.trained_models["LogisticRegression_OvR"] = model
        print("Huấn luyện LogisticRegression_OvR")
        return model
    
    def train_svm_ovo(self):
        """Huấn luyện SVM với chiến lược One-vs-One."""
        if self.X_train is None:
            self.prepare_training_data()
        
        model = SVC(
            kernel='rbf',
            decision_function_shape='ovo',
            probability=True,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        self.trained_models["SVM_OvO"] = model
        print("Huấn luyện SVM_OvO")
        return model
    
    def train_perceptron(self):
        """Huấn luyện Perceptron đa lớp."""
        if self.X_train is None:
            self.prepare_training_data()
        
        model = Perceptron(random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        
        self.trained_models["Perceptron"] = model
        print("Huấn luyện Perceptron")
        return model
    
    def train_gradient_boosting(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """Huấn luyện Gradient Boosting - Kỹ thuật boosting ensemble."""
        if self.X_train is None:
            self.prepare_training_data()
        
        print("Huấn luyện Gradient Boosting...")
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        model_name = "GradientBoosting"
        self.trained_models[model_name] = model
        print(f"Hoàn thành huấn luyện {model_name}")
        return model
    
    def train_lightgbm(self, n_estimators=100, learning_rate=0.1, max_depth=6):
        """Huấn luyện LightGBM - Kỹ thuật boosting tiên tiến."""
        if self.X_train is None:
            self.prepare_training_data()
        
        if not LIGHTGBM_AVAILABLE:
            print("LightGBM không có sẵn, bỏ qua...")
            return None
            
        print("Huấn luyện LightGBM...")
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
            verbose=-1
        )
        model.fit(self.X_train, self.y_train)
        
        model_name = "LightGBM"
        self.trained_models[model_name] = model
        print(f"Hoàn thành huấn luyện {model_name}")
        return model
    
    def train_extra_trees(self, n_estimators=100, max_depth=None):
        """Huấn luyện Extra Trees - Extreme randomization ensemble."""
        if self.X_train is None:
            self.prepare_training_data()
        
        print("Huấn luyện Extra Trees...")
        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        model_name = "ExtraTrees"
        self.trained_models[model_name] = model
        print(f"Hoàn thành huấn luyện {model_name}")
        return model
    
    def train_negative_correlation_ensemble(self, n_estimators=5, correlation_penalty=0.1):
        """Huấn luyện Negative Correlation Ensemble - Giảm tương quan giữa learners."""
        if self.X_train is None:
            self.prepare_training_data()
        
        print("Huấn luyện Negative Correlation Ensemble...")
        model = NegativeCorrelationEnsemble(
            n_estimators=n_estimators,
            correlation_penalty=correlation_penalty,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        model_name = "NegativeCorrelationEnsemble"
        self.trained_models[model_name] = model
        print(f"Hoàn thành huấn luyện {model_name}")
        return model
    
    def train_heterogeneous_ensemble(self, combination_method='weighted_vote'):
        """Huấn luyện Heterogeneous Ensemble - Kết hợp vision + classical models."""
        if self.X_train is None:
            self.prepare_training_data()
        
        print("Huấn luyện Heterogeneous Ensemble...")
        model = HeterogeneousEnsemble(
            combination_method=combination_method,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        model_name = f"HeterogeneousEnsemble_{combination_method}"
        self.trained_models[model_name] = model
        print(f"Hoàn thành huấn luyện {model_name}")
        return model
    
    def train_multilevel_deep_ensemble(self):
        """Huấn luyện Multi-level Deep Ensemble - Implicit ensemble qua feature fusion."""
        if self.X_train is None:
            self.prepare_training_data()
        
        print("Huấn luyện Multi-level Deep Ensemble...")
        model = MultiLevelDeepEnsemble(
            emotion_features_idx=list(range(4)),  # sad, angry, happy, relaxed
            tail_features_idx=list(range(4, 7)),  # down, up, mid
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        
        model_name = "MultiLevelDeepEnsemble"
        self.trained_models[model_name] = model
        print(f"Hoàn thành huấn luyện {model_name}")
        return model

    def train_all_models(self):
        """
        Huấn luyện tất cả thuật toán ML với đầy đủ 7 kỹ thuật ensemble learning.
        
        Triển khai đầy đủ theo tài liệu nghiên cứu ensemble methods:
        1. Bagging - Bootstrap Aggregating giảm phương sai
        2. Boosting - XGBoost, AdaBoost, GradientBoosting, LightGBM học mẫu khó
        3. Stacking - Meta-model kết hợp heterogeneous base models  
        4. Voting - Soft/Hard voting đơn giản hiệu quả
        5. Negative Correlation - Giảm tương quan tăng đa dạng
        6. Heterogeneous - Kết hợp vision + classical models
        7. Multi-level Deep - Implicit ensemble qua feature fusion
        """
        print("Bắt đầu huấn luyện tất cả thuật toán ML với 7 kỹ thuật ensemble đầy đủ...")
        
        # === CLASSICAL ML ALGORITHMS ===
        
        # 1. Logistic Regression (Multinomial/Softmax)
        self.train_logistic_regression()
        
        # 2. Logistic Regression (OvR)
        self.train_logistic_regression_ovr()
        
        # 3. Logistic Regression (OvO)
        self.train_ovo_classifier('LogisticRegression')
        
        # 4. SVM (RBF OvR)
        self.train_svm()
        
        # 5. SVM (OvO)
        self.train_svm_ovo()
        
        # 6. Decision Tree
        self.train_decision_tree()
        
        # 7. Random Forest (ensemble của Decision Trees)
        self.train_random_forest()
        
        # 8. Naive Bayes
        self.train_naive_bayes()
        
        # 9. K-Nearest Neighbors
        self.train_knn()
        
        # 10. Linear Discriminant Analysis
        self.train_lda()
        
        # 11. Quadratic Discriminant Analysis
        self.train_qda()
        
        # 12. Multi-layer Perceptron (Neural Network)
        self.train_mlp()
        
        # 13. Perceptron
        self.train_perceptron()
        
        # 14. One-vs-Rest meta-strategy
        self.train_ovr_classifier('SVM')
        
        # === 7 ENSEMBLE LEARNING TECHNIQUES THEO TÀI LIỆU ===
        
        print("\n=== Triển khai 7 kỹ thuật Ensemble Learning ===")
        
        # 1. BAGGING - Bootstrap Aggregating (Homogeneous ensemble)
        self.train_bagging_classifier()
        
        # 2. BOOSTING - Sequential ensemble học mẫu khó
        self.train_xgboost()                    # XGBoost - Extreme Gradient Boosting
        self.train_adaboost()                   # AdaBoost - Adaptive Boosting  
        self.train_gradient_boosting()          # Gradient Boosting Machine
        if LIGHTGBM_AVAILABLE:
            self.train_lightgbm()               # LightGBM - Light Gradient Boosting
        
        # 3. STACKING - Heterogeneous base + meta-model
        self.train_stacking_classifier()
        
        # 4. VOTING - Simple ensemble voting
        self.train_voting_classifier()          # Soft voting
        
        # 5. NEGATIVE CORRELATION ENSEMBLE - Giảm tương quan giữa learners
        self.train_negative_correlation_ensemble()
        
        # 6. HETEROGENEOUS ENSEMBLE - Vision + classical models
        self.train_heterogeneous_ensemble()
        
        # 7. MULTI-LEVEL DEEP ENSEMBLE - Implicit ensemble qua feature fusion
        self.train_multilevel_deep_ensemble()
        
        # Additional ensemble variants
        self.train_extra_trees()               # Extreme randomization ensemble
        
        print(f"\n=== HOÀN THÀNH HUẤN LUYỆN ===")
        print(f"Tổng số thuật toán đã huấn luyện: {len(self.trained_models)}")
        print("✅ Đã triển khai đầy đủ 7 kỹ thuật ensemble learning theo tài liệu nghiên cứu")
        print("✅ Kết hợp ResNet emotion detection + YOLO tail detection")
        print("✅ Pipeline ML hoàn chỉnh cho nhận diện cảm xúc chó")
        
        self.list_trained_models()
    
    def list_trained_models(self):
        """List all trained models."""
        if not self.trained_models:
            print("No models trained yet.")
            return
        
        print("\n=== Trained Models ===")
        for i, model_name in enumerate(self.trained_models.keys(), 1):
            print(f"{i:2d}. {model_name}")
    
    def get_model_info(self, model_name):
        """Get detailed information about a specific model."""
        if model_name not in self.trained_models:
            print(f"Model '{model_name}' not found.")
            return None
        
        model = self.trained_models[model_name]
        
        print(f"\n=== Model Information: {model_name} ===")
        print(f"Type: {type(model).__name__}")
        print(f"Parameters: {model.get_params()}")
        
        if hasattr(model, 'feature_importances_'):
            print(f"Feature Importances: {model.feature_importances_}")
        
        if hasattr(model, 'coef_'):
            print(f"Coefficients shape: {model.coef_.shape}")
        
        return model
    
    def predict_with_model(self, model_name, X=None):
        """
        Make predictions with a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the trained model
        X : array-like, optional
            Input features. If None, uses test data
            
        Returns:
        --------
        tuple
            (predictions, probabilities)
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        if X is None:
            if self.X_test is None:
                self.prepare_test_data()
            X = self.X_test
        
        model = self.trained_models[model_name]
        predictions = model.predict(X)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
        else:
            probabilities = None
        
        # Store predictions
        self.model_predictions[model_name] = {
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        return predictions, probabilities
    
    def evaluate_model(self, model_name):
        """Đánh giá một mô hình cụ thể trên dữ liệu test."""
        if self.X_test is None or self.y_test is None:
            self.prepare_test_data()
        
        predictions, _ = self.predict_with_model(model_name)
        accuracy = accuracy_score(self.y_test, predictions)
        
        print(f"\n=== Model Evaluation: {model_name} ===")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, predictions, 
                                  target_names=self.label_encoder.classes_))
        
        return accuracy
    
    def generate_meta_training_data(self):
        """
        Generate training data for meta-learner using predictions on test_for_train dataset.
        
        Returns:
        --------
        pd.DataFrame
            Dataset with original features plus model predictions
        """
        if self.test_for_train_data is None:
            raise ValueError("No test-for-train data loaded.")
        
        if not self.trained_models:
            raise ValueError("No models trained yet.")
        
        # Prepare test_for_train data
        self.prepare_test_for_train_data()
        
        # Start with original data
        meta_data = self.test_for_train_data.copy()
        
        # Add predictions from each model
        for model_name in self.trained_models.keys():
            predictions, probabilities = self.predict_with_model(model_name, self.X_test_for_train)
            
            if probabilities is not None:
                # Add probability columns for each class
                for i, class_name in enumerate(self.label_encoder.classes_):
                    col_name = f"{model_name}_{class_name}"
                    meta_data[col_name] = probabilities[:, i]
            else:
                # Add only prediction column
                meta_data[f"{model_name}_pred"] = predictions
        
        print(f"Generated meta-training data: {meta_data.shape}")
        print(f"Added predictions from {len(self.trained_models)} models")
        
        return meta_data
    
    def save_meta_training_data(self, output_path, format='csv'):
        """
        Save meta-training data to file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the file
        format : str, default='csv'
            Output format ('csv', 'xlsx')
        """
        meta_data = self.generate_meta_training_data()
        
        if format == 'csv':
            meta_data.to_csv(output_path, index=False)
        elif format == 'xlsx':
            meta_data.to_excel(output_path, index=False)
        else:
            raise ValueError("format must be 'csv' or 'xlsx'")
        
        print(f"Meta-training data saved to: {output_path}")
        return meta_data 