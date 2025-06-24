"""
Module cho Phần III: Lựa chọn thuật toán để nhận diện

Module này cung cấp chức năng huấn luyện meta-learner để chọn
thuật toán tốt nhất dựa trên đặc trưng cảm xúc ResNet và YOLO tail.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class EnsembleMetaLearner:
    """
    Lớp meta-learner cho việc lựa chọn thuật toán nhận diện cảm xúc chó.
    
    Huấn luyện meta-model để quyết định thuật toán ML nào sử dụng
    dựa trên đặc trưng đầu vào (ResNet emotion + YOLO tail detection).
    """
    
    def __init__(self, random_state=42):
        """
        Khởi tạo EnsembleMetaLearner.
        
        Parameters:
        -----------
        random_state : int, default=42
            Seed ngẫu nhiên để có kết quả nhất quán
        """
        self.random_state = random_state
        self.emotion_features = ['sad', 'angry', 'happy', 'relaxed']
        self.tail_features = ['down', 'up', 'mid']
        self.base_features = self.emotion_features + self.tail_features
        
        # Data storage
        self.meta_train_data = None
        self.meta_test_data = None
        
        # Processed data
        self.X_meta_train = None
        self.y_meta_train = None
        self.X_meta_test = None
        self.y_meta_test = None
        
        # Models and preprocessing
        self.meta_model = None
        self.algorithm_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        
        # Available algorithms from meta-training data
        self.available_algorithms = []
        self.algorithm_performance = {}
        
    def load_meta_training_data(self, file_path, filename_col='filename'):
        """
        Tải dữ liệu meta-training được tạo từ Phần II.
        
        Dữ liệu này chứa đặc trưng gốc cộng với dự đoán từ tất cả mô hình đã huấn luyện.
        
        Parameters:
        -----------
        file_path : str
            Đường dẫn tới file dữ liệu meta-training
        filename_col : str, default='filename'
            Tên cột chứa tên file
            
        Returns:
        --------
        pd.DataFrame
            Dataset meta-training đã tải
        """
        # Load data based on file format
        if file_path.endswith('.csv'):
            self.meta_train_data = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            self.meta_train_data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or XLSX.")
        
        # Identify available algorithms from column names
        self._identify_algorithms()
        
        print(f"Loaded meta-training data: {self.meta_train_data.shape}")
        print(f"Available algorithms: {self.available_algorithms}")
        
        return self.meta_train_data
    
    def load_meta_test_data(self, file_path, filename_col='filename'):
        """
        Tải dữ liệu meta-test để đánh giá meta-learner.
        
        Parameters:
        -----------
        file_path : str
            Đường dẫn tới file dữ liệu meta-test
        filename_col : str, default='filename'
            Tên cột chứa tên file
            
        Returns:
        --------
        pd.DataFrame
            Dataset meta-test đã tải
        """
        if file_path.endswith('.csv'):
            self.meta_test_data = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            self.meta_test_data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or XLSX.")
        
        print(f"Loaded meta-test data: {self.meta_test_data.shape}")
        return self.meta_test_data
    
    def _identify_algorithms(self):
        """Identify available algorithms from meta-training data columns."""
        if self.meta_train_data is None:
            raise ValueError("No meta-training data loaded.")
        
        # Get all columns except base features and label
        base_cols = [self.meta_train_data.columns[0]] + self.base_features + [self.meta_train_data.columns[-1]]
        prediction_cols = [col for col in self.meta_train_data.columns if col not in base_cols]
        
        # Extract algorithm names
        algorithms = set()
        for col in prediction_cols:
            if '_' in col:
                algorithm_name = col.split('_')[0]
                algorithms.add(algorithm_name)
        
        self.available_algorithms = sorted(list(algorithms))
    
    def analyze_algorithm_performance(self):
        """
        Analyze the performance of each algorithm on the meta-training data.
        
        This determines which algorithm performs best for each sample,
        creating the target labels for meta-learner training.
        
        Returns:
        --------
        dict
            Performance analysis results
        """
        if self.meta_train_data is None:
            raise ValueError("No meta-training data loaded.")
        
        # Get true labels
        true_labels = self.meta_train_data.iloc[:, -1].values
        
        # Get base features (emotion + tail)
        base_feature_cols = []
        for feature in self.base_features:
            for col in self.meta_train_data.columns:
                if feature.lower() in col.lower():
                    base_feature_cols.append(col)
                    break
        
        if len(base_feature_cols) != 7:
            raise ValueError(f"Expected 7 base features, found {len(base_feature_cols)}")
        
        # Analyze each sample to find best performing algorithm
        best_algorithms = []
        algorithm_confidences = []
        
        for idx in range(len(self.meta_train_data)):
            sample_confidences = {}
            true_label = true_labels[idx]
            
            # Check confidence for each algorithm
            for algorithm in self.available_algorithms:
                # Find columns for this algorithm
                algo_cols = [col for col in self.meta_train_data.columns 
                           if col.startswith(f"{algorithm}_")]
                
                if algo_cols:
                    # Get the confidence for the true class
                    true_class_col = None
                    for col in algo_cols:
                        if true_label.lower() in col.lower():
                            true_class_col = col
                            break
                    
                    if true_class_col:
                        confidence = self.meta_train_data.iloc[idx][true_class_col]
                        sample_confidences[algorithm] = confidence
            
            # Find algorithm with highest confidence for true class
            if sample_confidences:
                best_algo = max(sample_confidences, key=sample_confidences.get)
                best_confidence = sample_confidences[best_algo]
                
                best_algorithms.append(best_algo)
                algorithm_confidences.append(best_confidence)
            else:
                # Fallback to first available algorithm
                best_algorithms.append(self.available_algorithms[0])
                algorithm_confidences.append(0.5)
        
        # Store performance analysis
        self.algorithm_performance = {
            'best_algorithms': best_algorithms,
            'confidences': algorithm_confidences,
            'algorithm_distribution': pd.Series(best_algorithms).value_counts().to_dict()
        }
        
        print("\n=== Algorithm Performance Analysis ===")
        print("Algorithm distribution:")
        for algo, count in self.algorithm_performance['algorithm_distribution'].items():
            percentage = (count / len(best_algorithms)) * 100
            print(f"  {algo}: {count} samples ({percentage:.1f}%)")
        
        return self.algorithm_performance
    
    def prepare_meta_training_data(self):
        """
        Prepare data for training the meta-learner.
        
        Features: Original emotion + tail features (7 features)
        Labels: Best performing algorithm for each sample
        """
        if self.meta_train_data is None:
            raise ValueError("No meta-training data loaded.")
        
        if not self.algorithm_performance:
            self.analyze_algorithm_performance()
        
        # Extract base features (columns 1-7: emotions + tail)
        self.X_meta_train = self.meta_train_data.iloc[:, 1:8].values
        
        # Use best performing algorithms as labels
        self.y_meta_train = np.array(self.algorithm_performance['best_algorithms'])
        
        # Encode algorithm names
        self.y_meta_train = self.algorithm_encoder.fit_transform(self.y_meta_train)
        
        # Scale features
        self.X_meta_train = self.feature_scaler.fit_transform(self.X_meta_train)
        
        print(f"Meta-training data prepared: {self.X_meta_train.shape[0]} samples, {self.X_meta_train.shape[1]} features")
        print(f"Algorithm classes: {self.algorithm_encoder.classes_}")
    
    def prepare_meta_test_data(self):
        """
        Prepare meta-test data for evaluation.
        """
        if self.meta_test_data is None:
            raise ValueError("No meta-test data loaded.")
        
        # Extract base features
        self.X_meta_test = self.meta_test_data.iloc[:, 1:8].values
        
        # If test data has algorithm labels, extract them
        if len(self.meta_test_data.columns) > 8:
            # Assume similar structure and analyze performance
            true_labels = self.meta_test_data.iloc[:, -1].values
            
            # Simplified: use same analysis approach
            best_algorithms = []
            for idx in range(len(self.meta_test_data)):
                sample_confidences = {}
                true_label = true_labels[idx]
                
                for algorithm in self.available_algorithms:
                    algo_cols = [col for col in self.meta_test_data.columns 
                               if col.startswith(f"{algorithm}_")]
                    
                    if algo_cols:
                        true_class_col = None
                        for col in algo_cols:
                            if true_label.lower() in col.lower():
                                true_class_col = col
                                break
                        
                        if true_class_col:
                            confidence = self.meta_test_data.iloc[idx][true_class_col]
                            sample_confidences[algorithm] = confidence
                
                if sample_confidences:
                    best_algo = max(sample_confidences, key=sample_confidences.get)
                    best_algorithms.append(best_algo)
                else:
                    best_algorithms.append(self.available_algorithms[0])
            
            self.y_meta_test = self.algorithm_encoder.transform(best_algorithms)
        
        # Scale features
        self.X_meta_test = self.feature_scaler.transform(self.X_meta_test)
        
        print(f"Meta-test data prepared: {self.X_meta_test.shape[0]} samples")
    
    def train_meta_learner(self, algorithm='DecisionTree', **kwargs):
        """
        Train the meta-learner to select algorithms.
        
        Parameters:
        -----------
        algorithm : str, default='DecisionTree'
            Meta-learning algorithm ('DecisionTree', 'RandomForest', 'LogisticRegression')
        **kwargs
            Additional parameters for the algorithm
        """
        if self.X_meta_train is None:
            self.prepare_meta_training_data()
        
        if algorithm == 'DecisionTree':
            self.meta_model = DecisionTreeClassifier(
                random_state=self.random_state,
                **kwargs
            )
        elif algorithm == 'RandomForest':
            self.meta_model = RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=kwargs.get('n_estimators', 100),
                **{k: v for k, v in kwargs.items() if k != 'n_estimators'}
            )
        elif algorithm == 'LogisticRegression':
            self.meta_model = LogisticRegression(
                random_state=self.random_state,
                max_iter=kwargs.get('max_iter', 1000),
                **{k: v for k, v in kwargs.items() if k != 'max_iter'}
            )
        else:
            raise ValueError("Unsupported algorithm. Choose 'DecisionTree', 'RandomForest', or 'LogisticRegression'")
        
        self.meta_model.fit(self.X_meta_train, self.y_meta_train)
        
        # Evaluate using cross-validation
        cv_scores = cross_val_score(self.meta_model, self.X_meta_train, self.y_meta_train, cv=5)
        
        print(f"\n=== Meta-Learner Training Complete ===")
        print(f"Algorithm: {algorithm}")
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.meta_model
    
    def predict_best_algorithm(self, emotion_features, tail_features):
        """
        Predict the best algorithm for given features.
        
        Parameters:
        -----------
        emotion_features : array-like
            [sad, angry, happy, relaxed] confidence values
        tail_features : array-like
            [down, up, mid] confidence values
            
        Returns:
        --------
        tuple
            (predicted_algorithm, confidence_scores)
        """
        if self.meta_model is None:
            raise ValueError("Meta-learner not trained yet. Use train_meta_learner() first.")
        
        # Combine features
        features = np.array(emotion_features + tail_features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.feature_scaler.transform(features)
        
        # Predict
        prediction = self.meta_model.predict(features_scaled)[0]
        predicted_algorithm = self.algorithm_encoder.inverse_transform([prediction])[0]
        
        # Get confidence scores if available
        if hasattr(self.meta_model, 'predict_proba'):
            confidence_scores = self.meta_model.predict_proba(features_scaled)[0]
        else:
            confidence_scores = None
        
        return predicted_algorithm, confidence_scores
    
    def predict_best_algorithms_batch(self, X):
        """
        Predict best algorithms for a batch of samples.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix (n_samples, 7_features)
            
        Returns:
        --------
        tuple
            (predicted_algorithms, confidence_scores)
        """
        if self.meta_model is None:
            raise ValueError("Meta-learner not trained yet.")
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Predict
        predictions = self.meta_model.predict(X_scaled)
        predicted_algorithms = self.algorithm_encoder.inverse_transform(predictions)
        
        # Get confidence scores
        if hasattr(self.meta_model, 'predict_proba'):
            confidence_scores = self.meta_model.predict_proba(X_scaled)
        else:
            confidence_scores = None
        
        return predicted_algorithms, confidence_scores
    
    def evaluate_meta_learner(self):
        """
        Đánh giá meta-learner trên dữ liệu test.
        
        Returns:
        --------
        dict
            Kết quả đánh giá
        """
        if self.X_meta_test is None:
            self.prepare_meta_test_data()
        
        if self.y_meta_test is None:
            raise ValueError("No test labels available for evaluation.")
        
        # Predict
        predictions = self.meta_model.predict(self.X_meta_test)
        accuracy = accuracy_score(self.y_meta_test, predictions)
        
        # Detailed evaluation
        predicted_algorithms = self.algorithm_encoder.inverse_transform(predictions)
        true_algorithms = self.algorithm_encoder.inverse_transform(self.y_meta_test)
        
        results = {
            'accuracy': accuracy,
            'classification_report': classification_report(
                self.y_meta_test, predictions, 
                target_names=self.algorithm_encoder.classes_,
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(self.y_meta_test, predictions),
            'predicted_algorithms': predicted_algorithms,
            'true_algorithms': true_algorithms
        }
        
        print(f"\n=== Meta-Learner Evaluation ===")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_meta_test, predictions, 
                                  target_names=self.algorithm_encoder.classes_))
        
        return results
    
    def get_feature_importance(self):
        """
        Get feature importance from the meta-learner.
        
        Returns:
        --------
        dict
            Feature importance scores
        """
        if self.meta_model is None:
            raise ValueError("Meta-learner not trained yet.")
        
        if hasattr(self.meta_model, 'feature_importances_'):
            importance_scores = self.meta_model.feature_importances_
            feature_importance = dict(zip(self.base_features, importance_scores))
            
            print("\n=== Feature Importance ===")
            for feature, importance in sorted(feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True):
                print(f"{feature}: {importance:.4f}")
            
            return feature_importance
        else:
            print("Feature importance not available for this model type.")
            return None
    
    def get_algorithm_selection_rules(self, max_depth=3):
        """
        Extract decision rules from the meta-learner (if it's a decision tree).
        
        Parameters:
        -----------
        max_depth : int, default=3
            Maximum depth for rule extraction
            
        Returns:
        --------
        list
            List of decision rules
        """
        if not isinstance(self.meta_model, DecisionTreeClassifier):
            print("Rule extraction only available for DecisionTree meta-learner.")
            return None
        
        from sklearn.tree import export_text
        
        rules = export_text(
            self.meta_model,
            feature_names=self.base_features,
            class_names=self.algorithm_encoder.classes_,
            max_depth=max_depth
        )
        
        print("\n=== Algorithm Selection Rules ===")
        print(rules)
        
        return rules
    
    def analyze_algorithm_distribution(self):
        """
        Analyze the distribution of algorithm selections.
        
        Returns:
        --------
        dict
            Algorithm distribution analysis
        """
        if not self.algorithm_performance:
            raise ValueError("No algorithm performance analysis available.")
        
        distribution = self.algorithm_performance['algorithm_distribution']
        total_samples = sum(distribution.values())
        
        analysis = {
            'total_samples': total_samples,
            'algorithm_counts': distribution,
            'algorithm_percentages': {
                algo: (count / total_samples) * 100 
                for algo, count in distribution.items()
            }
        }
        
        print("\n=== Algorithm Selection Distribution ===")
        print(f"Total samples analyzed: {total_samples}")
        print("\nAlgorithm usage:")
        for algo in sorted(analysis['algorithm_percentages'].keys()):
            count = analysis['algorithm_counts'][algo]
            percentage = analysis['algorithm_percentages'][algo]
            print(f"  {algo}: {count} samples ({percentage:.1f}%)")
        
        return analysis
    
    def save_meta_model(self, model_path):
        """
        Save the trained meta-model.
        
        Parameters:
        -----------
        model_path : str
            Path to save the model
        """
        if self.meta_model is None:
            raise ValueError("No meta-model to save.")
        
        import joblib
        
        model_data = {
            'meta_model': self.meta_model,
            'algorithm_encoder': self.algorithm_encoder,
            'feature_scaler': self.feature_scaler,
            'available_algorithms': self.available_algorithms,
            'base_features': self.base_features
        }
        
        joblib.dump(model_data, model_path)
        print(f"Meta-model saved to: {model_path}")
    
    def load_meta_model(self, model_path):
        """
        Tải meta-model đã được huấn luyện.
        
        Parameters:
        -----------
        model_path : str
            Đường dẫn để tải model
        """
        import joblib
        
        model_data = joblib.load(model_path)
        
        self.meta_model = model_data['meta_model']
        self.algorithm_encoder = model_data['algorithm_encoder']
        self.feature_scaler = model_data['feature_scaler']
        self.available_algorithms = model_data['available_algorithms']
        self.base_features = model_data['base_features']
        
        print(f"Meta-model loaded from: {model_path}")
        print(f"Available algorithms: {self.available_algorithms}")
    
    def demonstrate_prediction(self, sample_features=None):
        """
        Demonstrate algorithm selection with sample features.
        
        Parameters:
        -----------
        sample_features : list, optional
            [sad, angry, happy, relaxed, down, up, mid] features
            If None, uses a random sample
        """
        if sample_features is None:
            # Generate random sample features
            sample_features = np.random.rand(7).tolist()
            print("Using random sample features:")
        else:
            print("Using provided sample features:")
        
        emotion_part = sample_features[:4]
        tail_part = sample_features[4:7]
        
        print(f"  Emotion features (sad, angry, happy, relaxed): {emotion_part}")
        print(f"  Tail features (down, up, mid): {tail_part}")
        
        # Predict best algorithm
        best_algo, confidence = self.predict_best_algorithm(emotion_part, tail_part)
        
        print(f"\nRecommended algorithm: {best_algo}")
        
        if confidence is not None:
            print("Algorithm confidence scores:")
            for i, algo in enumerate(self.algorithm_encoder.classes_):
                print(f"  {algo}: {confidence[i]:.4f}")
        
        return best_algo, confidence 