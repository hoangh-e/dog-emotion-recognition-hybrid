"""
Module Data Pipeline cho Dog Emotion Recognition

Module này cung cấp các chức năng để:
1. Download và xử lý dữ liệu từ Roboflow
2. Sử dụng YOLO và ResNet để tạo features
3. Chuẩn hóa dữ liệu đầu vào cho training và prediction
4. Tạo dataset với format phù hợp cho ML models
"""

import os
import yaml
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. YOLO functionality will be limited.")

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. ResNet functionality will be limited.")


class RoboflowDataProcessor:
    """
    Xử lý dữ liệu từ Roboflow dataset với format YOLOv12
    """
    
    def __init__(self, dataset_path, yolo_tail_model_path=None, resnet_emotion_model_path=None):
        """
        Khởi tạo processor
        
        Parameters:
        -----------
        dataset_path : str
            Đường dẫn đến thư mục dataset từ Roboflow
        yolo_tail_model_path : str, optional
            Đường dẫn đến model YOLO detect tail status
        resnet_emotion_model_path : str, optional
            Đường dẫn đến model ResNet emotion detection
        """
        self.dataset_path = Path(dataset_path)
        self.yolo_tail_model_path = yolo_tail_model_path
        self.resnet_emotion_model_path = resnet_emotion_model_path
        
        # Load data.yaml để lấy thông tin classes
        self.data_yaml_path = self.dataset_path / "data.yaml"
        self.class_info = self._load_data_yaml()
        
        # Initialize models
        self.yolo_tail_model = None
        self.resnet_emotion_model = None
        self._load_models()
        
        # Scalers cho chuẩn hóa
        self.emotion_scaler = StandardScaler()  # Z-score cho emotion features
        self.tail_scaler = None  # Không cần scaler cho binary features
        
    def _load_data_yaml(self):
        """Load thông tin từ data.yaml"""
        try:
            with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
                data_info = yaml.safe_load(f)
            return data_info
        except Exception as e:
            print(f"Error loading data.yaml: {e}")
            return None
    
    def _load_models(self):
        """Load các models YOLO và ResNet"""
        # Load YOLO tail detection model
        if self.yolo_tail_model_path and YOLO_AVAILABLE:
            try:
                self.yolo_tail_model = YOLO(self.yolo_tail_model_path)
                print("YOLO tail detection model loaded successfully")
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
        
        # Load ResNet emotion model
        if self.resnet_emotion_model_path and TORCH_AVAILABLE:
            try:
                self.resnet_emotion_model = torch.load(self.resnet_emotion_model_path, map_location='cpu')
                self.resnet_emotion_model.eval()
                print("ResNet emotion model loaded successfully")
            except Exception as e:
                print(f"Error loading ResNet model: {e}")
    
    def get_emotion_labels_from_yaml(self):
        """
        Trích xuất emotion labels từ data.yaml
        
        Returns:
        --------
        dict
            Mapping từ class names đến emotion labels
        """
        if not self.class_info or 'names' not in self.class_info:
            return {}
        
        # Mapping từ class names trong YAML đến emotion categories
        emotion_mapping = {}
        emotion_classes = ['angry', 'happy', 'relaxed', 'sad']
        tail_classes = ['down', 'up', 'mid']
        
        for class_name in self.class_info['names']:
            class_lower = class_name.lower()
            if class_lower in emotion_classes:
                emotion_mapping[class_name] = class_lower
            elif class_lower in tail_classes:
                emotion_mapping[class_name] = f"tail_{class_lower}"
        
        return emotion_mapping
    
    def process_image_with_yolo_tail(self, image_path):
        """
        Xử lý ảnh với YOLO để detect tail status
        
        Parameters:
        -----------
        image_path : str
            Đường dẫn đến ảnh
            
        Returns:
        --------
        dict
            Confident scores cho tail status: {'down': float, 'up': float, 'mid': float}
        """
        if not self.yolo_tail_model:
            # Return default values nếu không có model
            return {'down': 0.33, 'up': 0.33, 'mid': 0.34}
        
        try:
            # Run inference
            results = self.yolo_tail_model(image_path)
            
            # Extract confident scores cho tail classes
            tail_scores = {'down': 0.0, 'up': 0.0, 'mid': 0.0}
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        class_name = self.yolo_tail_model.names[class_id].lower()
                        
                        if class_name in tail_scores:
                            tail_scores[class_name] = max(tail_scores[class_name], confidence)
            
            # Normalize để tổng = 1
            total = sum(tail_scores.values())
            if total > 0:
                for key in tail_scores:
                    tail_scores[key] /= total
            else:
                # Default uniform distribution
                for key in tail_scores:
                    tail_scores[key] = 1.0 / len(tail_scores)
            
            return tail_scores
            
        except Exception as e:
            print(f"Error processing image with YOLO: {e}")
            return {'down': 0.33, 'up': 0.33, 'mid': 0.34}
    
    def process_image_with_resnet_emotion(self, image_path):
        """
        Xử lý ảnh với ResNet để detect emotion
        
        Parameters:
        -----------
        image_path : str
            Đường dẫn đến ảnh
            
        Returns:
        --------
        dict
            Confident scores cho emotions: {'sad': float, 'angry': float, 'happy': float, 'relaxed': float}
        """
        if not self.resnet_emotion_model:
            # Return default values nếu không có model
            return {'sad': 0.25, 'angry': 0.25, 'happy': 0.25, 'relaxed': 0.25}
        
        try:
            # Load và preprocess image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            
            # Convert to tensor
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                outputs = self.resnet_emotion_model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1).squeeze().numpy()
            
            # Map probabilities to emotion classes
            emotion_classes = ['sad', 'angry', 'happy', 'relaxed']
            emotion_scores = {}
            
            for i, emotion in enumerate(emotion_classes):
                if i < len(probabilities):
                    emotion_scores[emotion] = float(probabilities[i])
                else:
                    emotion_scores[emotion] = 0.0
            
            # Normalize để tổng = 1
            total = sum(emotion_scores.values())
            if total > 0:
                for key in emotion_scores:
                    emotion_scores[key] /= total
            else:
                # Default uniform distribution
                for key in emotion_scores:
                    emotion_scores[key] = 0.25
            
            return emotion_scores
            
        except Exception as e:
            print(f"Error processing image with ResNet: {e}")
            return {'sad': 0.25, 'angry': 0.25, 'happy': 0.25, 'relaxed': 0.25}
    
    def get_manual_label_from_filename(self, image_path):
        """
        Lấy manual label từ tên file hoặc từ annotation
        
        Parameters:
        -----------
        image_path : str
            Đường dẫn đến ảnh
            
        Returns:
        --------
        str
            Manual emotion label ('sad', 'angry', 'happy', 'relaxed')
        """
        # Thử lấy từ tên file trước
        filename = Path(image_path).stem.lower()
        
        emotion_keywords = {
            'sad': ['sad', 'buon', 'buồn'],
            'angry': ['angry', 'gian', 'giận', 'tuc', 'tức'],
            'happy': ['happy', 'vui', 'vui_ve', 'vui_vẻ'],
            'relaxed': ['relaxed', 'thu_gian', 'thư_giãn', 'binh_thuong', 'bình_thường']
        }
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in filename:
                    return emotion
        
        # Nếu không tìm thấy trong filename, thử từ class mapping trong YAML
        emotion_mapping = self.get_emotion_labels_from_yaml()
        
        # Thử tìm annotation file tương ứng
        annotation_path = str(image_path).replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt')
        
        if os.path.exists(annotation_path):
            try:
                with open(annotation_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Lấy class_id từ dòng đầu tiên
                        class_id = int(lines[0].split()[0])
                        if class_id < len(self.class_info['names']):
                            class_name = self.class_info['names'][class_id]
                            if class_name in emotion_mapping:
                                mapped_emotion = emotion_mapping[class_name]
                                if mapped_emotion in ['sad', 'angry', 'happy', 'relaxed']:
                                    return mapped_emotion
            except Exception as e:
                print(f"Error reading annotation file {annotation_path}: {e}")
        
        # Default fallback
        return 'happy'  # Default emotion nếu không xác định được
    
    def create_training_dataset(self, output_path, split='train'):
        """
        Tạo training dataset từ Roboflow data
        
        Parameters:
        -----------
        output_path : str
            Đường dẫn file output (CSV)
        split : str
            Split để xử lý ('train', 'val', 'test')
            
        Returns:
        --------
        pd.DataFrame
            Dataset đã tạo
        """
        # Đường dẫn đến thư mục images
        images_dir = self.dataset_path / split / 'images'
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Lấy danh sách tất cả ảnh
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f'*{ext}')))
            image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_files)} images in {images_dir}")
        
        # Tạo dataset
        dataset_rows = []
        
        for i, image_path in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {image_path.name}")
            
            try:
                # Get emotion features từ ResNet
                emotion_scores = self.process_image_with_resnet_emotion(image_path)
                
                # Get tail features từ YOLO
                tail_scores = self.process_image_with_yolo_tail(image_path)
                
                # Get manual label
                manual_label = self.get_manual_label_from_filename(image_path)
                
                # Tạo row cho dataset
                row = {
                    'filename': image_path.name,
                    'sad': emotion_scores['sad'],
                    'angry': emotion_scores['angry'],
                    'happy': emotion_scores['happy'],
                    'relaxed': emotion_scores['relaxed'],
                    'down': tail_scores['down'],
                    'up': tail_scores['up'],
                    'mid': tail_scores['mid'],
                    'label': manual_label
                }
                
                dataset_rows.append(row)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Tạo DataFrame
        df = pd.DataFrame(dataset_rows)
        
        # Lưu dataset
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to: {output_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Emotion distribution:\n{df['label'].value_counts()}")
        
        return df


class DataNormalizer:
    """
    Chuẩn hóa dữ liệu đầu vào cho training và prediction
    """
    
    def __init__(self):
        """Khởi tạo normalizer"""
        self.emotion_scaler = StandardScaler()  # Z-score cho emotion features
        self.tail_scaler = None  # Không cần scaler cho binary features
        self.is_fitted = False
        
        # Feature columns
        self.emotion_cols = ['sad', 'angry', 'happy', 'relaxed']
        self.tail_cols = ['down', 'up', 'mid']
        
    def fit(self, X_emotion, X_tail=None):
        """
        Fit scalers trên training data
        
        Parameters:
        -----------
        X_emotion : array-like, shape (n_samples, 4)
            Emotion features [sad, angry, happy, relaxed]
        X_tail : array-like, shape (n_samples, 3), optional
            Tail features [down, up, mid] - đã là binary hoặc probabilities
        """
        # Fit emotion scaler với Z-score normalization
        self.emotion_scaler.fit(X_emotion)
        
        # Tail features không cần fit scaler (binary hoặc probabilities)
        # Nhưng có thể dùng MaxAbsScaler nếu cần
        if X_tail is not None and X_tail.max() > 1.0:
            # Nếu tail values > 1, có thể cần MaxAbsScaler
            self.tail_scaler = MaxAbsScaler()
            self.tail_scaler.fit(X_tail)
        
        self.is_fitted = True
        print("DataNormalizer fitted successfully")
        print(f"Emotion features mean: {self.emotion_scaler.mean_}")
        print(f"Emotion features std: {self.emotion_scaler.scale_}")
        
    def transform(self, X_emotion, X_tail=None):
        """
        Transform features sử dụng fitted scalers
        
        Parameters:
        -----------
        X_emotion : array-like, shape (n_samples, 4)
            Emotion features để transform
        X_tail : array-like, shape (n_samples, 3), optional
            Tail features để transform
            
        Returns:
        --------
        tuple
            (normalized_emotion_features, normalized_tail_features)
        """
        if not self.is_fitted:
            raise ValueError("DataNormalizer must be fitted before transform")
        
        # Transform emotion features với Z-score
        X_emotion_normalized = self.emotion_scaler.transform(X_emotion)
        
        # Transform tail features (hoặc pass through nếu binary)
        if X_tail is not None:
            if self.tail_scaler is not None:
                X_tail_normalized = self.tail_scaler.transform(X_tail)
            else:
                # Pass through cho binary features
                X_tail_normalized = np.array(X_tail)
        else:
            X_tail_normalized = None
        
        return X_emotion_normalized, X_tail_normalized
    
    def fit_transform(self, X_emotion, X_tail=None):
        """
        Fit và transform trong một bước
        
        Parameters:
        -----------
        X_emotion : array-like, shape (n_samples, 4)
            Emotion features
        X_tail : array-like, shape (n_samples, 3), optional
            Tail features
            
        Returns:
        --------
        tuple
            (normalized_emotion_features, normalized_tail_features)
        """
        self.fit(X_emotion, X_tail)
        return self.transform(X_emotion, X_tail)
    
    def inverse_transform_emotion(self, X_emotion_normalized):
        """
        Inverse transform cho emotion features
        
        Parameters:
        -----------
        X_emotion_normalized : array-like
            Normalized emotion features
            
        Returns:
        --------
        array-like
            Original scale emotion features
        """
        if not self.is_fitted:
            raise ValueError("DataNormalizer must be fitted before inverse_transform")
        
        return self.emotion_scaler.inverse_transform(X_emotion_normalized)
    
    def normalize_dataset(self, df, fit=True):
        """
        Chuẩn hóa toàn bộ dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset với emotion và tail columns
        fit : bool, default=True
            Có fit scalers hay không (False cho test data)
            
        Returns:
        --------
        pd.DataFrame
            Dataset đã chuẩn hóa
        """
        df_normalized = df.copy()
        
        # Extract features
        X_emotion = df[self.emotion_cols].values
        X_tail = df[self.tail_cols].values if all(col in df.columns for col in self.tail_cols) else None
        
        if fit:
            # Fit và transform
            X_emotion_norm, X_tail_norm = self.fit_transform(X_emotion, X_tail)
        else:
            # Chỉ transform
            X_emotion_norm, X_tail_norm = self.transform(X_emotion, X_tail)
        
        # Update DataFrame
        df_normalized[self.emotion_cols] = X_emotion_norm
        if X_tail_norm is not None:
            df_normalized[self.tail_cols] = X_tail_norm
        
        return df_normalized
    
    def save_scalers(self, filepath):
        """
        Lưu scalers đã fit
        
        Parameters:
        -----------
        filepath : str
            Đường dẫn file để lưu scalers
        """
        if not self.is_fitted:
            raise ValueError("DataNormalizer must be fitted before saving")
        
        scaler_data = {
            'emotion_scaler': self.emotion_scaler,
            'tail_scaler': self.tail_scaler,
            'emotion_cols': self.emotion_cols,
            'tail_cols': self.tail_cols,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(scaler_data, filepath)
        print(f"Scalers saved to: {filepath}")
    
    def load_scalers(self, filepath):
        """
        Load scalers đã lưu
        
        Parameters:
        -----------
        filepath : str
            Đường dẫn file chứa scalers
        """
        scaler_data = joblib.load(filepath)
        
        self.emotion_scaler = scaler_data['emotion_scaler']
        self.tail_scaler = scaler_data['tail_scaler']
        self.emotion_cols = scaler_data['emotion_cols']
        self.tail_cols = scaler_data['tail_cols']
        self.is_fitted = scaler_data['is_fitted']
        
        print(f"Scalers loaded from: {filepath}")


def create_sample_roboflow_structure(base_path):
    """
    Tạo cấu trúc thư mục mẫu cho Roboflow dataset (để test)
    
    Parameters:
    -----------
    base_path : str
        Đường dẫn base để tạo structure
    """
    base_path = Path(base_path)
    
    # Tạo cấu trúc thư mục
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = base_path / split / 'images'
        labels_dir = base_path / split / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo data.yaml mẫu
    data_yaml_content = """train: ../train/images
val: ../valid/images
test: ../test/images

nc: 6
names: ['Angry', 'Down', 'Happy', 'Relaxed', 'Sad', 'Up']
"""
    
    with open(base_path / 'data.yaml', 'w') as f:
        f.write(data_yaml_content)
    
    print(f"Sample Roboflow structure created at: {base_path}")


def demo_data_pipeline():
    """
    Demo sử dụng data pipeline
    """
    print("=== Demo Data Pipeline ===")
    
    # 1. Tạo sample structure (nếu cần)
    sample_path = Path("./sample_roboflow_data")
    if not sample_path.exists():
        create_sample_roboflow_structure(sample_path)
    
    # 2. Initialize processor
    processor = RoboflowDataProcessor(
        dataset_path=sample_path,
        yolo_tail_model_path=None,  # Sẽ dùng dummy data
        resnet_emotion_model_path=None  # Sẽ dùng dummy data
    )
    
    # 3. Demo emotion labels từ YAML
    emotion_labels = processor.get_emotion_labels_from_yaml()
    print(f"Emotion labels from YAML: {emotion_labels}")
    
    # 4. Demo normalizer
    normalizer = DataNormalizer()
    
    # Tạo sample data
    np.random.seed(42)
    n_samples = 100
    
    # Emotion features (probabilities)
    emotion_data = np.random.dirichlet([1, 1, 1, 1], n_samples)
    
    # Tail features (binary hoặc probabilities)
    tail_data = np.random.dirichlet([1, 1, 1], n_samples)
    
    print(f"\nOriginal emotion data shape: {emotion_data.shape}")
    print(f"Original emotion data sample:\n{emotion_data[:3]}")
    print(f"Original emotion data mean: {emotion_data.mean(axis=0)}")
    print(f"Original emotion data std: {emotion_data.std(axis=0)}")
    
    # Fit và transform
    emotion_norm, tail_norm = normalizer.fit_transform(emotion_data, tail_data)
    
    print(f"\nNormalized emotion data sample:\n{emotion_norm[:3]}")
    print(f"Normalized emotion data mean: {emotion_norm.mean(axis=0)}")
    print(f"Normalized emotion data std: {emotion_norm.std(axis=0)}")
    
    # Test inverse transform
    emotion_recovered = normalizer.inverse_transform_emotion(emotion_norm)
    print(f"\nRecovered emotion data sample:\n{emotion_recovered[:3]}")
    print(f"Recovery error: {np.mean(np.abs(emotion_data - emotion_recovered))}")
    
    print("\n=== Demo completed successfully ===")


if __name__ == "__main__":
    demo_data_pipeline()
