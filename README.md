# Dog Emotion Recognition Hybrid System

Hệ thống nhận diện cảm xúc chó sử dụng kết hợp Deep Learning và Machine Learning với pipeline hoàn chỉnh từ YOLO head detection đến ResNet emotion classification và ensemble learning.

## 📋 Tổng quan

Dự án này triển khai hệ thống nhận diện cảm xúc chó toàn diện với:

1. **YOLO Head Detection**: Phát hiện vùng đầu chó trong ảnh
2. **Deep Learning Classification**: 16 họ thuật toán phân loại cảm xúc từ 2010-2025
3. **Machine Learning Pipeline**: 7 kỹ thuật ensemble learning
4. **Meta-Learning**: Tự động lựa chọn thuật toán tối ưu
5. **Production Pipeline**: Roboflow integration và deployment tools

### 🎯 Cảm xúc được nhận diện
- **Sad** (buồn)
- **Angry** (tức giận) 
- **Happy** (vui vẻ)
- **Relaxed** (thư giãn)

## ✨ Tính năng chính

### 🧠 Deep Learning Models

#### **CNN Kinh điển (2012-2016)**
- **AlexNet (2012)**: CNN 8 tầng, ReLU, dropout - mở đầu kỷ nguyên deep learning
- **VGGNet (2014)**: Bộ lọc 3×3 nhỏ, mạng sâu (VGG16, VGG19)
- **Inception/GoogLeNet (2014-2016)**: Module Inception song song (Inception v3, GoogLeNet)
- **ResNet (2015)**: Skip connections cho mạng cực sâu (ResNet50, ResNet101)
- **DenseNet (2017)**: Dense connectivity (DenseNet121, 169, 201)

#### **Mobile-Optimized Networks (2016-2019)**
- **SqueezeNet (2016)**: Module "Fire", 1/50 tham số AlexNet (SqueezeNet 1.0, 1.1)
- **MobileNet (2017-2019)**: Depthwise separable convolution (v2, v3 Large, v3 Small)
- **ShuffleNet (2018)**: Grouped conv + channel shuffle (v2 x0.5, x1.0, x1.5, x2.0)

#### **AutoML Architectures (2018-2019)**
- **EfficientNet (2019)**: Compound scaling (EfficientNet B0-B7)

#### **Transformer & Modern Models (2020-2025)**
- **Vision Transformer (ViT, 2020)**: Transformer cho ảnh patch (ViT-B/16, ViT-L/16, ViT-H/14)
- **Swin Transformer (2021)**: Shifted Windows attention (Swin-T, Swin-S, Swin-B, Swin v2)
- **ConvNeXt (2022)**: CNN hiện đại hóa (ConvNeXt Tiny, Small, Base, Large)
- **PURe (2025)**: Product units thay conv (PURe34, PURe50)

### 🤖 Machine Learning Pipeline

#### **7 Kỹ thuật Ensemble Learning**
1. **Bagging**: Bootstrap Aggregating giảm phương sai
2. **Boosting**: XGBoost, AdaBoost, GradientBoosting, LightGBM
3. **Stacking**: Meta-model kết hợp heterogeneous base models
4. **Voting**: Soft/Hard voting ensemble
5. **Negative Correlation**: Giảm tương quan giữa learners
6. **Heterogeneous**: Kết hợp vision + classical models  
7. **Multi-level Deep**: Implicit ensemble qua feature fusion

#### **Classical ML Algorithms**
- Logistic Regression (Multinomial, OvR, OvO)
- SVM (RBF, Linear)
- Decision Tree & Random Forest
- Naive Bayes & K-Nearest Neighbors
- Linear/Quadratic Discriminant Analysis
- Multi-layer Perceptron & Perceptron

### 🎯 Meta-Learning
- **Algorithm Selection**: Tự động chọn thuật toán tốt nhất
- **Feature-based Selection**: Dựa trên đặc trưng emotion + tail
- **Decision Tree Rules**: Quy tắc lựa chọn có thể diễn giải

### 🔧 Production Tools
- **Roboflow Integration**: Xử lý dataset tự động
- **Data Pipeline**: Chuẩn hóa và tiền xử lý nâng cao
- **Bbox Validation**: Kiểm tra chất lượng head detection
- **Colab Support**: Notebook demo và training

## 🚀 Cài đặt

### Yêu cầu hệ thống
```bash
pip install -r requirements.txt
```

### Dependencies chính
```txt
# Deep Learning Frameworks
torch>=1.12.0
torchvision>=0.13.0
timm>=0.6.0
transformers>=4.20.0

# Machine Learning
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0

# Computer Vision
opencv-python>=4.5.0
Pillow>=8.0.0

# Data Processing
pandas>=1.3.0
numpy>=1.21.0
```

## 💻 Sử dụng

### Package Usage

#### **Deep Learning Models**

```python
# AlexNet
from dog_emotion_classification.alexnet import load_alexnet_model, predict_emotion_alexnet
model, transform = load_alexnet_model('alexnet_emotion.pth')
result = predict_emotion_alexnet('dog_image.jpg', model, transform)

# VGG Networks
from dog_emotion_classification.vgg import load_vgg_model, predict_emotion_vgg
model, transform = load_vgg_model('vgg16_emotion.pth', architecture='vgg16')
result = predict_emotion_vgg('dog_image.jpg', model, transform)

# DenseNet
from dog_emotion_classification.densenet import load_densenet_model, predict_emotion_densenet
model, transform = load_densenet_model('densenet121_emotion.pth', architecture='densenet121')
result = predict_emotion_densenet('dog_image.jpg', model, transform)

# Inception/GoogLeNet
from dog_emotion_classification.inception import load_inception_model, predict_emotion_inception
model, transform = load_inception_model('inception_v3_emotion.pth', architecture='inception_v3')
result = predict_emotion_inception('dog_image.jpg', model, transform)

# MobileNet
from dog_emotion_classification.mobilenet import load_mobilenet_model, predict_emotion_mobilenet
model, transform = load_mobilenet_model('mobilenet_v2_emotion.pth', architecture='mobilenet_v2')
result = predict_emotion_mobilenet('dog_image.jpg', model, transform)

# EfficientNet
from dog_emotion_classification.efficientnet import load_efficientnet_model, predict_emotion_efficientnet
model, transform = load_efficientnet_model('efficientnet_b0_emotion.pth', architecture='efficientnet_b0')
result = predict_emotion_efficientnet('dog_image.jpg', model, transform)

# Vision Transformer
from dog_emotion_classification.vit import load_vit_model, predict_emotion_vit
model, transform = load_vit_model('vit_b_16_emotion.pth', architecture='vit_b_16')
result = predict_emotion_vit('dog_image.jpg', model, transform)

# ConvNeXt
from dog_emotion_classification.convnext import load_convnext_model, predict_emotion_convnext
model, transform = load_convnext_model('convnext_tiny_emotion.pth', architecture='convnext_tiny')
result = predict_emotion_convnext('dog_image.jpg', model, transform)

# SqueezeNet
from dog_emotion_classification.squeezenet import load_squeezenet_model, predict_emotion_squeezenet
model, transform = load_squeezenet_model('squeezenet1_0_emotion.pth', architecture='squeezenet1_0')
result = predict_emotion_squeezenet('dog_image.jpg', model, transform)

# ShuffleNet
from dog_emotion_classification.shufflenet import load_shufflenet_model, predict_emotion_shufflenet
model, transform = load_shufflenet_model('shufflenet_v2_x1_0_emotion.pth', architecture='shufflenet_v2_x1_0')
result = predict_emotion_shufflenet('dog_image.jpg', model, transform)

# Swin Transformer
from dog_emotion_classification.swin import load_swin_model, predict_emotion_swin
model, transform = load_swin_model('swin_t_emotion.pth', architecture='swin_t')
result = predict_emotion_swin('dog_image.jpg', model, transform)

# ResNet (existing)
from dog_emotion_classification.resnet import load_resnet_model, predict_emotion_resnet
model, transform = load_resnet_model('resnet50_emotion.pth', architecture='resnet50')
result = predict_emotion_resnet('dog_image.jpg', model, transform)

# PURe Networks (existing)
from dog_emotion_classification.pure import load_pure_model, predict_emotion_pure
model, transform = load_pure_model('pure34_emotion.pth', architecture='pure34')
result = predict_emotion_pure('dog_image.jpg', model, transform)
```

#### **Machine Learning Pipeline**

```python
from dog_emotion_ml import EmotionMLClassifier

# Initialize classifier
classifier = EmotionMLClassifier()

# Load datasets
classifier.load_train_dataset('train_data.csv')
classifier.load_test_dataset('test_data.csv')
classifier.load_test_for_train_dataset('test_for_train_data.csv')

# Train all algorithms with 7 ensemble techniques
classifier.train_all_models()

# Generate meta-training data for algorithm selection
meta_data = classifier.generate_meta_training_data()
classifier.save_meta_training_data('meta_training_data.csv')
```

#### **Meta-Learning Algorithm Selection**

```python
from dog_emotion_ml import EnsembleMetaLearner

# Initialize meta-learner
meta_learner = EnsembleMetaLearner()

# Load meta-training data
meta_learner.load_meta_training_data('meta_training_data.csv')

# Train meta-learner
meta_learner.train_meta_learner(algorithm='RandomForest')

# Predict best algorithm for new features
emotion_features = [0.8, 0.1, 0.05, 0.05]  # [sad, angry, happy, relaxed]
tail_features = [0.2, 0.7, 0.1]            # [down, up, mid]
best_algo, confidence = meta_learner.predict_best_algorithm(emotion_features, tail_features)
print(f"Recommended algorithm: {best_algo}")
```

#### **Data Pipeline & Roboflow Integration**

```python
from dog_emotion_ml import RoboflowDataProcessor

# Process Roboflow dataset
processor = RoboflowDataProcessor(
    dataset_path='path/to/roboflow/dataset',
    yolo_tail_model_path='yolo_tail.pt',
    resnet_emotion_model_path='resnet_emotion.pth'
)

# Create training dataset
dataset = processor.create_training_dataset('output_dataset.csv', split='train')
```

## 📊 Hiệu năng

### Deep Learning Models
- **16 họ thuật toán** từ AlexNet (2012) đến Swin Transformer (2021)
- **50+ biến thể kiến trúc** với input size và parameters khác nhau
- **Head bbox cropping** để tăng độ chính xác
- **Ensemble ready** tích hợp với ML pipeline

### Ensemble Learning
- **7 kỹ thuật ensemble** theo tài liệu nghiên cứu
- **20+ base algorithms** kết hợp đa dạng
- **Meta-learning selection** tự động chọn thuật toán
- **Cross-validation** đánh giá robust

## 🛠️ Requirements

```txt
# Core dependencies for dog emotion recognition ML package
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0
joblib>=1.1.0

# Ensemble learning algorithms
xgboost>=1.5.0
lightgbm>=3.3.0

# Data processing and visualization
openpyxl>=3.0.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Computer vision and image processing
opencv-python>=4.5.0
Pillow>=8.0.0

# Deep learning frameworks for classification algorithms
torch>=1.12.0
torchvision>=0.13.0
timm>=0.6.0
transformers>=4.20.0

# YOLO detection (optional for tail detection)
# ultralytics>=8.0.0

# YAML processing for Roboflow data.yaml
PyYAML>=6.0

# Optional dependencies for advanced features
# Install with: pip install -r requirements.txt 
cursor-notebook-mcp==0.2.3
```

## 📁 Cấu trúc dự án

```
dog-emotion-recognition-hybrid/
├── dog_emotion_classification/          # Deep Learning Models Package
│   ├── __init__.py
│   ├── alexnet.py                      # AlexNet (2012)
│   ├── vgg.py                          # VGGNet (2014)
│   ├── inception.py                    # Inception/GoogLeNet (2014-2016)
│   ├── resnet.py                       # ResNet (2015)
│   ├── densenet.py                     # DenseNet (2017)
│   ├── squeezenet.py                   # SqueezeNet (2016)
│   ├── mobilenet.py                    # MobileNet (2017-2019)
│   ├── shufflenet.py                   # ShuffleNet (2018)
│   ├── efficientnet.py                 # EfficientNet (2019)
│   ├── vit.py                          # Vision Transformer (2020)
│   ├── swin.py                         # Swin Transformer (2021)
│   ├── convnext.py                     # ConvNeXt (2022)
│   ├── pure.py                         # PURe Networks (2025)
│   ├── pure34.py                       # PURe34 specific
│   └── pure50.py                       # PURe50 specific
├── dog_emotion_ml/                     # ML Pipeline Package
│   ├── __init__.py
│   ├── emotion_ml.py                   # Main ML classifier
│   ├── ensemble_config.py              # Ensemble configurations
│   ├── ensemble_meta.py                # Meta-learning
│   ├── data_pipeline.py                # Data processing
│   └── utils.py                        # Utilities
├── server-stream/                      # Web Application
├── Documents/                          # Documentation
├── requirements.txt                    # Dependencies
└── README.md                          # This file
```

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) - Deep Learning Framework
- [Scikit-learn](https://scikit-learn.org/) - Machine Learning Library
- [Roboflow](https://roboflow.com/) - Computer Vision Platform
- [YOLO](https://github.com/ultralytics/ultralytics) - Object Detection
- [Timm](https://github.com/rwightman/pytorch-image-models) - PyTorch Image Models
