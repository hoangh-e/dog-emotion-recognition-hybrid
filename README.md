# Gói Machine Learning Hybrid cho Nhận Diện Cảm Xúc Chó

Gói machine learning chuyên nghiệp cho việc nhận diện cảm xúc chó với **7 kỹ thuật ensemble learning đầy đủ** theo tài liệu nghiên cứu, kết hợp đặc trưng từ ResNet emotion detection và YOLO tail detection.

## Tổng Quan

### Pipeline Machine Learning Hai Giai Đoạn

**Giai đoạn II**: Huấn luyện nhiều thuật toán ML với đầy đủ 7 kỹ thuật ensemble trên đặc trưng kết hợp từ ResNet emotion detection (buồn, tức giận, vui vẻ, thư giãn) và YOLO tail status detection (xuống, lên, giữa).

**Giai đoạn III**: Meta-learner lựa chọn thuật toán tối ưu cho mỗi dự đoán dựa trên đặc trưng đầu vào.

### 7 Kỹ Thuật Ensemble Learning Được Triển Khai

1. **Bagging** - Bootstrap Aggregating giảm phương sai thông qua bootstrap sampling
2. **Boosting** - XGBoost, AdaBoost, GradientBoosting, LightGBM học tập trung vào mẫu khó
3. **Stacking** - Meta-learner kết hợp predictions từ heterogeneous base models
4. **Voting** - Soft/Hard voting ensemble đơn giản hiệu quả
5. **Negative Correlation** - Giảm tương quan giữa learners để tăng diversity
6. **Heterogeneous** - Kết hợp vision models với classical ML models
7. **Multi-level Deep** - Implicit ensemble qua feature fusion và engineering

### Đặc Điểm Nổi Bật

- ✅ **Triển khai đầy đủ 7 kỹ thuật ensemble** theo tài liệu nghiên cứu
- ✅ **Pipeline chuyên nghiệp** với data validation và preprocessing
- ✅ **Tự động phát hiện và xử lý** các vấn đề dữ liệu
- ✅ **Hỗ trợ đa định dạng** (CSV, TXT, XLSX) với auto-detection
- ✅ **Cấu hình linh hoạt** cho từng kỹ thuật ensemble
- ✅ **Comments và documentation hoàn toàn bằng tiếng Việt**

## Cài Đặt

### Yêu Cầu Hệ Thống

```bash
# Core dependencies
pip install pandas>=1.3.0 numpy>=1.21.0 scikit-learn>=1.0.0 scipy>=1.7.0 joblib>=1.1.0

# Ensemble learning algorithms
pip install xgboost>=1.5.0 lightgbm>=3.3.0

# Data processing and visualization
pip install openpyxl>=3.0.0 matplotlib>=3.5.0 seaborn>=0.11.0
```

Hoặc cài đặt tất cả từ requirements.txt:

```bash
pip install -r requirements.txt
```

### Cài Đặt Gói

```bash
# Clone hoặc tải xuống gói
# Điều hướng đến thư mục gói
pip install -e .
```

## Sử Dụng Nhanh

```python
from dog_emotion_ml import EmotionMLClassifier, EnsembleMetaLearner

# Giai đoạn II: Huấn luyện nhiều mô hình ML
classifier = EmotionMLClassifier()
classifier.load_train_dataset('train_data.csv')
classifier.train_all_models()

# Tạo dữ liệu meta-training
meta_data = classifier.generate_meta_training_data()
classifier.save_meta_training_data('meta_train_data.csv')

# Giai đoạn III: Huấn luyện meta-learner
meta_learner = EnsembleMetaLearner()
meta_learner.load_meta_training_data('meta_train_data.csv')
meta_learner.train_meta_learner()

# Dự đoán thuật toán tốt nhất
emotion_features = [0.1, 0.8, 0.05, 0.05]  # buồn, tức giận, vui vẻ, thư giãn
tail_features = [0.2, 0.7, 0.1]  # xuống, lên, giữa
best_algo, confidence = meta_learner.predict_best_algorithm(emotion_features, tail_features)
print(f"Thuật toán được đề xuất: {best_algo}")
```

## Định Dạng Dữ Liệu

### Định Dạng Dataset Đầu Vào

File dữ liệu (CSV, TXT, hoặc XLSX) cần chứa các cột sau:

| Cột | Kiểu | Mô Tả | Ví Dụ |
|-----|------|-------|-------|
| filename | text | Tên file hoặc đường dẫn ảnh | "dog_001.jpg" |
| sad | float | Độ tin cậy cảm xúc buồn (0-1) | 0.15 |
| angry | float | Độ tin cậy cảm xúc tức giận (0-1) | 0.05 |
| happy | float | Độ tin cậy cảm xúc vui vẻ (0-1) | 0.75 |
| relaxed | float | Độ tin cậy cảm xúc thư giãn (0-1) | 0.05 |
| down | float | Độ tin cậy đuôi xuống (0-1) | 0.1 |
| up | float | Độ tin cậy đuôi lên (0-1) | 0.8 |
| mid | float | Độ tin cậy đuôi giữa (0-1) | 0.1 |
| label | text | Nhãn cảm xúc thực tế | "happy" |

**Lưu ý:**
- Tên cột không phân biệt hoa thường và hỗ trợ biến thể (ví dụ: "tail_down", "down_tail")
- Gói tự động phát hiện và ánh xạ tên cột
- Giá trị độ tin cậy cảm xúc nên tổng khoảng 1.0 cho mỗi mẫu
- Giá trị độ tin cậy đuôi nên tổng khoảng 1.0 cho mỗi mẫu

## Tài Liệu API

## Lớp EmotionMLClassifier

Lớp chính để huấn luyện nhiều thuật toán ML trên dữ liệu nhận diện cảm xúc chó.

### Constructor

```python
EmotionMLClassifier(random_state=42)
```

**Tham số:**
- `random_state` (int): Trạng thái ngẫu nhiên để có kết quả nhất quán

### Phương Thức Tải Dữ Liệu

#### load_train_dataset()

```python
load_train_dataset(file_path, filename_col='filename', emotion_cols=None, tail_cols=None, label_col='label')
```

Tải dataset huấn luyện từ file.

**Tham số:**
- `file_path` (str): Đường dẫn đến file dataset (CSV, TXT, hoặc XLSX)
- `filename_col` (str): Tên cột chứa tên file
- `emotion_cols` (list, tùy chọn): Danh sách tên cột cảm xúc (tự động phát hiện nếu None)
- `tail_cols` (list, tùy chọn): Danh sách tên cột đuôi (tự động phát hiện nếu None)
- `label_col` (str): Tên cột nhãn

**Trả về:**
- `pd.DataFrame`: Dataset đã tải và xử lý

#### load_test_dataset()

```python
load_test_dataset(file_path, filename_col='filename', emotion_cols=None, tail_cols=None, label_col='label')
```

Tải dataset test với các tham số giống `load_train_dataset()`.

#### load_test_for_train_dataset()

```python
load_test_for_train_dataset(file_path, filename_col='filename', emotion_cols=None, tail_cols=None, label_col='label')
```

Tải dataset để tạo dữ liệu meta-training với các tham số giống `load_train_dataset()`.

### Phương Thức Kiểm Tra Chất Lượng Dữ Liệu

#### check_data_anomalies()

```python
check_data_anomalies(dataset_name='train')
```

Kiểm tra bất thường dữ liệu bao gồm giá trị thiếu, xác suất không hợp lệ, và outliers.

**Tham số:**
- `dataset_name` (str): Dataset cần kiểm tra ('train', 'test', 'test_for_train')

**Trả về:**
- `dict`: Thông tin bất thường với đề xuất xử lý dữ liệu

#### display_anomalies_summary()

```python
display_anomalies_summary(dataset_name='train')
```

Hiển thị tóm tắt định dạng về bất thường dữ liệu.

**Tham số:**
- `dataset_name` (str): Dataset cần phân tích

### Phương Thức Lọc Dữ Liệu

#### filter_missing_values()

```python
filter_missing_values(dataset_name='train', method='drop', fill_value=0.0)
```

Lọc giá trị thiếu từ dataset.

**Tham số:**
- `dataset_name` (str): Dataset đích
- `method` (str): Phương thức xử lý ('drop' hoặc 'fill')
- `fill_value` (float): Giá trị điền khi method='fill'

#### filter_invalid_probabilities()

```python
filter_invalid_probabilities(dataset_name='train', method='clip')
```

Lọc giá trị xác suất không hợp lệ.

**Tham số:**
- `dataset_name` (str): Dataset đích
- `method` (str): Phương thức xử lý ('clip' hoặc 'drop')

#### filter_outliers()

```python
filter_outliers(dataset_name='train', method='iqr', factor=1.5)
```

Lọc outliers từ dataset.

**Tham số:**
- `dataset_name` (str): Dataset đích
- `method` (str): Phương thức phát hiện ('iqr' hoặc 'zscore')
- `factor` (float): Ngưỡng phát hiện

### Phương Thức Huấn Luyện Mô Hình

#### Huấn Luyện Thuật Toán Riêng Lẻ

```python
# Thuật toán cơ bản
train_logistic_regression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
train_svm(kernel='rbf', decision_function_shape='ovr', C=1.0)
train_decision_tree(max_depth=None, min_samples_split=2)
train_random_forest(n_estimators=100, max_depth=None)
train_xgboost(n_estimators=100, max_depth=6, learning_rate=0.1)
train_adaboost(n_estimators=50, learning_rate=1.0)
train_naive_bayes()
train_knn(n_neighbors=5)
train_lda()
train_qda()
train_mlp(hidden_layer_sizes=(100,), max_iter=500)

# Meta-classifiers
train_ovr_classifier(base_estimator_name='LogisticRegression')
train_ovo_classifier(base_estimator_name='LogisticRegression')

# Phương pháp ensemble
train_bagging_classifier(base_estimator_name='DecisionTree', n_estimators=10)
train_voting_classifier(voting='soft')
train_stacking_classifier(final_estimator_name='LogisticRegression')
```

#### train_all_models()

```python
train_all_models()
```

Huấn luyện tất cả mô hình có sẵn với tham số mặc định.

### Phương Thức Thông Tin Mô Hình

#### list_trained_models()

```python
list_trained_models()
```

Hiển thị danh sách tất cả mô hình đã huấn luyện.

#### get_model_info()

```python
get_model_info(model_name)
```

Lấy thông tin chi tiết về mô hình cụ thể.

**Tham số:**
- `model_name` (str): Tên mô hình đã huấn luyện

**Trả về:**
- Đối tượng mô hình với thông tin chi tiết

### Phương Thức Dự Đoán

#### predict_with_model()

```python
predict_with_model(model_name, X=None)
```

Thực hiện dự đoán với mô hình cụ thể.

**Tham số:**
- `model_name` (str): Tên mô hình đã huấn luyện
- `X` (array-like, tùy chọn): Đặc trưng đầu vào (sử dụng dữ liệu test nếu None)

**Trả về:**
- `tuple`: (predictions, probabilities)

#### evaluate_model()

```python
evaluate_model(model_name)
```

Đánh giá hiệu suất mô hình trên dữ liệu test.

**Tham số:**
- `model_name` (str): Tên mô hình đã huấn luyện

**Trả về:**
- `float`: Điểm độ chính xác

### Tạo Dữ Liệu Meta-Training

#### generate_meta_training_data()

```python
generate_meta_training_data()
```

Tạo dữ liệu huấn luyện cho meta-learner sử dụng dự đoán trên dataset test_for_train.

**Trả về:**
- `pd.DataFrame`: Dataset với đặc trưng gốc cộng dự đoán mô hình

#### save_meta_training_data()

```python
save_meta_training_data(output_path, format='csv')
```

Lưu dữ liệu meta-training vào file.

**Tham số:**
- `output_path` (str): Đường dẫn lưu file
- `format` (str): Định dạng đầu ra ('csv' hoặc 'xlsx')

## Lớp EnsembleMetaLearner

Lớp meta-learner để lựa chọn thuật toán dựa trên đặc trưng đầu vào.

### Constructor

```python
EnsembleMetaLearner(random_state=42)
```

**Tham số:**
- `random_state` (int): Trạng thái ngẫu nhiên để có kết quả nhất quán

### Phương Thức Tải Dữ Liệu

#### load_meta_training_data()

```python
load_meta_training_data(file_path, filename_col='filename')
```

Tải dữ liệu meta-training được tạo từ EmotionMLClassifier.

**Tham số:**
- `file_path` (str): Đường dẫn đến file dữ liệu meta-training
- `filename_col` (str): Tên cột filename

**Trả về:**
- `pd.DataFrame`: Dataset meta-training đã tải

#### load_meta_test_data()

```python
load_meta_test_data(file_path, filename_col='filename')
```

Tải dữ liệu meta-test để đánh giá.

### Phương Thức Phân Tích

#### analyze_algorithm_performance()

```python
analyze_algorithm_performance()
```

Phân tích hiệu suất của mỗi thuật toán để xác định lựa chọn tốt nhất.

**Trả về:**
- `dict`: Kết quả phân tích hiệu suất

#### analyze_algorithm_distribution()

```python
analyze_algorithm_distribution()
```

Phân tích phân phối lựa chọn thuật toán.

**Trả về:**
- `dict`: Phân tích phân phối thuật toán

### Phương Thức Huấn Luyện

#### train_meta_learner()

```python
train_meta_learner(algorithm='DecisionTree', **kwargs)
```

Huấn luyện meta-learner để lựa chọn thuật toán.

**Tham số:**
- `algorithm` (str): Thuật toán meta-learning ('DecisionTree', 'RandomForest', 'LogisticRegression')
- `**kwargs`: Tham số bổ sung cho thuật toán

**Trả về:**
- Meta-model đã huấn luyện

### Phương Thức Dự Đoán

#### predict_best_algorithm()

```python
predict_best_algorithm(emotion_features, tail_features)
```

Dự đoán thuật toán tốt nhất cho đặc trưng đã cho.

**Tham số:**
- `emotion_features` (array-like): Giá trị đặc trưng [buồn, tức giận, vui vẻ, thư giãn]
- `tail_features` (array-like): Giá trị đặc trưng [xuống, lên, giữa]

**Trả về:**
- `tuple`: (thuật toán dự đoán, điểm độ tin cậy)

#### predict_best_algorithms_batch()

```python
predict_best_algorithms_batch(X)
```

Dự đoán thuật toán tốt nhất cho một batch mẫu.

**Tham số:**
- `X` (array-like): Ma trận đặc trưng (n_samples, 7_features)

**Trả về:**
- `tuple`: (thuật toán dự đoán, điểm độ tin cậy)

### Phương Thức Đánh Giá

#### evaluate_meta_learner()

```python
evaluate_meta_learner()
```

Đánh giá meta-learner trên dữ liệu test.

**Trả về:**
- `dict`: Kết quả đánh giá

#### get_feature_importance()

```python
get_feature_importance()
```

Lấy tầm quan trọng đặc trưng từ meta-learner.

**Trả về:**
- `dict`: Điểm tầm quan trọng đặc trưng

#### get_algorithm_selection_rules()

```python
get_algorithm_selection_rules(max_depth=3)
```

Trích xuất quy tắc quyết định từ meta-learner (nếu là decision tree).

**Tham số:**
- `max_depth` (int): Độ sâu tối đa để trích xuất quy tắc

**Trả về:**
- `list`: Danh sách quy tắc quyết định

### Phương Thức Lưu/Tải Mô Hình

#### save_meta_model()

```python
save_meta_model(model_path)
```

Lưu meta-model đã huấn luyện.

**Tham số:**
- `model_path` (str): Đường dẫn lưu mô hình

#### load_meta_model()

```python
load_meta_model(model_path)
```

Tải meta-model đã huấn luyện.

**Tham số:**
- `model_path` (str): Đường dẫn tải mô hình

#### demonstrate_prediction()

```python
demonstrate_prediction(sample_features=None)
```

Demo lựa chọn thuật toán với đặc trưng mẫu.

**Tham số:**
- `sample_features` (list, tùy chọn): Đặc trưng [buồn, tức giận, vui vẻ, thư giãn, xuống, lên, giữa]

## Ví Dụ Sử Dụng Hoàn Chỉnh

### Ví Dụ 1: Pipeline Cơ Bản

```python
from dog_emotion_ml import EmotionMLClassifier, EnsembleMetaLearner

# Bước 1: Khởi tạo và tải dữ liệu
classifier = EmotionMLClassifier(random_state=42)
classifier.load_train_dataset('data/train.csv')
classifier.load_test_dataset('data/test.csv')
classifier.load_test_for_train_dataset('data/test_for_train.csv')

# Bước 2: Kiểm tra và làm sạch dữ liệu
classifier.display_anomalies_summary('train')
classifier.filter_missing_values('train', method='fill')
classifier.filter_invalid_probabilities('train', method='clip')

# Bước 3: Huấn luyện mô hình
classifier.train_all_models()

# Bước 4: Đánh giá mô hình
classifier.list_trained_models()
for model_name in classifier.trained_models:
    accuracy = classifier.evaluate_model(model_name)
    print(f"{model_name}: {accuracy:.4f}")

# Bước 5: Tạo dữ liệu meta-training
meta_data = classifier.generate_meta_training_data()
classifier.save_meta_training_data('meta_train.csv')

# Bước 6: Huấn luyện meta-learner
meta_learner = EnsembleMetaLearner(random_state=42)
meta_learner.load_meta_training_data('meta_train.csv')
meta_learner.train_meta_learner(algorithm='DecisionTree')

# Bước 7: Sử dụng meta-learner
emotion_features = [0.1, 0.2, 0.6, 0.1]  # buồn, tức giận, vui vẻ, thư giãn
tail_features = [0.1, 0.8, 0.1]  # xuống, lên, giữa
best_algo, confidence = meta_learner.predict_best_algorithm(emotion_features, tail_features)
print(f"Thuật toán được đề xuất: {best_algo}")
```

### Ví Dụ 2: Sử Dụng Với Dữ Liệu Thực

```python
import pandas as pd
from dog_emotion_ml import EmotionMLClassifier

# Tải dữ liệu từ file Excel
classifier = EmotionMLClassifier()
train_data = classifier.load_train_dataset('real_data.xlsx')

# Kiểm tra cấu trúc dữ liệu
print("Cấu trúc dữ liệu:")
print(train_data.info())
print("\nMẫu dữ liệu:")
print(train_data.head())

# Huấn luyện mô hình cụ thể
classifier.train_xgboost(n_estimators=200, max_depth=8)
classifier.train_random_forest(n_estimators=150)
classifier.train_svm(kernel='rbf', C=2.0)

# Đánh giá chi tiết
for model_name in classifier.trained_models:
    print(f"\n=== {model_name} ===")
    classifier.get_model_info(model_name)
    accuracy = classifier.evaluate_model(model_name)
```

### Ví Dụ 3: Batch Prediction

```python
import numpy as np
from dog_emotion_ml import EnsembleMetaLearner

# Tải meta-learner đã huấn luyện
meta_learner = EnsembleMetaLearner()
meta_learner.load_meta_model('trained_meta_model.pkl')

# Chuẩn bị dữ liệu batch
batch_features = np.array([
    [0.1, 0.2, 0.6, 0.1, 0.1, 0.8, 0.1],  # Mẫu 1
    [0.8, 0.1, 0.05, 0.05, 0.7, 0.2, 0.1],  # Mẫu 2
    [0.05, 0.05, 0.05, 0.85, 0.2, 0.1, 0.7],  # Mẫu 3
])

# Dự đoán batch
predicted_algos, confidences = meta_learner.predict_best_algorithms_batch(batch_features)

for i, (algo, conf) in enumerate(zip(predicted_algos, confidences)):
    print(f"Mẫu {i+1}: {algo} (độ tin cậy: {conf.max():.3f})")
```

## Cấu Trúc Thư Mục

```
dog-emotion-recognition-hybrid/
├── dog_emotion_ml/              # Gói chính
│   ├── __init__.py             # Khởi tạo gói
│   ├── emotion_ml.py           # Lớp EmotionMLClassifier
│   ├── ensemble_meta.py        # Lớp EnsembleMetaLearner
│   └── utils.py                # Các hàm tiện ích
├── Documents/                   # Tài liệu kỹ thuật
│   ├── khái quát.txt           # Tổng quan dự án
│   ├── Các thuật toán ML Multi-Classification.md
│   └── Các kỹ thuật Ensemble ML.md
├── Dog_Emotion_Recognition_Demo.ipynb  # Demo notebook
├── colab_demo.py               # Script demo cho Colab
├── example_usage.py            # Ví dụ sử dụng
├── requirements.txt            # Dependencies
├── setup.py                    # Cài đặt gói
└── README.md                   # Tài liệu này
```

## Thuật Toán Được Hỗ Trợ

### Thuật Toán Cơ Bản
- **Logistic Regression**: Hồi quy logistic đa lớp
- **Support Vector Machine**: SVM với kernel RBF/Linear
- **Decision Tree**: Cây quyết định
- **Random Forest**: Rừng ngẫu nhiên
- **XGBoost**: Gradient boosting tối ưu
- **AdaBoost**: Adaptive boosting
- **Naive Bayes**: Bayes ngây thơ Gaussian
- **K-Nearest Neighbors**: K láng giềng gần nhất
- **Linear/Quadratic Discriminant Analysis**: Phân tích phân biệt
- **Multi-layer Perceptron**: Mạng nơ-ron đa lớp

### Phương Pháp Meta-Classification
- **One-vs-Rest**: Một-so-với-còn-lại
- **One-vs-One**: Một-so-với-một

### Phương Pháp Ensemble
- **Bagging**: Bootstrap aggregating
- **Voting**: Bỏ phiếu cứng/mềm
- **Stacking**: Xếp chồng với meta-learner

## Tính Năng Nâng Cao

### Xử Lý Dữ Liệu
- Tự động phát hiện cột
- Kiểm tra chất lượng dữ liệu
- Lọc giá trị thiếu và outliers
- Chuẩn hóa xác suất

### Phân Tích Mô Hình
- Đánh giá hiệu suất chi tiết
- Tầm quan trọng đặc trưng
- Quy tắc quyết định
- Phân tích phân phối thuật toán

### Tiện Ích
- Tạo dữ liệu mẫu
- Trực quan hóa kết quả
- Xuất báo cáo Excel
- Lưu/tải mô hình

## Xử Lý Sự Cố

### Lỗi Thường Gặp

**1. ImportError: No module named 'dog_emotion_ml'**
```bash
# Cài đặt gói ở chế độ development
pip install -e .
```

**2. ValueError: Could not find column for emotion**
```python
# Chỉ định rõ tên cột
classifier.load_train_dataset('data.csv', 
                             emotion_cols=['sad_conf', 'angry_conf', 'happy_conf', 'relaxed_conf'],
                             tail_cols=['tail_down', 'tail_up', 'tail_mid'])
```

**3. Lỗi xác suất không hợp lệ**
```python
# Sử dụng filter để làm sạch dữ liệu
classifier.filter_invalid_probabilities('train', method='clip')
```

**4. Lỗi thiếu dữ liệu**
```python
# Xử lý giá trị thiếu
classifier.filter_missing_values('train', method='fill', fill_value=0.0)
```

### Tối Ưu Hiệu Suất

**1. Giảm số lượng thuật toán**
```python
# Chỉ huấn luyện các thuật toán chính
classifier.train_xgboost()
classifier.train_random_forest()
classifier.train_svm()
```

**2. Tối ưu tham số**
```python
# Sử dụng tham số tối ưu cho dữ liệu lớn
classifier.train_xgboost(n_estimators=50, max_depth=4)
classifier.train_random_forest(n_estimators=50)
```

**3. Sử dụng cross-validation**
```python
# Đánh giá mô hình với cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier.trained_models['XGBoost'], X, y, cv=5)
```

## Đóng Góp

Chúng tôi hoan nghênh các đóng góp từ cộng đồng. Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## Giấy Phép

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## Liên Hệ

- **Tác giả**: Dog Emotion Recognition Team
- **Email**: [email liên hệ]
- **GitHub**: https://github.com/hoangh-e/dog-emotion-recognition-hybrid

## Trích Dẫn

Nếu bạn sử dụng gói này trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@software{dog_emotion_recognition_hybrid,
  title={Dog Emotion Recognition Hybrid ML Package},
  author={Dog Emotion Recognition Team},
  year={2024},
  url={https://github.com/hoangh-e/dog-emotion-recognition-hybrid}
}
```

---

*Tài liệu này được cập nhật lần cuối: [Ngày hiện tại]*