"""
Dog Emotion Recognition Hybrid ML Package - Demo Script for Jupyter Notebook

This script can be converted to a Jupyter notebook using the cell markers.
Each cell is marked with # %% for code cells and # %% [markdown] for markdown cells.
"""

# %% [markdown]
# # Dog Emotion Recognition Hybrid ML Package - Demo trên Google Colab
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hoangh-e/dog-emotion-recognition-hybrid/blob/main/demo_notebook.ipynb)
#
# Demo này sẽ hướng dẫn bạn cách sử dụng package Dog Emotion Recognition Hybrid ML để:
# - Huấn luyện nhiều thuật toán ML trên dữ liệu cảm xúc chó
# - Sử dụng meta-learner để chọn thuật toán tốt nhất
# - Thực hiện dự đoán và đánh giá kết quả

# %% [markdown]
# ## 1. Cài đặt và Thiết lập

# %%
# Clone repository từ GitHub
import subprocess
import sys
import os

print("Clone repository từ GitHub...")
result = subprocess.run(['git', 'clone', 'https://github.com/hoangh-e/dog-emotion-recognition-hybrid.git'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print("Clone thành công!")
else:
    print("Repository có thể đã tồn tại hoặc có lỗi clone")

# %%
# Chuyển vào thư mục project
os.chdir('dog-emotion-recognition-hybrid')
print("Thư mục hiện tại:", os.getcwd())

# %%
# Cài đặt các dependencies cần thiết
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'])
print("Hoàn thành cài đặt!")

# %%
# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dog_emotion_ml import EmotionMLClassifier, EnsembleMetaLearner
from dog_emotion_ml.utils import generate_sample_dataset, analyze_class_distribution

print("Đã import thành công các module!")

# %% [markdown]
# ## 2. Tạo Dữ liệu Mẫu
#
# Tạo dữ liệu mẫu để demo (trong thực tế bạn sẽ sử dụng dữ liệu thật từ ResNet và YOLO)

# %%
# Tạo dữ liệu mẫu
print("Tạo dữ liệu mẫu...")

# Tạo các dataset
train_data = generate_sample_dataset(n_samples=800, noise_level=0.1, random_state=42)
test_data = generate_sample_dataset(n_samples=200, noise_level=0.1, random_state=123)
test_for_train_data = generate_sample_dataset(n_samples=300, noise_level=0.1, random_state=456)

# Lưu vào file CSV
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
test_for_train_data.to_csv('test_for_train_data.csv', index=False)

print(f"Đã tạo dữ liệu:")
print(f"   - Training: {len(train_data)} mẫu")
print(f"   - Test: {len(test_data)} mẫu")
print(f"   - Test-for-train: {len(test_for_train_data)} mẫu")

# Hiển thị mẫu dữ liệu
print("\nMẫu dữ liệu training:")
print(train_data.head())

# %%
# Phân tích phân phối class
print("Phân tích phân phối cảm xúc trong dữ liệu training:")
try:
    class_stats = analyze_class_distribution(train_data, label_col='label', title="Phân phối cảm xúc - Training Data")
except:
    # Fallback nếu không có matplotlib
    class_counts = train_data['label'].value_counts()
    print("Phân phối cảm xúc:")
    for emotion, count in class_counts.items():
        percentage = (count / len(train_data)) * 100
        print(f"  {emotion}: {count} mẫu ({percentage:.1f}%)")

# %% [markdown]
# ## 3. Phần II: Huấn luyện các thuật toán ML
#
# Huấn luyện tất cả các thuật toán ML có sẵn trong package

# %%
# Khởi tạo classifier
print("Khởi tạo EmotionMLClassifier...")
classifier = EmotionMLClassifier(random_state=42)

# Tải dữ liệu
print("Tải dữ liệu...")
classifier.load_train_dataset('train_data.csv')
classifier.load_test_dataset('test_data.csv')
classifier.load_test_for_train_dataset('test_for_train_data.csv')

print("Đã tải xong dữ liệu!")

# %%
# Kiểm tra chất lượng dữ liệu
print("Kiểm tra chất lượng dữ liệu training...")
anomalies = classifier.display_anomalies_summary('train')

# Xử lý dữ liệu bất thường nếu có
if anomalies['summary']['total_missing'] > 0:
    print("Phát hiện dữ liệu thiếu, đang xử lý...")
    classifier.filter_missing_values('train', method='fill', fill_value=0.0)

if anomalies['summary']['total_invalid_probs'] > 0:
    print("Phát hiện xác suất không hợp lệ, đang xử lý...")
    classifier.filter_invalid_probabilities('train', method='clip')

# %%
# Huấn luyện các thuật toán ML
print("Bắt đầu huấn luyện các thuật toán ML...")
print("Đang huấn luyện các thuật toán cơ bản...")

# Huấn luyện tất cả 19 thuật toán ML
print("Đang huấn luyện tất cả 19 thuật toán ML...")
classifier.train_all_models()

print(f"\nĐã hoàn thành huấn luyện {len(classifier.trained_models)} thuật toán!")

# %%
# Hiển thị danh sách các mô hình đã huấn luyện
print("Danh sách các mô hình đã huấn luyện:")
classifier.list_trained_models()

# %%
# Đánh giá hiệu suất các mô hình
print("Đánh giá hiệu suất các mô hình trên dữ liệu test:")
print("="*50)

model_scores = {}
for model_name in classifier.trained_models.keys():
    try:
        accuracy = classifier.evaluate_model(model_name)
        model_scores[model_name] = accuracy
        print(f"\n{model_name}: {accuracy:.4f}")
    except Exception as e:
        print(f"\nLỗi đánh giá {model_name}: {e}")

# Hiển thị ranking
if model_scores:
    print("\nRanking các mô hình (theo độ chính xác):")
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (model_name, score) in enumerate(sorted_models, 1):
        print(f"  {i}. {model_name}: {score:.4f}")

# %% [markdown]
# ## 4. Tạo dữ liệu Meta-Training
#
# Tạo dữ liệu để huấn luyện meta-learner (Phần III)

# %%
# Tạo dữ liệu meta-training
print("Tạo dữ liệu meta-training...")
meta_data = classifier.generate_meta_training_data()

# Lưu dữ liệu meta-training
classifier.save_meta_training_data('meta_train_data.csv')

print(f"Đã tạo dữ liệu meta-training với shape: {meta_data.shape}")
print(f"Đã lưu vào file: meta_train_data.csv")

# Hiển thị mẫu dữ liệu meta
print("\nMẫu dữ liệu meta-training:")
print(meta_data.head())
print("\nCác cột trong dữ liệu meta:")
for i, col in enumerate(meta_data.columns):
    print(f"  {i+1:2d}. {col}")

# %% [markdown]
# ## 5. Phần III: Demo Tải Dữ Liệu cho Meta-Learning
#
# Demo cách tải dữ liệu cho meta-learner (chưa thực hiện huấn luyện)

# %%
# Khởi tạo meta-learner
print("Khởi tạo EnsembleMetaLearner...")
meta_learner = EnsembleMetaLearner(random_state=42)

# Tải dữ liệu meta-training
print("Tải dữ liệu meta-training...")
meta_learner.load_meta_training_data('meta_train_data.csv')

# Hiển thị thông tin dữ liệu đã tải
print(f"Dữ liệu meta-training đã tải:")
print(f"  - Kích thước: {meta_learner.meta_train_data.shape}")
print(f"  - Các thuật toán có sẵn: {meta_learner.available_algorithms}")
print(f"  - Số cột: {len(meta_learner.meta_train_data.columns)}")

# Hiển thị mẫu dữ liệu
print("\nMẫu dữ liệu meta-training (5 dòng đầu):")
print(meta_learner.meta_train_data.head())

print("\nLƯU Ý: Đây chỉ là demo tải dữ liệu cho Phần III.")
print("Trong thực tế, bạn sẽ tiếp tục với:")
print("  1. Phân tích hiệu suất thuật toán: meta_learner.analyze_algorithm_performance()")
print("  2. Chuẩn bị dữ liệu training: meta_learner.prepare_meta_training_data()")
print("  3. Huấn luyện meta-learner: meta_learner.train_meta_learner()")
print("  4. Đánh giá và sử dụng meta-learner cho dự đoán")

# %%
# Tạo meta-learner giả để demo có thể tiếp tục
print("Tạo meta-learner giả để demo có thể tiếp tục...")
meta_learner.analyze_algorithm_performance()
meta_learner.train_meta_learner(algorithm='DecisionTree', max_depth=5)
print("Meta-learner giả đã sẵn sàng cho demo!")

# %% [markdown]
# ## 6. Test và Demo Dự đoán
#
# Demo cách sử dụng meta-learner để dự đoán thuật toán tốt nhất

# %%
# Demo dự đoán với các mẫu cụ thể
print("Demo dự đoán với các mẫu cụ thể:")

# Các mẫu test
test_samples = [
    {
        'name': 'Chó vui vẻ với đuôi lên',
        'emotion': [0.1, 0.05, 0.8, 0.05],  # buồn, tức giận, vui vẻ, thư giãn
        'tail': [0.1, 0.8, 0.1]  # xuống, lên, giữa
    },
    {
        'name': 'Chó buồn với đuôi xuống',
        'emotion': [0.7, 0.1, 0.1, 0.1],
        'tail': [0.8, 0.1, 0.1]
    },
    {
        'name': 'Chó tức giận với đuôi cứng',
        'emotion': [0.1, 0.7, 0.1, 0.1],
        'tail': [0.2, 0.2, 0.6]
    },
    {
        'name': 'Chó thư giãn với đuôi giữa',
        'emotion': [0.1, 0.05, 0.1, 0.75],
        'tail': [0.1, 0.2, 0.7]
    }
]

for sample in test_samples:
    print(f"\n{sample['name']}:")
    print(f"   Đặc trưng cảm xúc: {sample['emotion']}")
    print(f"   Đặc trưng đuôi: {sample['tail']}")
    
    try:
        best_algo, confidence = meta_learner.predict_best_algorithm(
            sample['emotion'], sample['tail']
        )
        print(f"   Thuật toán được đề xuất: {best_algo}")
        
        if confidence is not None:
            print(f"   Độ tin cậy các thuật toán:")
            for i, algo in enumerate(meta_learner.algorithm_encoder.classes_):
                print(f"     {algo}: {confidence[i]:.4f}")
    except Exception as e:
        print(f"   Lỗi dự đoán: {e}")

# %% [markdown]
# ## 7. Lưu và Tải Mô Hình
#
# Demo cách lưu và tải lại meta-model

# %%
# Lưu meta-model
print("Lưu meta-model...")
meta_learner.save_meta_model('trained_meta_model.pkl')
print("Đã lưu meta-model!")

# Tải lại mô hình
print("\nTải lại meta-model...")
new_meta_learner = EnsembleMetaLearner()
new_meta_learner.load_meta_model('trained_meta_model.pkl')
print("Đã tải lại meta-model thành công!")

# Test mô hình đã tải
print("\nTest mô hình đã tải:")
emotion_features = [0.2, 0.1, 0.6, 0.1]
tail_features = [0.1, 0.7, 0.2]
best_algo, confidence = new_meta_learner.predict_best_algorithm(emotion_features, tail_features)
print(f"Thuật toán đề xuất: {best_algo}")
print("Model hoạt động bình thường sau khi tải lại!")

# %% [markdown]
# ## 8. Tóm Tắt Kết Quả
#
# Tóm tắt toàn bộ quy trình đã thực hiện

# %%
# Tóm tắt kết quả
print("\n" + "=" * 60)
print("TÓM TẮT KẾT QUẢ DEMO")
print("=" * 60)

print(f"\nPHẦN II - Huấn luyện ML:")
print(f"   Số thuật toán đã huấn luyện: {len(classifier.trained_models)}")
print(f"   Dữ liệu training: {len(train_data)} mẫu")
print(f"   Dữ liệu test: {len(test_data)} mẫu")

if model_scores:
    best_model = max(model_scores, key=model_scores.get)
    best_score = model_scores[best_model]
    print(f"   Mô hình tốt nhất: {best_model} ({best_score:.4f})")

print(f"\nPHẦN III - Meta-Learning:")
print(f"   Meta-learner đã được huấn luyện")
print(f"   Dữ liệu meta-training: {meta_data.shape[0]} mẫu")
print(f"   Có thể đề xuất thuật toán tối ưu cho từng mẫu")

print(f"\nCác thuật toán có sẵn: {meta_learner.available_algorithms}")

# %%
print(f"\nHOÀN THÀNH DEMO!")
print("\nWorkflow đã thực hiện:")
print("   1. Cài đặt package từ GitHub")
print("   2. Tạo dữ liệu mẫu")
print("   3. Huấn luyện nhiều thuật toán ML")
print("   4. Đánh giá hiệu suất các thuật toán")
print("   5. Tạo dữ liệu meta-training")
print("   6. Huấn luyện meta-learner")
print("   7. Test dự đoán thuật toán tối ưu")
print("   8. Lưu và tải lại mô hình")

print("\nPackage sẵn sàng sử dụng cho dự án thực tế!")

# %% [markdown]
# ## 9. Hướng Dẫn Sử Dụng Với Dữ Liệu Thực
#
# ### Chuẩn bị dữ liệu:
# ```
# 1. CHUẨN BỊ DỮ LIỆU:
#    - Định dạng file: CSV, TXT, hoặc XLSX
#    - Các cột bắt buộc: filename, sad, angry, happy, relaxed, down, up, mid, label
#    - Giá trị độ tin cậy: 0-1, tổng emotion ≈ 1, tổng tail ≈ 1
# 
# 2. HUẤN LUYỆN:
#    classifier = EmotionMLClassifier()
#    classifier.load_train_dataset('your_train_data.csv')
#    classifier.train_all_models()
# 
# 3. META-LEARNING:
#    meta_data = classifier.generate_meta_training_data()
#    meta_learner = EnsembleMetaLearner()
#    meta_learner.load_meta_training_data('meta_train_data.csv')
#    meta_learner.train_meta_learner()
# 
# 4. DỰ ĐOÁN:
#    emotion_features = [sad, angry, happy, relaxed]
#    tail_features = [down, up, mid]
#    best_algo, confidence = meta_learner.predict_best_algorithm(emotion_features, tail_features)
# 
# 5. LƯU MÔ HÌNH:
#    meta_learner.save_meta_model('my_model.pkl')
#    
# 6. SỬ DỤNG MÔ HÌNH ĐÃ LƯU:
#    meta_learner.load_meta_model('my_model.pkl')
# ``` 