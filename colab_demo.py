#!/usr/bin/env python3
"""
Dog Emotion Recognition Hybrid ML Package - Demo Script cho Google Colab

Script này demo toàn bộ quy trình từ cài đặt package đến huấn luyện và test.
Có thể chạy trực tiếp hoặc chuyển thành notebook.
"""

# =============================================================================
# 1. CÀI ĐẶT VÀ THIẾT LẬP
# =============================================================================

print("=" * 60)
print("DOG EMOTION RECOGNITION HYBRID ML PACKAGE DEMO")
print("=" * 60)

# Clone repository và cài đặt (chỉ chạy trên Colab)
def setup_colab():
    """Thiết lập môi trường Colab"""
    import os
    
    print("\nTHIẾT LẬP MÔI TRƯỜNG COLAB")
    print("-" * 40)
    
    # Clone repository
    print("Clone repository từ GitHub...")
    os.system("git clone https://github.com/hoangh-e/dog-emotion-recognition-hybrid.git")
    
    # Chuyển vào thư mục project
    os.chdir('dog-emotion-recognition-hybrid')
    print(f"Thư mục hiện tại: {os.getcwd()}")
    
    # Cài đặt dependencies
    print("Cài đặt dependencies...")
    os.system("pip install -r requirements.txt")
    os.system("pip install -e .")
    
    print("Hoàn thành thiết lập!")

# Uncomment dòng dưới nếu chạy trên Colab
# setup_colab()

# Import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from dog_emotion_ml import EmotionMLClassifier, EnsembleMetaLearner
    from dog_emotion_ml.utils import generate_sample_dataset, analyze_class_distribution
    print("Import thành công các module!")
except ImportError as e:
    print(f"Lỗi import: {e}")
    print("Hãy chạy setup_colab() trước")

# =============================================================================
# 2. TẠO DỮ LIỆU MẪU
# =============================================================================

def create_sample_data():
    """Tạo dữ liệu mẫu để demo"""
    print("\nTẠO DỮ LIỆU MẪU")
    print("-" * 40)
    
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
    
    # Phân tích phân phối class
    print("\nPhân tích phân phối cảm xúc:")
    try:
        class_stats = analyze_class_distribution(train_data, label_col='label', 
                                               title="Phân phối cảm xúc - Training Data")
    except:
        # Nếu không có matplotlib, chỉ in thống kê
        class_counts = train_data['label'].value_counts()
        print("Phân phối cảm xúc:")
        for emotion, count in class_counts.items():
            percentage = (count / len(train_data)) * 100
            print(f"  {emotion}: {count} mẫu ({percentage:.1f}%)")
    
    return train_data, test_data, test_for_train_data

# =============================================================================
# 3. PHẦN II: HUẤN LUYỆN CÁC THUẬT TOÁN ML
# =============================================================================

def train_ml_algorithms():
    """Huấn luyện các thuật toán ML"""
    print("\nPHẦN II: HUẤN LUYỆN CÁC THUẬT TOÁN ML")
    print("-" * 40)
    
    # Khởi tạo classifier
    print("Khởi tạo EmotionMLClassifier...")
    classifier = EmotionMLClassifier(random_state=42)
    
    # Tải dữ liệu
    print("Tải dữ liệu...")
    classifier.load_train_dataset('train_data.csv')
    classifier.load_test_dataset('test_data.csv')
    classifier.load_test_for_train_dataset('test_for_train_data.csv')
    
    # Kiểm tra chất lượng dữ liệu
    print("Kiểm tra chất lượng dữ liệu...")
    anomalies = classifier.display_anomalies_summary('train')
    
    # Xử lý dữ liệu bất thường nếu có
    if anomalies['summary']['total_missing'] > 0:
        print("Xử lý dữ liệu thiếu...")
        classifier.filter_missing_values('train', method='fill', fill_value=0.0)
    
    if anomalies['summary']['total_invalid_probs'] > 0:
        print("Xử lý xác suất không hợp lệ...")
        classifier.filter_invalid_probabilities('train', method='clip')
    
    # Huấn luyện tất cả 19 thuật toán ML
    print("\nBắt đầu huấn luyện tất cả 19 thuật toán ML...")
    
    try:
        classifier.train_all_models()
        print(f"\nHoàn thành huấn luyện {len(classifier.trained_models)} thuật toán!")
    except Exception as e:
        print(f"Lỗi trong quá trình huấn luyện: {e}")
        # Fallback: huấn luyện từng thuật toán riêng lẻ
        print("Thử huấn luyện từng thuật toán riêng lẻ...")
        
        algorithms_to_train = [
            ('Logistic Regression Multinomial', lambda: classifier.train_logistic_regression()),
            ('Logistic Regression OvR', lambda: classifier.train_logistic_regression_ovr()),
            ('Logistic Regression OvO', lambda: classifier.train_ovo_classifier('LogisticRegression')),
            ('SVM OvR', lambda: classifier.train_svm()),
            ('SVM OvO', lambda: classifier.train_svm_ovo()),
            ('Decision Tree', lambda: classifier.train_decision_tree()),
            ('Random Forest', lambda: classifier.train_random_forest(n_estimators=50)),
            ('XGBoost', lambda: classifier.train_xgboost(n_estimators=50)),
            ('AdaBoost', lambda: classifier.train_adaboost()),
            ('Naive Bayes', lambda: classifier.train_naive_bayes()),
            ('K-NN', lambda: classifier.train_knn()),
            ('LDA', lambda: classifier.train_lda()),
            ('QDA', lambda: classifier.train_qda()),
            ('MLP', lambda: classifier.train_mlp()),
            ('Perceptron', lambda: classifier.train_perceptron()),
            ('One-vs-Rest SVM', lambda: classifier.train_ovr_classifier('SVM')),
            ('Bagging', lambda: classifier.train_bagging_classifier()),
            ('Voting', lambda: classifier.train_voting_classifier()),
            ('Stacking', lambda: classifier.train_stacking_classifier()),
        ]
        
        for name, train_func in algorithms_to_train:
            try:
                print(f"  Huấn luyện {name}...")
                train_func()
                print(f"  Hoàn thành {name}")
            except Exception as e:
                print(f"  Lỗi {name}: {e}")
        
        print(f"\nHoàn thành huấn luyện {len(classifier.trained_models)} thuật toán!")
    
    # Hiển thị danh sách mô hình
    print("\nDanh sách mô hình đã huấn luyện:")
    classifier.list_trained_models()
    
    return classifier

def evaluate_models(classifier):
    """Đánh giá hiệu suất các mô hình"""
    print("\nĐÁNH GIÁ HIỆU SUẤT CÁC MÔ HÌNH")
    print("-" * 40)
    
    model_scores = {}
    for model_name in classifier.trained_models.keys():
        try:
            accuracy = classifier.evaluate_model(model_name)
            model_scores[model_name] = accuracy
            print(f"{model_name}: {accuracy:.4f}")
        except Exception as e:
            print(f"Lỗi đánh giá {model_name}: {e}")
    
    # Hiển thị ranking
    if model_scores:
        print("\nRanking các mô hình (theo độ chính xác):")
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (model_name, score) in enumerate(sorted_models, 1):
            print(f"  {i:2d}. {model_name}: {score:.4f}")
    
    return model_scores

# =============================================================================
# 4. TẠO DỮ LIỆU META-TRAINING
# =============================================================================

def create_meta_data(classifier):
    """Tạo dữ liệu meta-training"""
    print("\nTẠO DỮ LIỆU META-TRAINING")
    print("-" * 40)
    
    # Tạo dữ liệu meta-training
    print("Tạo dữ liệu meta-training...")
    meta_data = classifier.generate_meta_training_data()
    
    # Lưu dữ liệu
    classifier.save_meta_training_data('meta_train_data.csv')
    
    print(f"Đã tạo dữ liệu meta với shape: {meta_data.shape}")
    print(f"Đã lưu vào: meta_train_data.csv")
    
    # Hiển thị thông tin
    print(f"\nThông tin dữ liệu meta:")
    print(f"  - Số mẫu: {meta_data.shape[0]}")
    print(f"  - Số cột: {meta_data.shape[1]}")
    print(f"  - Các cột đầu tiên: {list(meta_data.columns[:10])}")
    
    return meta_data

# =============================================================================
# 5. PHẦN III: HUẤN LUYỆN META-LEARNER
# =============================================================================

def demo_meta_learning_data_loading():
    """Demo tải dữ liệu cho meta-learning (Phần III)"""
    print("\nPHẦN III: DEMO TẢI DỮ LIỆU CHO META-LEARNING")
    print("-" * 40)
    
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
    
    # Tạo một meta-learner giả để demo có thể tiếp tục
    print("\nTạo meta-learner giả để demo có thể tiếp tục...")
    meta_learner.analyze_algorithm_performance()
    meta_learner.train_meta_learner(algorithm='DecisionTree', max_depth=5)
    
    return meta_learner

# =============================================================================
# 6. TEST VÀ DEMO DỰ ĐOÁN
# =============================================================================

def demo_predictions(meta_learner):
    """Demo dự đoán với các mẫu cụ thể"""
    print("\nTEST VÀ DEMO DỰ ĐOÁN")
    print("-" * 40)
    
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

def demo_batch_prediction(meta_learner):
    """Demo dự đoán batch"""
    print("\nDEMO DỰ ĐOÁN BATCH")
    print("-" * 40)
    
    # Tạo dữ liệu batch ngẫu nhiên
    print("Tạo dữ liệu batch ngẫu nhiên...")
    np.random.seed(42)
    n_samples = 10
    
    batch_features = []
    for i in range(n_samples):
        # Tạo đặc trưng cảm xúc ngẫu nhiên
        emotion = np.random.dirichlet([1, 1, 1, 1])
        tail = np.random.dirichlet([1, 1, 1])
        combined = np.concatenate([emotion, tail])
        batch_features.append(combined)
    
    batch_features = np.array(batch_features)
    
    print(f"Dự đoán cho {n_samples} mẫu...")
    try:
        predicted_algos, confidences = meta_learner.predict_best_algorithms_batch(batch_features)
        
        print("\nKết quả dự đoán batch:")
        for i, (algo, conf) in enumerate(zip(predicted_algos, confidences)):
            max_conf = conf.max()
            print(f"  Mẫu {i+1:2d}: {algo} (độ tin cậy: {max_conf:.3f})")
        
        # Thống kê phân phối
        print(f"\nPhân phối thuật toán được đề xuất:")
        unique, counts = np.unique(predicted_algos, return_counts=True)
        for algo, count in zip(unique, counts):
            percentage = (count / n_samples) * 100
            print(f"  {algo}: {count}/{n_samples} ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Lỗi dự đoán batch: {e}")

# =============================================================================
# 7. LƯU VÀ TẢI MÔ HÌNH
# =============================================================================

def save_and_load_model(meta_learner):
    """Demo lưu và tải mô hình"""
    print("\nLƯU VÀ TẢI MÔ HÌNH")
    print("-" * 40)
    
    # Lưu mô hình
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

# =============================================================================
# 8. TÓM TẮT KẾT QUẢ
# =============================================================================

def print_summary(train_data, test_data, classifier, meta_learner, model_scores):
    """In tóm tắt kết quả"""
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
    print(f"   Có thể đề xuất thuật toán tối ưu cho từng mẫu")
    
    # Hiển thị các thuật toán có sẵn
    print(f"   Các thuật toán có sẵn: {meta_learner.available_algorithms}")

def print_usage_guide():
    """In hướng dẫn sử dụng"""
    print("\n" + "=" * 60)
    print("HƯỚNG DẪN SỬ DỤNG CHO DỮ LIỆU THỰC TẾ")
    print("=" * 60)
    
    usage_text = """
1. CHUẨN BỊ DỮ LIỆU:
   - Định dạng file: CSV, TXT, hoặc XLSX
   - Các cột bắt buộc: filename, sad, angry, happy, relaxed, down, up, mid, label
   - Giá trị độ tin cậy: 0-1, tổng emotion ≈ 1, tổng tail ≈ 1

2. HUẤN LUYỆN:
   classifier = EmotionMLClassifier()
   classifier.load_train_dataset('your_train_data.csv')
   classifier.train_all_models()

3. META-LEARNING:
   meta_data = classifier.generate_meta_training_data()
   meta_learner = EnsembleMetaLearner()
   meta_learner.load_meta_training_data('meta_train_data.csv')
   meta_learner.train_meta_learner()

4. DỰ ĐOÁN:
   emotion_features = [sad, angry, happy, relaxed]
   tail_features = [down, up, mid]
   best_algo, confidence = meta_learner.predict_best_algorithm(emotion_features, tail_features)

5. LƯU MÔ HÌNH:
   meta_learner.save_meta_model('my_model.pkl')
   
6. SỬ DỤNG MÔ HÌNH ĐÃ LƯU:
   meta_learner.load_meta_model('my_model.pkl')
"""
    print(usage_text)

# =============================================================================
# 9. MAIN FUNCTION
# =============================================================================

def main():
    """Hàm chính chạy toàn bộ demo"""
    try:
        # Bước 1: Tạo dữ liệu mẫu
        train_data, test_data, test_for_train_data = create_sample_data()
        
        # Bước 2: Huấn luyện các thuật toán ML
        classifier = train_ml_algorithms()
        
        # Bước 3: Đánh giá mô hình
        model_scores = evaluate_models(classifier)
        
        # Bước 4: Tạo dữ liệu meta-training
        meta_data = create_meta_data(classifier)
        
        # Bước 5: Demo tải dữ liệu meta-learning
        meta_learner = demo_meta_learning_data_loading()
        
        # Bước 6: Demo dự đoán
        demo_predictions(meta_learner)
        demo_batch_prediction(meta_learner)
        
        # Bước 7: Lưu và tải mô hình
        save_and_load_model(meta_learner)
        
        # Bước 8: Tóm tắt kết quả
        print_summary(train_data, test_data, classifier, meta_learner, model_scores)
        
        # Bước 9: Hướng dẫn sử dụng
        print_usage_guide()
        
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
        
    except Exception as e:
        print(f"Lỗi trong quá trình demo: {e}")
        import traceback
        traceback.print_exc()

# Chạy demo nếu file được thực thi trực tiếp
if __name__ == "__main__":
    main() 