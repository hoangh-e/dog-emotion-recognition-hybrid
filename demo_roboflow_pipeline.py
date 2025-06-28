"""
Demo script cho Roboflow Data Pipeline

Script này demo cách sử dụng các chức năng mới trong package dog_emotion_ml
để xử lý dữ liệu từ Roboflow và chuẩn hóa features theo yêu cầu.

Các tính năng được demo:
1. RoboflowDataProcessor - Xử lý dữ liệu từ Roboflow
2. DataNormalizer - Chuẩn hóa nâng cao (Z-score cho emotion, pass-through cho tail)
3. EmotionMLClassifier với advanced normalization
4. Tích hợp pipeline hoàn chỉnh
"""

import pandas as pd
import numpy as np
from pathlib import Path

def demo_roboflow_processor():
    """Demo RoboflowDataProcessor - Xử lý dữ liệu từ Roboflow"""
    print("=== Demo RoboflowDataProcessor ===")
    print("Chức năng: Xử lý dataset từ Roboflow, tích hợp YOLO + ResNet")
    
    from dog_emotion_ml import RoboflowDataProcessor, create_sample_roboflow_structure
    
    # 1. Tạo sample Roboflow structure để test
    sample_path = Path("./sample_roboflow_data")
    if not sample_path.exists():
        create_sample_roboflow_structure(sample_path)
        print(f"✅ Tạo cấu trúc Roboflow mẫu tại: {sample_path}")
    
    # 2. Khởi tạo processor (không có real models, sẽ dùng dummy data)
    processor = RoboflowDataProcessor(
        dataset_path=sample_path,
        yolo_tail_model_path=None,  # Sẽ dùng dummy confidence scores
        resnet_emotion_model_path=None  # Sẽ dùng dummy confidence scores
    )
    
    # 3. Demo lấy emotion labels từ data.yaml
    emotion_labels = processor.get_emotion_labels_from_yaml()
    print(f"📋 Emotion labels từ data.yaml: {emotion_labels}")
    
    # 4. Demo xử lý một ảnh (dummy)
    print("\n🖼️  Demo xử lý ảnh đơn lẻ:")
    dummy_image_path = "dummy_image.jpg"
    
    emotion_scores = processor.process_image_with_resnet_emotion(dummy_image_path)
    print(f"   Emotion scores: {emotion_scores}")
    
    tail_scores = processor.process_image_with_yolo_tail(dummy_image_path)
    print(f"   Tail scores: {tail_scores}")
    
    manual_label = processor.get_manual_label_from_filename(dummy_image_path)
    print(f"   Manual label: {manual_label}")
    
    print("✅ Demo RoboflowDataProcessor hoàn thành!")


def demo_data_normalizer():
    """Demo DataNormalizer - Chuẩn hóa nâng cao theo yêu cầu"""
    print("\n=== Demo DataNormalizer ===")
    print("Chức năng: Z-score cho emotion features, pass-through cho tail features")
    
    from dog_emotion_ml import DataNormalizer
    
    # 1. Tạo sample data
    np.random.seed(42)
    n_samples = 100
    
    # Emotion features (probabilities summing to ~1)
    emotion_data = np.random.dirichlet([1, 1, 1, 1], n_samples)
    print(f"📊 Dữ liệu emotion gốc:")
    print(f"   Shape: {emotion_data.shape}")
    print(f"   Sample:\n{emotion_data[:3]}")
    print(f"   Mean: {emotion_data.mean(axis=0)}")
    print(f"   Std: {emotion_data.std(axis=0)}")
    
    # Tail features (binary hoặc probabilities)
    tail_data = np.random.dirichlet([1, 1, 1], n_samples)
    print(f"\n📊 Dữ liệu tail gốc:")
    print(f"   Shape: {tail_data.shape}")
    print(f"   Sample:\n{tail_data[:3]}")
    
    # 2. Khởi tạo normalizer
    normalizer = DataNormalizer()
    
    # 3. Fit và transform
    emotion_norm, tail_norm = normalizer.fit_transform(emotion_data, tail_data)
    
    print(f"\n🔄 Sau chuẩn hóa:")
    print(f"   Emotion (Z-score) sample:\n{emotion_norm[:3]}")
    print(f"   Emotion mean: {emotion_norm.mean(axis=0)}")
    print(f"   Emotion std: {emotion_norm.std(axis=0)}")
    
    print(f"\n   Tail (pass-through) sample:\n{tail_norm[:3]}")
    print(f"   Tail mean: {tail_norm.mean(axis=0)}")
    
    # 4. Test inverse transform
    emotion_recovered = normalizer.inverse_transform_emotion(emotion_norm)
    recovery_error = np.mean(np.abs(emotion_data - emotion_recovered))
    print(f"\n🔍 Kiểm tra inverse transform:")
    print(f"   Recovery error: {recovery_error:.6f}")
    
    # 5. Demo normalize_dataset
    print(f"\n📋 Demo normalize_dataset:")
    sample_df = pd.DataFrame({
        'filename': [f'img_{i}.jpg' for i in range(10)],
        'sad': emotion_data[:10, 0],
        'angry': emotion_data[:10, 1], 
        'happy': emotion_data[:10, 2],
        'relaxed': emotion_data[:10, 3],
        'down': tail_data[:10, 0],
        'up': tail_data[:10, 1],
        'mid': tail_data[:10, 2],
        'label': ['happy'] * 10
    })
    
    print("   Dataset gốc:")
    print(sample_df.head())
    
    normalized_df = normalizer.normalize_dataset(sample_df, fit=False)  # Already fitted
    print("\n   Dataset sau chuẩn hóa:")
    print(normalized_df.head())
    
    print("✅ Demo DataNormalizer hoàn thành!")


def demo_emotion_ml_with_advanced_normalization():
    """Demo EmotionMLClassifier với advanced normalization"""
    print("\n=== Demo EmotionMLClassifier với Advanced Normalization ===")
    print("Chức năng: Tích hợp chuẩn hóa nâng cao vào pipeline ML")
    
    from dog_emotion_ml import EmotionMLClassifier
    
    # 1. Tạo sample dataset
    np.random.seed(42)
    n_samples = 200
    
    # Tạo sample data theo format yêu cầu
    emotion_data = np.random.dirichlet([1, 1, 1, 1], n_samples)
    tail_data = np.random.dirichlet([1, 1, 1], n_samples)
    labels = np.random.choice(['sad', 'angry', 'happy', 'relaxed'], n_samples)
    
    # Tạo DataFrame
    sample_df = pd.DataFrame({
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
    
    # Lưu sample data
    train_file = 'sample_train_data.csv'
    test_file = 'sample_test_data.csv'
    
    sample_df[:150].to_csv(train_file, index=False)
    sample_df[150:].to_csv(test_file, index=False)
    
    print(f"📁 Tạo dữ liệu mẫu:")
    print(f"   Training: {train_file} ({len(sample_df[:150])} mẫu)")
    print(f"   Test: {test_file} ({len(sample_df[150:])} mẫu)")
    
    # 2. Khởi tạo classifier
    classifier = EmotionMLClassifier(random_state=42)
    
    # 3. Tải dữ liệu
    classifier.load_train_dataset(train_file)
    classifier.load_test_dataset(test_file)
    
    # 4. Chuẩn bị dữ liệu với advanced normalization
    print(f"\n🔄 Chuẩn bị dữ liệu với advanced normalization...")
    classifier.prepare_training_data(use_advanced_normalization=True)
    classifier.prepare_test_data(use_advanced_normalization=True)
    
    # 5. Huấn luyện một vài models để test
    print(f"\n🤖 Huấn luyện models...")
    classifier.train_logistic_regression()
    classifier.train_random_forest()
    classifier.train_xgboost()
    
    # 6. Đánh giá models
    print(f"\n📊 Đánh giá models:")
    for model_name in classifier.trained_models:
        accuracy = classifier.evaluate_model(model_name)
        print(f"   {model_name}: {accuracy:.4f}")
    
    # 7. Demo normalize_features_advanced method
    print(f"\n🔧 Demo normalize_features_advanced method:")
    test_emotion = emotion_data[:5]
    test_tail = tail_data[:5]
    
    normalized_features = classifier.normalize_features_advanced(
        test_emotion, test_tail, fit=False
    )
    print(f"   Normalized features shape: {normalized_features.shape}")
    print(f"   Normalized features sample:\n{normalized_features[:2]}")
    
    # Clean up
    Path(train_file).unlink(missing_ok=True)
    Path(test_file).unlink(missing_ok=True)
    print(f"\n🧹 Đã dọn dẹp file tạm")
    print("✅ Demo EmotionMLClassifier hoàn thành!")


def demo_roboflow_integration():
    """Demo tích hợp với Roboflow (giả lập)"""
    print("\n=== Demo Roboflow Integration ===")
    print("Chức năng: Tích hợp hoàn chỉnh từ Roboflow đến ML training")
    
    from dog_emotion_ml import EmotionMLClassifier
    
    # Khởi tạo classifier
    classifier = EmotionMLClassifier(random_state=42)
    
    # Demo create_dataset_from_roboflow (sẽ fail vì không có real Roboflow data)
    print("🔗 Demo create_dataset_from_roboflow (simulation):")
    print("   Trong thực tế, method này sẽ:")
    print("   1. Tải dữ liệu từ Roboflow dataset")
    print("   2. Sử dụng YOLO model để detect tail status")
    print("   3. Sử dụng ResNet model để detect emotion")
    print("   4. Kết hợp với manual labels từ data.yaml")
    print("   5. Tạo dataset CSV với format chuẩn")
    
    # Giả lập kết quả
    print(f"\n📋 Kết quả mong đợi:")
    print("   - Dataset CSV với cột: filename, sad, angry, happy, relaxed, down, up, mid, label")
    print("   - Emotion features từ ResNet (confidence scores)")
    print("   - Tail features từ YOLO (binary detection)")
    print("   - Manual labels từ data.yaml")
    
    print("✅ Demo Roboflow Integration hoàn thành!")


def demo_complete_pipeline():
    """Demo pipeline hoàn chỉnh từ đầu đến cuối"""
    print("\n=== Demo Complete Pipeline ===")
    print("Pipeline hoàn chỉnh: Roboflow → Normalization → ML Training → Meta-Learning")
    
    from dog_emotion_ml import EmotionMLClassifier, EnsembleMetaLearner, demo_data_pipeline
    
    # Chạy demo pipeline
    demo_data_pipeline()
    
    print("✅ Demo Complete Pipeline hoàn thành!")


def main():
    """Chạy tất cả demos"""
    print("🚀 DOG EMOTION RECOGNITION - ROBOFLOW PIPELINE DEMO")
    print("=" * 60)
    print("Demo các tính năng mới trong package dog_emotion_ml v2.1.0:")
    print("- RoboflowDataProcessor: Xử lý dữ liệu từ Roboflow")
    print("- DataNormalizer: Chuẩn hóa nâng cao theo yêu cầu")
    print("- Advanced normalization trong EmotionMLClassifier")
    print("- Pipeline tích hợp hoàn chỉnh")
    print("=" * 60)
    
    try:
        # Demo từng component
        demo_roboflow_processor()
        demo_data_normalizer()
        demo_emotion_ml_with_advanced_normalization()
        demo_roboflow_integration()
        demo_complete_pipeline()
        
        print("\n" + "=" * 60)
        print("🎉 TẤT CẢ DEMOS ĐÃ HOÀN THÀNH THÀNH CÔNG!")
        print("Package dog_emotion_ml v2.1.0 đã sẵn sàng sử dụng!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình demo: {e}")
        print("Vui lòng kiểm tra lại cài đặt package và dependencies.")


if __name__ == "__main__":
    main() 