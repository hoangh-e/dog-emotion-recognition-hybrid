# Pure Networks Module Guide

## Tổng quan

Module `dog_emotion_classification.pure` cung cấp implementation hoàn chỉnh của **Product Unit Residual Networks (Pure)** cho bài toán phân loại cảm xúc chó. Module hỗ trợ nhiều kiến trúc khác nhau và bao gồm đầy đủ các tính năng training và inference.

## Các Kiến trúc được Hỗ trợ

| Kiến trúc | Blocks | Mô tả | Parameters (4 classes) |
|-----------|--------|--------|------------------------|
| **Pure18** | [2, 2, 2, 2] | Basic Product Blocks | ~11.2M |
| **Pure34** | [3, 4, 6, 3] | Basic Product Blocks | ~21.3M |
| **Pure50** | [3, 4, 6, 3] | Bottleneck Product Blocks | ~23.5M |
| **Pure101** | [3, 4, 23, 3] | Bottleneck Product Blocks | ~42.5M |
| **Pure152** | [3, 8, 36, 3] | Bottleneck Product Blocks | ~58.2M |

## Cài đặt và Import

```python
from dog_emotion_classification.pure import (
    # Model architectures
    Pure18, Pure34, Pure50, Pure101, Pure152, get_pure_model,
    # Training utilities
    PureTrainer, get_pure_transforms,
    # Inference utilities
    load_pure_model, predict_emotion_pure,
    # Download utilities
    download_pure_model
)
```

## 1. Tạo Model

### Cách 1: Sử dụng Constructor trực tiếp

```python
# Tạo các model cụ thể
model = Pure34(num_classes=4, input_size=512)
model = Pure50(num_classes=4, input_size=512)
```

### Cách 2: Sử dụng Factory function (Khuyến nghị)

```python
# Linh hoạt hơn, có thể thay đổi architecture dễ dàng
model = get_pure_model('pure50', num_classes=4, input_size=512)
model = get_pure_model('pure101', num_classes=4, input_size=512)
```

## 2. Training

### Setup cơ bản

```python
import torch
from torch.utils.data import DataLoader
from dog_emotion_classification.pure import (
    get_pure_model, PureTrainer, get_pure_transforms
)

# 1. Tạo model
model = get_pure_model('pure50', num_classes=4, input_size=512)

# 2. Tạo trainer
trainer = PureTrainer(
    model=model,
    device='cuda',
    checkpoint_dir='checkpoints'
)

# 3. Setup training parameters
trainer.setup_training(
    learning_rate=1e-4,
    weight_decay=1e-4,
    step_size=10,
    gamma=0.1
)
```

### Chuẩn bị Data

```python
# Tạo transforms cho training và validation
train_transform = get_pure_transforms(input_size=512, is_training=True)
val_transform = get_pure_transforms(input_size=512, is_training=False)

# Tạo DataLoaders (sử dụng dataset của bạn)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
```

### Bắt đầu Training

```python
# Training với validation
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    save_best=True,
    save_interval=5
)

# Training chỉ với train set
trainer.train(
    train_loader=train_loader,
    epochs=30,
    save_best=True
)
```

### Load Checkpoint

```python
# Load checkpoint để tiếp tục training
trainer.load_checkpoint('checkpoints/best_model.pth')
```

## 3. Inference

### Load Model từ Checkpoint

```python
model, transform = load_pure_model(
    model_path='checkpoints/best_model.pth',
    architecture='pure50',
    num_classes=4,
    input_size=512,
    device='cuda'
)
```

### Prediction trên một ảnh

```python
result = predict_emotion_pure(
    image_path='path/to/image.jpg',
    model=model,
    transform=transform,
    head_bbox=None,  # Optional: [x1, y1, x2, y2]
    device='cuda',
    emotion_classes=['sad', 'angry', 'happy', 'relaxed']
)

# Result format
# {
#     'sad': 0.2,
#     'angry': 0.1,
#     'happy': 0.6,
#     'relaxed': 0.1,
#     'predicted': True
# }
```

### Batch Prediction

```python
import cv2
import torch
from PIL import Image

def predict_batch(image_paths, model, transform, device='cuda'):
    """Predict emotions for multiple images"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for img_path in image_paths:
            # Load and preprocess image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            
            # Transform and predict
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            
            # Map to emotion classes
            emotion_scores = {
                'sad': float(probs[0]),
                'angry': float(probs[1]), 
                'happy': float(probs[2]),
                'relaxed': float(probs[3])
            }
            results.append(emotion_scores)
    
    return results
```

## 4. Scripts có sẵn

### Training Scripts

```bash
# Train Pure34
python [train]_head_classification_pure34.py

# Train Pure50
python [train]_head_classification_pure50.py
```

### Demo Script

```bash
# Chạy demo để hiểu cách sử dụng module
python demo_pure_training.py
```

## 5. Ví dụ Hoàn chỉnh

### Training từ đầu

```python
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class DogEmotionDataset(Dataset):
    def __init__(self, root, labels_csv, transform=None):
        self.root = root
        df = pd.read_csv(labels_csv)
        self.items = df[['filename', 'label']].values
        
        unique_labels = sorted(df['label'].unique())
        self.label2index = {name: i for i, name in enumerate(unique_labels)}
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn, label_str = self.items[idx]
        label_idx = self.label2index[label_str]
        img_path = os.path.join(self.root, label_str, fn)
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label_idx

# Complete training workflow
def train_complete():
    # 1. Create model
    model = get_pure_model('pure50', num_classes=4, input_size=512)
    
    # 2. Setup trainer
    trainer = PureTrainer(model, device='cuda', checkpoint_dir='checkpoints')
    trainer.setup_training(learning_rate=1e-4, weight_decay=1e-4)
    
    # 3. Prepare data
    train_transform = get_pure_transforms(512, is_training=True)
    dataset = DogEmotionDataset('data/images', 'data/labels.csv', train_transform)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 4. Train
    trainer.train(train_loader=train_loader, epochs=30, save_best=True)
    
    print(f"✅ Training completed! Best accuracy: {trainer.best_acc:.4f}")

if __name__ == "__main__":
    train_complete()
```

### Inference và Evaluation

```python
def evaluate_model():
    # Load trained model
    model, transform = load_pure_model(
        model_path='checkpoints/best_model.pth',
        architecture='pure50',
        num_classes=4,
        input_size=512,
        device='cuda'
    )
    
    # Test on images
    test_images = ['test1.jpg', 'test2.jpg', 'test3.jpg']
    
    for img_path in test_images:
        result = predict_emotion_pure(img_path, model, transform)
        
        print(f"📷 {img_path}:")
        for emotion, score in result.items():
            if emotion != 'predicted':
                print(f"   {emotion}: {score:.3f}")
        print()

if __name__ == "__main__":
    evaluate_model()
```

## 6. Tips và Best Practices

### Model Selection

- **Pure18/Pure34**: Nhanh hơn, ít parameters, phù hợp cho prototype
- **Pure50**: Cân bằng tốt giữa accuracy và tốc độ
- **Pure101/Pure152**: Accuracy cao nhất nhưng chậm hơn

### Training Tips

```python
# Sử dụng learning rate scheduling
trainer.setup_training(
    learning_rate=1e-4,    # Bắt đầu với LR cao hơn
    weight_decay=1e-4,     # Regularization
    step_size=10,          # Giảm LR mỗi 10 epochs
    gamma=0.1              # Giảm LR xuống 10%
)

# Mixed precision training (tùy chọn)
# Cần thêm vào trainer nếu muốn tăng tốc
```

### Data Augmentation

```python
# Training transforms đã bao gồm augmentation phù hợp
train_transform = get_pure_transforms(input_size=512, is_training=True)
# - RandomResizedCrop
# - RandomHorizontalFlip  
# - ColorJitter
# - Normalization

# Inference không có augmentation
val_transform = get_pure_transforms(input_size=512, is_training=False)
```

### Memory Optimization

```python
# Giảm batch size nếu bị out of memory
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)  # Thay vì 8

# Sử dụng gradient accumulation
# Cần modify trainer để accumulate gradients qua nhiều batches
```

## 7. Troubleshooting

### Lỗi thường gặp

1. **Out of Memory**: Giảm batch_size hoặc input_size
2. **Model không converge**: Thử learning rate thấp hơn (1e-5)
3. **Import error**: Đảm bảo đã cài đặt đúng dependencies

### Dependencies

```bash
pip install torch torchvision tqdm pandas opencv-python pillow gdown
```

## 8. Tích hợp với Pipeline hiện tại

Để tích hợp với notebook `[Data] Tạo data cho ML.ipynb`:

```python
# Thay thế phần load model cũ
from dog_emotion_classification.pure import load_pure_model, predict_emotion_pure

# Load Pure model
pure_model, pure_transform = load_pure_model(
    model_path="pure50_best.pth",
    architecture="pure50", 
    num_classes=4,
    input_size=512
)

# Thay thế function prediction
def predict_emotion_classification(image_path, model, transform, head_bbox=None):
    return predict_emotion_pure(image_path, model, transform, head_bbox)
```

---

## Tài liệu tham khảo

- **Paper gốc**: "Deep residual learning with product units"
- **Based on ResNet**: He et al. "Deep Residual Learning for Image Recognition"
- **Product Units**: Multiplication-based feature interactions thay vì addition

Với module này, bạn có thể dễ dàng thử nghiệm với các kiến trúc Pure khác nhau và tối ưu hóa model cho bài toán phân loại cảm xúc chó! 