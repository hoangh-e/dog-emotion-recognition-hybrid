# 🚀 Multi-Model Support Package Update Guide

## 📦 **ĐÃ CẬP NHẬT PACKAGE:**

### ✅ **Files được thêm/cập nhật:**
1. **`dog_emotion_classification/pure50.py`** - Pure50 wrapper functions
2. **`dog_emotion_classification/resnet.py`** - ResNet50/101 support
3. **`dog_emotion_classification/__init__.py`** - Import đầy đủ các functions

### ✅ **Functions mới available:**

#### **Pure50 Support:**
```python
from dog_emotion_classification import (
    load_pure50_model, 
    predict_emotion_pure50,
    get_pure50_transforms,
    create_pure50_model
)
```

#### **ResNet Support:**
```python
from dog_emotion_classification import (
    load_resnet_emotion_model,    # Generic ResNet loader
    predict_emotion_resnet,       # Generic ResNet prediction
    load_resnet50_model,          # ResNet50 specific
    load_resnet101_model,         # ResNet101 specific
    predict_emotion_resnet50,     # ResNet50 prediction
    predict_emotion_resnet101,    # ResNet101 prediction
    get_resnet_transforms,        # ResNet transforms
    create_resnet_emotion_model   # Create ResNet model
)
```

## 🔧 **CÁCH SỬ DỤNG TRONG NOTEBOOK:**

### **Bước 1: Cập nhật Cell 7 - Model Loading**

Thay thế cell hiện tại bằng:

```python
# ==========================================
# CELL 7: Load Emotion Models (UPDATED)
# ==========================================

# Import ALL model functions
from dog_emotion_classification import (
    load_pure34_model, predict_emotion_pure34,
    load_pure50_model, predict_emotion_pure50,
    load_resnet_emotion_model, predict_emotion_resnet
)

# Storage for loaded models
loaded_models = {}
model_transforms = {}
model_predict_functions = {}

print("🔄 Loading emotion classification models...")
print("=" * 50)

for model_name, config in ENABLED_MODELS.items():
    try:
        print(f"Loading {model_name} ({config['type']})...")
        
        if config['type'] == 'pure34':
            model, transform = load_pure34_model(
                model_path=config['path'],
                num_classes=len(config['classes']),
                device=device
            )
            predict_func = predict_emotion_pure34
            
        elif config['type'] == 'pure50':
            model, transform = load_pure50_model(
                model_path=config['path'],
                num_classes=len(config['classes']),
                input_size=config['input_size'],
                device=device
            )
            predict_func = predict_emotion_pure50
            
        elif config['type'] == 'resnet':
            model, transform = load_resnet_emotion_model(
                model_path=config['path'],
                architecture=config['architecture'],
                num_classes=len(config['classes']),
                input_size=config['input_size'],
                device=device
            )
            predict_func = predict_emotion_resnet
            
        else:
            print(f"❌ Unknown model type: {config['type']}")
            continue
            
        # Store successfully loaded models
        loaded_models[model_name] = model
        model_transforms[model_name] = transform
        model_predict_functions[model_name] = predict_func
        print(f"✅ {model_name} loaded successfully")
            
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")

# Update multi-model mode
MULTI_MODEL_MODE = len(loaded_models) >= 2
print(f"📊 Loaded {len(loaded_models)} models - Multi-model: {MULTI_MODEL_MODE}")
```

### **Bước 2: Cập nhật Main Processing Loop**

Thay thế main processing loop để support multiple models:

```python
# Results storage for each model
model_results = {model_name: [] for model_name in loaded_models.keys()}

for i, image_path in enumerate(image_files):
    try:
        # Shared detection (head + tail)
        head_result = predict_head_detection(image_path, yolo_head_model)
        tail_result = predict_tail_detection(image_path, yolo_tail_model)
        
        if not head_result['detected'] or not tail_result['detected']:
            continue
            
        manual_label = get_manual_label_from_filename(image_path)
        
        # Process with ALL loaded models
        for model_name in loaded_models.keys():
            try:
                model = loaded_models[model_name]
                transform = model_transforms[model_name]
                predict_func = model_predict_functions[model_name]
                
                emotion_result = predict_func(
                    image_path=image_path,
                    model=model,
                    transform=transform,
                    head_bbox=head_result['bbox'],
                    device=device
                )
                
                if emotion_result['predicted']:
                    row = {
                        'filename': image_path.name,
                        'sad': emotion_result['sad'],
                        'angry': emotion_result['angry'],
                        'happy': emotion_result['happy'],
                        'relaxed': emotion_result['relaxed'],
                        'down': tail_result['down'],
                        'up': tail_result['up'],
                        'mid': tail_result['mid'],
                        'label': manual_label,
                        'head_confidence': head_result['confidence'],
                        'model_name': model_name
                    }
                    model_results[model_name].append(row)
                    
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                
    except Exception as e:
        print(f"❌ Error processing {image_path.name}: {e}")
```

### **Bước 3: Cập nhật Data Saving**

```python
# Save separate CSV for each model
MODEL_OUTPUTS = {}

for model_name, results in model_results.items():
    if len(results) > 0:
        df = pd.DataFrame(results)
        
        # Normalize probabilities
        emotion_cols = ['sad', 'angry', 'happy', 'relaxed']
        emotion_sums = df[emotion_cols].sum(axis=1)
        for col in emotion_cols:
            df[col] = df[col] / emotion_sums
            
        # Save files
        raw_csv = f"{OUTPUT_DIR}/raw_predictions_{model_name}.csv"
        processed_csv = f"{OUTPUT_DIR}/processed_dataset_{model_name}.csv"
        
        df.to_csv(raw_csv, index=False)
        df.to_csv(processed_csv, index=False)
        
        MODEL_OUTPUTS[model_name] = {
            'raw_csv': raw_csv,
            'processed_csv': processed_csv
        }
        
        print(f"✅ {model_name}: {len(df)} predictions saved")
```

## 🎯 **KẾT QUẢ MONG ĐỢI:**

Sau khi cập nhật, bạn sẽ có:

### ✅ **Multi-model loading:**
```
✅ pure34_30e loaded successfully (Pure34)
✅ pure50_30e loaded successfully (Pure50)  
✅ pure50_50e loaded successfully (Pure50)
✅ resnet50_50e loaded successfully (RESNET50)
✅ resnet50_30e loaded successfully (RESNET50)
✅ resnet101_30e loaded successfully (RESNET101)
```

### ✅ **Multi-model processing:**
```
📊 Loaded 6 models - Multi-model: True
🔄 Starting multi-model image processing pipeline...
   pure34_30e     : 479 predictions
   pure50_30e     : 479 predictions
   pure50_50e     : 479 predictions
   resnet50_50e   : 479 predictions
   resnet50_30e   : 479 predictions
   resnet101_30e  : 479 predictions
```

### ✅ **Output files:**
```
📂 PURE34_30E: ✅ raw_csv, ✅ processed_csv
📂 PURE50_30E: ✅ raw_csv, ✅ processed_csv
📂 PURE50_50E: ✅ raw_csv, ✅ processed_csv
📂 RESNET50_50E: ✅ raw_csv, ✅ processed_csv
📂 RESNET50_30E: ✅ raw_csv, ✅ processed_csv
📂 RESNET101_30E: ✅ raw_csv, ✅ processed_csv
📂 COMPARISON: ✅ comparison_csv, ✅ analysis_json
```

## 🚨 **LƯU Ý QUAN TRỌNG:**

1. **Model files phải tồn tại:** Đảm bảo tất cả model files đã được download
2. **Input size khác nhau:** Pure models dùng 512x512, ResNet dùng 224x224
3. **Architecture matching:** ResNet config phải có `architecture` field
4. **Memory usage:** Loading nhiều models cùng lúc sẽ tốn RAM/VRAM

## 🔧 **TROUBLESHOOTING:**

### **Nếu model không load được:**
```python
# Debug model loading
for model_name, config in EMOTION_MODELS.items():
    print(f"{model_name}: {config['enabled']} - {os.path.exists(config['path'])}")
```

### **Nếu muốn force enable multi-model:**
```python
MULTI_MODEL_MODE = True  # Force enable for testing
```

### **Nếu memory không đủ:**
```python
# Load models tuần tự thay vì cùng lúc
SEQUENTIAL_PROCESSING = True
``` 