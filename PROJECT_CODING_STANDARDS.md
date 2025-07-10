# Project Coding Standards & Error Handling Guidelines

## 🚫 **STRICT NO-FALLBACK POLICY**

### **Nguyên tắc cốt lõi: "FAIL FAST, FAIL CLEAR"**

Tất cả code trong project này PHẢI tuân thủ nguyên tắc **"No Silent Failures"** - không được phép che giấu lỗi bằng cách trả về dữ liệu giả hoặc fallback data.

---

## 1. **ERROR HANDLING REQUIREMENTS**

### ✅ **ĐƯỢC PHÉP:**
```python
# ✅ Raise exception với thông báo rõ ràng
def predict_emotion(image_path, model):
    try:
        result = model.predict(image_path)
        return result
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        raise RuntimeError(f"Model prediction failed: {e}")

# ✅ Return None với log warning khi optional operation fails
def extract_head_bbox(image_path):
    try:
        bbox = detect_head(image_path)
        return bbox
    except Exception as e:
        print(f"⚠️  Head detection failed: {e}")
        return None  # OK vì đây là optional feature

# ✅ Validate input và raise exception
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Continue with loading...
```

### 🚫 **NGHIÊM CẤM:**
```python
# 🚫 KHÔNG BAO GIỜ làm thế này
def predict_emotion(image_path, model):
    try:
        result = model.predict(image_path)
        return result
    except Exception as e:
        print(f"Error: {e}")
        # ❌ NGHIÊM CẤM: Trả về dữ liệu giả
        return {
            'angry': 0.25,
            'happy': 0.25, 
            'relaxed': 0.25,
            'sad': 0.25
        }

# 🚫 KHÔNG BAO GIỜ tạo fallback model
def load_model(model_path):
    try:
        return torch.load(model_path)
    except Exception:
        # ❌ NGHIÊM CẤM: Tạo model giả
        return create_dummy_model()

# 🚫 KHÔNG BAO GIỜ sử dụng random data
def get_prediction():
    try:
        return real_prediction()
    except Exception:
        # ❌ NGHIÊM CẤM: Random data
        return np.random.rand(4)
```

---

## 2. **EXCEPTION HANDLING STANDARDS**

### **2.1 Exception Types**
```python
# Sử dụng exception types phù hợp
FileNotFoundError(f"Model file not found: {path}")
ValueError(f"Invalid architecture: {arch}")
RuntimeError(f"Model prediction failed: {error}")
ImportError(f"Required package not installed: {package}")
```

### **2.2 Error Messages**
```python
# ✅ Error message format chuẩn
def handle_error(operation, error):
    print(f"❌ Error in {operation}: {error}")
    raise RuntimeError(f"{operation} failed: {error}")

# ✅ Thông báo chi tiết với context
raise ValueError(f"Unsupported architecture '{arch}'. Supported: {supported_archs}")
```

### **2.3 Logging Standards**
```python
# ✅ Logging levels
print(f"✅ Success: {message}")          # Success
print(f"⚠️  Warning: {message}")         # Warning  
print(f"❌ Error: {message}")            # Error
print(f"🔄 Loading: {message}")          # Progress
print(f"🏗️  Created: {message}")         # Creation
print(f"📦 Loading from: {message}")     # Data loading
```

---

## 3. **FUNCTION DESIGN PRINCIPLES**

### **3.1 Single Responsibility**
```python
# ✅ Mỗi function chỉ làm một việc
def load_model(model_path):
    """Only loads the model, doesn't handle prediction"""
    
def predict_emotion(image, model):
    """Only handles prediction, doesn't load model"""
    
def preprocess_image(image_path):
    """Only handles image preprocessing"""
```

### **3.2 Clear Return Types**
```python
# ✅ Return type rõ ràng
def load_model(model_path) -> Tuple[nn.Module, transforms.Compose]:
    """Returns (model, transform) or raises exception"""
    
def predict_emotion(image, model) -> Dict[str, float]:
    """Returns emotion scores or raises exception"""
```

### **3.3 Input Validation**
```python
# ✅ Validate inputs ngay đầu function
def predict_emotion(image_path, model, transform):
    if not isinstance(image_path, (str, Image.Image)):
        raise TypeError(f"Invalid image_path type: {type(image_path)}")
    
    if model is None:
        raise ValueError("Model cannot be None")
        
    if transform is None:
        raise ValueError("Transform cannot be None")
```

---

## 4. **TESTING STANDARDS**

### **4.1 No Dummy Data in Tests**
```python
# ✅ Test với real data hoặc proper mocks
def test_model_loading():
    with pytest.raises(FileNotFoundError):
        load_model("nonexistent_path.pth")

# 🚫 KHÔNG sử dụng dummy data
def test_prediction():
    # ❌ NGHIÊM CẤM
    dummy_input = torch.randn(1, 3, 224, 224)
```

### **4.2 Test Error Conditions**
```python
# ✅ Test các error conditions
def test_invalid_architecture():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        load_model("path.pth", architecture="invalid_arch")
```

---

## 5. **CODE REVIEW CHECKLIST**

### **5.1 Mandatory Checks**
- [ ] Không có fallback data (0.0, 0.25, random values)
- [ ] Không có dummy/fake models
- [ ] Tất cả exceptions được raise properly
- [ ] Error messages có context rõ ràng
- [ ] Không có silent failures
- [ ] Input validation đầy đủ
- [ ] Return types consistent

### **5.2 Red Flags - REJECT CODE**
- [ ] `return {emotion: 0.0 for emotion in emotions}`
- [ ] `torch.randn()` trong production code
- [ ] `create_dummy_*()` functions
- [ ] `except: pass` hoặc `except: return default`
- [ ] Hardcoded probability values
- [ ] Random number generation cho predictions

---

## 6. **ENFORCEMENT**

### **6.1 Automated Checks**
```bash
# Tìm kiếm các pattern cấm
grep -r "0\.25.*for.*emotion" .
grep -r "0\.0.*for.*emotion" .
grep -r "torch\.randn" .
grep -r "create.*dummy" .
grep -r "random.*prediction" .
```

### **6.2 Manual Review**
- Mọi PR phải được review bởi ít nhất 1 người
- Reviewer phải check compliance với standards này
- Không merge code vi phạm standards

### **6.3 Consequences**
- **First violation**: Code review rejection + explanation
- **Repeated violations**: Mandatory training on error handling
- **Persistent violations**: Code access restriction

---

## 7. **EXAMPLES OF PROPER ERROR HANDLING**

### **7.1 Model Loading**
```python
def load_model(model_path, architecture, num_classes, device):
    # Validate inputs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    try:
        # Attempt to load
        checkpoint = torch.load(model_path, map_location=device)
        model = create_model(architecture, num_classes)
        model.load_state_dict(checkpoint, strict=True)
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model from {model_path}: {e}")
        raise RuntimeError(f"Model loading failed: {e}")
```

### **7.2 Prediction**
```python
def predict_emotion(image_path, model, transform, device):
    # Validate inputs
    if not isinstance(image_path, (str, Image.Image)):
        raise TypeError(f"Invalid image type: {type(image_path)}")
    
    try:
        # Load and preprocess
        image = load_image(image_path)
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Return real results
        return {
            emotion: float(prob) 
            for emotion, prob in zip(EMOTION_CLASSES, probs)
        }
        
    except Exception as e:
        print(f"❌ Prediction failed for {image_path}: {e}")
        raise RuntimeError(f"Emotion prediction failed: {e}")
```

---

## 8. **SUMMARY**

### **🎯 Key Principles:**
1. **Fail Fast**: Detect errors early and raise exceptions immediately
2. **Fail Clear**: Provide detailed error messages with context
3. **No Silent Failures**: Never hide errors with fallback data
4. **Real Data Only**: No dummy, fake, or random data in production
5. **Explicit Validation**: Check all inputs and preconditions
6. **Consistent Errors**: Use appropriate exception types

### **🚫 Absolute Prohibitions:**
- Fallback emotion scores (0.0, 0.25, etc.)
- Dummy/fake models
- Random data generation
- Silent exception swallowing
- Hardcoded prediction values
- Default probability distributions

### **✅ Always Remember:**
> "It's better to crash with a clear error message than to silently return wrong results"

---

**Last Updated**: [Current Date]  
**Enforcement**: Mandatory for all contributors  
**Review**: Required before any code merge 