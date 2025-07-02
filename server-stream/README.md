# 🐕 Dog Emotion Recognition - Web Interface

Giao diện web local cho hệ thống nhận diện cảm xúc chó sử dụng AI/Deep Learning.

## ✨ Tính năng chính

### 📦 3 Module chính:
1. **Add Model**: Upload và quản lý các AI models
   - Hỗ trợ YOLO head detection models (.pt)
   - Hỗ trợ emotion classification models (.pth): Pure34, Pure50, ResNet
   - Validation và lưu trữ models

2. **Single Prediction**: Dự đoán cảm xúc cho 1 ảnh
   - Chọn models từ danh sách đã upload
   - Hiển thị kết quả với bounding box và confidence score
   - 4 loại cảm xúc: Happy, Sad, Angry, Relaxed

3. **Batch Prediction**: Xử lý hàng loạt ảnh
   - **Folder upload**: Chọn nhiều ảnh từ máy
   - **ZIP file**: Upload và extract file ZIP
   - **Roboflow**: Download dataset từ Roboflow (đang phát triển)
   - Export kết quả dưới dạng JSON

## 🚀 Cài đặt và Chạy

### 📍 Chạy Local

#### Bước 1: Cài đặt dependencies
```bash
# Di chuyển vào thư mục server-stream
cd server-stream

# Cài đặt packages
pip install -r requirements.txt
```

#### Bước 2: Chạy ứng dụng
```bash
python app.py
```

#### Bước 3: Truy cập web interface
Mở trình duyệt và truy cập: **http://localhost:5000**

### ☁️ Chạy trên Google Colab

#### 🎯 Cách nhanh nhất (Khuyên dùng):
```python
# Clone repository và cài đặt
!git clone https://github.com/your-repo/dog-emotion-recognition-hybrid.git
%cd dog-emotion-recognition-hybrid/server-stream

# Cài đặt tất cả dependencies
!pip install flask werkzeug pillow opencv-python torch torchvision ultralytics numpy pandas pyngrok

# Chạy setup tự động cho Colab
exec(open('colab_setup.py').read())
```

#### 🔧 Cách thủ công (nếu gặp lỗi):
```python
import os
import sys
import threading
import time
from pyngrok import ngrok

# Di chuyển đến thư mục server
%cd /content/dog-emotion-recognition-hybrid/server-stream

# Thêm vào Python path
sys.path.insert(0, os.getcwd())

# Import và cấu hình Flask
from app import app
app.config['DEBUG'] = False
app.config['TESTING'] = False

# Chạy Flask trong background thread
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# Đợi Flask khởi động
time.sleep(3)

# Tạo ngrok tunnel
public_url = ngrok.connect(5000)
print(f"🌍 Public URL: {public_url}")
print(f"📱 Local URL: http://localhost:5000")

# Giữ server chạy
try:
    while True:
        time.sleep(60)
        print(f"🔄 Server vẫn chạy tại: {public_url}")
except KeyboardInterrupt:
    print("🛑 Server đã dừng")
    ngrok.kill()
```

#### 📖 Hướng dẫn chi tiết
Xem file [COLAB_INSTRUCTIONS.md](COLAB_INSTRUCTIONS.md) để có hướng dẫn đầy đủ và troubleshooting.

## 📋 Hướng dẫn sử dụng

### 1️⃣ Upload Models
1. Truy cập **Add Model** từ menu
2. Nhập tên model (unique)
3. Chọn loại model:
   - **Head-object-detection**: Cho YOLO models
   - **Classification**: Cho emotion models
4. Upload file model (.pt hoặc .pth)
5. Click **Upload Model**

**⚠️ Lưu ý**: Cần có ít nhất 1 model mỗi loại để thực hiện prediction.

### 2️⃣ Single Prediction
1. Truy cập **Single Prediction**
2. Chọn Head Detection Model
3. Chọn Classification Model  
4. Upload ảnh chó
5. Click **Predict Emotion**
6. Xem kết quả với bounding box và confidence

### 3️⃣ Batch Prediction
1. Truy cập **Batch Prediction**
2. Chọn models
3. Chọn input method:
   - **Folder Images**: Chọn nhiều ảnh
   - **ZIP File**: Upload file ZIP chứa ảnh
   - **Roboflow**: Nhập script download (đang phát triển)
4. Click **Start Batch Processing**
5. Download kết quả JSON khi hoàn thành

## 🔧 Cấu trúc Project

```
server-stream/
├── app.py                 # Flask main application
├── model_manager.py       # Quản lý models
├── prediction_engine.py   # Engine xử lý prediction
├── requirements.txt       # Dependencies
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── add_model.html
│   ├── predict_single.html
│   └── predict_batch.html
├── models/               # Thư mục lưu models (auto-created)
├── uploads/              # Thư mục upload tạm (auto-created)
└── results/              # Thư mục kết quả (auto-created)
```

## 🛠️ Yêu cầu hệ thống

- Python 3.8+
- PyTorch 1.9+
- CUDA (khuyên dùng cho GPU acceleration)
- RAM: Tối thiểu 4GB, khuyên dùng 8GB+
- Disk: Ít nhất 2GB trống cho models

## 📝 Supported Models

### Head Detection Models (.pt):
- YOLO v8/v11 models trained for dog head detection
- Input: RGB images
- Output: Bounding boxes với confidence scores

### Classification Models (.pth):
- **Pure34**: Product-Unit Residual Network 34 layers
- **Pure50**: Product-Unit Residual Network 50 layers  
- **ResNet**: Standard ResNet architectures
- Input size: 224x224 (ResNet) hoặc 512x512 (Pure models)
- Output: 4 classes [sad, angry, happy, relaxed]

## 🚨 Troubleshooting

### Lỗi thường gặp:

1. **"No module named 'dog_emotion_classification'"**
   ```bash
   # Đảm bảo chạy từ thư mục gốc project
   cd dog-emotion-recognition-hybrid/server-stream
   python app.py
   ```

2. **"CUDA out of memory"**
   - Giảm batch size hoặc chuyển sang CPU
   - Thêm `device='cpu'` trong prediction_engine.py

3. **"Model loading failed"**
   - Kiểm tra định dạng file model (.pt/.pth)
   - Đảm bảo model tương thích với architecture

4. **Port 5000 đã được sử dụng**
   ```python
   # Thay đổi port trong app.py
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

## 📞 Support

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra console logs
2. Đảm bảo đã cài đúng dependencies
3. Verify models đã được upload đúng format

---

**🎯 Happy Dog Emotion Recognition!** 🐕‍🦺 