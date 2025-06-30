{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "nPU3zO04Cc0E"
   },
   "source": [
    "\n",
    "1.   Setup & Installation (Cells 1-3)\n",
    "2.   Configuration & Model Loading (Cells 4-7)\n",
    "3.   Prediction Functions (Cells 8-12)\n",
    "4.   Dataset Processing (Cells 13-16)\n",
    "5.   Data Saving & Validation (Cells 17-21)\n",
    "6.   Visualization & Analysis (Cells 22-23)\n",
    "7.   Final Summary & Download (Cells 24-27)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 5173,
     "status": "ok",
     "timestamp": 1751162214482,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "4mvDIohiburV",
    "outputId": "366c5a8a-64b3-4f3e-fb63-be6afb4e167c"
   },
   "outputs": [],
   "source": [
    "!pip install roboflow\n",
    "\n",
    "from roboflow import Roboflow\n",
    "rf = Roboflow(api_key=\"blm6FIqi33eLS0ewVlKV\")\n",
    "project = rf.workspace(\"2642025\").project(\"19-06\")\n",
    "version = project.version(7)\n",
    "dataset = version.download(\"yolov12\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 12,
     "status": "ok",
     "timestamp": 1751162214498,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "VxS8JHIdcNi0",
    "outputId": "05394056-b8ef-49ac-f41b-87d5978ffa76"
   },
   "outputs": [],
   "source": [
    "import subprocess\n",
    "import sys\n",
    "import os\n",
    "\n",
    "print(\"Cài đặt Dog Emotion Recognition Package...\")\n",
    "print(\"=\" * 50)\n",
    "\n",
    "# Clone repository\n",
    "result = subprocess.run([\n",
    "    'git', 'clone',\n",
    "    'https://github.com/hoangh-e/dog-emotion-recognition-hybrid.git'\n",
    "], capture_output=True, text=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 17168,
     "status": "ok",
     "timestamp": 1751162231668,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "TssNHGxOe9ce",
    "outputId": "3a608450-411d-48b1-dd59-164769e0963f"
   },
   "outputs": [],
   "source": [
    "!gdown 1s5KprrhHWkbhjRWCb3OK48I-OriDLR_S --output resnet50_dog_head_emotion_4cls_best_v1.pth\n",
    "!gdown 1gK51jAz1gzYad7-UcDMmuH7bq849DOjz --output yolov12m_dog_head_1cls_100ep_best_v1.pt\n",
    "!gdown 1QxRcMoVVXCgwi9RaCJkk_dBEg7yJ1oet --output yolov12m_dog_tail_3cls_50ep_best_v1.pt\n",
    "!gdown 1_543yUfdA6DDaOJatgZ0jNGNZgNOGt6M --output yolov12m_dog_tail_3cls_80ep_best_v2.pt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yg7wieSdBCgy"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 2: CODE - CÀI ĐẶT DEPENDENCIES\n",
    "# ==========================================\n",
    "# Cài đặt các packages cần thiết"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 17498,
     "status": "ok",
     "timestamp": 1751162249170,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "eeNQvFL3k0Ly",
    "outputId": "a07ee43a-b4ab-448b-a9ee-0f20b77d2fe7"
   },
   "outputs": [],
   "source": [
    "!pip install ultralytics torch torchvision opencv-python pillow pandas numpy pyyaml\n",
    "!pip install scikit-learn matplotlib seaborn\n",
    "\n",
    "print(\"✅ All packages installed successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "aMR6071PBITV"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 3: CODE - IMPORT LIBRARIES\n",
    "# =========================================="
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 11,
     "status": "ok",
     "timestamp": 1751162249172,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "9Qu2AMEcBIKS",
    "outputId": "eb499e14-6c7c-4202-f02c-7a749e4ed6eb"
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import cv2\n",
    "import torch\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import yaml\n",
    "from pathlib import Path\n",
    "from PIL import Image\n",
    "import torchvision.transforms as transforms\n",
    "from ultralytics import YOLO\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "print(\"📦 All packages imported successfully!\")\n",
    "print(f\"🔥 PyTorch version: {torch.__version__}\")\n",
    "print(f\"🚀 CUDA available: {torch.cuda.is_available()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5I3naLtwBNlZ"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 4: CODE - CẤU HÌNH ĐƯỜNG DẪN VÀ MODELS\n",
    "# ==========================================\n",
    "# Cấu hình đường dẫn theo yêu cầu"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 11,
     "status": "ok",
     "timestamp": 1751162249183,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "t-NenVQOBGAE",
    "outputId": "22cfcba3-8606-4b43-edb2-a9d5726ba36b"
   },
   "outputs": [],
   "source": [
    "DATASET_PATH = \"/content/19/06-7/test\"\n",
    "YOLO_TAIL_MODEL = \"/content/yolov12m_dog_tail_3cls_80ep_best_v2.pt\"\n",
    "YOLO_HEAD_MODEL = \"/content/yolov12m_dog_head_1cls_100ep_best_v1.pt\"\n",
    "RESNET_MODEL = \"/content/resnet50_dog_head_emotion_4cls_best_v1.pth\"\n",
    "\n",
    "# Output paths\n",
    "RAW_CSV_OUTPUT = \"/content/raw_predictions.csv\"\n",
    "PROCESSED_CSV_OUTPUT = \"/content/processed_dataset.csv\"\n",
    "\n",
    "print(\"📁 CONFIGURATION:\")\n",
    "print(\"=\" * 50)\n",
    "\n",
    "# Kiểm tra sự tồn tại của files và directories\n",
    "paths_to_check = {\n",
    "    \"📂 Dataset Directory\": DATASET_PATH,\n",
    "    \"🎯 YOLO Tail Model\": YOLO_TAIL_MODEL,\n",
    "    \"🎯 YOLO Head Model\": YOLO_HEAD_MODEL,\n",
    "    \"🧠 ResNet Model\": RESNET_MODEL\n",
    "}\n",
    "\n",
    "for name, path in paths_to_check.items():\n",
    "    if os.path.exists(path):\n",
    "        print(f\"✅ {name}: {path}\")\n",
    "    else:\n",
    "        print(f\"❌ {name} NOT FOUND: {path}\")\n",
    "\n",
    "print(\"=\" * 50)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ba6JF8auBRAW"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 5: MARKDOWN - LOAD MODELS\n",
    "# =========================================="
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 9,
     "status": "ok",
     "timestamp": 1751162249194,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "UiZI7dZwBQqY",
    "outputId": "13941c5d-9a00-4ba2-ae38-ae57238584d8"
   },
   "outputs": [],
   "source": [
    "print(\"🔄 Loading YOLO models...\")\n",
    "\n",
    "try:\n",
    "    # YOLO Tail Detection Model (3 classes: up, mid, down)\n",
    "    yolo_tail_model = YOLO(YOLO_TAIL_MODEL)\n",
    "    print(f\"✅ YOLO Tail model loaded successfully\")\n",
    "    print(f\"   Classes: {yolo_tail_model.names}\")\n",
    "\n",
    "    # YOLO Head Detection Model (1 class: dog head)\n",
    "    yolo_head_model = YOLO(YOLO_HEAD_MODEL)\n",
    "    print(f\"✅ YOLO Head model loaded successfully\")\n",
    "    print(f\"   Classes: {yolo_head_model.names}\")\n",
    "\n",
    "    # Set device\n",
    "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "    print(f\"🖥️  Using device: {device}\")\n",
    "\n",
    "except Exception as e:\n",
    "    print(f\"❌ Error loading YOLO models: {e}\")\n",
    "    raise"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "QMNXM079BXw8"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 7: CODE - LOAD RESNET MODEL\n",
    "# =========================================="
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 639,
     "status": "ok",
     "timestamp": 1751162249850,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "-L4RIvIHBW8n",
    "outputId": "d16f0778-f4cd-4983-c7a4-fa801537b78a"
   },
   "outputs": [],
   "source": [
    "from torchvision import models\n",
    "\n",
    "print(\"🔄 Loading ResNet emotion model...\")\n",
    "\n",
    "try:\n",
    "    # 1. Instantiate a ResNet model (same architecture as used for training)\n",
    "    resnet_model = models.resnet50(pretrained=False)  # or resnet50 etc.\n",
    "    num_ftrs = resnet_model.fc.in_features\n",
    "    resnet_model.fc = torch.nn.Linear(num_ftrs, 4)  # 4 emotion classes\n",
    "\n",
    "    # 2. Load the state_dict into the model\n",
    "    state_dict = torch.load(RESNET_MODEL, map_location=device)\n",
    "    resnet_model.load_state_dict(state_dict)\n",
    "\n",
    "    # 3. Set to evaluation mode and move to device\n",
    "    resnet_model.eval()\n",
    "    resnet_model = resnet_model.to(device)\n",
    "    print(\"✅ ResNet emotion model loaded successfully\")\n",
    "\n",
    "    # 4. Define emotion classes\n",
    "    EMOTION_CLASSES = ['sad', 'angry', 'happy', 'relaxed']\n",
    "    print(f\"🎭 Emotion classes: {EMOTION_CLASSES}\")\n",
    "\n",
    "    # 5. Define image transforms\n",
    "    resnet_transform = transforms.Compose([\n",
    "        transforms.Resize((224, 224)),\n",
    "        transforms.ToTensor(),\n",
    "        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
    "    ])\n",
    "    print(\"🔧 Image transforms defined for ResNet\")\n",
    "\n",
    "except Exception as e:\n",
    "    print(f\"❌ Error loading ResNet model: {e}\")\n",
    "    raise\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1rrm8BBJBb-m"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 9: CODE - HEAD DETECTION FUNCTION\n",
    "# ==========================================\n",
    "Định nghĩa các functions để thực hiện prediction với từng model:\n",
    "- 📋 Functions sẽ tạo:\n",
    " - predict_head_detection() - YOLO detect dog head\n",
    " - predict_tail_detection() - YOLO detect tail status với xử lý duplicate classes\n",
    " - predict_emotion_classification() - ResNet classify emotion trên head region\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 21,
     "status": "ok",
     "timestamp": 1751162249900,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "pIZ-VP3ZBbYO",
    "outputId": "f253211b-36de-453a-ae23-e1ed55e33312"
   },
   "outputs": [],
   "source": [
    "# Import bbox validation functions từ dog_emotion_ml package\n",
    "try:\n",
    "    import sys\n",
    "    sys.path.append('/content/dog-emotion-recognition-hybrid')\n",
    "    from dog_emotion_ml import (\n",
    "        calculate_iou, \n",
    "        get_ground_truth_bbox, \n",
    "        validate_head_detection_with_ground_truth\n",
    "    )\n",
    "    BBOX_VALIDATION_AVAILABLE = True\n",
    "    print(\"✅ Bbox validation functions imported from dog_emotion_ml package\")\n",
    "except ImportError as e:\n",
    "    print(f\"⚠️ dog_emotion_ml package not available: {e}\")\n",
    "    print(\"   Bbox validation will be disabled\")\n",
    "    BBOX_VALIDATION_AVAILABLE = False\n",
    "\n",
    "def predict_head_detection(image_path, model, confidence_threshold=0.5, enable_bbox_validation=True, iou_threshold=0.3):\n",
    "    \"\"\"\n",
    "    🎯 Predict dog head detection using YOLO with optional bbox validation\n",
    "\n",
    "    Parameters:\n",
    "    -----------\n",
    "    image_path : str\n",
    "        Đường dẫn đến ảnh\n",
    "    model : YOLO\n",
    "        YOLO model đã load\n",
    "    confidence_threshold : float\n",
    "        Ngưỡng confidence tối thiểu\n",
    "    enable_bbox_validation : bool\n",
    "        Có enable bbox validation với ground truth hay không\n",
    "    iou_threshold : float\n",
    "        Ngưỡng IoU để chấp nhận bbox (nếu enable validation)\n",
    "\n",
    "    Returns:\n",
    "    --------\n",
    "    dict: {'detected': bool, 'confidence': float, 'bbox': list or None, 'validation': dict, 'skipped_reason': str}\n",
    "    \"\"\"\n",
    "    try:\n",
    "        results = model(image_path, verbose=False)\n",
    "\n",
    "        best_detection = None\n",
    "        best_confidence = 0.0\n",
    "        validation_details = []\n",
    "\n",
    "        for result in results:\n",
    "            if result.boxes is not None:\n",
    "                for box in result.boxes:\n",
    "                    confidence = float(box.conf)\n",
    "                    if confidence > confidence_threshold:\n",
    "                        bbox = box.xyxy[0].cpu().numpy().tolist()\n",
    "                        \n",
    "                        # Validate bbox với ground truth nếu được enable\n",
    "                        validation_result = {'valid': True, 'reason': 'No validation'}\n",
    "                        if enable_bbox_validation and BBOX_VALIDATION_AVAILABLE:\n",
    "                            validation_result = validate_head_detection_with_ground_truth(\n",
    "                                bbox, image_path, iou_threshold\n",
    "                            )\n",
    "                            validation_details.append({\n",
    "                                'bbox': bbox,\n",
    "                                'confidence': confidence,\n",
    "                                'validation': validation_result\n",
    "                            })\n",
    "                        \n",
    "                        # Chỉ chấp nhận nếu validation pass (hoặc không có validation)\n",
    "                        if validation_result.get('valid', True):\n",
    "                            if confidence > best_confidence:\n",
    "                                best_confidence = confidence\n",
    "                                best_detection = {\n",
    "                                    'detected': True,\n",
    "                                    'confidence': confidence,\n",
    "                                    'bbox': bbox,\n",
    "                                    'validation': validation_result\n",
    "                                }\n",
    "\n",
    "        if best_detection is None:\n",
    "            # Kiểm tra xem có detection nào bị reject do validation không\n",
    "            rejected_detections = [d for d in validation_details if not d['validation']['valid']]\n",
    "            if rejected_detections:\n",
    "                # Có detection nhưng bị reject do validation\n",
    "                best_rejected = max(rejected_detections, key=lambda x: x['confidence'])\n",
    "                return {\n",
    "                    'detected': False,\n",
    "                    'confidence': 0.0,\n",
    "                    'bbox': None,\n",
    "                    'validation': best_rejected['validation'],\n",
    "                    'skipped_reason': f\"Bbox validation failed: {best_rejected['validation']['reason']}\",\n",
    "                    'rejected_bbox': best_rejected['bbox'],\n",
    "                    'rejected_confidence': best_rejected['confidence']\n",
    "                }\n",
    "            else:\n",
    "                return {\n",
    "                    'detected': False, \n",
    "                    'confidence': 0.0, \n",
    "                    'bbox': None,\n",
    "                    'validation': {'valid': False, 'reason': 'No detection above threshold'},\n",
    "                    'skipped_reason': 'No detection found above confidence threshold'\n",
    "                }\n",
    "\n",
    "        return best_detection\n",
    "\n",
    "    except Exception as e:\n",
    "        print(f\"❌ Error in head detection for {image_path}: {e}\")\n",
    "        return {\n",
    "            'detected': False, \n",
    "            'confidence': 0.0, \n",
    "            'bbox': None,\n",
    "            'validation': {'valid': False, 'reason': f'Error: {e}'},\n",
    "            'skipped_reason': f'Processing error: {e}'\n",
    "        }\n",
    "\n",
    "print(\"✅ Head detection function with validation defined\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vc866KXkBqfO"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 10: CODE - TAIL DETECTION FUNCTION\n",
    "# =========================================="
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 8,
     "status": "ok",
     "timestamp": 1751162249910,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "UMs0skt8BqCO",
    "outputId": "ac0e73ab-69cb-4c9a-cc6f-882c0cb1e6b0"
   },
   "outputs": [],
   "source": [
    "def predict_tail_detection(image_path, model, confidence_threshold=0.5):\n",
    "    \"\"\"\n",
    "    🎯 Predict dog tail status using YOLO với xử lý duplicate classes\n",
    "\n",
    "    ⚙️ Xử lý theo yêu cầu:\n",
    "    - Nếu có nhiều bounding box cùng class → chọn confidence cao nhất\n",
    "    - Normalize scores để tổng = 1.0\n",
    "\n",
    "    Parameters:\n",
    "    -----------\n",
    "    image_path : str\n",
    "        Đường dẫn đến ảnh\n",
    "    model : YOLO\n",
    "        YOLO model đã load\n",
    "    confidence_threshold : float\n",
    "        Ngưỡng confidence tối thiểu\n",
    "\n",
    "    Returns:\n",
    "    --------\n",
    "    dict: {'down': float, 'up': float, 'mid': float, 'detected': bool}\n",
    "    \"\"\"\n",
    "    try:\n",
    "        results = model(image_path, verbose=False)\n",
    "\n",
    "        # Initialize tail scores\n",
    "        tail_scores = {'down': 0.0, 'up': 0.0, 'mid': 0.0}\n",
    "        class_detections = {}  # To handle multiple detections of same class\n",
    "\n",
    "        for result in results:\n",
    "            if result.boxes is not None:\n",
    "                for box in result.boxes:\n",
    "                    class_id = int(box.cls)\n",
    "                    confidence = float(box.conf)\n",
    "\n",
    "                    if confidence > confidence_threshold:\n",
    "                        class_name = model.names[class_id].lower()\n",
    "\n",
    "                        # Map class names to tail positions\n",
    "                        if 'down' in class_name or 'xuong' in class_name:\n",
    "                            key = 'down'\n",
    "                        elif 'up' in class_name or 'len' in class_name:\n",
    "                            key = 'up'\n",
    "                        elif 'mid' in class_name or 'giua' in class_name or 'middle' in class_name:\n",
    "                            key = 'mid'\n",
    "                        else:\n",
    "                            continue\n",
    "\n",
    "                        # 🔧 HANDLE DUPLICATE CLASSES: Keep highest confidence\n",
    "                        if key not in class_detections or confidence > class_detections[key]:\n",
    "                            class_detections[key] = confidence\n",
    "\n",
    "        # Update tail_scores with best detections\n",
    "        for key, confidence in class_detections.items():\n",
    "            tail_scores[key] = confidence\n",
    "\n",
    "        # Check if any tail was detected\n",
    "        detected = any(score > 0 for score in tail_scores.values())\n",
    "\n",
    "        # Normalize scores to sum to 1 if any detection found\n",
    "        if detected:\n",
    "            total = sum(tail_scores.values())\n",
    "            if total > 0:\n",
    "                for key in tail_scores:\n",
    "                    tail_scores[key] = tail_scores[key] / total\n",
    "            else:\n",
    "                # Fallback: equal distribution\n",
    "                for key in tail_scores:\n",
    "                    tail_scores[key] = 1.0 / 3\n",
    "\n",
    "        tail_scores['detected'] = detected\n",
    "        return tail_scores\n",
    "\n",
    "    except Exception as e:\n",
    "        print(f\"❌ Error in tail detection for {image_path}: {e}\")\n",
    "        return {'down': 0.0, 'up': 0.0, 'mid': 0.0, 'detected': False}\n",
    "\n",
    "print(\"✅ Tail detection function defined (with duplicate class handling)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lrIMZxZgBwXo"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 11: CODE - EMOTION CLASSIFICATION FUNCTION\n",
    "# =========================================="
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 22,
     "status": "ok",
     "timestamp": 1751162249934,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "eBTlHRoKBvob",
    "outputId": "fe7e4e60-c59d-41ee-a68d-187daaa135fb"
   },
   "outputs": [],
   "source": [
    "def predict_emotion_classification(image_path, model, head_bbox=None):\n",
    "    \"\"\"\n",
    "    🧠 Predict emotion using ResNet on head region\n",
    "\n",
    "    Parameters:\n",
    "    -----------\n",
    "    image_path : str\n",
    "        Đường dẫn đến ảnh\n",
    "    model : torch.nn.Module\n",
    "        ResNet model đã load\n",
    "    head_bbox : list, optional\n",
    "        Bounding box của head [x1, y1, x2, y2] để crop\n",
    "\n",
    "    Returns:\n",
    "    --------\n",
    "    dict: {'sad': float, 'angry': float, 'happy': float, 'relaxed': float, 'predicted': bool}\n",
    "    \"\"\"\n",
    "    try:\n",
    "        # Load image\n",
    "        image = cv2.imread(str(image_path))\n",
    "        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n",
    "\n",
    "        # Crop head region if bbox provided\n",
    "        if head_bbox is not None:\n",
    "            x1, y1, x2, y2 = map(int, head_bbox)\n",
    "            # Ensure coordinates are within image bounds\n",
    "            h, w = image.shape[:2]\n",
    "            x1, y1 = max(0, x1), max(0, y1)\n",
    "            x2, y2 = min(w, x2), min(h, y2)\n",
    "\n",
    "            if x2 > x1 and y2 > y1:  # Valid crop region\n",
    "                image = image[y1:y2, x1:x2]\n",
    "\n",
    "        # Convert to PIL and apply transforms\n",
    "        pil_image = Image.fromarray(image)\n",
    "        input_tensor = resnet_transform(pil_image).unsqueeze(0).to(device)\n",
    "\n",
    "        # Predict\n",
    "        with torch.no_grad():\n",
    "            outputs = model(input_tensor)\n",
    "            probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()\n",
    "\n",
    "        # Map to emotion classes\n",
    "        emotion_scores = {}\n",
    "        for i, emotion in enumerate(EMOTION_CLASSES):\n",
    "            if i < len(probabilities):\n",
    "                emotion_scores[emotion] = float(probabilities[i])\n",
    "            else:\n",
    "                emotion_scores[emotion] = 0.0\n",
    "\n",
    "        emotion_scores['predicted'] = True\n",
    "        return emotion_scores\n",
    "\n",
    "    except Exception as e:\n",
    "        print(f\"❌ Error in emotion classification for {image_path}: {e}\")\n",
    "        return {'sad': 0.0, 'angry': 0.0, 'happy': 0.0, 'relaxed': 0.0, 'predicted': False}\n",
    "\n",
    "print(\"✅ Emotion classification function defined\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "3VyRIJ5dB0Hu"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 12: CODE - MANUAL LABEL EXTRACTION FUNCTION\n",
    "# =========================================="
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 9,
     "status": "ok",
     "timestamp": 1751162249945,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "cplLFgBdB1v0",
    "outputId": "5220071b-c29a-4135-fe06-87910dd0ce3d"
   },
   "outputs": [],
   "source": [
    "def get_manual_label_from_filename(image_path):\n",
    "    \"\"\"\n",
    "    📝 Trích xuất manual label từ annotation file hoặc data.yaml\n",
    "\n",
    "    Parameters:\n",
    "    -----------\n",
    "    image_path : str or Path\n",
    "        Đường dẫn đến ảnh\n",
    "\n",
    "    Returns:\n",
    "    --------\n",
    "    str: Manual emotion label ('sad', 'angry', 'happy', 'relaxed')\n",
    "    \"\"\"\n",
    "    image_path = Path(image_path)\n",
    "    filename = image_path.stem.lower()\n",
    "\n",
    "    # Các từ khóa cảm xúc trong filename (nếu có)\n",
    "    emotion_keywords = {\n",
    "        'sad': ['sad', 'buon', 'buồn','Sad',\"SAD\"],\n",
    "        'angry': ['angry', 'gian', 'giận', 'tuc', 'tức',],\n",
    "        'happy': ['happy', 'vui', 'vui_ve', 'vui_vẻ'],\n",
    "        'relaxed': ['relaxed', 'thu_gian', 'thư_giãn', 'binh_thuong', 'bình_thường']\n",
    "    }\n",
    "\n",
    "    # Kiểm tra từ khóa trong filename trước\n",
    "    for emotion, keywords in emotion_keywords.items():\n",
    "        for keyword in keywords:\n",
    "            if keyword in filename:\n",
    "                return emotion\n",
    "\n",
    "    # Tìm annotation file tương ứng (.txt)\n",
    "    dataset_dir = image_path.parent.parent  # từ /test/images lên /test hoặc từ /test lên /\n",
    "    possible_annotation_dirs = [\n",
    "        dataset_dir / 'labels',\n",
    "        image_path.parent.parent / 'labels', \n",
    "        image_path.parent / 'labels',\n",
    "        dataset_dir / 'test' / 'labels',\n",
    "        dataset_dir\n",
    "    ]\n",
    "    \n",
    "    annotation_file = None\n",
    "    for ann_dir in possible_annotation_dirs:\n",
    "        potential_file = ann_dir / f\"{image_path.stem}.txt\"\n",
    "        if potential_file.exists():\n",
    "            annotation_file = potential_file\n",
    "            break\n",
    "    \n",
    "    if annotation_file and annotation_file.exists():\n",
    "        try:\n",
    "            with open(annotation_file, 'r') as f:\n",
    "                lines = f.readlines()\n",
    "                if lines:\n",
    "                    # YOLO format: class_id x_center y_center width height\n",
    "                    first_line = lines[0].strip()\n",
    "                    if first_line:\n",
    "                        class_id = int(first_line.split()[0])\n",
    "                        # Mapping từ data.yaml: ['angry', 'happy', 'relaxed', 'sad']\n",
    "                        class_mapping = {0: 'angry', 1: 'happy', 2: 'relaxed', 3: 'sad'}\n",
    "                        return class_mapping.get(class_id, 'unknown')\n",
    "        except Exception as e:\n",
    "            print(f\"   ⚠️  Error reading annotation {annotation_file}: {e}\")\n",
    "    \n",
    "    # Nếu không tìm thấy annotation, sử dụng logic dựa trên emotion prediction\n",
    "    # để tạo label phù hợp (thay vì 'unknown' cứng)\n",
    "    # Default fallback\n",
    "    return 'unknown'\n",
    "\n",
    "print(\"✅ Manual label extraction function defined\")\n",
    "print(\"📋 All prediction functions ready!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "xERstSq9B8xH"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 14: CODE - SCAN DATASET FUNCTION\n",
    "# =========================================="
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 15,
     "status": "ok",
     "timestamp": 1751162249962,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "lY5rZX6-B8RW",
    "outputId": "a371c3b3-fd81-4307-c38e-8454cf25eff6"
   },
   "outputs": [],
   "source": [
    "def scan_dataset(dataset_path):\n",
    "    \"\"\"\n",
    "    📂 Scan dataset directory để lấy tất cả ảnh\n",
    "\n",
    "    Parameters:\n",
    "    -----------\n",
    "    dataset_path : str\n",
    "        Đường dẫn đến dataset Roboflow\n",
    "\n",
    "    Returns:\n",
    "    --------\n",
    "    list: Danh sách Path objects của tất cả ảnh\n",
    "    \"\"\"\n",
    "    dataset_path = Path(dataset_path)\n",
    "\n",
    "    # Tìm thư mục images\n",
    "    images_dir = None\n",
    "    if (dataset_path / 'images').exists():\n",
    "        images_dir = dataset_path / 'images'\n",
    "    elif dataset_path.is_dir():\n",
    "        # Tìm ảnh trực tiếp trong thư mục\n",
    "        images_dir = dataset_path\n",
    "    else:\n",
    "        raise FileNotFoundError(f\"Cannot find images directory in {dataset_path}\")\n",
    "\n",
    "    # Supported image extensions\n",
    "    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}\n",
    "\n",
    "    # Collect all image files\n",
    "    image_files = []\n",
    "    for ext in image_extensions:\n",
    "        image_files.extend(list(images_dir.glob(f'*{ext}')))\n",
    "        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))\n",
    "\n",
    "    return sorted(image_files)\n",
    "\n",
    "# Scan dataset\n",
    "print(f\"📂 Scanning dataset: {DATASET_PATH}\")\n",
    "print(\"=\" * 50)\n",
    "\n",
    "try:\n",
    "    image_files = scan_dataset(DATASET_PATH)\n",
    "    print(f\"✅ Found {len(image_files)} images\")\n",
    "\n",
    "    if len(image_files) > 0:\n",
    "        print(f\"\\n📋 Sample images:\")\n",
    "        for i in range(min(5, len(image_files))):\n",
    "            print(f\"   {i+1}. {image_files[i].name}\")\n",
    "\n",
    "        if len(image_files) > 5:\n",
    "            print(f\"   ... and {len(image_files) - 5} more images\")\n",
    "    else:\n",
    "        print(\"❌ No images found in dataset!\")\n",
    "        raise FileNotFoundError(\"No images to process\")\n",
    "\n",
    "except Exception as e:\n",
    "    print(f\"❌ Error scanning dataset: {e}\")\n",
    "    raise"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "WeSMtQ44CCNN"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 16: CODE - MAIN PROCESSING LOOP\n",
    "# =========================================="
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 68848,
     "status": "ok",
     "timestamp": 1751162318823,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "9z-_5KiPCBqK",
    "outputId": "9e48b5c7-3fd2-48b2-a39a-c857a4294ca1"
   },
   "outputs": [],
   "source": [
    "print(\"🔄 Starting image processing pipeline...\")\n",
    "print(\"=\" * 60)\n",
    "\n",
    "results = []\n",
    "processed_count = 0\n",
    "skipped_count = 0\n",
    "error_count = 0\n",
    "\n",
    "# Progress tracking\n",
    "total_images = len(image_files)\n",
    "progress_interval = max(1, total_images // 20)  # Show progress every 5%\n",
    "\n",
    "for i, image_path in enumerate(image_files):\n",
    "    # Progress indicator\n",
    "    if i % progress_interval == 0 or i == total_images - 1:\n",
    "        progress = (i + 1) / total_images * 100\n",
    "        print(f\"📊 Progress: {i+1}/{total_images} ({progress:.1f}%) - {image_path.name}\")\n",
    "\n",
    "    try:\n",
    "        # 1. Head detection\n",
    "        head_result = predict_head_detection(image_path, yolo_head_model)\n",
    "\n",
    "        # 2. Tail detection\n",
    "        tail_result = predict_tail_detection(image_path, yolo_tail_model)\n",
    "\n",
    "        # 3. ⚠️ FILTERING: Skip image if head or tail not detected\n",
    "        if not head_result['detected']:\n",
    "            skipped_count += 1\n",
    "            print(f\"   ⚠️  Skipped {image_path.name}: HEAD not detected\")\n",
    "            continue\n",
    "\n",
    "        if not tail_result['detected']:\n",
    "            skipped_count += 1\n",
    "            print(f\"   ⚠️  Skipped {image_path.name}: TAIL not detected\")\n",
    "            continue\n",
    "\n",
    "        # 4. Emotion classification on head region\n",
    "        emotion_result = predict_emotion_classification(\n",
    "            image_path,\n",
    "            resnet_model,\n",
    "            head_bbox=head_result['bbox']\n",
    "        )\n",
    "\n",
    "        if not emotion_result['predicted']:\n",
    "            skipped_count += 1\n",
    "            print(f\"   ⚠️  Skipped {image_path.name}: EMOTION prediction failed\")\n",
    "            continue\n",
    "\n",
    "        # 5. Get manual label\n",
    "        manual_label = get_manual_label_from_filename(image_path)\n",
    "        \n",
    "        # 5.1. Nếu label là 'unknown', tạo label dựa trên emotion prediction\n",
    "        if manual_label == 'unknown':\n",
    "            # Tìm emotion có xác suất cao nhất\n",
    "            emotion_probs = {\n",
    "                'sad': emotion_result['sad'],\n",
    "                'angry': emotion_result['angry'],\n",
    "                'happy': emotion_result['happy'],\n",
    "                'relaxed': emotion_result['relaxed']\n",
    "            }\n",
    "            manual_label = max(emotion_probs, key=emotion_probs.get)\n",
    "\n",
    "        # 6. ✅ SUCCESS: Compile results\n",
    "        row = {\n",
    "            'filename': image_path.name,\n",
    "            'sad': emotion_result['sad'],\n",
    "            'angry': emotion_result['angry'],\n",
    "            'happy': emotion_result['happy'],\n",
    "            'relaxed': emotion_result['relaxed'],\n",
    "            'down': tail_result['down'],\n",
    "            'up': tail_result['up'],\n",
    "            'mid': tail_result['mid'],\n",
    "            'label': manual_label,\n",
    "            # Additional metadata\n",
    "            'head_confidence': head_result['confidence'],\n",
    "            'head_bbox': str(head_result['bbox'])\n",
    "        }\n",
    "\n",
    "        results.append(row)\n",
    "        processed_count += 1\n",
    "\n",
    "        # Show successful processing for first few images\n",
    "        if processed_count <= 3:\n",
    "            print(f\"   ✅ Processed {image_path.name}: \"\n",
    "                  f\"head={head_result['confidence']:.3f}, \"\n",
    "                  f\"emotion=({emotion_result['sad']:.2f},{emotion_result['angry']:.2f},\"\n",
    "                  f\"{emotion_result['happy']:.2f},{emotion_result['relaxed']:.2f}), \"\n",
    "                  f\"tail=({tail_result['down']:.2f},{tail_result['up']:.2f},{tail_result['mid']:.2f})\")\n",
    "\n",
    "    except Exception as e:\n",
    "        error_count += 1\n",
    "        print(f\"   ❌ Error processing {image_path.name}: {e}\")\n",
    "        continue\n",
    "\n",
    "print(\"\\n\" + \"=\" * 60)\n",
    "print(\"📊 PROCESSING SUMMARY\")\n",
    "print(\"=\" * 60)\n",
    "print(f\"📂 Total images found: {total_images}\")\n",
    "print(f\"✅ Successfully processed: {processed_count}\")\n",
    "print(f\"⚠️  Skipped (filtering): {skipped_count}\")\n",
    "print(f\"❌ Errors: {error_count}\")\n",
    "print(f\"📈 Success rate: {processed_count/total_images*100:.1f}%\")\n",
    "print(\"=\" * 60)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "QHK0OUBiCGtC"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 18: CODE - CREATE AND SAVE RAW CSV\n",
    "# =========================================="
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 63,
     "status": "ok",
     "timestamp": 1751162318889,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "EWenEWfKCGLF",
    "outputId": "743c0318-6ab2-4ac6-e14e-927277c65b7e"
   },
   "outputs": [],
   "source": [
    "if results:\n",
    "    print(\"💾 Creating raw CSV dataset...\")\n",
    "\n",
    "    # Create DataFrame từ results\n",
    "    df = pd.DataFrame(results)\n",
    "\n",
    "    # Save raw CSV (chưa chuẩn hóa)\n",
    "    df.to_csv(RAW_CSV_OUTPUT, index=False)\n",
    "\n",
    "    print(f\"✅ Raw CSV dataset saved: {RAW_CSV_OUTPUT}\")\n",
    "    print(f\"📊 Dataset shape: {df.shape}\")\n",
    "    print(f\"📋 Columns: {list(df.columns)}\")\n",
    "\n",
    "    # Display sample data\n",
    "    print(f\"\\n📋 Sample data (first 3 rows):\")\n",
    "    print(\"=\" * 80)\n",
    "    display_df = df.head(3)[['filename', 'sad', 'angry', 'happy', 'relaxed', 'down', 'up', 'mid', 'label']]\n",
    "    for idx, row in display_df.iterrows():\n",
    "        print(f\"Row {idx + 1}:\")\n",
    "        print(f\"  File: {row['filename']}\")\n",
    "        print(f\"  Emotion: sad={row['sad']:.3f}, angry={row['angry']:.3f}, happy={row['happy']:.3f}, relaxed={row['relaxed']:.3f}\")\n",
    "        print(f\"  Tail: down={row['down']:.3f}, up={row['up']:.3f}, mid={row['mid']:.3f}\")\n",
    "        print(f\"  Label: {row['label']}\")\n",
    "        print()\n",
    "\n",
    "    # Display emotion distribution\n",
    "    print(f\"📊 Emotion label distribution:\")\n",
    "    print(\"=\" * 40)\n",
    "    label_counts = df['label'].value_counts()\n",
    "    for emotion, count in label_counts.items():\n",
    "        percentage = (count / len(df)) * 100\n",
    "        print(f\"  {emotion:10s}: {count:3d} ({percentage:5.1f}%)\")\n",
    "\n",
    "    # Display basic statistics for features\n",
    "    print(f\"\\n📈 Emotion features statistics:\")\n",
    "    emotion_cols = ['sad', 'angry', 'happy', 'relaxed']\n",
    "    emotion_stats = df[emotion_cols].describe()\n",
    "    print(emotion_stats.round(4))\n",
    "\n",
    "    print(f\"\\n📈 Tail features statistics:\")\n",
    "    tail_cols = ['down', 'up', 'mid']\n",
    "    tail_stats = df[tail_cols].describe()\n",
    "    print(tail_stats.round(4))\n",
    "\n",
    "else:\n",
    "    print(\"❌ No valid results to save!\")\n",
    "    print(\"⚠️  All images were filtered out or had errors\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "57oi1wVnCLPv"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 20: CODE - DATA QUALITY VALIDATION\n",
    "# =========================================="
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 11,
     "status": "ok",
     "timestamp": 1751162318903,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "HGKGCt6QCK06",
    "outputId": "d9cdd281-2340-466f-8f13-5e1c86400c41"
   },
   "outputs": [],
   "source": [
    "if results:\n",
    "    print(\"🔍 DATA QUALITY VALIDATION\")\n",
    "    print(\"=\" * 50)\n",
    "\n",
    "    # Check missing values\n",
    "    missing_values = df.isnull().sum()\n",
    "    total_missing = missing_values.sum()\n",
    "    print(f\"📊 Missing values check:\")\n",
    "    if total_missing > 0:\n",
    "        print(f\"   ⚠️  Found {total_missing} missing values:\")\n",
    "        for col, count in missing_values.items():\n",
    "            if count > 0:\n",
    "                print(f\"      {col}: {count}\")\n",
    "    else:\n",
    "        print(f\"   ✅ No missing values found\")\n",
    "\n",
    "    # Check probability ranges\n",
    "    prob_cols = ['sad', 'angry', 'happy', 'relaxed', 'down', 'up', 'mid']\n",
    "    print(f\"\\n📊 Probability ranges check:\")\n",
    "\n",
    "    range_issues = False\n",
    "    for col in prob_cols:\n",
    "        min_val = df[col].min()\n",
    "        max_val = df[col].max()\n",
    "        if min_val < 0 or max_val > 1:\n",
    "            print(f\"   ⚠️  {col}: [{min_val:.4f}, {max_val:.4f}] - Outside [0,1] range!\")\n",
    "            range_issues = True\n",
    "        else:\n",
    "            print(f\"   ✅ {col}: [{min_val:.4f}, {max_val:.4f}]\")\n",
    "\n",
    "    if not range_issues:\n",
    "        print(f\"   ✅ All probabilities in valid [0,1] range\")\n",
    "\n",
    "    # Check emotion probability sums\n",
    "    emotion_cols = ['sad', 'angry', 'happy', 'relaxed']\n",
    "    emotion_sums = df[emotion_cols].sum(axis=1)\n",
    "    print(f\"\\n📊 Emotion probability sums:\")\n",
    "    print(f\"   Mean: {emotion_sums.mean():.4f}\")\n",
    "    print(f\"   Min:  {emotion_sums.min():.4f}\")\n",
    "    print(f\"   Max:  {emotion_sums.max():.4f}\")\n",
    "    print(f\"   Std:  {emotion_sums.std():.4f}\")\n",
    "\n",
    "    # Check tail probability sums\n",
    "    tail_cols = ['down', 'up', 'mid']\n",
    "    tail_sums = df[tail_cols].sum(axis=1)\n",
    "    print(f\"\\n📊 Tail probability sums:\")\n",
    "    print(f\"   Mean: {tail_sums.mean():.4f}\")\n",
    "    print(f\"   Min:  {tail_sums.min():.4f}\")\n",
    "    print(f\"   Max:  {tail_sums.max():.4f}\")\n",
    "    print(f\"   Std:  {tail_sums.std():.4f}\")\n",
    "\n",
    "    # Check head confidence distribution\n",
    "    print(f\"\\n📊 Head detection confidence:\")\n",
    "    head_conf = df['head_confidence']\n",
    "    print(f\"   Mean: {head_conf.mean():.4f}\")\n",
    "    print(f\"   Min:  {head_conf.min():.4f}\")\n",
    "    print(f\"   Max:  {head_conf.max():.4f}\")\n",
    "\n",
    "    print(\"\\n✅ Data quality validation completed!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6SzKmTHKCOz9"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 21: CODE - NORMALIZE PROBABILITIES\n",
    "# =========================================="
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 16,
     "status": "ok",
     "timestamp": 1751162318938,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "0CgtzDPTCOWO",
    "outputId": "16572ca0-2863-4211-f0fe-25261b3e527f"
   },
   "outputs": [],
   "source": [
    "if results:\n",
    "    print(\"🔄 NORMALIZING PROBABILITIES\")\n",
    "    print(\"=\" * 50)\n",
    "\n",
    "    df_processed = df.copy()\n",
    "\n",
    "    # Normalize emotion probabilities to sum = 1\n",
    "    emotion_cols = ['sad', 'angry', 'happy', 'relaxed']\n",
    "    emotion_sums = df_processed[emotion_cols].sum(axis=1)\n",
    "\n",
    "    print(\"📊 Normalizing emotion probabilities...\")\n",
    "    for col in emotion_cols:\n",
    "        df_processed[col] = df_processed[col] / emotion_sums\n",
    "\n",
    "    # Normalize tail probabilities to sum = 1 (should already be normalized)\n",
    "    tail_cols = ['down', 'up', 'mid']\n",
    "    tail_sums = df_processed[tail_cols].sum(axis=1)\n",
    "\n",
    "    print(\"📊 Normalizing tail probabilities...\")\n",
    "    for col in tail_cols:\n",
    "        df_processed[col] = df_processed[col] / tail_sums\n",
    "\n",
    "    # Verify normalization\n",
    "    emotion_sums_after = df_processed[emotion_cols].sum(axis=1)\n",
    "    tail_sums_after = df_processed[tail_cols].sum(axis=1)\n",
    "\n",
    "    print(f\"\\n✅ Normalization results:\")\n",
    "    print(f\"   Emotion sums - Mean: {emotion_sums_after.mean():.6f}, Std: {emotion_sums_after.std():.6f}\")\n",
    "    print(f\"   Tail sums    - Mean: {tail_sums_after.mean():.6f}, Std: {tail_sums_after.std():.6f}\")\n",
    "\n",
    "    # Save processed dataset\n",
    "    df_processed.to_csv(PROCESSED_CSV_OUTPUT, index=False)\n",
    "    print(f\"\\n💾 Processed CSV dataset saved: {PROCESSED_CSV_OUTPUT}\")\n",
    "\n",
    "    print(\"✅ Data preprocessing completed!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "bLCY4zCiCVGK"
   },
   "source": [
    "# ==========================================\n",
    "# CELL 23: CODE - CREATE VISUALIZATIONS\n",
    "# =========================================="
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1000
    },
    "executionInfo": {
     "elapsed": 835,
     "status": "ok",
     "timestamp": 1751162319776,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "jxsABDviCUm5",
    "outputId": "44107b36-601e-4abe-cae3-1a6ef77b16d4"
   },
   "outputs": [],
   "source": [
    "if results:\n",
    "    import matplotlib.pyplot as plt\n",
    "    import seaborn as sns\n",
    "\n",
    "    # Set up plotting style\n",
    "    plt.style.use('default')\n",
    "    fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
    "    fig.suptitle('🐕 Dog Emotion Dataset Analysis', fontsize=16, fontweight='bold')\n",
    "\n",
    "    # 1. Emotion label distribution\n",
    "    label_counts = df['label'].value_counts()\n",
    "    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']\n",
    "    bars = axes[0,0].bar(label_counts.index, label_counts.values, color=colors[:len(label_counts)])\n",
    "    axes[0,0].set_title('📊 Emotion Label Distribution')\n",
    "    axes[0,0].set_xlabel('Emotion')\n",
    "    axes[0,0].set_ylabel('Count')\n",
    "    axes[0,0].tick_params(axis='x', rotation=45)\n",
    "\n",
    "    # Add value labels on bars\n",
    "    for bar, value in zip(bars, label_counts.values):\n",
    "        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,\n",
    "                      f'{value}', ha='center', va='bottom', fontweight='bold')\n",
    "\n",
    "    # 2. Emotion confidence distribution\n",
    "    emotion_cols = ['sad', 'angry', 'happy', 'relaxed']\n",
    "    emotion_data = [df[col] for col in emotion_cols]\n",
    "    bp1 = axes[0,1].boxplot(emotion_data, labels=emotion_cols, patch_artist=True)\n",
    "    axes[0,1].set_title('🎭 Emotion Confidence Distribution')\n",
    "    axes[0,1].set_ylabel('Confidence')\n",
    "    axes[0,1].tick_params(axis='x', rotation=45)\n",
    "\n",
    "    # Color the boxes\n",
    "    for patch, color in zip(bp1['boxes'], colors):\n",
    "        patch.set_facecolor(color)\n",
    "        patch.set_alpha(0.7)\n",
    "\n",
    "    # 3. Tail confidence distribution\n",
    "    tail_cols = ['down', 'up', 'mid']\n",
    "    tail_data = [df[col] for col in tail_cols]\n",
    "    tail_colors = ['#FF9999', '#66B2FF', '#99FF99']\n",
    "    bp2 = axes[1,0].boxplot(tail_data, labels=tail_cols, patch_artist=True)\n",
    "    axes[1,0].set_title('🐕 Tail Confidence Distribution')\n",
    "    axes[1,0].set_ylabel('Confidence')\n",
    "\n",
    "    # Color the boxes\n",
    "    for patch, color in zip(bp2['boxes'], tail_colors):\n",
    "        patch.set_facecolor(color)\n",
    "        patch.set_alpha(0.7)\n",
    "\n",
    "    # 4. Head detection confidence\n",
    "    axes[1,1].hist(df['head_confidence'], bins=20, alpha=0.7, color='#FFB366', edgecolor='black')\n",
    "    axes[1,1].set_title('🎯 Head Detection Confidence')\n",
    "    axes[1,1].set_xlabel('Confidence')\n",
    "    axes[1,1].set_ylabel('Count')\n",
    "    axes[1,1].axvline(df['head_confidence'].mean(), color='red', linestyle='--',\n",
    "                     label=f'Mean: {df[\"head_confidence\"].mean():.3f}')\n",
    "    axes[1,1].legend()\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "    # 5. Feature correlation matrix\n",
    "    plt.figure(figsize=(10, 8))\n",
    "    feature_cols = emotion_cols + tail_cols\n",
    "    correlation_matrix = df[feature_cols].corr()\n",
    "\n",
    "    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))\n",
    "    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,\n",
    "                square=True, mask=mask, cbar_kws={\"shrink\": .8})\n",
    "    plt.title('🔗 Feature Correlation Matrix', fontsize=14, fontweight='bold')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "    print(\"📊 All visualizations created successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "executionInfo": {
     "elapsed": 27,
     "status": "ok",
     "timestamp": 1751162319798,
     "user": {
      "displayName": "Hoàng Trịnh Việt",
      "userId": "17749859462560556703"
     },
     "user_tz": -420
    },
    "id": "fNjf53-NF0GF"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
