import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO
import base64
from io import BytesIO
import traceback

# Add parent directory to path to import dog_emotion_classification
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from dog_emotion_classification.pure34 import load_pure34_model, predict_emotion_pure34
    from dog_emotion_classification.pure50 import load_pure50_model, predict_emotion_pure50
    from dog_emotion_classification.pure import load_pure_model, predict_emotion_pure
    from dog_emotion_classification.resnet import load_resnet_model, predict_emotion_resnet
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import classification modules: {e}")
    MODELS_AVAILABLE = False

class PredictionEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded_models = {}
        self.emotion_classes = ['sad', 'angry', 'happy', 'relaxed']
    
    def _load_head_detection_model(self, model_info):
        """Load YOLO head detection model"""
        model_key = f"head_{model_info['name']}"
        if model_key not in self.loaded_models:
            try:
                model = YOLO(model_info['file_path'])
                self.loaded_models[model_key] = model
                return model
            except Exception as e:
                raise Exception(f"Failed to load head detection model: {str(e)}")
        return self.loaded_models[model_key]
    
    def _load_classification_model(self, model_info):
        """Load emotion classification model based on type"""
        if not MODELS_AVAILABLE:
            raise Exception("Classification modules not available. Please check dog_emotion_classification package.")
            
        model_key = f"classification_{model_info['name']}"
        if model_key not in self.loaded_models:
            try:
                model_path = model_info['file_path']
                
                # Determine model type from filename or metadata
                filename = model_info['filename'].lower()
                
                if 'pure34' in filename:
                    model = load_pure34_model(model_path, num_classes=4, device=self.device)
                    model_type = 'pure34'
                elif 'pure50' in filename:
                    model = load_pure50_model(model_path, num_classes=4, device=self.device)
                    model_type = 'pure50'
                elif 'pure' in filename:
                    model = load_pure_model(model_path, num_classes=4, device=self.device)
                    model_type = 'pure'
                elif 'resnet' in filename:
                    model = load_resnet_model(model_path, num_classes=4, device=self.device)
                    model_type = 'resnet'
                else:
                    # Default to pure34 if type cannot be determined
                    model = load_pure34_model(model_path, num_classes=4, device=self.device)
                    model_type = 'pure34'
                
                self.loaded_models[model_key] = {'model': model, 'type': model_type}
                return self.loaded_models[model_key]
                
            except Exception as e:
                raise Exception(f"Failed to load classification model: {str(e)}")
        return self.loaded_models[model_key]
    
    def _detect_head(self, image_path, head_model):
        """Detect dog head using YOLO model"""
        try:
            results = head_model(image_path)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the first detection (highest confidence)
                box = results[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                return {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence)
                }
            else:
                return None
                
        except Exception as e:
            raise Exception(f"Head detection failed: {str(e)}")
    
    def _fallback_prediction(self, image_path, model, transform, head_bbox=None):
        """Fallback prediction method when classification modules are not available"""
        try:
            # Load and preprocess image
            if head_bbox:
                image = cv2.imread(image_path)
                x1, y1, x2, y2 = head_bbox['bbox']
                cropped = image[y1:y2, x1:x2]
                pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.open(image_path).convert('RGB')
            
            # Apply transform
            input_tensor = transform(pil_image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = self.emotion_classes[predicted.item()]
                confidence_score = confidence.item()
                
                # Get all probabilities
                all_probs = probabilities[0].cpu().numpy()
                prob_dict = {cls: float(prob) for cls, prob in zip(self.emotion_classes, all_probs)}
                
                return {
                    'predicted_class': predicted_class,
                    'confidence': confidence_score,
                    'probabilities': prob_dict
                }
                
        except Exception as e:
            raise Exception(f"Fallback prediction failed: {str(e)}")
    
    def _predict_emotion(self, image_path, classification_model_info, head_bbox=None):
        """Predict emotion using classification model"""
        try:
            model_data = classification_model_info
            model = model_data['model']
            model_type = model_data['type']
            
            # Create transform based on model type
            if model_type in ['pure34', 'pure50', 'pure']:
                input_size = 512
            else:  # resnet
                input_size = 224
            
            transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Predict based on model type
            if MODELS_AVAILABLE:
                if model_type == 'pure34':
                    result = predict_emotion_pure34(image_path, model, transform, head_bbox, self.device)
                elif model_type == 'pure50':
                    result = predict_emotion_pure50(image_path, model, transform, head_bbox, self.device)
                elif model_type == 'pure':
                    result = predict_emotion_pure(image_path, model, transform, head_bbox, self.device)
                elif model_type == 'resnet':
                    result = predict_emotion_resnet(image_path, model, transform, head_bbox, self.device)
                else:
                    raise Exception(f"Unknown model type: {model_type}")
            else:
                # Fallback prediction if modules not available
                result = self._fallback_prediction(image_path, model, transform, head_bbox)
            
            return result
            
        except Exception as e:
            raise Exception(f"Emotion prediction failed: {str(e)}")
    
    def _create_result_image(self, image_path, head_bbox, emotion_result):
        """Create result image with bounding box and emotion label"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Could not read image")
            
            # Draw bounding box if available
            if head_bbox:
                x1, y1, x2, y2 = head_bbox['bbox']
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add emotion label
                if emotion_result:
                    label = f"{emotion_result['predicted_class']} ({emotion_result['confidence']:.2f})"
                    cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.9, (0, 255, 0), 2)
            
            # Convert to base64 for web display
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return f"data:image/jpeg;base64,{img_base64}"
            
        except Exception as e:
            print(f"Warning: Could not create result image: {str(e)}")
            return None
    
    def predict_single(self, image_path, head_model_info, classification_model_info):
        """Predict emotion for a single image"""
        try:
            # Load models
            head_model = self._load_head_detection_model(head_model_info)
            classification_model = self._load_classification_model(classification_model_info)
            
            # Detect head
            head_detection = self._detect_head(image_path, head_model)
            
            # Predict emotion
            emotion_result = self._predict_emotion(image_path, classification_model, head_detection)
            
            # Create result image
            result_image = self._create_result_image(image_path, head_detection, emotion_result)
            
            return {
                'image_path': os.path.basename(image_path),
                'head_detection': head_detection,
                'emotion_prediction': emotion_result,
                'result_image': result_image
            }
            
        except Exception as e:
            return {
                'image_path': os.path.basename(image_path),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def predict_batch(self, image_paths, head_model_info, classification_model_info):
        """Predict emotions for multiple images"""
        results = []
        
        try:
            # Load models once for batch processing
            head_model = self._load_head_detection_model(head_model_info)
            classification_model = self._load_classification_model(classification_model_info)
            
            for image_path in image_paths:
                try:
                    # Detect head
                    head_detection = self._detect_head(image_path, head_model)
                    
                    # Predict emotion
                    emotion_result = self._predict_emotion(image_path, classification_model, head_detection)
                    
                    results.append({
                        'image_path': os.path.basename(image_path),
                        'head_detection': head_detection,
                        'emotion_prediction': emotion_result,
                        'status': 'success'
                    })
                    
                except Exception as e:
                    results.append({
                        'image_path': os.path.basename(image_path),
                        'error': str(e),
                        'status': 'error'
                    })
            
            return results
            
        except Exception as e:
            return [{'error': f"Batch prediction failed: {str(e)}"}] 