"""
🎯 UPDATED PREDICTION PIPELINE WITH BBOX VALIDATION
==================================================
File này demo cách sử dụng validation functions từ dog_emotion_ml package
để cập nhật quy trình prediction với bbox validation.

Cách sử dụng:
1. Import functions từ dog_emotion_ml package
2. Cập nhật predict_head_detection để sử dụng validation
3. Run prediction với enable_bbox_validation=True
"""

import os
import sys
from pathlib import Path

# Import từ dog_emotion_ml package
try:
    from dog_emotion_ml import (
        calculate_iou,
        get_ground_truth_bbox,
        validate_head_detection_with_ground_truth,
        validate_bbox_format
    )
    BBOX_VALIDATION_AVAILABLE = True
    print("✅ Bbox validation functions imported from dog_emotion_ml package")
except ImportError as e:
    print(f"⚠️ dog_emotion_ml package not available: {e}")
    print("   Bbox validation will be disabled")
    BBOX_VALIDATION_AVAILABLE = False


def predict_head_detection_with_validation(image_path, model, confidence_threshold=0.5, 
                                         enable_bbox_validation=True, iou_threshold=0.3):
    """
    🎯 Predict dog head detection using YOLO with optional bbox validation
    
    Parameters:
    -----------
    image_path : str
        Đường dẫn đến ảnh
    model : YOLO
        YOLO model đã load
    confidence_threshold : float
        Ngưỡng confidence tối thiểu (default: 0.5)
    enable_bbox_validation : bool
        Có enable bbox validation với ground truth hay không (default: True)
    iou_threshold : float
        Ngưỡng IoU để chấp nhận bbox (default: 0.3)
    
    Returns:
    --------
    dict: {
        'detected': bool, 
        'confidence': float, 
        'bbox': list or None,
        'validation': dict with validation details,
        'skipped_reason': str (if validation failed)
    }
    """
    try:
        results = model(image_path, verbose=False)
        
        best_detection = None
        best_confidence = 0.0
        validation_details = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf)
                    if confidence > confidence_threshold:
                        bbox = box.xyxy[0].cpu().numpy().tolist()
                        
                        # Validate bbox với ground truth nếu được enable
                        validation_result = {'valid': True, 'reason': 'No validation'}
                        if enable_bbox_validation and BBOX_VALIDATION_AVAILABLE:
                            validation_result = validate_head_detection_with_ground_truth(
                                bbox, image_path, iou_threshold
                            )
                            validation_details.append({
                                'bbox': bbox,
                                'confidence': confidence,
                                'validation': validation_result
                            })
                        
                        # Chỉ chấp nhận nếu validation pass (hoặc không có validation)
                        if validation_result.get('valid', True):
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_detection = {
                                    'detected': True,
                                    'confidence': confidence,
                                    'bbox': bbox,
                                    'validation': validation_result
                                }
        
        if best_detection is None:
            # Kiểm tra xem có detection nào bị reject do validation không
            rejected_detections = [d for d in validation_details if not d['validation']['valid']]
            if rejected_detections:
                # Có detection nhưng bị reject do validation
                best_rejected = max(rejected_detections, key=lambda x: x['confidence'])
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'bbox': None,
                    'validation': best_rejected['validation'],
                    'skipped_reason': f"Bbox validation failed: {best_rejected['validation']['reason']}",
                    'rejected_bbox': best_rejected['bbox'],
                    'rejected_confidence': best_rejected['confidence']
                }
            else:
                # Không có detection nào thỏa mãn confidence threshold
                return {
                    'detected': False,
                    'confidence': 0.0,
                    'bbox': None,
                    'validation': {'valid': False, 'reason': 'No detection above threshold'},
                    'skipped_reason': 'No detection found above confidence threshold'
                }
        
        return best_detection
        
    except Exception as e:
        print(f"❌ Error in head detection for {image_path}: {e}")
        return {
            'detected': False,
            'confidence': 0.0,
            'bbox': None,
            'validation': {'valid': False, 'reason': f'Error: {e}'},
            'skipped_reason': f'Processing error: {e}'
        }


def demo_validation_usage():
    """
    📝 Demo cách sử dụng validation functions
    """
    print("🎯 DEMO: Bbox Validation Functions Usage")
    print("=" * 50)
    
    # Demo 1: Calculate IoU
    bbox1 = [100, 100, 200, 200]  # x1, y1, x2, y2
    bbox2 = [150, 150, 250, 250]
    
    if BBOX_VALIDATION_AVAILABLE:
        iou = calculate_iou(bbox1, bbox2)
        print(f"📐 IoU between {bbox1} and {bbox2}: {iou:.3f}")
        
        # Demo 2: Validate bbox format
        bbox_valid = validate_bbox_format(bbox1)
        print(f"✅ Bbox format validation: {bbox_valid}")
        
        # Demo 3: Get ground truth (sẽ return None nếu không có file)
        sample_image = "/content/19/06-7/test/images/sample.jpg"
        if os.path.exists(sample_image):
            gt_bbox = get_ground_truth_bbox(sample_image)
            print(f"🏷️ Ground truth bbox for sample: {gt_bbox}")
        else:
            print("⚠️ Sample image not found for ground truth demo")
    else:
        print("⚠️ Validation functions not available")


def get_validation_statistics(predictions_list):
    """
    📊 Tính toán thống kê validation
    
    Parameters:
    -----------
    predictions_list : list
        List của prediction results với validation info
    
    Returns:
    --------
    dict: Validation statistics
    """
    stats = {
        'total_images': len(predictions_list),
        'detected': 0,
        'skipped_validation_failed': 0,
        'skipped_no_detection': 0,
        'validation_enabled_count': 0,
        'avg_iou': 0.0,
        'iou_distribution': {'high': 0, 'medium': 0, 'low': 0}
    }
    
    iou_scores = []
    
    for pred in predictions_list:
        if pred['detected']:
            stats['detected'] += 1
            
            # Check validation info
            validation = pred.get('validation', {})
            if 'iou' in validation:
                stats['validation_enabled_count'] += 1
                iou = validation['iou']
                iou_scores.append(iou)
                
                # Categorize IoU
                if iou >= 0.5:
                    stats['iou_distribution']['high'] += 1
                elif iou >= 0.3:
                    stats['iou_distribution']['medium'] += 1
                else:
                    stats['iou_distribution']['low'] += 1
        else:
            # Check skip reason
            skip_reason = pred.get('skipped_reason', '')
            if 'validation failed' in skip_reason.lower():
                stats['skipped_validation_failed'] += 1
            else:
                stats['skipped_no_detection'] += 1
    
    # Calculate average IoU
    if iou_scores:
        stats['avg_iou'] = sum(iou_scores) / len(iou_scores)
    
    return stats


def print_validation_summary(stats):
    """
    📋 In ra summary của validation results
    """
    print("\n🎯 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"📸 Total images processed: {stats['total_images']}")
    print(f"✅ Successfully detected: {stats['detected']}")
    print(f"❌ Skipped (validation failed): {stats['skipped_validation_failed']}")
    print(f"🔍 Skipped (no detection): {stats['skipped_no_detection']}")
    print(f"📊 Validation enabled: {stats['validation_enabled_count']}")
    
    if stats['validation_enabled_count'] > 0:
        print(f"\n📐 Average IoU: {stats['avg_iou']:.3f}")
        print("📈 IoU Distribution:")
        print(f"   🟢 High (≥0.5): {stats['iou_distribution']['high']}")
        print(f"   🟡 Medium (0.3-0.5): {stats['iou_distribution']['medium']}")
        print(f"   🔴 Low (<0.3): {stats['iou_distribution']['low']}")
    
    # Calculate success rate
    success_rate = (stats['detected'] / stats['total_images']) * 100 if stats['total_images'] > 0 else 0
    print(f"\n🎯 Detection Success Rate: {success_rate:.1f}%")
    
    if stats['skipped_validation_failed'] > 0:
        reject_rate = (stats['skipped_validation_failed'] / stats['total_images']) * 100
        print(f"⚠️ Validation Rejection Rate: {reject_rate:.1f}%")


# Configuration constants để dễ sử dụng
BBOX_VALIDATION_CONFIG = {
    'ENABLE_BBOX_VALIDATION': True,  # Bật/tắt validation
    'IoU_THRESHOLD': 0.3,           # Ngưỡng IoU (0.3 = balanced, 0.5 = strict)
    'CONFIDENCE_THRESHOLD': 0.5      # Ngưỡng confidence cho YOLO
}

if __name__ == "__main__":
    print("🎯 Bbox Validation Demo")
    demo_validation_usage()
    print("\n✅ Updated prediction pipeline ready to use!")
    print("\n📝 Usage in notebook:")
    print("   from updated_prediction_with_validation import predict_head_detection_with_validation")
    print("   result = predict_head_detection_with_validation(image_path, model, enable_bbox_validation=True)") 