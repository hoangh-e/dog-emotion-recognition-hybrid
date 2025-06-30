"""
🧪 Test Bbox Validation Functions từ dog_emotion_ml package
============================================================
File này test các validation functions đã được tích hợp vào package.
"""

import sys
import os
from pathlib import Path

def test_validation_functions():
    """Test validation functions từ dog_emotion_ml package"""
    
    print("🧪 Testing Bbox Validation Functions")
    print("=" * 50)
    
    # Test import
    try:
        from dog_emotion_ml import (
            calculate_iou,
            get_ground_truth_bbox,
            validate_head_detection_with_ground_truth,
            validate_bbox_format
        )
        print("✅ Successfully imported validation functions from dog_emotion_ml")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 1: Calculate IoU
    print("\n📐 Test 1: Calculate IoU")
    bbox1 = [100, 100, 200, 200]  # x1, y1, x2, y2
    bbox2 = [150, 150, 250, 250]
    bbox3 = [300, 300, 400, 400]  # No overlap
    
    iou1 = calculate_iou(bbox1, bbox2)
    iou2 = calculate_iou(bbox1, bbox3)
    
    print(f"   IoU between overlapping boxes: {iou1:.3f}")
    print(f"   IoU between non-overlapping boxes: {iou2:.3f}")
    
    assert iou1 > 0, "Overlapping boxes should have IoU > 0"
    assert iou2 == 0, "Non-overlapping boxes should have IoU = 0"
    print("   ✅ IoU calculation test passed")
    
    # Test 2: Validate bbox format
    print("\n🔍 Test 2: Validate bbox format")
    
    valid_bbox = [10, 10, 50, 50]
    invalid_bbox = [50, 50, 10, 10]  # x2 < x1, y2 < y1
    
    valid_result = validate_bbox_format(valid_bbox)
    invalid_result = validate_bbox_format(invalid_bbox)
    
    print(f"   Valid bbox result: {valid_result['valid']}")
    print(f"   Invalid bbox result: {invalid_result['valid']}")
    
    assert valid_result['valid'] == True, "Valid bbox should pass validation"
    assert invalid_result['valid'] == False, "Invalid bbox should fail validation"
    print("   ✅ Bbox format validation test passed")
    
    # Test 3: Ground truth bbox (sẽ return None nếu không có file)
    print("\n🏷️ Test 3: Get ground truth bbox")
    
    sample_image_path = "/nonexistent/path/image.jpg"
    gt_bbox = get_ground_truth_bbox(sample_image_path)
    
    print(f"   Ground truth for nonexistent image: {gt_bbox}")
    assert gt_bbox is None, "Should return None for nonexistent image"
    print("   ✅ Ground truth bbox test passed")
    
    # Test 4: Head detection validation (simulated)
    print("\n✅ Test 4: Head detection validation")
    
    predicted_bbox = [100, 100, 200, 200]
    nonexistent_image = "/nonexistent/path/image.jpg"
    
    validation_result = validate_head_detection_with_ground_truth(
        predicted_bbox, nonexistent_image, iou_threshold=0.3
    )
    
    print(f"   Validation result: {validation_result}")
    assert validation_result['valid'] == False, "Should fail when no ground truth available"
    assert validation_result['reason'] == 'No ground truth available'
    print("   ✅ Head detection validation test passed")
    
    print("\n🎉 All tests passed successfully!")
    return True


def demo_usage():
    """Demo cách sử dụng trong thực tế"""
    
    print("\n🎯 DEMO: Practical Usage")
    print("=" * 50)
    
    try:
        from dog_emotion_ml import calculate_iou, validate_bbox_format
        
        # Demo realistic scenario
        detected_bbox = [120, 80, 220, 180]  # YOLO prediction
        ground_truth_bbox = [110, 75, 210, 185]  # Ground truth
        
        # Calculate overlap
        iou = calculate_iou(detected_bbox, ground_truth_bbox)
        print(f"📐 IoU between prediction and ground truth: {iou:.3f}")
        
        # Determine if acceptable
        threshold = 0.3
        acceptable = iou >= threshold
        print(f"✅ Acceptable (IoU ≥ {threshold}): {acceptable}")
        
        # Validate format
        bbox_validation = validate_bbox_format(detected_bbox)
        print(f"🔍 Bbox format valid: {bbox_validation['valid']}")
        
        # Real-world decision logic
        if acceptable and bbox_validation['valid']:
            print("🟢 ACCEPT: Detection passes all validation")
        else:
            print("🔴 REJECT: Detection fails validation")
            
    except ImportError:
        print("⚠️ dog_emotion_ml package not available for demo")


if __name__ == "__main__":
    print("🎯 Bbox Validation Test Suite")
    print("Testing functions from dog_emotion_ml package...")
    
    success = test_validation_functions()
    
    if success:
        demo_usage()
        
        print("\n📝 USAGE SUMMARY:")
        print("=" * 50)
        print("1. Import: from dog_emotion_ml import calculate_iou, ...")
        print("2. Use in prediction: validate_head_detection_with_ground_truth(...)")
        print("3. Check IoU: calculate_iou(predicted_bbox, ground_truth_bbox)")
        print("4. Validate format: validate_bbox_format(bbox)")
        print("\n✅ Ready to use in your notebook!")
    else:
        print("\n❌ Tests failed. Check dog_emotion_ml package installation.") 