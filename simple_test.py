#!/usr/bin/env python3
"""
Simple test for core YOLO head detection functionality
"""

import json
import re
from typing import Dict, Any, Optional, Union

def safe_json_parse(json_string: str, fix_common_errors: bool = True) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON string with automatic error fixing for common issues
    """
    if not json_string or not isinstance(json_string, str):
        return None
    
    # First try direct parsing
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        if not fix_common_errors:
            print(f"JSON parsing failed: {e}")
            return None
        
        print(f"⚠️  JSON parsing error: {e}")
        print("🔧 Attempting to fix common JSON syntax issues...")
        
        # Fix common JSON syntax errors
        fixed_json = _fix_json_syntax_errors(json_string, e)
        
        if fixed_json != json_string:
            try:
                result = json.loads(fixed_json)
                print("✅ Successfully fixed and parsed JSON")
                return result
            except json.JSONDecodeError as e2:
                print(f"JSON parsing still failed after fixes: {e2}")
                return None
        else:
            print(f"Could not fix JSON syntax error: {e}")
            return None

def _fix_json_syntax_errors(json_string: str, original_error: json.JSONDecodeError) -> str:
    """
    Attempt to fix common JSON syntax errors
    """
    fixed = json_string
    error_msg = str(original_error).lower()
    
    # Fix trailing commas - this is the most common issue
    print("🔧 Fixing trailing commas...")
    # Remove trailing commas before closing brackets/braces
    fixed = re.sub(r',(\s*[\]\}])', r'\1', fixed)
    
    # Fix: Expected ',' or ']' after array element
    if "expected" in error_msg and ("after array element" in error_msg or "after object member" in error_msg):
        print("🔧 Fixing trailing comma syntax...")
        # More aggressive trailing comma removal
        fixed = re.sub(r',(\s*[\]\}])', r'\1', fixed)
    
    # Fix: Multiple consecutive commas
    if "," in fixed:
        print("🔧 Fixing multiple consecutive commas...")
        fixed = re.sub(r',+', ',', fixed)
        # Remove commas that are followed immediately by closing brackets
        fixed = re.sub(r',(\s*[\]\}])', r'\1', fixed)
    
    # Fix: Comma before opening brace/bracket
    fixed = re.sub(r',(\s*[\[\{])', r'\1', fixed)
    
    # Handle "Expecting value" errors which often indicate trailing commas
    if "expecting value" in error_msg:
        print("🔧 Fixing 'expecting value' error (likely trailing comma)...")
        # Remove trailing commas more aggressively
        fixed = re.sub(r',(\s*[\]\}])', r'\1', fixed)
        # Remove trailing comma at end of string
        fixed = re.sub(r',\s*$', '', fixed)
    
    # Handle "Expecting property name" errors in objects
    if "expecting property name" in error_msg:
        print("🔧 Fixing 'expecting property name' error...")
        fixed = re.sub(r',(\s*\})', r'\1', fixed)
    
    return fixed

def validate_head_bbox_format(bbox_data: Union[str, list]) -> Optional[list]:
    """
    Validate and normalize head bounding box format
    """
    if bbox_data is None:
        return None
    
    # If it's a string, try to parse as JSON
    if isinstance(bbox_data, str):
        if not bbox_data.strip() or bbox_data.strip() == '[]':
            return None
        
        parsed_bbox = safe_json_parse(bbox_data)
        if parsed_bbox is None:
            return None
        bbox_data = parsed_bbox
    
    # Validate list format
    if not isinstance(bbox_data, list):
        return None
    
    if len(bbox_data) != 4:
        return None
    
    try:
        # Convert to float and validate
        bbox = [float(x) for x in bbox_data]
        
        # Basic sanity checks
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None
        
        if any(x < 0 for x in bbox):
            return None
        
        return bbox
    except (ValueError, TypeError):
        return None

def test_json_parsing():
    """Test JSON parsing with the specific error mentioned by user"""
    print("🧪 Testing JSON Parsing...")
    
    # Test the specific JSON error: "Expected ',' or ']' after array element in JSON at position 116894"
    test_cases = [
        '[217.25555419921875, 27.69806671142578, 432.82,]',  # Trailing comma - user's issue
        '[100, 200, 300,]',  # Simple trailing comma
        '{"x": 100, "y": 200,}',  # Trailing comma in object
        '[1, 2, 3]',  # Valid JSON
    ]
    
    success_count = 0
    for i, test_json in enumerate(test_cases):
        print(f"\n  Test {i+1}: {test_json}")
        
        # Test with fixing
        result_with_fix = safe_json_parse(test_json, fix_common_errors=True)
        
        if result_with_fix is not None:
            print(f"    ✅ Fixed: {result_with_fix}")
            success_count += 1
        else:
            print(f"    ❌ Could not fix")
    
    print(f"\n  Success rate: {success_count}/{len(test_cases)}")
    return success_count == len(test_cases)

def test_bbox_validation():
    """Test bounding box validation"""
    print("\n🧪 Testing Bounding Box Validation...")
    
    test_cases = [
        ([100, 50, 200, 150], [100.0, 50.0, 200.0, 150.0]),  # Valid bbox
        ('[100, 50, 200, 150]', [100.0, 50.0, 200.0, 150.0]),  # JSON string
        ('[100, 50, 200, 150,]', [100.0, 50.0, 200.0, 150.0]),  # Trailing comma
        ([200, 50, 100, 150], None),  # Invalid (x2 <= x1)
        ([], None),  # Empty
    ]
    
    success_count = 0
    for i, (input_bbox, expected) in enumerate(test_cases):
        result = validate_head_bbox_format(input_bbox)
        success = result == expected
        
        print(f"    Test {i+1}: {input_bbox} -> {result} {'✅' if success else '❌'}")
        if success:
            success_count += 1
    
    print(f"    Success rate: {success_count}/{len(test_cases)}")
    return success_count == len(test_cases)

def main():
    """Run all tests"""
    print("🚀 Testing Core YOLO Head Detection Functionality...")
    
    results = {
        'json_parsing': test_json_parsing(),
        'bbox_validation': test_bbox_validation()
    }
    
    # Summary
    print(f"\n📋 Test Results Summary:")
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"    {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n✅ Core functionality implemented successfully!")
        print("\n📝 Implementation Summary:")
        print("   1. ✅ JSON parsing error handling (fixes trailing commas)")
        print("   2. ✅ Bounding box validation with format checking")
        print("   3. ✅ Handles the specific error: 'Expected \",\" or \"]\" after array element'")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}") 