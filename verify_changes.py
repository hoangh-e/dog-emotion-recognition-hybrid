#!/usr/bin/env python3
"""
Script để verify các thay đổi đã được áp dụng trong notebooks
"""

import os
import re

def check_notebook_changes(notebook_path):
    """Kiểm tra các thay đổi trong notebook"""
    print(f"\n🔍 Checking {notebook_path}...")
    
    if not os.path.exists(notebook_path):
        print(f"   ❌ File not found: {notebook_path}")
        return False
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        "New import function": "convert_dataframe_4class_to_3class_merge_relaxed_sad" in content,
        "New constant": "EMOTION_CLASSES_3CLASS_MERGE" in content,
        "Merge comment": "merge relaxed+sad" in content,
        "Updated print statement": "Imported 3-class utility functions (merge relaxed+sad)" in content,
        "Updated data processing": "Converting to 3-class configuration (merge relaxed+sad)" in content,
        "Summary text updated": "3-CLASS MERGE" in content,
        "Dataset description": "3-Class Merge: relaxed+sad→sad" in content,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

def main():
    """Main verification function"""
    
    notebooks = [
        "[Colab]_AlexNet_Cross_Validation_Training.ipynb",
        "[Colab]_DenseNet121_Cross_Validation_Training.ipynb", 
        "[Colab]_EfficientNet_B0_Cross_Validation_Training.ipynb",
        "[Colab]_ResNet_Pretrained_Training.ipynb",
        "[Colab]_Vision_Transformer_Cross_Validation_Training.ipynb"
    ]
    
    print("🔍 VERIFYING NOTEBOOK CHANGES")
    print("=" * 50)
    
    all_notebooks_updated = True
    
    for notebook in notebooks:
        if not check_notebook_changes(notebook):
            all_notebooks_updated = False
    
    print("\n" + "=" * 50)
    if all_notebooks_updated:
        print("✅ ALL NOTEBOOKS SUCCESSFULLY UPDATED!")
        print("\n🎯 Expected Results:")
        print("   - Classes: ['angry', 'happy', 'sad']")
        print("   - 'sad' class contains both 'relaxed' and 'sad' samples")
        print("   - Total samples preserved (~4000 instead of ~3000)")
        print("   - Better class balance")
    else:
        print("❌ Some notebooks need additional updates")
        print("   Manual fixes may be required for remaining issues")

if __name__ == "__main__":
    main() 