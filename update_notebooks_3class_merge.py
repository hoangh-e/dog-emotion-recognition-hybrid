#!/usr/bin/env python3
"""
Script để cập nhật tất cả Colab training notebooks từ 3-class configuration
(removing 'sad') sang 3-class configuration mới (merging 'relaxed' + 'sad' → 'sad')
"""

import json
import re
import os

def update_notebook_content(content):
    """Update notebook content với cấu hình merge mới"""
    
    # 1. Thay đổi documentation trong markdown cells - handle escape characters
    content = content.replace(
        "angry, happy, relaxed (removed \\'sad\\' class)",
        "angry, happy, sad (merged \\'relaxed\\' + \\'sad\\' → \\'sad\\')"
    )
    
    # Alternative patterns for escaped quotes
    content = content.replace(
        "angry, happy, relaxed (removed \\\"sad\\\" class)",
        "angry, happy, sad (merged \\\"relaxed\\\" + \\\"sad\\\" → \\\"sad\\\")"
    )
    
    content = content.replace(
        "- **Emotion Classes**: [\\'angry\\', \\'happy\\', \\'relaxed\\']",
        "- **Emotion Classes**: [\\'angry\\', \\'happy\\', \\'sad\\']"
    )
    
    # Alternative pattern
    content = content.replace(
        "[\\'angry\\', \\'happy\\', \\'relaxed\\']",
        "[\\'angry\\', \\'happy\\', \\'sad\\']"
    )
    
    content = content.replace(
        "- **Class Mapping**: 0=angry, 1=happy, 2=relaxed",
        "- **Class Mapping**: 0=angry, 1=happy, 2=sad (merged relaxed+sad)"
    )
    
    # 2. Thay đổi import statements
    content = content.replace(
        "# Import utility functions for 3-class conversion",
        "# Import utility functions for 3-class conversion (merge relaxed+sad)"
    )
    
    content = content.replace(
        "convert_dataframe_4class_to_3class,",
        "convert_dataframe_4class_to_3class_merge_relaxed_sad,"
    )
    
    content = content.replace(
        "get_3class_emotion_classes,",
        "get_3class_emotion_classes_merge,"
    )
    
    content = content.replace(
        "EMOTION_CLASSES_3CLASS",
        "EMOTION_CLASSES_3CLASS_MERGE"
    )
    
    # 3. Thay đổi print statements
    content = content.replace(
        "print(\\\"✅ Imported 3-class utility functions\\\")",
        "print(\\\"✅ Imported 3-class utility functions (merge relaxed+sad)\\\")"
    )
    
    # Alternative pattern
    content = content.replace(
        "\\\"✅ Imported 3-class utility functions\\\"",
        "\\\"✅ Imported 3-class utility functions (merge relaxed+sad)\\\""
    )
    
    # 4. Thay đổi data preparation logic
    content = content.replace(
        "# Filter dataset for 3-class configuration",
        "# Filter dataset for 3-class configuration (merge relaxed+sad)"
    )
    
    content = content.replace(
        "print(\\\"\\\\n🔧 Converting to 3-class configuration...\\\")",
        "print(\\\"\\\\n🔧 Converting to 3-class configuration (merge relaxed+sad)...\\\")"
    )
    
    content = content.replace(
        "\\\"\\\\n🔧 Converting to 3-class configuration...\\\"",
        "\\\"\\\\n🔧 Converting to 3-class configuration (merge relaxed+sad)...\\\""
    )
    
    content = content.replace(
        "# Read labels CSV and filter out 'sad' class",
        "# Read labels CSV and merge 'relaxed' + 'sad' → 'sad'"
    )
    
    content = content.replace(
        "# Convert to 3-class by removing 'sad' samples",
        "# Convert to 3-class by merging 'relaxed' + 'sad' → 'sad'"
    )
    
    content = content.replace(
        "filtered_df = convert_dataframe_4class_to_3class(labels_df, 'label')",
        "filtered_df = convert_dataframe_4class_to_3class_merge_relaxed_sad(labels_df, 'label')"
    )
    
    # 5. Thay đổi summary text
    content = content.replace(
        "TRAINING SUMMARY (3-CLASS)",
        "TRAINING SUMMARY (3-CLASS MERGE)"
    )
    
    content = content.replace(
        "Dataset: Dog Emotion Recognition (3-Class)",
        "Dataset: Dog Emotion Recognition (3-Class Merge: relaxed+sad→sad)"
    )
    
    # 6. Special handling cho ResNet notebook
    content = content.replace(
        "# Determine if we need 3-class conversion\\n\",\\n        \"if 'sad' in original_classes and len(original_classes) == 4:",
        "# Determine if we need 3-class conversion (merge relaxed+sad)\\n\",\\n        \"if 'sad' in original_classes and 'relaxed' in original_classes and len(original_classes) == 4:"
    )
    
    # Thêm pattern cụ thể hơn cho ResNet
    content = content.replace(
        "if 'sad' in original_classes and len(original_classes) == 4:",
        "if 'sad' in original_classes and 'relaxed' in original_classes and len(original_classes) == 4:"
    )
    
    return content

def update_notebook_file(notebook_path):
    """Update một notebook file"""
    print(f"🔧 Updating {notebook_path}...")
    
    try:
        # Đọc file
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Backup original
        backup_path = notebook_path + '.backup'
        if not os.path.exists(backup_path):  # Only create backup if doesn't exist
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   ✅ Created backup: {backup_path}")
        
        # Update content
        updated_content = update_notebook_content(content)
        
        # Check if any changes were made
        if content != updated_content:
            # Write updated file
            with open(notebook_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            print(f"   ✅ Updated {notebook_path}")
            return True
        else:
            print(f"   ℹ️  No changes needed for {notebook_path}")
            return True
        
    except Exception as e:
        print(f"   ❌ Error updating {notebook_path}: {e}")
        return False

def main():
    """Main function để update tất cả notebooks"""
    
    notebooks_to_update = [
        "[Colab]_AlexNet_Cross_Validation_Training.ipynb",
        "[Colab]_DenseNet121_Cross_Validation_Training.ipynb", 
        "[Colab]_EfficientNet_B0_Cross_Validation_Training.ipynb",
        "[Colab]_ResNet_Pretrained_Training.ipynb",
        "[Colab]_Vision_Transformer_Cross_Validation_Training.ipynb"
    ]
    
    print("🚀 Starting notebook updates for 3-class merge configuration...")
    print("=" * 60)
    
    updated_count = 0
    
    for notebook in notebooks_to_update:
        if os.path.exists(notebook):
            if update_notebook_file(notebook):
                updated_count += 1
        else:
            print(f"❌ File not found: {notebook}")
    
    print("\n" + "=" * 60)
    print(f"✅ Updated {updated_count}/{len(notebooks_to_update)} notebooks")
    print("\n🎯 Configuration changes:")
    print("   From: ['angry', 'happy', 'relaxed'] (removed 'sad')")
    print("   To:   ['angry', 'happy', 'sad'] (merged 'relaxed' + 'sad' → 'sad')")
    print("\n🔧 Key changes made:")
    print("   - Updated import functions to use merge variants")
    print("   - Changed data processing logic from removal to merge")
    print("   - Updated documentation and comments")
    print("   - Modified summary texts")
    print("\n✅ All notebooks ready for testing!")

if __name__ == "__main__":
    main() 