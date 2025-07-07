#!/usr/bin/env python3
"""
Script to fix dataset paths in all [Colab] notebook files.
Updates the dataset extraction and path references to match the new structure.
"""

import os
import glob
import json
import re

def fix_notebook_paths(notebook_path):
    """Fix dataset paths in a single notebook file."""
    print(f"Processing: {notebook_path}")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Track if any changes were made
        changed = False
        
        # Pattern 1: Fix EXTRACT_PATH definition and usage
        pattern1 = r'EXTRACT_PATH = "data"'
        if pattern1 in content:
            content = re.sub(pattern1, '', content)
            changed = True
        
        # Pattern 2: Fix extract path in zipfile.extractall
        pattern2 = r'zip_ref\.extractall\(EXTRACT_PATH\)'
        if re.search(pattern2, content):
            content = re.sub(pattern2, 'zip_ref.extractall(".")', content)
            changed = True
        
        # Pattern 3: Fix data_root path construction
        pattern3 = r'data_root = os\.path\.join\(EXTRACT_PATH, "cropped_dataset_4k_face", "Dog Emotion"\)'
        if re.search(pattern3, content):
            content = re.sub(pattern3, 'data_root = os.path.join("cropped_dataset_4k_face", "Dog Emotion")', content)
            changed = True
        
        # Pattern 4: Fix condition check for extraction
        pattern4 = r'if not os\.path\.exists\(EXTRACT_PATH\):'
        if re.search(pattern4, content):
            content = re.sub(pattern4, 'if not os.path.exists("cropped_dataset_4k_face"):', content)
            changed = True
        
        # Pattern 5: Alternative pattern for data_root
        pattern5 = r'os\.path\.join\(EXTRACT_PATH, "cropped_dataset_4k_face", "Dog Emotion"\)'
        if re.search(pattern5, content):
            content = re.sub(pattern5, 'os.path.join("cropped_dataset_4k_face", "Dog Emotion")', content)
            changed = True
        
        # Pattern 6: Fix any remaining EXTRACT_PATH references
        pattern6 = r'EXTRACT_PATH'
        if re.search(pattern6, content):
            content = re.sub(pattern6, '"cropped_dataset_4k_face"', content)
            changed = True
        
        # Save the file if changes were made
        if changed:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Updated: {notebook_path}")
            return True
        else:
            print(f"  ➡️ No changes needed: {notebook_path}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error processing {notebook_path}: {e}")
        return False

def main():
    """Main function to process all [Colab] notebook files."""
    print("🔧 Fixing dataset paths in all [Colab] notebook files...")
    print("=" * 60)
    
    # Find all [Colab] notebook files
    colab_files = glob.glob("*[Colab]*.ipynb")
    
    if not colab_files:
        print("❌ No [Colab] notebook files found!")
        return
    
    print(f"📋 Found {len(colab_files)} [Colab] notebook files:")
    for file in colab_files:
        print(f"  - {file}")
    
    print("\n🚀 Starting path fixes...")
    print("-" * 60)
    
    # Process each file
    updated_count = 0
    for notebook_path in colab_files:
        if fix_notebook_paths(notebook_path):
            updated_count += 1
    
    print("-" * 60)
    print(f"🎉 Processing completed!")
    print(f"📊 Summary:")
    print(f"  - Total files processed: {len(colab_files)}")
    print(f"  - Files updated: {updated_count}")
    print(f"  - Files unchanged: {len(colab_files) - updated_count}")
    
    print("\n✅ All dataset paths have been fixed!")
    print("📂 New structure:")
    print("   cropped_dataset_4k_face/")
    print("   └── Dog Emotion/")
    print("       ├── angry/")
    print("       ├── happy/")
    print("       ├── relaxed/")
    print("       ├── sad/")
    print("       └── labels.csv")

if __name__ == "__main__":
    main()
