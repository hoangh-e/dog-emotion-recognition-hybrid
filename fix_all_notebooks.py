#!/usr/bin/env python3
"""
Script to fix common issues in all Colab training notebooks
=========================================================

This script addresses the following issues:
1. Dataset loading problems (Path imports, dataset structure)
2. num_workers=2 causing issues in Colab
3. Epochs set too high for testing
4. Missing imports and dependencies

Author: Dog Emotion Recognition Team
Date: 2024
"""

import os
import json
import re
from pathlib import Path

def fix_notebook_issues():
    """Fix common issues in all Colab notebooks"""
    
    print("🔧 Starting comprehensive notebook fixes...")
    
    # Get all notebook files
    notebook_files = []
    for file in os.listdir("."):
        if file.startswith("[Colab]") and file.endswith(".ipynb"):
            notebook_files.append(file)
    
    print(f"📁 Found {len(notebook_files)} notebooks to fix")
    
    fixes_applied = 0
    
    for notebook_file in notebook_files:
        print(f"\n🔍 Processing: {notebook_file}")
        
        try:
            # Read notebook
            with open(notebook_file, 'r', encoding='utf-8') as f:
                notebook_content = f.read()
            
            # Parse JSON
            notebook_data = json.loads(notebook_content)
            
            # Track if any changes were made
            changes_made = False
            
            # Fix issues in all cells
            for cell in notebook_data.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        source_text = ''.join(source)
                    else:
                        source_text = source
                    
                    original_source = source_text
                    
                    # Fix 1: Replace num_workers=2 with num_workers=0
                    source_text = re.sub(r'num_workers=2', 'num_workers=0', source_text)
                    
                    # Fix 2: Replace epochs=50 with epochs=10 for faster testing
                    source_text = re.sub(r'epochs\s*=\s*50', 'epochs = 10  # Reduced for faster testing', source_text)
                    
                    # Fix 3: Add missing imports at the beginning if needed
                    if 'import glob' not in source_text and 'glob.glob' in source_text:
                        source_text = 'import glob\n' + source_text
                    
                    # Fix 4: Replace Path usage with os.path for dataset loading
                    if 'Path(dataset_dir)' in source_text:
                        source_text = source_text.replace('Path(dataset_dir)', 'dataset_dir')
                        source_text = source_text.replace('class_dir.exists()', 'os.path.exists(class_dir)')
                        source_text = source_text.replace('class_dir.glob(', 'glob.glob(os.path.join(class_dir, ')
                        source_text = source_text.replace("'*.jpg')) + list(class_dir.glob('*.png'))", "'*.jpg')) + glob.glob(os.path.join(class_dir, '*.png'))")
                    
                    # Fix 5: Add robust dataset loading
                    if 'for class_name in emotion_classes:' in source_text and 'class_dir =' in source_text:
                        dataset_fix = '''
# Find dataset directory - check multiple possible locations
dataset_paths = ["dog_emotion_dataset", "dataset", "data"]
actual_dataset_dir = None

for path in dataset_paths:
    if os.path.exists(path):
        # Check if it contains emotion class directories
        class_dirs = [os.path.join(path, cls) for cls in emotion_classes]
        if any(os.path.exists(d) for d in class_dirs):
            actual_dataset_dir = path
            print(f"📊 Found dataset at: {actual_dataset_dir}")
            break

# If no dataset found, create sample dataset
if actual_dataset_dir is None:
    print("⚠️ No dataset found. Creating sample dataset...")
    actual_dataset_dir = "dataset"
    os.makedirs(actual_dataset_dir, exist_ok=True)
    
    # Create sample images for each class
    for class_name in emotion_classes:
        class_dir = os.path.join(actual_dataset_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Create 25 sample images per class for better cross-validation
        for i in range(25):
            # Create random RGB image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = os.path.join(class_dir, f"sample_{i:03d}.jpg")
            img.save(img_path)
        
        print(f"📁 Created {class_name}: 25 sample images")

# Collect all images and labels
all_images = []
all_labels = []
import glob

for class_name in emotion_classes:
    class_dir = os.path.join(actual_dataset_dir, class_name)
    if os.path.exists(class_dir):
        images = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            images.extend(glob.glob(os.path.join(class_dir, ext)))
        
        all_images.extend(images)
        all_labels.extend([class_to_idx[class_name]] * len(images))
        print(f"📁 {class_name}: {len(images)} images")
    else:
        print(f"⚠️ Class directory not found: {class_dir}")

print(f"\\n📊 Total dataset: {len(all_images)} images")
print(f"📊 Classes: {emotion_classes}")

# Ensure we have at least some data
if len(all_images) == 0:
    print("❌ No images found! Creating minimal dataset for testing...")
    
    # Create minimal dataset
    actual_dataset_dir = "dataset"
    os.makedirs(actual_dataset_dir, exist_ok=True)
    
    for class_name in emotion_classes:
        class_dir = os.path.join(actual_dataset_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Create 10 sample images per class
        for i in range(10):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = os.path.join(class_dir, f"sample_{i:03d}.jpg")
            img.save(img_path)
            all_images.append(img_path)
            all_labels.append(class_to_idx[class_name])
        
        print(f"📁 Created {class_name}: 10 sample images")
    
    print(f"✅ Created total: {len(all_images)} sample images")

# Convert to numpy arrays
all_images = np.array(all_images)
all_labels = np.array(all_labels)
'''
                        # Only add if not already present
                        if 'Find dataset directory - check multiple possible locations' not in source_text:
                            # Find the position to insert the fix
                            insert_pos = source_text.find('for class_name in emotion_classes:')
                            if insert_pos != -1:
                                source_text = source_text[:insert_pos] + dataset_fix + '\n\n' + source_text[insert_pos:]
                    
                    # Update cell source if changes were made
                    if source_text != original_source:
                        changes_made = True
                        # Convert back to list format
                        cell['source'] = source_text.split('\n')
                        # Add newline to each line except the last
                        for i in range(len(cell['source']) - 1):
                            cell['source'][i] += '\n'
            
            # Save notebook if changes were made
            if changes_made:
                with open(notebook_file, 'w', encoding='utf-8') as f:
                    json.dump(notebook_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Fixed: {notebook_file}")
                fixes_applied += 1
            else:
                print(f"ℹ️  No changes needed: {notebook_file}")
                
        except Exception as e:
            print(f"❌ Error processing {notebook_file}: {e}")
    
    print(f"\n🎉 Completed! Fixed {fixes_applied} notebooks out of {len(notebook_files)} total")

if __name__ == "__main__":
    fix_notebook_issues() 