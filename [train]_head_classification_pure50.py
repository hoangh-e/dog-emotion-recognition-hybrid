# -*- coding: utf-8 -*-
"""
[Train] Head Classification - Pure50

Training script for Pure50 emotion classification model using the new pure.py module.
Based on Product Unit Residual Networks with Bottleneck blocks.
"""

# Install required packages
import subprocess
import sys

def install_packages():
    """Install required packages"""
    packages = ['torch', 'torchvision', 'albumentations', 'pandas', 'tqdm', 'gdown']
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Uncomment to install packages in new environment
# install_packages()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import os
import zipfile
from PIL import Image
import time
from tqdm import tqdm

# Import our Pure module
from dog_emotion_classification.pure import (
    Pure50, get_pure_model, PureTrainer, 
    get_pure_transforms, download_model
)

# Configuration
class Config:
    """Training configuration"""
    
    # Model settings
    ARCHITECTURE = 'pure50'
    NUM_CLASSES = 4
    INPUT_SIZE = 512
    
    # Training settings
    BATCH_SIZE = 8
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    STEP_SIZE = 10
    GAMMA = 0.1
    
    # Data settings
    DATA_URL_ID = "1ZAgz5u64i3LDbwMFpBXjzsKt6FrhNGdW"  # Cropped dataset
    DATA_ZIP = "cropped_dataset_4k_face.zip"
    DATA_EXTRACT_PATH = "data"
    DATA_ROOT = "data/cropped_dataset_4k_face/Dog Emotion"
    LABELS_CSV = "data/cropped_dataset_4k_face/Dog Emotion/labels.csv"
    
    # Checkpoint settings
    CHECKPOINT_DIR = "checkpoints"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DogEmotionDataset(Dataset):
    """
    Dataset class for dog emotion classification.
    """
    
    def __init__(self, root, labels_csv, transform=None):
        self.root = root
        df = pd.read_csv(labels_csv)
        self.items = df[['filename', 'label']].values

        # Create label to index mapping
        unique_labels = sorted(df['label'].unique())
        self.label2index = {name: i for i, name in enumerate(unique_labels)}
        self.index2label = {i: name for name, i in self.label2index.items()}

        self.transform = transform
        
        print(f"📊 Dataset loaded: {len(self.items)} samples")
        print(f"🏷️  Labels: {list(self.label2index.keys())}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn, label_str = self.items[idx]
        label_idx = self.label2index[label_str]
        img_path = os.path.join(self.root, label_str, fn)
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label_idx
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                img = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                img = torch.zeros(3, 224, 224)
            return img, label_idx


def download_and_extract_data():
    """
    Download and extract the dataset.
    """
    print("📥 Downloading dataset...")
    
    # Download dataset
    if not os.path.exists(Config.DATA_ZIP):
        download_result = download_model(Config.DATA_URL_ID, Config.DATA_ZIP)
        if download_result is None:
            print("❌ Failed to download dataset")
            return False
    else:
        print(f"✅ Dataset already exists: {Config.DATA_ZIP}")
    
    # Extract dataset
    if not os.path.exists(Config.DATA_EXTRACT_PATH):
        print("📂 Extracting dataset...")
        with zipfile.ZipFile(Config.DATA_ZIP, 'r') as zip_ref:
            zip_ref.extractall(Config.DATA_EXTRACT_PATH)
        print("✅ Dataset extracted successfully")
    else:
        print(f"✅ Dataset already extracted: {Config.DATA_EXTRACT_PATH}")
    
    # Verify data structure
    if os.path.exists(Config.DATA_ROOT) and os.path.exists(Config.LABELS_CSV):
        print("✅ Dataset structure verified")
        return True
    else:
        print("❌ Dataset structure verification failed")
        return False


def create_data_loaders():
    """
    Create training data loader.
    """
    print("🔄 Creating data loaders...")
    
    # Create transforms
    train_transform = get_pure_transforms(Config.INPUT_SIZE, is_training=True)
    
    # Create dataset
    dataset = DogEmotionDataset(
        root=Config.DATA_ROOT,
        labels_csv=Config.LABELS_CSV,
        transform=train_transform
    )
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if Config.DEVICE == 'cuda' else False
    )
    
    print(f"✅ Data loader created: {len(train_loader)} batches")
    return train_loader, dataset.label2index


def train_pure50():
    """
    Main training function for Pure50 model.
    """
    print("🚀 Starting Pure50 training script")
    print(f"🎯 Architecture: {Config.ARCHITECTURE.upper()}")
    print(f"📱 Device: {Config.DEVICE}")
    print(f"🖼️  Input size: {Config.INPUT_SIZE}x{Config.INPUT_SIZE}")
    print(f"📊 Batch size: {Config.BATCH_SIZE}")
    print(f"🔄 Epochs: {Config.EPOCHS}")
    
    # Step 1: Download and prepare data
    if not download_and_extract_data():
        print("❌ Failed to prepare dataset")
        return
    
    # Step 2: Create data loaders
    train_loader, label2index = create_data_loaders()
    num_classes = len(label2index)
    
    # Step 3: Create model
    print(f"🏗️  Creating {Config.ARCHITECTURE.upper()} model...")
    model = get_pure_model(
        architecture=Config.ARCHITECTURE,
        num_classes=num_classes,
        input_size=Config.INPUT_SIZE
    )
    
    print(f"✅ Model created: {model.__class__.__name__}")
    print(f"🎯 Classes: {num_classes}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 4: Create trainer
    trainer = PureTrainer(
        model=model,
        device=Config.DEVICE,
        checkpoint_dir=Config.CHECKPOINT_DIR
    )
    
    # Step 5: Setup training
    trainer.setup_training(
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        step_size=Config.STEP_SIZE,
        gamma=Config.GAMMA
    )
    
    print("✅ Training setup complete")
    
    # Step 6: Load checkpoint if exists
    best_checkpoint = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_checkpoint):
        try:
            trainer.load_checkpoint(best_checkpoint)
            print("📦 Loaded existing checkpoint")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
    
    # Step 7: Start training
    print("🚀 Starting training...")
    start_time = time.time()
    
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=None,  # No validation split for now
            epochs=Config.EPOCHS,
            save_best=True,
            save_interval=5
        )
        
        total_time = time.time() - start_time
        print(f"🎉 Training completed in {total_time/60:.1f} minutes")
        print(f"🏆 Best accuracy: {trainer.best_acc:.4f}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 8: Save final model
    try:
        final_checkpoint = os.path.join(Config.CHECKPOINT_DIR, "final_model.pth")
        trainer.save_checkpoint("final_model.pth", Config.EPOCHS - 1, trainer.best_acc)
        print(f"💾 Final model saved: {final_checkpoint}")
    except Exception as e:
        print(f"⚠️  Could not save final model: {e}")


def test_model():
    """
    Test the trained model on a sample image.
    """
    print("🧪 Testing trained model...")
    
    # Load best model
    best_checkpoint = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(best_checkpoint):
        print("❌ No trained model found")
        return
    
    try:
        from dog_emotion_classification.pure import load_pure_model, predict_emotion_pure
        
        model, transform = load_pure_model(
            model_path=best_checkpoint,
            architecture=Config.ARCHITECTURE,
            num_classes=Config.NUM_CLASSES,
            input_size=Config.INPUT_SIZE,
            device=Config.DEVICE
        )
        
        print("✅ Model loaded for testing")
        
        # Find a test image
        if os.path.exists(Config.DATA_ROOT):
            for emotion_dir in os.listdir(Config.DATA_ROOT):
                emotion_path = os.path.join(Config.DATA_ROOT, emotion_dir)
                if os.path.isdir(emotion_path):
                    for img_file in os.listdir(emotion_path)[:1]:  # Test first image
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            test_image = os.path.join(emotion_path, img_file)
                            
                            print(f"🖼️  Testing image: {test_image}")
                            result = predict_emotion_pure(
                                image_path=test_image,
                                model=model,
                                transform=transform,
                                device=Config.DEVICE
                            )
                            
                            print("📊 Prediction results:")
                            for emotion, score in result.items():
                                if emotion != 'predicted':
                                    print(f"   {emotion}: {score:.4f}")
                            
                            return
        
        print("❌ No test images found")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    """
    Main execution
    """
    print("=" * 60)
    print("🐕 Dog Emotion Classification - Pure50 Training")
    print("=" * 60)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not found")
        sys.exit(1)
    
    # Start training
    train_pure50()
    
    # Test model
    test_model()
    
    print("=" * 60)
    print("🎉 Pure50 training script completed!")
    print("=" * 60) 