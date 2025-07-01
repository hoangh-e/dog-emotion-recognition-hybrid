#!/usr/bin/env python3
"""
Demo script showing how to use the new pure.py module for training and inference.

This script demonstrates:
1. Creating different Pure architectures (Pure18, Pure34, Pure50, Pure101, Pure152)
2. Training a Pure model using PureTrainer
3. Loading and using a trained model for prediction
4. Downloading pre-trained models

Usage:
    python demo_pure_training.py
"""

import torch
import os
from dog_emotion_classification.pure import (
    # Model architectures
    Pure18, Pure34, Pure50, Pure101, Pure152, get_pure_model,
    # Training utilities
    PureTrainer, get_pure_transforms,
    # Inference utilities  
    load_pure_model, predict_emotion_pure,
    # Download utilities
    download_pure_model
)

def demo_create_models():
    """
    Demo: Create different Pure model architectures
    """
    print("🏗️ === Demo: Creating Pure Models ===")
    
    architectures = ['pure18', 'pure34', 'pure50', 'pure101', 'pure152']
    
    for arch in architectures:
        print(f"\n📊 Creating {arch.upper()}...")
        try:
            model = get_pure_model(arch, num_classes=4, input_size=512)
            params = sum(p.numel() for p in model.parameters())
            print(f"   ✅ {model.__class__.__name__}: {params:,} parameters")
        except Exception as e:
            print(f"   ❌ Error: {e}")


def demo_training_setup():
    """
    Demo: Setting up training for a Pure model
    """
    print("\n🚀 === Demo: Training Setup ===")
    
    # Create model
    print("📱 Creating Pure50 model...")
    model = get_pure_model('pure50', num_classes=4, input_size=512)
    
    # Create trainer
    print("🏋️ Setting up trainer...")
    trainer = PureTrainer(model, device='cuda', checkpoint_dir='demo_checkpoints')
    
    # Setup training components
    trainer.setup_training(
        learning_rate=1e-4,
        weight_decay=1e-4,
        step_size=10,
        gamma=0.1
    )
    
    print("✅ Training setup complete!")
    print(f"   📊 Model: {model.__class__.__name__}")
    print(f"   🎯 Classes: {model.num_classes}")
    print(f"   📏 Input size: {model.input_size}x{model.input_size}")
    print(f"   🔧 Optimizer: {trainer.optimizer.__class__.__name__}")
    print(f"   📈 Scheduler: {trainer.scheduler.__class__.__name__}")


def demo_transforms():
    """
    Demo: Data transforms for Pure models
    """
    print("\n🔄 === Demo: Data Transforms ===")
    
    # Training transforms (with augmentation)
    train_transform = get_pure_transforms(input_size=512, is_training=True)
    print("🔄 Training transforms (with augmentation):")
    for i, transform in enumerate(train_transform.transforms):
        print(f"   {i+1}. {transform.__class__.__name__}")
    
    # Inference transforms (no augmentation)
    val_transform = get_pure_transforms(input_size=512, is_training=False)
    print("\n🔍 Inference transforms (no augmentation):")
    for i, transform in enumerate(val_transform.transforms):
        print(f"   {i+1}. {transform.__class__.__name__}")


def demo_model_saving_loading():
    """
    Demo: Saving and loading Pure models
    """
    print("\n💾 === Demo: Model Saving/Loading ===")
    
    # Create and save a model
    print("📱 Creating Pure34 model...")
    model = get_pure_model('pure34', num_classes=4, input_size=512)
    
    # Save model
    os.makedirs('demo_models', exist_ok=True)
    model_path = 'demo_models/demo_pure34.pth'
    
    print(f"💾 Saving model to {model_path}...")
    torch.save(model.state_dict(), model_path)
    
    # Load model
    print(f"📂 Loading model from {model_path}...")
    try:
        loaded_model, transform = load_pure_model(
            model_path=model_path,
            architecture='pure34',
            num_classes=4,
            input_size=512,
            device='cpu'  # Use CPU for demo
        )
        print("✅ Model loaded successfully!")
        print(f"   📊 Model: {loaded_model.__class__.__name__}")
        print(f"   🎯 Classes: {loaded_model.num_classes}")
        print(f"   📏 Input size: {loaded_model.input_size}x{loaded_model.input_size}")
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")


def demo_prediction():
    """
    Demo: Using Pure model for prediction (simulated)
    """
    print("\n🔮 === Demo: Model Prediction ===")
    
    # Create a model for demo
    model = get_pure_model('pure34', num_classes=4, input_size=512)
    model.eval()
    
    # Create transform
    transform = get_pure_transforms(input_size=512, is_training=False)
    
    print("🎯 Prediction function signature:")
    print("   predict_emotion_pure(")
    print("       image_path,")
    print("       model,") 
    print("       transform,")
    print("       head_bbox=None,")
    print("       device='cuda',")
    print("       emotion_classes=['sad', 'angry', 'happy', 'relaxed']")
    print("   )")
    
    print("\n📝 Example usage:")
    print("   result = predict_emotion_pure(")
    print("       image_path='path/to/image.jpg',")
    print("       model=model,")
    print("       transform=transform")
    print("   )")
    print("   # Returns: {'sad': 0.2, 'angry': 0.1, 'happy': 0.6, 'relaxed': 0.1, 'predicted': True}")


def demo_architecture_comparison():
    """
    Demo: Compare different Pure architectures
    """
    print("\n📊 === Demo: Architecture Comparison ===")
    
    architectures = [
        ('pure18', 'Basic blocks: [2, 2, 2, 2]'),
        ('pure34', 'Basic blocks: [3, 4, 6, 3]'), 
        ('pure50', 'Bottleneck blocks: [3, 4, 6, 3]'),
        ('pure101', 'Bottleneck blocks: [3, 4, 23, 3]'),
        ('pure152', 'Bottleneck blocks: [3, 8, 36, 3]')
    ]
    
    print("🏗️ Pure Architecture Comparison:")
    print("   " + "="*60)
    print(f"   {'Architecture':<12} {'Parameters':<12} {'Description'}")
    print("   " + "="*60)
    
    for arch, desc in architectures:
        try:
            model = get_pure_model(arch, num_classes=4, input_size=512)
            params = sum(p.numel() for p in model.parameters())
            print(f"   {arch.upper():<12} {params/1e6:.1f}M{'':<7} {desc}")
        except Exception as e:
            print(f"   {arch.upper():<12} {'Error':<12} {str(e)[:40]}...")
    
    print("   " + "="*60)


def demo_training_workflow():
    """
    Demo: Complete training workflow
    """
    print("\n🔄 === Demo: Complete Training Workflow ===")
    
    print("📝 Step-by-step training workflow:")
    print()
    print("1️⃣ Create model:")
    print("   model = get_pure_model('pure50', num_classes=4, input_size=512)")
    print()
    print("2️⃣ Setup trainer:")
    print("   trainer = PureTrainer(model, device='cuda', checkpoint_dir='checkpoints')")
    print("   trainer.setup_training(learning_rate=1e-4, weight_decay=1e-4)")
    print()
    print("3️⃣ Prepare data:")
    print("   train_transform = get_pure_transforms(512, is_training=True)")
    print("   dataset = YourDataset(transform=train_transform)")
    print("   train_loader = DataLoader(dataset, batch_size=8, shuffle=True)")
    print()
    print("4️⃣ Start training:")
    print("   trainer.train(")
    print("       train_loader=train_loader,")
    print("       val_loader=val_loader,  # optional")
    print("       epochs=30,")
    print("       save_best=True")
    print("   )")
    print()
    print("5️⃣ Load trained model:")
    print("   model, transform = load_pure_model(")
    print("       model_path='checkpoints/best_model.pth',")
    print("       architecture='pure50',")
    print("       num_classes=4")
    print("   )")
    print()
    print("6️⃣ Make predictions:")
    print("   result = predict_emotion_pure(image_path, model, transform)")


def main():
    """
    Main demo function
    """
    print("🐕 === Pure Networks Demo ===")
    print("Demonstrating the dog_emotion_classification.pure module")
    print()
    
    # Run all demos
    demo_create_models()
    demo_transforms() 
    demo_training_setup()
    demo_model_saving_loading()
    demo_prediction()
    demo_architecture_comparison()
    demo_training_workflow()
    
    print("\n🎉 === Demo Complete ===")
    print("For actual training, see:")
    print("   - [train]_head_classification_pure34.py (Pure34)")
    print("   - [train]_head_classification_pure50.py (Pure50)")
    print()
    print("For more architectures, simply change the architecture parameter:")
    print("   get_pure_model('pure101', num_classes=4, input_size=512)")


if __name__ == "__main__":
    main() 