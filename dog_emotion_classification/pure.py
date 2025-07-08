"""
PURe Networks (Product-Unit Residual Networks) implementation for dog emotion classification.

Based on the paper: "Deep residual learning with product units"
This implementation supports multiple architectures: Pure18, Pure34, Pure50, Pure101, Pure152

Supports both training and inference for dog emotion classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import time
from tqdm import tqdm
import subprocess


class ProductUnit2D(nn.Module):
    """
    2D Product Unit layer for convolutional processing.
    
    This layer implements multiplicative feature interactions using:
    y(i,j) = exp(sum(w(m,n) * log(x(i+m,j+n))))
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ProductUnit2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Trainable weights for product unit
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Trainable threshold parameter to ensure positivity
        self.threshold = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=5.0)
        
    def forward(self, x):
        # Ensure inputs are positive by clamping with softplus threshold
        eps = 1e-7
        min_val = F.softplus(self.threshold) + eps
        x_clamped = torch.clamp(x, min=min_val)
        
        # Apply logarithm transformation
        log_x = torch.log(x_clamped)
        
        # Perform convolution in log space
        conv_output = F.conv2d(log_x, self.weight, self.bias, self.stride, self.padding)
        
        # Apply exponential to return to original space
        output = torch.exp(conv_output)
        
        return output


class BasicProductBlock(nn.Module):
    """
    Basic Product Unit Block for Pure18/Pure34.
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicProductBlock, self).__init__()
        
        # First convolution (standard)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution replaced with product unit
        self.prod_unit = ProductUnit2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        # First conv + BN (no ReLU in PURe)
        out = self.conv1(x)
        out = self.bn1(out)
        
        # Product unit + BN
        out = self.prod_unit(out)
        out = self.bn2(out)
        
        # Downsample identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)
            
        # Add residual connection
        out += identity
        
        return out


class BottleneckProductBlock(nn.Module):
    """
    Bottleneck Product Unit Block for Pure50/Pure101/Pure152.
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckProductBlock, self).__init__()
        
        # 1x1 conv to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 product unit
        self.prod_unit = ProductUnit2D(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv to expand channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        # 1x1 conv + BN
        out = self.conv1(x)
        out = self.bn1(out)
        
        # Product unit + BN
        out = self.prod_unit(out)
        out = self.bn2(out)
        
        # 1x1 conv + BN
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Downsample identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)
            
        # Add residual connection
        out += identity
        
        return out


class PUReNet(nn.Module):
    """
    Generic Product Unit Residual Network (PURe) model.
    
    Supports multiple architectures:
    - Pure18: [2, 2, 2, 2] with BasicProductBlock
    - Pure34: [3, 4, 6, 3] with BasicProductBlock
    - Pure50: [3, 4, 6, 3] with BottleneckProductBlock
    - Pure101: [3, 4, 23, 3] with BottleneckProductBlock  
    - Pure152: [3, 8, 36, 3] with BottleneckProductBlock
    """
    
    def __init__(self, block, layers, num_classes=4, input_size=512):
        super(PUReNet, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers with product units
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
            
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def Pure18(num_classes=4, input_size=512):
    """Pure18 model with basic product blocks."""
    return PUReNet(BasicProductBlock, [2, 2, 2, 2], num_classes, input_size)


def Pure34(num_classes=4, input_size=512):
    """Pure34 model with basic product blocks."""
    return PUReNet(BasicProductBlock, [3, 4, 6, 3], num_classes, input_size)


def Pure50(num_classes=4, input_size=512):
    """Pure50 model with bottleneck product blocks."""
    return PUReNet(BottleneckProductBlock, [3, 4, 6, 3], num_classes, input_size)


def Pure101(num_classes=4, input_size=512):
    """Pure101 model with bottleneck product blocks."""
    return PUReNet(BottleneckProductBlock, [3, 4, 23, 3], num_classes, input_size)


def Pure152(num_classes=4, input_size=512):
    """Pure152 model with bottleneck product blocks."""
    return PUReNet(BottleneckProductBlock, [3, 8, 36, 3], num_classes, input_size)


def get_pure_model(architecture, num_classes=4, input_size=512):
    """
    Get a Pure model by architecture name.
    
    Args:
        architecture (str): Model architecture ('pure18', 'pure34', 'pure50', 'pure101', 'pure152')
        num_classes (int): Number of output classes
        input_size (int): Input image size
        
    Returns:
        torch.nn.Module: Pure model
    """
    arch_lower = architecture.lower()
    
    if arch_lower == 'pure18':
        return Pure18(num_classes, input_size)
    elif arch_lower == 'pure34':
        return Pure34(num_classes, input_size)
    elif arch_lower == 'pure50':
        return Pure50(num_classes, input_size)
    elif arch_lower == 'pure101':
        return Pure101(num_classes, input_size)
    elif arch_lower == 'pure152':
        return Pure152(num_classes, input_size)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Supported: pure18, pure34, pure50, pure101, pure152")


class PureTrainer:
    """
    Trainer class for Pure models with comprehensive training functionality.
    """
    
    def __init__(self, model, device='cuda', checkpoint_dir='checkpoints'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training components
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.best_acc = 0.0
        self.train_losses = []
        self.train_accuracies = []
        
    def setup_training(self, learning_rate=1e-4, weight_decay=1e-4, step_size=10, gamma=0.1):
        """
        Setup training components (criterion, optimizer, scheduler).
        """
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=step_size, 
            gamma=gamma
        )
        
    def train_epoch(self, dataloader):
        """
        Train the model for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """
        Evaluate the model on validation/test data.
        
        Args:
            dataloader: Evaluation data loader
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader=None, epochs=30, save_best=True, save_interval=5):
        """
        Full training loop with validation and checkpointing.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            save_best: Whether to save the best model
            save_interval: Epoch interval for saving checkpoints
        """
        print(f"🚀 Starting training for {epochs} epochs on {self.device}")
        print(f"📊 Model: {self.model.__class__.__name__}")
        print(f"🎯 Classes: {self.model.num_classes}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Timing
            epoch_time = time.time() - start_time
            eta = epoch_time * (epochs - epoch - 1)
            
            # Logging
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Time: {epoch_time:.1f}s | ETA: {eta/60:.1f}m | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            
            if val_loader is not None:
                print(f"                    Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                current_acc = val_acc
            else:
                current_acc = train_acc
            
            if self.scheduler is not None:
                print(f"                    LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if save_best and current_acc > self.best_acc:
                self.best_acc = current_acc
                self.save_checkpoint(f"best_model.pth", epoch, current_acc)
                print("✅ Saved new best model!")
            
            # Save periodic checkpoints
            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth", epoch, current_acc)
                
        print(f"🎉 Training completed! Best accuracy: {self.best_acc:.4f}")
    
    def save_checkpoint(self, filename, epoch, accuracy):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            epoch: Current epoch
            accuracy: Current accuracy
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'accuracy': accuracy,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
        }
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
    def load_checkpoint(self, filepath, load_optimizer=True, load_scheduler=True):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if load_scheduler and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.best_acc = checkpoint.get('accuracy', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"📊 Accuracy: {self.best_acc:.4f}")


def get_pure_transforms(input_size=512, is_training=True):
    """
    Get data transforms for Pure models.
    
    Args:
        input_size (int): Input image size
        is_training (bool): Whether for training (includes augmentation)
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def predict_emotion_pure(image_path, model, transform, head_bbox=None, device='cuda', 
                        emotion_classes=['angry', 'happy', 'relaxed', 'sad']):
    """
    Predict dog emotion using Pure model.
    
    Args:
        image_path (str): Path to the image file
        model: Loaded Pure model
        transform: Image preprocessing transform
        head_bbox (list, optional): Bounding box [x1, y1, x2, y2] to crop head region
        device (str): Device to run inference on
        emotion_classes (list): List of emotion class names
        
    Returns:
        dict: Emotion probabilities and prediction status
    """
    try:
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Crop head region if bbox provided
        if head_bbox is not None:
            x1, y1, x2, y2 = map(int, head_bbox)
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:  # Valid crop region
                image = image[y1:y2, x1:x2]
        
        # Convert to PIL and apply transforms
        pil_image = Image.fromarray(image)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        
        # Map to emotion classes
        emotion_scores = {}
        for i, emotion in enumerate(emotion_classes):
            if i < len(probabilities):
                emotion_scores[emotion] = float(probabilities[i])
            else:
                emotion_scores[emotion] = 0.0
                
        emotion_scores['predicted'] = True
        return emotion_scores
        
    except Exception as e:
        print(f"❌ Error in Pure emotion classification for {image_path}: {e}")
        result = {emotion: 0.0 for emotion in emotion_classes}
        result['predicted'] = False
        return result


def load_pure_model(model_path, architecture='pure34', num_classes=4, input_size=512, device='cuda'):
    """
    Load a trained Pure model from checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint
        architecture (str): Model architecture ('pure18', 'pure34', 'pure50', etc.)
        num_classes (int): Number of emotion classes
        input_size (int): Input image size
        device (str): Device to load the model on
        
    Returns:
        tuple: (model, transform) where model is the loaded Pure model 
               and transform is the image preprocessing pipeline
    """
    print(f"🔄 Loading {architecture.upper()} model from: {model_path}")
    
    # Create model
    model = get_pure_model(architecture, num_classes, input_size)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state dict
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        
        # Create transform
        transform = get_pure_transforms(input_size, is_training=False)
        
        print(f"✅ {architecture.upper()} model loaded successfully on {device}")
        print(f"📏 Input size: {input_size}x{input_size}")
        print(f"🎯 Classes: {num_classes}")
        
        return model, transform
        
    except Exception as e:
        print(f"❌ Error loading {architecture.upper()} model: {e}")
        raise


def download_model(file_id, output_path):
    """
    Download a model file using gdown.
    
    Args:
        file_id (str): Google Drive file ID
        output_path (str): Path where to save the downloaded model
        
    Returns:
        str: Path to the downloaded model file or None if failed
    """
    try:
        # Download using gdown
        cmd = f"gdown {file_id} -O {output_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"✅ Model downloaded successfully: {output_path}")
            return output_path
        else:
            print(f"❌ Error downloading model: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None


# Model download links (example - replace with actual file IDs)
MODEL_URLS = {
    'pure34': '11Oy8lqKF7MeMWV89SR-kN6sNLwNi-jjQ',  # Example ID
    'pure50': 'your_pure50_file_id_here',
    'pure101': 'your_pure101_file_id_here',
}


def download_pure_model(architecture='pure34', output_dir='models'):
    """
    Download a pre-trained Pure model.
    
    Args:
        architecture (str): Model architecture to download
        output_dir (str): Directory to save the model
        
    Returns:
        str: Path to downloaded model or None if failed
    """
    if architecture.lower() not in MODEL_URLS:
        print(f"❌ No download URL available for {architecture}")
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{architecture}_best.pth")
    
    file_id = MODEL_URLS[architecture.lower()]
    return download_model(file_id, output_path) 