"""
Nutrition5k Paper Experiment Implementation

This script implements the exact training setup described in the original
Nutrition5k paper (Thames et al. 2021), including:
- InceptionV2 architecture
- RMSprop optimizer with 0.9 momentum
- Weight decay of 1e-6
- Learning rate of 1e-4
- 256x256 image size with 224x224 crops
- Early stopping with patience 5
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import numpy as np
import json

# Import the dataset class
from nutrition5k_dataloader import Nutrition5kPaperDataset, create_train_val_split

# Import the model class
from nutrition5k_inceptionv2_model import build_nutrition5k_inceptionv2_model

# Constants from the paper (modified to keep at 256x256)
IMG_SIZE = 256         # Resize size
BATCH_SIZE = 32        # From the paper
LEARNING_RATE = 1e-4   # From the paper
WEIGHT_DECAY = 1e-6    # From the paper
MOMENTUM = 0.9         # From the paper (for RMSprop)


class EarlyStopping:
    """Early stopping as implemented in the paper (patience=5)"""
    
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_loss, epoch):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_epoch = epoch
            return False
            
        if val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.best_epoch = epoch
            self.counter = 0
            
        return self.early_stop


class Trainer:
    """Trainer class implementing the exact training procedure from the paper"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        output_dir,
        early_stopping_patience=5
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Tensorboard
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_metrics = {}
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            rgb = batch['rgb'].to(self.device)
            depth = batch['depth'].to(self.device)
            calories = batch['calorie'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            calorie_pred = self.model(rgb, depth)
            
            # Compute loss (MSE for calorie prediction)
            loss = self.criterion(calorie_pred.squeeze(), calories)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                rgb = batch['rgb'].to(self.device)
                depth = batch['depth'].to(self.device)
                calories = batch['calorie'].to(self.device)
                
                # Forward pass
                calorie_pred = self.model(rgb, depth)
                
                # Compute loss
                loss = self.criterion(calorie_pred.squeeze(), calories)
                total_loss += loss.item()
                
                # Store predictions and targets for metrics
                all_predictions.extend(calorie_pred.squeeze().cpu().numpy())
                all_targets.extend(calories.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        mae = np.mean(np.abs(predictions - targets))
        
        return avg_loss, mae
    
    def train(self, num_epochs):
        """Full training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, mae = self.validate_epoch()
            
            # Log metrics
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('MAE', mae, epoch)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_metrics = {
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'mae': mae,
                }
                
                # Save model checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'mae': mae,
                }, os.path.join(self.output_dir, 'best_model.pth'))
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"MAE: {mae:.2f}")
            
            # Early stopping
            if self.early_stopping(val_loss, epoch):
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best epoch: {self.early_stopping.best_epoch+1}")
                break
        
        self.writer.close()
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def train_nutrition5k_paper_model(
    fusion='middle',
    data_root='../Nutrition5K/train',
    csv_path='../Nutrition5K/nutrition5k_train.csv',
    output_dir='../experiments/nutrition5k_paper',
    num_epochs=100,
    num_workers=4,
    dropout_rate=0.4,
    early_stopping_patience=5,
    pretrained=False
):
    """
    Train the Nutrition5k model with InceptionV2 using the exact paper configuration
    
    Args:
        fusion: Fusion type ('early', 'middle', or 'late')
        data_root: Root directory for dataset
        csv_path: Path to CSV with calorie values
        output_dir: Directory to save outputs
        num_epochs: Maximum number of epochs
        num_workers: Number of data loader workers
        dropout_rate: Dropout rate
        early_stopping_patience: Patience for early stopping
        pretrained: Whether to use pretrained weights
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print(f"TRAINING: Nutrition5k Paper (InceptionV2) + {fusion.capitalize()} Fusion")
    print("="*60)
    
    # Create train/val split
    print("Creating train/validation split...")
    train_csv, val_csv = create_train_val_split(
        csv_path,
        val_ratio=0.15,
        random_seed=42
    )
    
    # Create datasets with paper preprocessing (modified to keep 256x256)
    train_dataset = Nutrition5kPaperDataset(
        csv_path=train_csv,
        data_root=data_root,
        split='train',
        img_size=IMG_SIZE,
        use_augmentation=True
    )
    
    val_dataset = Nutrition5kPaperDataset(
        csv_path=val_csv,
        data_root=data_root,
        split='val',
        img_size=IMG_SIZE,
        use_augmentation=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Build the model with InceptionV2
    model = build_nutrition5k_inceptionv2_model(
        fusion=fusion,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    model = model.to(device)
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer - RMSprop with momentum 0.9 as in the paper
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Optimizer: RMSprop with momentum {MOMENTUM}")
    
    # Create experiment directory
    exp_name = f"nutrition5k_paper_{fusion}_fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save experiment configuration
    config = {
        'fusion': fusion,
        'pretrained': pretrained,
        'dropout_rate': dropout_rate,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'momentum': MOMENTUM,
        'batch_size': BATCH_SIZE,
        'img_size': IMG_SIZE,
        'num_epochs': num_epochs,
        'early_stopping_patience': early_stopping_patience
    }
    
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        output_dir=exp_dir,
        early_stopping_patience=early_stopping_patience
    )
    
    # Train the model
    trainer.train(num_epochs)
    
    print(f"\nExperiment completed! Results saved to: {exp_dir}")
    return trainer.best_metrics


if __name__ == "__main__":
    # Run experiment with each fusion type as in the paper
    
    # Middle fusion (primary approach in the paper)
    middle_fusion_results = train_nutrition5k_paper_model(fusion='middle')
    
    # Early fusion (tested in the paper)
    early_fusion_results = train_nutrition5k_paper_model(fusion='early')
    
    # Late fusion (tested in the paper)
    late_fusion_results = train_nutrition5k_paper_model(fusion='late')
    
    # Compare results
    print("\n" + "="*80)
    print("NUTRITION5K PAPER INCEPTIONV2 RESULTS COMPARISON")
    print("="*80)
    
    results = [
        ("InceptionV2 (Early Fusion)", early_fusion_results),
        ("InceptionV2 (Middle Fusion)", middle_fusion_results),
        ("InceptionV2 (Late Fusion)", late_fusion_results)
    ]
    
    print(f"{'Experiment':<30} {'Val Loss':<10} {'MAE':<10} {'Best Epoch':<12}")
    print("-" * 80)
    
    for name, metrics in results:
        val_loss = metrics['val_loss']
        mae = metrics['mae']
        epoch = metrics['epoch']
        
        print(f"{name:<30} {val_loss:<10.4f} {mae:<10.2f} {epoch:<12}")
