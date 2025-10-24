# Add this to your notebook to run the InceptionV2 experiment from the Nutrition5k paper

import sys
import os
sys.path.append('/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
import json
import matplotlib.pyplot as plt

# Import the paper implementation
from nutrition5k_inceptionv2_model import build_nutrition5k_inceptionv2_model
from nutrition5k_dataloader import Nutrition5kPaperDataset, create_train_val_split

# Constants from the paper (modified to keep at 256x256)
IMG_SIZE = 256        # Resize size
BATCH_SIZE = 32       # Batch size
LEARNING_RATE = 1e-4  # Learning rate
WEIGHT_DECAY = 1e-6   # Weight decay
MOMENTUM = 0.9        # Momentum for RMSprop
NUM_EPOCHS = 100      # Maximum number of epochs
EARLY_STOPPING_PATIENCE = 5  # Early stopping patience
NUM_WORKERS = 4       # Data loader workers

# Output directory
OUTPUT_DIR = '../experiments/nutrition5k_paper_experiments'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Early stopping class as used in the paper
class EarlyStopping:
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

# Function to train the model
def train_nutrition5k_paper_model(fusion='middle'):
    """
    Train the Nutrition5k model with InceptionV2 using the exact paper configuration
    """
    print("="*60)
    print(f"TRAINING: Nutrition5k Paper (InceptionV2) + {fusion.capitalize()} Fusion")
    print("="*60)
    
    # Create train/val split
    print("Creating train/validation split...")
    train_csv, val_csv = create_train_val_split(
        CSV_PATH,  # Use the global CSV_PATH from your notebook
        val_ratio=0.15,
        random_seed=42
    )
    
    # Create datasets with paper preprocessing (modified to keep 256x256)
    train_dataset = Nutrition5kPaperDataset(
        csv_path=train_csv,
        data_root=DATA_ROOT,  # Use the global DATA_ROOT from your notebook
        split='train',
        img_size=IMG_SIZE,
        use_augmentation=True
    )
    
    val_dataset = Nutrition5kPaperDataset(
        csv_path=val_csv,
        data_root=DATA_ROOT,  # Use the global DATA_ROOT from your notebook
        split='val',
        img_size=IMG_SIZE,
        use_augmentation=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Build the model with InceptionV2
    model = build_nutrition5k_inceptionv2_model(
        fusion=fusion,
        pretrained=False,
        dropout_rate=0.4
    )
    model = model.to(device)
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Loss function - MSE as in the paper
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
    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save experiment configuration
    config = {
        'fusion': fusion,
        'pretrained': False,
        'dropout_rate': 0.4,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'momentum': MOMENTUM,
        'batch_size': BATCH_SIZE,
        'img_size': IMG_SIZE,
        'num_epochs': NUM_EPOCHS,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE
    }
    
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Tensorboard writer
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'))
    
    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # Tracking metrics
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    mae_values = []
    
    # Training loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # ---- Training phase ----
        model.train()
        train_loss = 0.0
        
        # Create progress bar
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            calories = batch['calorie'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            calorie_pred = model(rgb, depth)
            
            # Compute loss
            loss = criterion(calorie_pred.squeeze(), calories)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # ---- Validation phase ----
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move to device
                rgb = batch['rgb'].to(device)
                depth = batch['depth'].to(device)
                calories = batch['calorie'].to(device)
                
                # Forward pass
                calorie_pred = model(rgb, depth)
                
                # Compute loss
                loss = criterion(calorie_pred.squeeze(), calories)
                val_loss += loss.item()
                
                # Store predictions for metrics
                all_predictions.extend(calorie_pred.squeeze().cpu().numpy())
                all_targets.extend(calories.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate MAE
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        mae = np.mean(np.abs(predictions - targets))
        mae_values.append(mae)
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('MAE', mae, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'mae': mae,
            }
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'mae': mae,
            }, os.path.join(exp_dir, 'best_model.pth'))
            
            print(f"✓ New best model saved! (Val Loss: {val_loss:.4f})")
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"MAE: {mae:.2f}")
        
        # Check early stopping
        if early_stopping(val_loss, epoch):
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Best epoch: {early_stopping.best_epoch+1}")
            break
    
    writer.close()
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(mae_values, label='MAE', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (calories)')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'training_curves.png'))
    plt.show()
    
    print(f"\nExperiment completed! Results saved to: {exp_dir}")
    print(f"Best validation loss: {best_val_loss:.4f} (Epoch {best_metrics['epoch']})")
    print(f"Best MAE: {best_metrics['mae']:.2f}")
    
    return best_metrics

# Run the experiment with Middle Fusion (the primary approach in the paper)
# Uncomment this line to run the training
# middle_fusion_results = train_nutrition5k_paper_model(fusion='middle')

# For early and late fusion experiments:
# early_fusion_results = train_nutrition5k_paper_model(fusion='early')
# late_fusion_results = train_nutrition5k_paper_model(fusion='late')

# You can compare the results like this:
# results = {
#     "InceptionV2 (Early Fusion)": early_fusion_results,
#     "InceptionV2 (Middle Fusion)": middle_fusion_results,
#     "InceptionV2 (Late Fusion)": late_fusion_results
# }
# 
# print(f"{'Experiment':<30} {'Val Loss':<10} {'MAE':<10} {'Best Epoch':<12}")
# print("-" * 80)
# for name, metrics in results.items():
#     print(f"{name:<30} {metrics['val_loss']:<10.4f} {metrics['mae']:<10.2f} {metrics['epoch']:<12}")
