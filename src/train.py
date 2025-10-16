"""
Training Script for Calorie Prediction with Multi-Task Learning
Implements regularization, early stopping, and weight decay
"""

import os
import argparse
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Nutrition5KDataset, create_train_val_split
from model import MultiStreamCaloriePredictor, build_model


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning:
    Total Loss = MSE Loss (Calories) + BCE Loss (Segmentation)
    
    Can also be used for single-task (calorie prediction only) by setting use_segmentation=False
    """
    
    def __init__(self, calorie_weight: float = 1.0, seg_weight: float = 1.0, use_segmentation: bool = True):
        super().__init__()
        self.calorie_weight = calorie_weight
        self.seg_weight = seg_weight
        self.use_segmentation = use_segmentation
        
        self.mse_loss = nn.MSELoss()
        if self.use_segmentation:
            self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, calorie_pred, seg_pred, calorie_target, seg_target):
        """
        Args:
            calorie_pred: (B, 1) predicted calories
            seg_pred: (B, 1, H, W) predicted segmentation logits or None if use_segmentation=False
            calorie_target: (B,) or (B, 1) target calories
            seg_target: (B, 1, H, W) target segmentation masks or dummy tensor
        
        Returns:
            total_loss, calorie_loss, seg_loss
        """
        # Ensure shapes match
        if calorie_target.dim() == 1:
            calorie_target = calorie_target.unsqueeze(1)
        
        # Calorie loss (MSE)
        calorie_loss = self.mse_loss(calorie_pred, calorie_target)
        
        # Segmentation loss (BCE with logits) - only if enabled
        if self.use_segmentation and seg_pred is not None:
            seg_loss = self.bce_loss(seg_pred, seg_target)
            total_loss = self.calorie_weight * calorie_loss + self.seg_weight * seg_loss
        else:
            seg_loss = torch.tensor(0.0, device=calorie_pred.device)
            total_loss = self.calorie_weight * calorie_loss
        
        return total_loss, calorie_loss, seg_loss


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max' - whether lower or higher metric is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


class Trainer:
    """Training manager with all the bells and whistles"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        output_dir,
        early_stopping_patience=15,
        scheduler_step_on_batch=False
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.scheduler_step_on_batch = scheduler_step_on_batch
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=0.1,
            mode='min'
        )
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
        
        # Tracking
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Track best metrics
        self.best_metrics = {
            'epoch': 0,
            'train_loss': 0.0,
            'train_calorie_loss': 0.0,
            'train_seg_loss': 0.0,
            'val_loss': float('inf'),
            'val_calorie_loss': 0.0,
            'val_seg_loss': 0.0,
            'mae': 0.0,
            'mse': 0.0,
            'mape': 0.0
        }
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_calorie_loss = 0
        total_seg_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            rgb = batch['rgb'].to(self.device)
            depth = batch['depth'].to(self.device)
            calorie = batch['calorie'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            calorie_pred, seg_pred = self.model(rgb, depth)
            
            # Calculate loss
            loss, calorie_loss, seg_loss = self.criterion(
                calorie_pred, seg_pred, calorie, mask
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Step scheduler if it's per-batch (Cosine, OneCycle)
            if self.scheduler is not None and self.scheduler_step_on_batch:
                self.scheduler.step()
            
            # Track losses
            total_loss += loss.item()
            total_calorie_loss += calorie_loss.item()
            total_seg_loss += seg_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cal': f'{calorie_loss.item():.4f}',
                'seg': f'{seg_loss.item():.4f}'
            })
        
        # Calculate average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_calorie_loss = total_calorie_loss / len(self.train_loader)
        avg_seg_loss = total_seg_loss / len(self.train_loader)
        
        return avg_loss, avg_calorie_loss, avg_seg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        total_calorie_loss = 0
        total_seg_loss = 0
        
        # For calorie metrics
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        for batch in pbar:
            # Move to device
            rgb = batch['rgb'].to(self.device)
            depth = batch['depth'].to(self.device)
            calorie = batch['calorie'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            calorie_pred, seg_pred = self.model(rgb, depth)
            
            # Calculate loss
            loss, calorie_loss, seg_loss = self.criterion(
                calorie_pred, seg_pred, calorie, mask
            )
            
            # Track losses
            total_loss += loss.item()
            total_calorie_loss += calorie_loss.item()
            total_seg_loss += seg_loss.item()
            
            # Collect predictions for metrics
            all_predictions.extend(calorie_pred.cpu().numpy().flatten())
            all_targets.extend(calorie.cpu().numpy().flatten())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cal': f'{calorie_loss.item():.4f}',
                'seg': f'{seg_loss.item():.4f}'
            })
        
        # Calculate average losses
        avg_loss = total_loss / len(self.val_loader)
        avg_calorie_loss = total_calorie_loss / len(self.val_loader)
        avg_seg_loss = total_seg_loss / len(self.val_loader)
        
        # Calculate calorie metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100
        
        return avg_loss, avg_calorie_loss, avg_seg_loss, mae, mse, mape
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Model parameters: {self.model.get_num_parameters():,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_cal_loss, train_seg_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_cal_loss, val_seg_loss, mae, mse, mape = self.validate(epoch)
            
            # Learning rate scheduling (only for epoch-based schedulers)
            if self.scheduler is not None and not self.scheduler_step_on_batch:
                # ReduceLROnPlateau needs val_loss
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Loss/train_calorie', train_cal_loss, epoch)
            self.writer.add_scalar('Loss/val_calorie', val_cal_loss, epoch)
            self.writer.add_scalar('Loss/train_seg', train_seg_loss, epoch)
            self.writer.add_scalar('Loss/val_seg', val_seg_loss, epoch)
            self.writer.add_scalar('Metrics/MAE', mae, epoch)
            self.writer.add_scalar('Metrics/MSE', mse, epoch)
            self.writer.add_scalar('Metrics/MAPE', mape, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} (Cal: {train_cal_loss:.4f}, Seg: {train_seg_loss:.4f})")
            print(f"  Val Loss:   {val_loss:.4f} (Cal: {val_cal_loss:.4f}, Seg: {val_seg_loss:.4f})")
            print(f"  Calorie Metrics - MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                
                # Update best metrics
                self.best_metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_calorie_loss': train_cal_loss,
                    'train_seg_loss': train_seg_loss,
                    'val_loss': val_loss,
                    'val_calorie_loss': val_cal_loss,
                    'val_seg_loss': val_seg_loss,
                    'mae': mae,
                    'mse': mse,
                    'mape': mape
                }
                
                # Save best metrics to file
                self.save_best_metrics()
                
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model saved! (Val Loss: {val_loss:.4f})")
            
            # Regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.early_stopping(val_loss, epoch):
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best epoch was {self.early_stopping.best_epoch} with val loss {self.early_stopping.best_score:.4f}")
                print(f"{'='*60}\n")
                break
        
        # Save final model
        self.save_checkpoint(epoch, is_best=False, name='final_model.pth')
        self.writer.close()
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"\nBest Results (Epoch {self.best_metrics['epoch']}):")
        print(f"  Validation Loss: {self.best_metrics['val_loss']:.4f}")
        print(f"  Training Loss:   {self.best_metrics['train_loss']:.4f}")
        print(f"  Val Calorie Loss: {self.best_metrics['val_calorie_loss']:.4f}")
        print(f"  Val Seg Loss:     {self.best_metrics['val_seg_loss']:.4f}")
        print(f"  MAE:  {self.best_metrics['mae']:.2f} kcal")
        print(f"  MSE:  {self.best_metrics['mse']:.2f}")
        print(f"  MAPE: {self.best_metrics['mape']:.2f}%")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, epoch, is_best=False, name=None):
        """Save model checkpoint"""
        if name is None:
            name = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(self.output_dir, name)
        torch.save(checkpoint, checkpoint_path)
    
    def save_best_metrics(self):
        """Save best metrics to JSON file"""
        metrics_path = os.path.join(self.output_dir, 'best_metrics.json')
        
        # Convert all values to native Python types for JSON serialization
        serializable_metrics = {
            key: float(value) if isinstance(value, (np.floating, np.integer)) else value
            for key, value in self.best_metrics.items()
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        print(f"  ✓ Best metrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Calorie Prediction Model')
    
    # Data
    parser.add_argument('--data_root', type=str, default='../Nutrition5K/train',
                        help='Path to training data directory')
    parser.add_argument('--csv_path', type=str, default='../Nutrition5K/nutrition5k_train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation split ratio')
    
    # Model architecture (modular system)
    parser.add_argument('--encoder', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 
                                'early_fusion_resnet18', 'early_fusion_resnet34'],
                        help='Encoder architecture')
    parser.add_argument('--fusion', type=str, default='middle',
                        choices=['middle', 'middle_attention', 'additive', 
                                'cross_modal_attention', 'gated', 'inception',
                                'late_average', 'late_weighted'],
                        help='Fusion strategy (ignored for early_fusion encoders)')
    parser.add_argument('--regression_head', type=str, default='standard',
                        choices=['standard', 'deep', 'light', 'minimal', 'se'],
                        help='Regression head type')
    parser.add_argument('--segmentation_head', type=str, default='standard',
                        choices=['standard', 'light', 'fpn'],
                        help='Segmentation head type')
    
    # Model hyperparameters
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate for regularization')
    parser.add_argument('--fusion_channels', type=int, default=512,
                        help='Number of channels after fusion')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='linear',
                        choices=['linear', 'cosine_warm_restarts', 'reduce_on_plateau', 'one_cycle', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience')
    
    # Loss weights
    parser.add_argument('--calorie_weight', type=float, default=1.0,
                        help='Weight for calorie loss')
    parser.add_argument('--seg_weight', type=float, default=0.5,
                        help='Weight for segmentation loss (only used if --use_segmentation)')
    parser.add_argument('--use_segmentation', action='store_true', default=False,
                        help='Enable segmentation as auxiliary task (default: disabled)')
    parser.add_argument('--no_segmentation', dest='use_segmentation', action='store_false',
                        help='Disable segmentation task (calorie prediction only) - this is the default')
    
    # Data augmentation
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for resizing')
    parser.add_argument('--augment', dest='use_augment', action='store_true', default=True,
                        help='Enable data augmentation (default: enabled)')
    parser.add_argument('--no_augment', dest='use_augment', action='store_false',
                        help='Disable data augmentation')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='../outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (default: timestamp)')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print(f"Output directory: {output_dir}")
    
    # Create train/val split
    print("\nCreating train/val split...")
    train_csv, val_csv = create_train_val_split(
        args.csv_path,
        val_ratio=args.val_ratio,
        random_seed=args.seed
    )
    
    # Create datasets
    print("Loading datasets...")
    print(f"Data augmentation: {'ENABLED' if args.use_augment else 'DISABLED'}")
    
    train_dataset = Nutrition5KDataset(
        csv_path=train_csv,
        data_root=args.data_root,
        split='train',
        augment=args.use_augment,
        img_size=args.img_size,
        use_segmentation=args.use_segmentation
    )
    
    val_dataset = Nutrition5KDataset(
        csv_path=val_csv,
        data_root=args.data_root,
        split='val',
        augment=False,  # Never augment validation data
        img_size=args.img_size,
        use_segmentation=args.use_segmentation
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model (use modular system)
    print("\nBuilding model...")
    print(f"Architecture: {args.encoder} + {args.fusion} fusion")
    print(f"Heads: {args.regression_head} (regression), {args.segmentation_head} (segmentation)")
    print(f"Segmentation task: {'ENABLED' if args.use_segmentation else 'DISABLED (calorie prediction only)'}")
    
    model = build_model(
        encoder=args.encoder,
        fusion=args.fusion,
        regression_head=args.regression_head,
        segmentation_head=args.segmentation_head,
        pretrained=False,  # Training from scratch as per constraints
        dropout_rate=args.dropout,
        fusion_channels=args.fusion_channels,
        use_segmentation=args.use_segmentation
    )
    model = model.to(device)
    
    print(f"Model has {model.get_num_parameters():,} trainable parameters")
    
    # Create loss function
    criterion = MultiTaskLoss(
        calorie_weight=args.calorie_weight,
        seg_weight=args.seg_weight,
        use_segmentation=args.use_segmentation
    )
    
    # Create optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler based on user choice
    if args.scheduler == 'linear':
        # Linear decay from initial LR to 0
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=args.num_epochs
        )
        scheduler_step_on_batch = False  # Step after each epoch
        
    elif args.scheduler == 'cosine_warm_restarts':
        # Cosine Annealing with Warm Restarts (Best for limited data)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,         # Restart every 10 epochs
            T_mult=2,       # Double the period after each restart (10, 20, 40...)
            eta_min=1e-7    # Minimum learning rate
        )
        scheduler_step_on_batch = True  # Step after each batch
        
    elif args.scheduler == 'reduce_on_plateau':
        # ReduceLROnPlateau (Original, more conservative settings)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,     # Less aggressive than 0.5
            patience=10,    # More patient than 5
            min_lr=1e-7
        )
        scheduler_step_on_batch = False  # Step after each epoch
        
    elif args.scheduler == 'one_cycle':
        # OneCycleLR (Fast training)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr * 10,
            epochs=args.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # 30% warmup
            anneal_strategy='cos'
        )
        scheduler_step_on_batch = True  # Step after each batch
        
    else:  # args.scheduler == 'none'
        scheduler = None
        scheduler_step_on_batch = False
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_step_on_batch=scheduler_step_on_batch
    )
    
    # Train
    trainer.train(args.num_epochs)


if __name__ == '__main__':
    main()

