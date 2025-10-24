import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import json

# Import from your existing modules
from comp90086 import (
    Nutrition5KDataset, 
    Trainer, 
    get_warmup_cosine_scheduler,
    create_train_val_split
)

# Import the nutrition5k model implementation
from nutrition5k_inceptionv3_model import build_nutrition5k_model

def train_nutrition5k_model(
    fusion_type='middle',
    data_root='../Nutrition5K/train',
    csv_path='../Nutrition5K/nutrition5k_train.csv',
    output_dir='../experiments/nutrition5k_experiments',
    batch_size=32,
    num_epochs=40,
    img_size=224,
    num_workers=4,
    dropout_rate=0.4,
    learning_rate=3e-4,
    weight_decay=1e-6,
    early_stopping_patience=15,
    warmup_ratio=0.1,
    min_lr_ratio=0.05,
    pretrained=False
):
    """Train the Nutrition5k model with InceptionV3 and specified fusion type"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("="*60)
    print(f"TRAINING: Nutrition5k InceptionV3 + {fusion_type.capitalize()} Fusion")
    print("="*60)
    
    # Create train/val split
    print("Creating train/validation split...")
    train_csv, val_csv = create_train_val_split(
        csv_path,
        val_ratio=0.15,
        random_seed=42
    )
    
    # Create datasets
    train_dataset = Nutrition5KDataset(
        csv_path=train_csv,
        data_root=data_root,
        split='train',
        augment=True,  # Use data augmentation for training
        img_size=img_size
    )
    
    val_dataset = Nutrition5KDataset(
        csv_path=val_csv,
        data_root=data_root,
        split='val',
        augment=False,  # Never augment validation
        img_size=img_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Build model with the specified fusion type
    fusion_channels = 2048  # For InceptionV3
    model = build_nutrition5k_model(
        fusion=fusion_type,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        fusion_channels=fusion_channels
    )
    model = model.to(device)
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    
    # Learning rate scheduler
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = get_warmup_cosine_scheduler(
        optimizer, 
        warmup_steps=warmup_steps, 
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio
    )
    
    # Create experiment directory
    exp_name = f"nutrition5k_{fusion_type}_fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save experiment configuration
    config = {
        'fusion': fusion_type,
        'pretrained': pretrained,
        'dropout_rate': dropout_rate,
        'fusion_channels': fusion_channels,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'img_size': img_size,
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
        scheduler=scheduler,
        device=device,
        output_dir=exp_dir,
        early_stopping_patience=early_stopping_patience,
        scheduler_step_on_batch=False
    )
    
    # Train the model
    trainer.train(num_epochs)
    
    print(f"\nExperiment completed! Results saved to: {exp_dir}")
    return trainer.best_metrics

if __name__ == "__main__":
    # Train with middle fusion (original Nutrition5k approach)
    middle_fusion_results = train_nutrition5k_model(fusion_type='middle')
    
    # Train with early fusion
    early_fusion_results = train_nutrition5k_model(fusion_type='early')
    
    # Train with late fusion
    late_fusion_results = train_nutrition5k_model(fusion_type='late')
    
    # Compare results
    print("\n" + "="*80)
    print("NUTRITION5K INCEPTIONV3 RESULTS COMPARISON")
    print("="*80)
    
    results = [
        ("InceptionV3 (Early Fusion)", early_fusion_results),
        ("InceptionV3 (Middle Fusion)", middle_fusion_results),
        ("InceptionV3 (Late Fusion)", late_fusion_results)
    ]
    
    print(f"{'Experiment':<30} {'Val Loss':<10} {'MAE':<10} {'Best Epoch':<12}")
    print("-" * 80)
    
    for name, metrics in results:
        val_loss = metrics['val_loss']
        mae = metrics['mae']
        epoch = metrics['epoch']
        
        print(f"{name:<30} {val_loss:<10.4f} {mae:<10.2f} {epoch:<12}")
