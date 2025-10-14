"""
Visualization utilities for debugging and understanding the model
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image


def visualize_sample(sample, save_path=None):
    """
    Visualize a single dataset sample with RGB, depth, and mask
    
    Args:
        sample: Dictionary from dataset with 'rgb', 'depth', 'mask', 'calorie'
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB image (denormalize)
    rgb = sample['rgb'].numpy()
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    rgb = rgb * std + mean
    rgb = np.clip(rgb, 0, 1)
    rgb = np.transpose(rgb, (1, 2, 0))
    
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Depth image
    depth = sample['depth'].numpy()[0]  # Remove channel dimension
    axes[1].imshow(depth, cmap='viridis')
    axes[1].set_title('Depth Image')
    axes[1].axis('off')
    
    # Mask
    mask = sample['mask'].numpy()[0]
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Segmentation Mask')
    axes[2].axis('off')
    
    # Add calorie info
    calorie = sample['calorie'].item()
    dish_id = sample['dish_id']
    fig.suptitle(f"Dish: {dish_id} | Calories: {calorie:.2f} kcal", fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_prediction(rgb_img, depth_img, pred_calories, pred_mask, gt_calories=None, save_path=None):
    """
    Visualize prediction results
    
    Args:
        rgb_img: RGB image (numpy array or PIL Image)
        depth_img: Depth image (numpy array or PIL Image)
        pred_calories: Predicted calorie value
        pred_mask: Predicted segmentation mask (numpy array)
        gt_calories: Ground truth calories (optional)
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Convert PIL to numpy if needed
    if isinstance(rgb_img, Image.Image):
        rgb_img = np.array(rgb_img)
    if isinstance(depth_img, Image.Image):
        depth_img = np.array(depth_img)
    
    # RGB
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')
    
    # Depth
    axes[0, 1].imshow(depth_img, cmap='viridis')
    axes[0, 1].set_title('Depth Image')
    axes[0, 1].axis('off')
    
    # Predicted mask
    axes[1, 0].imshow(pred_mask, cmap='gray')
    axes[1, 0].set_title('Predicted Segmentation')
    axes[1, 0].axis('off')
    
    # Overlay
    rgb_normalized = rgb_img.astype(float) / 255.0 if rgb_img.max() > 1 else rgb_img
    mask_overlay = np.zeros_like(rgb_normalized)
    mask_overlay[:, :, 0] = pred_mask  # Red channel
    blended = 0.7 * rgb_normalized + 0.3 * mask_overlay
    axes[1, 1].imshow(np.clip(blended, 0, 1))
    axes[1, 1].set_title('RGB + Mask Overlay')
    axes[1, 1].axis('off')
    
    # Title
    title = f"Predicted Calories: {pred_calories:.2f} kcal"
    if gt_calories is not None:
        error = abs(pred_calories - gt_calories)
        title += f"\nGround Truth: {gt_calories:.2f} kcal (Error: {error:.2f} kcal)"
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_augmentations(dataset, idx=0, num_samples=5, save_path=None):
    """
    Visualize the effect of augmentations on a single sample
    
    Args:
        dataset: Dataset object with augmentation enabled
        idx: Index of sample to augment
        num_samples: Number of augmented versions to show
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        sample = dataset[idx]
        
        # RGB
        rgb = sample['rgb'].numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        rgb = rgb * std + mean
        rgb = np.clip(rgb, 0, 1)
        rgb = np.transpose(rgb, (1, 2, 0))
        
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f'RGB (Aug {i+1})')
        axes[i, 0].axis('off')
        
        # Depth
        depth = sample['depth'].numpy()[0]
        axes[i, 1].imshow(depth, cmap='viridis')
        axes[i, 1].set_title(f'Depth (Aug {i+1})')
        axes[i, 1].axis('off')
        
        # Mask
        mask = sample['mask'].numpy()[0]
        axes[i, 2].imshow(mask, cmap='gray')
        axes[i, 2].set_title(f'Mask (Aug {i+1})')
        axes[i, 2].axis('off')
    
    fig.suptitle(f"Augmentation Variations for Dish: {sample['dish_id']}", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(train_losses, val_losses, train_cal_losses, val_cal_losses, save_path=None):
    """
    Plot training curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_cal_losses: List of training calorie losses
        val_cal_losses: List of validation calorie losses
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Total loss
    axes[0].plot(epochs, train_losses, label='Train', marker='o', markersize=3)
    axes[0].plot(epochs, val_losses, label='Validation', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Calorie loss
    axes[1].plot(epochs, train_cal_losses, label='Train', marker='o', markersize=3)
    axes[1].plot(epochs, val_cal_losses, label='Validation', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Calorie Loss (MSE)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    """Test visualization on a sample"""
    import sys
    sys.path.append('.')
    
    from dataset import Nutrition5KDataset
    
    # Create dataset
    csv_path = '../Nutrition5K/nutrition5k_train.csv'
    data_root = '../Nutrition5K/train'
    
    dataset = Nutrition5KDataset(csv_path, data_root, split='train', augment=True)
    
    # Visualize first sample
    print("Visualizing first sample...")
    sample = dataset[0]
    visualize_sample(sample, save_path='sample_visualization.png')
    
    # Visualize augmentations
    print("\nVisualizing augmentations...")
    visualize_augmentations(dataset, idx=0, num_samples=5, save_path='augmentation_examples.png')
    
    print("\nVisualization complete!")

