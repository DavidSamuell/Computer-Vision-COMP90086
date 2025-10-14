"""
Inference script for trained calorie prediction model
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import Normalize

from model import MultiStreamCaloriePredictor


class CaloriePredictor:
    """Wrapper for easy inference"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = MultiStreamCaloriePredictor(
            pretrained=False,
            dropout_rate=0.4  # Dropout is disabled during inference anyway
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Normalization
        self.color_normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Using device: {self.device}")
    
    def preprocess_rgb(self, rgb_path, img_size=224):
        """Load and preprocess RGB image"""
        try:
            img = Image.open(rgb_path).convert('RGB')
            img = TF.resize(img, (img_size, img_size))
            tensor = TF.to_tensor(img)
            tensor = self.color_normalize(tensor)
            return tensor.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f"Error loading RGB image: {e}")
            return None
    
    def preprocess_depth(self, depth_path, img_size=224):
        """Load and preprocess depth image"""
        try:
            img = Image.open(depth_path).convert('L')
            img = TF.resize(img, (img_size, img_size))
            tensor = TF.to_tensor(img)
            # Normalize depth to [0, 1]
            tensor = tensor / 65535.0 if tensor.max() > 1.0 else tensor
            return tensor.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f"Error loading depth image: {e}")
            return None
    
    @torch.no_grad()
    def predict(self, rgb_path, depth_path, img_size=224):
        """
        Predict calories for a single dish
        
        Args:
            rgb_path: Path to RGB image
            depth_path: Path to depth image
            img_size: Image size for resizing
        
        Returns:
            calories: Predicted calorie value
            segmentation: Predicted segmentation mask (numpy array)
        """
        # Preprocess
        rgb = self.preprocess_rgb(rgb_path, img_size)
        depth = self.preprocess_depth(depth_path, img_size)
        
        if rgb is None or depth is None:
            return None, None
        
        # Move to device
        rgb = rgb.to(self.device)
        depth = depth.to(self.device)
        
        # Predict
        calorie_pred, seg_pred = self.model(rgb, depth)
        
        # Convert to numpy
        calories = calorie_pred.cpu().item()
        segmentation = torch.sigmoid(seg_pred).cpu().numpy()[0, 0]  # (H, W)
        
        return calories, segmentation
    
    @torch.no_grad()
    def predict_batch(self, rgb_paths, depth_paths, img_size=224, batch_size=16):
        """
        Predict calories for multiple dishes
        
        Args:
            rgb_paths: List of RGB image paths
            depth_paths: List of depth image paths
            img_size: Image size for resizing
            batch_size: Batch size for processing
        
        Returns:
            List of predicted calories
        """
        all_calories = []
        
        for i in range(0, len(rgb_paths), batch_size):
            batch_rgb_paths = rgb_paths[i:i+batch_size]
            batch_depth_paths = depth_paths[i:i+batch_size]
            
            # Preprocess batch
            rgb_batch = []
            depth_batch = []
            
            for rgb_path, depth_path in zip(batch_rgb_paths, batch_depth_paths):
                rgb = self.preprocess_rgb(rgb_path, img_size)
                depth = self.preprocess_depth(depth_path, img_size)
                
                if rgb is not None and depth is not None:
                    rgb_batch.append(rgb)
                    depth_batch.append(depth)
                else:
                    all_calories.append(None)
            
            if len(rgb_batch) == 0:
                continue
            
            # Stack into batch
            rgb_batch = torch.cat(rgb_batch, dim=0).to(self.device)
            depth_batch = torch.cat(depth_batch, dim=0).to(self.device)
            
            # Predict
            calorie_pred, _ = self.model(rgb_batch, depth_batch)
            
            # Convert to list
            calories = calorie_pred.cpu().numpy().flatten().tolist()
            all_calories.extend(calories)
        
        return all_calories


def main():
    parser = argparse.ArgumentParser(description='Predict calories from RGB and depth images')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--rgb', type=str, required=True,
                        help='Path to RGB image')
    parser.add_argument('--depth', type=str, required=True,
                        help='Path to depth image')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for resizing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save_mask', type=str, default=None,
                        help='Path to save segmentation mask (optional)')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = CaloriePredictor(args.checkpoint, device=args.device)
    
    # Predict
    print(f"\nPredicting for:")
    print(f"  RGB: {args.rgb}")
    print(f"  Depth: {args.depth}")
    
    calories, segmentation = predictor.predict(args.rgb, args.depth, args.img_size)
    
    if calories is not None:
        print(f"\n{'='*50}")
        print(f"Predicted Calories: {calories:.2f} kcal")
        print(f"{'='*50}\n")
        
        # Save segmentation mask if requested
        if args.save_mask is not None:
            mask_img = (segmentation * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_img)
            mask_pil.save(args.save_mask)
            print(f"Segmentation mask saved to: {args.save_mask}")
    else:
        print("\nFailed to predict. Check image paths.")


if __name__ == '__main__':
    main()

