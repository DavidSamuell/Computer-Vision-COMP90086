"""
Analyze and visualize grid search results
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


def load_results(results_file):
    """Load grid search results"""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    return pd.read_csv(results_file)


def print_summary(df):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nTotal configurations tested: {len(df)}")
    print(f"Best validation loss: {df['best_val_loss'].min():.4f}")
    print(f"Worst validation loss: {df['best_val_loss'].max():.4f}")
    print(f"Mean validation loss: {df['best_val_loss'].mean():.4f}")
    print(f"Std validation loss: {df['best_val_loss'].std():.4f}")
    
    print("\n" + "-"*80)
    print("TOP 10 CONFIGURATIONS")
    print("-"*80)
    
    df_sorted = df.sort_values('best_val_loss')
    for idx, (i, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        print(f"\nRank {idx}:")
        print(f"  Val Loss: {row['best_val_loss']:.4f}")
        print(f"  LR: {row['lr']:.6f}, Dropout: {row['dropout']:.2f}, Weight Decay: {row['weight_decay']:.6f}")
        print(f"  Batch Size: {int(row['batch_size'])}, Seg Weight: {row['seg_weight']:.2f}")
    
    print("\n" + "="*80)


def plot_hyperparameter_effects(df, save_dir):
    """Plot effect of each hyperparameter"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Hyperparameters to plot
    params = ['lr', 'dropout', 'weight_decay', 'batch_size', 'seg_weight']
    
    for param in params:
        if param not in df.columns:
            continue
        
        plt.figure(figsize=(10, 6))
        
        # Group by parameter and calculate mean/std
        grouped = df.groupby(param)['best_val_loss'].agg(['mean', 'std', 'count'])
        
        # Plot
        plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                     marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
        
        plt.xlabel(param.replace('_', ' ').title(), fontsize=14)
        plt.ylabel('Validation Loss', fontsize=14)
        plt.title(f'Effect of {param.replace("_", " ").title()} on Validation Loss', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        if param in ['lr', 'weight_decay']:
            plt.xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{param}_effect.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {param}_effect.png")


def plot_pairwise_effects(df, save_dir):
    """Plot pairwise effects of hyperparameters"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Key parameter pairs
    pairs = [
        ('lr', 'dropout'),
        ('lr', 'weight_decay'),
        ('dropout', 'weight_decay'),
        ('lr', 'seg_weight')
    ]
    
    for param1, param2 in pairs:
        if param1 not in df.columns or param2 not in df.columns:
            continue
        
        # Skip if either parameter has only one value
        if df[param1].nunique() == 1 or df[param2].nunique() == 1:
            continue
        
        plt.figure(figsize=(10, 8))
        
        # Create pivot table
        pivot = df.pivot_table(values='best_val_loss', index=param2, columns=param1, aggfunc='mean')
        
        # Plot heatmap
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Val Loss'})
        
        plt.xlabel(param1.replace('_', ' ').title(), fontsize=14)
        plt.ylabel(param2.replace('_', ' ').title(), fontsize=14)
        plt.title(f'Validation Loss: {param1.title()} vs {param2.title()}', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{param1}_vs_{param2}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {param1}_vs_{param2}.png")


def plot_distribution(df, save_dir):
    """Plot distribution of validation losses"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(df['best_val_loss'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Validation Loss', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Validation Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(df['best_val_loss'], vert=True)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss Box Plot', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: loss_distribution.png")


def export_best_config(df, output_file):
    """Export best configuration to JSON"""
    best_row = df.loc[df['best_val_loss'].idxmin()]
    
    config = {
        'run_id': int(best_row['run_id']),
        'best_val_loss': float(best_row['best_val_loss']),
        'lr': float(best_row['lr']),
        'dropout': float(best_row['dropout']),
        'weight_decay': float(best_row['weight_decay']),
        'batch_size': int(best_row['batch_size']),
        'calorie_weight': float(best_row['calorie_weight']),
        'seg_weight': float(best_row['seg_weight'])
    }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nBest configuration saved to: {output_file}")
    print(json.dumps(config, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Analyze Grid Search Results')
    
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to grid_search_results.csv')
    parser.add_argument('--output_dir', type=str, default='grid_search_analysis',
                        help='Output directory for plots and analysis')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_file}")
    df = load_results(args.results_file)
    
    # Print summary
    print_summary(df)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    print("-" * 80)
    
    plot_distribution(df, args.output_dir)
    plot_hyperparameter_effects(df, args.output_dir)
    plot_pairwise_effects(df, args.output_dir)
    
    # Export best configuration
    best_config_file = os.path.join(args.output_dir, 'best_configuration.json')
    export_best_config(df, best_config_file)
    
    # Save full sorted results
    df_sorted = df.sort_values('best_val_loss')
    sorted_file = os.path.join(args.output_dir, 'sorted_results.csv')
    df_sorted.to_csv(sorted_file, index=False)
    print(f"\nSorted results saved to: {sorted_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

