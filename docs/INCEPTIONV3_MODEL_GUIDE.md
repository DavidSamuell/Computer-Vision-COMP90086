# InceptionV3 Model Architecture Guide

## Overview

This document provides an overview of the InceptionV3-based model implementation that replicates the architecture used in the original Nutrition5k paper. The implementation supports three different fusion strategies: early, middle, and late fusion.

## Architecture Details

### InceptionV3 Encoder

The InceptionV3 encoder serves as the backbone for feature extraction, with the following characteristics:

- **Based on**: Google's InceptionV3 architecture
- **Output Channels**: 2048 (4x more than ResNet-18's 512)
- **Input Support**: Modified to support both RGB (3-channel) and depth (1-channel) inputs
- **Feature Map Size**: H/32 × W/32 (32x downsampling from input resolution)

### Fusion Strategies

The implementation supports three fusion methods to combine RGB and depth information:

#### 1. Early Fusion

![Early Fusion](https://i.imgur.com/UgGq0Hj.png)

- **Mechanism**: Concatenates RGB and depth at the input level (4-channel input)
- **Parameters**: ~25M (single encoder)
- **Benefits**: 
  - Reduced parameter count
  - Enables learning of joint features from the very beginning
- **Drawbacks**: 
  - Each modality might not develop specialized features
  - Less expressive than dual-stream approaches

#### 2. Middle Fusion (Default in Nutrition5k Paper)

![Middle Fusion](https://i.imgur.com/8yJTXHh.png)

- **Mechanism**: Processes RGB and depth through separate encoders, then concatenates features and applies 1x1 convolution
- **Parameters**: ~50M (dual encoders)
- **Benefits**: 
  - Modality-specific feature learning
  - Feature interaction at mid-level
  - Most common approach in multi-modal fusion
- **Drawbacks**: 
  - Higher parameter count
  - More computation required

#### 3. Late Fusion

![Late Fusion](https://i.imgur.com/RYr0isp.png)

- **Mechanism**: Processes RGB and depth through separate encoders, applies global pooling to each, then concatenates and feeds through FC layers
- **Parameters**: ~50M (dual encoders)
- **Benefits**: 
  - Independent feature extractors
  - Simpler fusion mechanism
  - Good when modalities are highly complementary
- **Drawbacks**: 
  - Limited cross-modal interaction
  - May miss important correlation between modalities

### Regression Head

The regression head is designed to effectively map the high-dimensional features to calorie predictions:

- **Input**: 2048-channel feature maps (for middle/late fusion) or 4-channel inputs (for early fusion)
- **Architecture**: 
  - Global average pooling
  - Fully connected layers (2048→512→256→1)
  - ReLU activations between layers
  - Dropout (configurable rate, default 0.4)
- **Output**: Single calorie prediction value

## Usage

### Training the InceptionV3 Model

You can train the InceptionV3 model directly from the notebook:

```python
# Import the model builder
import sys
sys.path.append('/data/projects/punim0478/setiawand/Computer-Vision-COMP90086/src')
from nutrition5k_inceptionv3_model import build_nutrition5k_model

# Create and train model with middle fusion (default in original paper)
model = build_nutrition5k_model(
    fusion='middle',              # Options: 'early', 'middle', 'late'
    pretrained=False,             # Whether to use pretrained weights
    dropout_rate=0.4,             # Dropout rate for regularization
    fusion_channels=2048          # Output channels from fusion module
)
```

Alternatively, you can use the provided script:

```bash
# Run from the project root directory
bash run_nutrition5k_experiment.sh
```

### Experiment with Different Fusion Types

To compare different fusion strategies:

```python
# Middle fusion (original Nutrition5k paper approach)
middle_fusion_results = train_nutrition5k_model(fusion_type='middle')

# Early fusion
early_fusion_results = train_nutrition5k_model(fusion_type='early')

# Late fusion
late_fusion_results = train_nutrition5k_model(fusion_type='late')

# Compare results
results = {
    "InceptionV3 (Early Fusion)": early_fusion_results,
    "InceptionV3 (Middle Fusion)": middle_fusion_results,
    "InceptionV3 (Late Fusion)": late_fusion_results
}
compare_nutrition5k_results(results)
```

## Technical Details

### InceptionV3 vs ResNet Comparison

| Property             | InceptionV3           | ResNet-18              | ResNet-34              |
|----------------------|----------------------|------------------------|------------------------|
| Output Channels      | 2048                 | 512                    | 512                    |
| Parameters (single)  | ~24M                 | ~11M                   | ~21M                   |
| Parameters (dual)    | ~48M                 | ~22M                   | ~42M                   |
| Architecture         | Inception modules    | Residual blocks        | Residual blocks        |
| Receptive Field      | Larger               | Smaller                | Medium                 |
| Training Speed       | Slower               | Faster                 | Medium                 |
| Feature Complexity   | Multi-scale features | Basic features         | Better features        |

### Implementation Details

- **File Structure**:
  - `nutrition5k_inceptionv3_model.py`: Contains the InceptionV3 encoder and Nutrition5k model implementations
  - `inception_fusion_modules.py`: Contains different fusion module implementations
  - `nutrition5k_experiment.py`: Script for running experiments with different fusion types

- **Modifications to Original InceptionV3**:
  - Added support for 1-channel depth input
  - Removed auxiliary classifier (used for training only)
  - Configured for regression task instead of classification

## Hyperparameter Recommendations

For training the InceptionV3 model:

| Hyperparameter   | Recommended Value | Notes                                    |
|------------------|-------------------|------------------------------------------|
| Learning Rate    | 3e-4 to 1e-4      | Lower than ResNet due to model size      |
| Batch Size       | 16-32             | Adjust based on available GPU memory     |
| Dropout Rate     | 0.4-0.5           | Prevent overfitting                      |
| Weight Decay     | 1e-6              | Light regularization                     |
| Early Stopping   | 15 epochs         | Patience for validation loss improvement |
| Epochs           | 40-60             | More epochs may help with larger model   |

## Performance Considerations

When using the InceptionV3 architecture:

1. **GPU Memory**: InceptionV3 requires more memory than ResNet models
2. **Training Time**: Expect longer training times (1.5-2x compared to ResNet-18)
3. **Data Efficiency**: InceptionV3 may require more data to avoid overfitting
4. **Feature Quality**: Should provide richer features for complex tasks

## References

1. Szegedy, C., et al. (2016). "Rethinking the inception architecture for computer vision." CVPR 2016.
2. Gorbonos, E., et al. (2021). "Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food." CVPR 2021.

## Next Steps

1. **Compare Fusion Strategies**: Run experiments with all three fusion types
2. **Hyperparameter Tuning**: Fine-tune learning rates and regularization for InceptionV3
3. **Ensemble Models**: Consider combining predictions from different fusion approaches
4. **Ablation Studies**: Compare against ResNet models to measure improvement
