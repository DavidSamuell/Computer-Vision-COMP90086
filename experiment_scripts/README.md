# Experiment Scripts

Pre-configured training scripts for architecture experiments following the plan in `/plan/arc.plan.md`.

## Quick Start

```bash
cd experiment_scripts
chmod +x *.sh

# Run baseline first, then experiments
bash exp0_baseline_r18_no_seg.sh  # Baseline
bash exp1_early_fusion_r34.sh
bash exp2_fpn_seg_r34.sh
bash exp3_early_fpn_r34.sh
```

## Experiment Overview

### Phase 0: Baseline (Run This First)

**Exp 0: Simple ResNet-18 Baseline (No Segmentation)**
```bash
bash exp0_baseline_r18_no_seg.sh
```
- Simplest architecture: ResNet-18 dual encoder
- No segmentation task (calorie prediction only)
- Establishes baseline performance
- Expected: ~130-150 kcal MAE
- Params: 22.4M

### Phase 0.5: New Fusion Methods (Based on Analysis Results)

**Exp 6: Inception Fusion**
```bash
bash exp6_inception_fusion_r18.sh          # With segmentation
bash exp6_inception_fusion_r18_no_seg.sh   # Without segmentation
```
- Multi-scale fusion with parallel convolution branches
- Replaces simple middle fusion with Inception-style processing
- Expected: 2-5% improvement over middle fusion
- More parameters in fusion module (~4x)

**Exp 7: Late Fusion**
```bash
bash exp7_late_fusion_weighted_r18.sh      # Learned weights
bash exp7_late_fusion_average_r18.sh       # Simple averaging
```
- Separate RGB and Depth processing streams
- Ensemble-like behavior, more robust to noisy modalities
- Expected: 3-8% improvement, especially if one modality is unreliable

### Phase 1: High Priority (Run These Next)

**Exp 1: Early Fusion ResNet-34**
```bash
bash exp1_early_fusion_r34.sh
```
- Single encoder with 4-channel input (RGB+Depth)
- ~50% fewer parameters than dual encoder
- Expected: 10-15% improvement
- Params: 24.2M (vs 46.0M for dual encoder)

**Exp 2: FPN Segmentation**
```bash
bash exp2_fpn_seg_r34.sh
```
- Multi-scale features for segmentation
- Better segmentation → better calorie prediction
- Expected: 15-25% better segmentation, 5-10% better calories
- Params: 44.4M

**Exp 3: Early Fusion + FPN** (Recommended Best Combo)
```bash
bash exp3_early_fpn_r34.sh
```
- Combines benefits of Exp 1 & 2
- Fewer parameters + better segmentation
- Expected: 20-30% total improvement
- Params: 22.6M (lowest with best features!)

### Phase 2: Advanced Fusion (Try If Phase 1 Successful)

**Exp 4: Cross-Modal Attention**
```bash
bash exp4_cross_attn_r34.sh
```
- RGB and Depth attend to each other
- Learns spatial importance per modality
- Expected: 5-10% improvement
- Params: 45.0M

**Exp 5: Gated Fusion**
```bash
bash exp5_gated_r34.sh
```
- Adaptive fusion weights per sample
- Good if one modality is noisy
- Expected: 3-7% improvement
- Params: 44.6M

### Phase 3: Optimization & Scaling (Based on Best Results)

**Exp 8: Hyperparameter Tuning**
```bash
bash exp8_hyperparameter_tuning_best.sh
```
- Fine-tune learning rate for best configuration (early fusion)
- Tests multiple LR values around current best
- Expected: 1-3% improvement through better optimization

**Exp 9: ResNet-34 Scaling**
```bash
bash exp9_resnet34_best_configs.sh
```
- Test top 3 configurations with ResNet-34
- Better feature extraction capacity
- Expected: 3-7% improvement from deeper network

### Phase 4: Regression Head Optimization

**Exp 10: Quick Regression Head Test**
```bash
bash exp10_regression_heads_quick.sh
```
- Test minimal, light, and SE heads on current best config
- Quick 30-epoch runs to identify promising heads
- Run NOW in parallel with fusion experiments

**Exp 11: Full Regression Head Comparison**
```bash
bash exp11_regression_heads_full.sh     # Edit script with best fusion first!
```
- Test all 5 regression heads on best fusion method
- Full 40-epoch runs for accurate comparison
- Run AFTER fusion experiments complete

**Exp 12: Advanced Regression Techniques**
```bash
bash exp12_advanced_regression_techniques.sh
```
- Test Huber loss, label smoothing, optimizer variants
- Advanced techniques for final optimization
- Run on best fusion + regression head combination

## Decision Tree

```
Recommended Execution Order:

Phase 1 (Parallel - Start Now):
  ├─ Run Exp 6 & 7 (Fusion methods) - 4-6 hours
  └─ Run Exp 10 (Quick regression heads) - 3-4 hours

Phase 2 (After Phase 1 results):
  ├─ Identify best fusion method from Exp 6-7
  ├─ Edit Exp 11 script with best fusion
  └─ Run Exp 11 (Full regression heads) - 6-8 hours

Phase 3 (Scaling & Optimization):
  ├─ Run Exp 8 (Hyperparameter tuning) - 10-15 hours
  ├─ Run Exp 9 (ResNet-34 scaling) - 6-9 hours
  └─ Run Exp 12 (Advanced techniques) - 6-8 hours

Target: Get below 60 MAE (currently 63.78)
```

## Monitoring Results

View training in real-time:
```bash
tensorboard --logdir=../outputs/experiments
```

Compare experiments:
```bash
# After training multiple experiments
tensorboard --logdir=../outputs/experiments --port=6006
```

## Expected Results

| Experiment | Current Best MAE | Target MAE | Improvement | Priority |
|------------|------------------|------------|-------------|----------|
| **Current Best** | 63.78 | - | Baseline | - |
| Exp 6 (Inception) | 63.78 | 61-63 | 0-4% | ⭐⭐⭐ |
| Exp 7 (Late Fusion) | 63.78 | 60-62 | 3-6% | ⭐⭐⭐ |
| Exp 10 (Quick Heads) | 63.78 | 61-64 | 0-4% | ⭐⭐ |
| Exp 11 (Full Heads) | Best Fusion | 59-62 | 2-6% | ⭐⭐⭐ |
| Exp 8 (HP Tuning) | 63.78 | 60-62 | 3-6% | ⭐⭐⭐ |
| Exp 9 (ResNet-34) | 63.78 | 58-61 | 4-9% | ⭐⭐⭐ |
| Exp 12 (Advanced) | Best Combo | 57-60 | 1-3% | ⭐⭐ |
| **Target Goal** | - | <60 | >6% | - |

### Regression Head Expectations (for 3K dataset):
- **minimal** (512→1): May underfit, very fast
- **light** (512→128→1): **RECOMMENDED** for limited data  
- **standard** (512→256→128→1): Current baseline
- **se** (standard + attention): May help, slight overhead
- **deep** (5-layer): Likely to overfit on 3K samples

## Tips

1. **Run in order**: Phase 1 before Phase 2
2. **Check overfitting**: If train/val gap > 30%, stop and increase regularization
3. **Early stopping**: All scripts use patience=15, will stop automatically
4. **Best combo**: Exp 3 (Early Fusion + FPN) is recommended starting point
5. **Compare carefully**: Use TensorBoard to compare validation curves

## Customization

To modify hyperparameters, edit the `.sh` files directly. Key parameters:
- `--lr`: Learning rate (default: 0.0005)
- `--dropout`: Dropout rate (default: 0.4)
- `--weight_decay`: L2 regularization (default: 0.0001)
- `--batch_size`: Batch size (default: 64)
- `--seg_weight`: Segmentation loss weight (default: 0.5)

## Troubleshooting

**Out of Memory:**
```bash
# Reduce batch size in script
--batch_size 32  # or 16
```

**Overfitting:**
```bash
# Increase regularization
--dropout 0.5
--weight_decay 0.0005
```

**Underfitting:**
```bash
# Try more complex model (if not overfitting)
bash exp4_cross_attn_r34.sh
```

