# Architecture Experimentation Plan for Calorie Prediction

## Context

- **Task**: Calorie prediction from RGB + Depth images with segmentation auxiliary task
- **Constraint**: ~3000 training samples, no pretrained models available
- **Current**: ResNet-18/34 dual encoders + middle fusion + standard heads
- **Goal**: Improve performance while avoiding overfitting on limited data

## Problem Analysis

**Key Challenges:**

1. Limited training data makes complex models prone to overfitting
2. No pretrained weights available to leverage
3. Need to effectively fuse RGB and depth modalities
4. Multi-task learning (calorie regression + segmentation) needs balanced architecture

**Current Architecture Limitations:**

- Two separate encoders = 2x parameters (prone to overfitting with limited data)
- Simple middle fusion may not capture cross-modal interactions optimally
- Segmentation head only uses deepest features (missing fine details)
- No explicit mechanism for modality importance weighting

---

## Recommended Experiments

### Phase 1: Low-Hanging Fruit (High Priority)

#### Experiment 1.1: Early Fusion Architecture

**Rationale**: Reduce parameters by ~50% to combat overfitting

**Architecture**:

```
Input: Concatenate RGB (3ch) + Depth (1ch) = 4 channels
       ↓
Single Encoder: ResNet-34 with 4-channel input
       ↓
Regression Head + Segmentation Head
```

**Benefits**:

- Half the encoder parameters (one encoder vs two)
- Forces joint RGB-Depth representation learning from the start
- Less prone to overfitting
- Faster training and inference

**Expected Impact**: 10-15% improvement in validation loss, reduced overfitting

**Implementation**:

- Modify ResNet first conv layer: `Conv2d(4, 64, ...)`
- Compare with current dual-encoder approach
- Test with ResNet-18, ResNet-34

**Testing Strategy**:

```bash
python train.py --encoder early_fusion_resnet34 --experiment_name early_fusion_exp
```

---

#### Experiment 1.2: FPN Segmentation Head

**Rationale**: Current segmentation uses only deep features, missing spatial details

**Architecture**:

```
Encoder outputs multiple feature maps at different scales
    layer1 (1/4 resolution) → 64 channels
    layer2 (1/8 resolution) → 128 channels  
    layer3 (1/16 resolution) → 256 channels
    layer4 (1/32 resolution) → 512 channels
         ↓
FPN: Progressively upsample and combine with skip connections
         ↓
Better segmentation masks
```

**Benefits**:

- Multi-scale features improve segmentation quality
- Better segmentation → better shared representations → better calorie prediction
- Standard in semantic segmentation tasks
- Modest parameter increase (~5-10%)

**Expected Impact**: 15-25% improvement in segmentation IoU, 5-10% better calorie MAE

**Implementation**:

- Add FPNSegmentationHead to heads.py
- Use skip connections from encoder intermediate layers
- Similar to U-Net but lighter

**Testing Strategy**:

```bash
python train.py --encoder resnet34 --segmentation_head fpn --experiment_name fpn_seg_exp
```

---

### Phase 2: Better Fusion Strategies (Medium Priority)

Try these **only if** Phase 1 shows promise and no overfitting.

#### Experiment 2.1: Cross-Modal Attention Fusion

**Rationale**: Let RGB and Depth attend to each other before fusion

**Architecture**:

```
RGB features (512 ch)  ─┐
                        ├─→ Cross-Attention ──→ Attended features
Depth features (512 ch) ─┘                      
                                ↓
                        Concatenate + Conv1x1
                                ↓
                          Fused features
```

**Mechanism**:

```python
# RGB queries Depth
rgb_attended = attention(query=rgb_feat, key=depth_feat, value=depth_feat)
# Depth queries RGB  
depth_attended = attention(query=depth_feat, key=rgb_feat, value=rgb_feat)
# Combine
fused = conv1x1(concat([rgb_attended, depth_attended]))
```

**Benefits**:

- Model learns which modality to trust for each spatial region
- Better than simple concatenation
- Only ~10% more parameters than middle fusion

**Expected Impact**: 5-10% improvement over middle fusion if modalities are complementary

**Testing Strategy**:

```bash
python train.py --encoder resnet34 --fusion cross_modal_attention
```

---

#### Experiment 2.2: Gated Fusion

**Rationale**: Learn adaptive fusion weights per sample

**Architecture**:

```
RGB feat + Depth feat → Gate Network → Fusion weights
                                 ↓
fused = gate * RGB + (1 - gate) * Depth
```

**Benefits**:

- Adapts to each sample's characteristics
- Handles cases where one modality is noisy or less informative
- Minimal parameter overhead

**Expected Impact**: 3-7% improvement, especially on samples with poor depth quality

**Testing Strategy**:

```bash
python train.py --encoder resnet34 --fusion gated
```

---

### Phase 3: Advanced Techniques (Low Priority - Only if Still Improving)

Try these **only if** Phase 2 successful and validation loss still decreasing.

#### Experiment 3.1: Multi-Scale Fusion

**Rationale**: Fuse features at multiple encoder depths, not just final layer

**Architecture**:

```
RGB Encoder          Depth Encoder
   ↓                      ↓
layer1 ─────┬────── layer1  (Fusion 1/4 res)
   ↓        │          ↓
layer2 ─────┼──┬────  layer2  (Fusion 1/8 res)
   ↓        │  │       ↓
layer3 ─────┼──┼──┬── layer3  (Fusion 1/16 res)
   ↓        │  │  │    ↓
layer4 ─────┴──┴──┴── layer4  (Fusion 1/32 res)
             ↓
    Aggregate all fused features
```

**Benefits**:

- Captures both low-level and high-level cross-modal interactions
- Better for tasks requiring fine details (like segmentation)
- Common in medical imaging

**Expected Impact**: 5-10% improvement in segmentation, 2-5% in regression

---

#### Experiment 3.2: SE (Squeeze-and-Excitation) Blocks in Heads

**Rationale**: Add channel attention to emphasize important features

**Architecture**:

```
Features → Global Pool → FC → ReLU → FC → Sigmoid → Channel Weights
              ↓                                           ↓
         Apply weights to features → Regression/Segmentation
```

**Benefits**:

- Minimal parameters (~1% increase)
- Proven in ResNet-SE variants
- Easy to add, low risk

**Expected Impact**: 2-5% improvement, marginal but safe to try

---

## Experimental Strategy

### Recommended Testing Order:

**Week 1: Foundation**

1. Baseline: Current ResNet-34 + middle fusion (already done)
2. **Test Early Fusion + ResNet-34** (highest priority)
3. **Test Current + FPN Segmentation**

**Week 2: Best Combination from Week 1**

4. Combine best from steps 2-3 (e.g., Early Fusion + FPN)
5. Fine-tune hyperparameters if needed

**Week 3: Advanced Fusion (if not overfitting)**

6. Test Cross-Modal Attention
7. Test Gated Fusion

**Week 4: Final Refinements**

8. Multi-scale fusion if still improving
9. SE blocks if marginal gains needed

### Decision Criteria:

**Try Next Experiment If:**

- Validation loss improving
- Training/validation gap < 20%
- Model converging within 30-40 epochs

**Stop and Regularize If:**

- Training/validation gap > 30% (overfitting)
- Validation loss plateauing early (< 10 epochs)
- Try: increase dropout, weight decay, or simplify architecture

**Go Back to Simpler Model If:**

- Complex model (e.g., multi-scale) shows worse validation than simple
- Training unstable or not converging

---

## Specific Recommendations by Scenario

### If Currently Overfitting (train loss << val loss):

1. **Try Early Fusion** (fewer parameters)
2. Increase dropout: 0.4 → 0.5
3. Increase weight decay: 0.0001 → 0.0005
4. Consider ResNet-18 instead of ResNet-34

### If Currently Underfitting (train loss ≈ val loss, both high):

1. **Try ResNet-34 + FPN + Cross-Modal Attention**
2. Try deeper regression head
3. Consider multi-scale fusion
4. Decrease weight decay

### If Segmentation Poor (helps calorie too):

1. **Add FPN Segmentation Head** (top priority)
2. Increase seg_weight: 0.5 → 0.7
3. Try multi-scale fusion

### If One Modality Seems Weak:

1. **Try Gated Fusion** (learns to ignore noisy modality)
2. Check data quality (corrupt depth maps?)
3. Try cross-modal attention

---

## Implementation Notes

All experiments should:

- Use same hyperparameters initially: lr=0.0005, dropout=0.4, batch=64
- Train for 50 epochs with early stopping (patience=15)
- Use AdamW optimizer + ReduceLROnPlateau scheduler
- Save TensorBoard logs for comparison
- Track: val_loss, calorie_MAE, segmentation_IoU

**Naming Convention**:

```bash
outputs/
  ├── exp1_early_fusion_r34/
  ├── exp2_fpn_seg_r34/
  ├── exp3_early_fpn_r34/
  ├── exp4_cross_attn_r34/
  └── exp5_gated_r34/
```

---

## Expected Outcomes

**Conservative Estimates**:

- Early Fusion: 10-15% improvement, significantly faster
- FPN Segmentation: 15-20% better segmentation, 5-10% better calories
- Best Combination (Early + FPN): 20-30% total improvement
- Advanced Fusion: Additional 5-10% if no overfitting

**Success Metrics**:

- Validation loss < 9000 (current ~10300)
- Calorie MAE < 100 kcal
- Segmentation IoU > 0.7
- Training/validation gap < 25%

---

## Risk Mitigation

**If Results Worse:**

- Early Fusion worse → RGB and Depth need separate processing, keep dual encoders
- FPN worse → Deep features sufficient, keep standard head
- Attention worse → Simple fusion adequate, stick with middle/additive

**Always have fallback**: Keep current ResNet-34 + middle fusion as baseline.

**Don't over-engineer**: If early fusion + FPN gives good results, stop there. More complexity = more risk with limited data.

---

## Summary Table

| Experiment | Priority | Parameter Change | Expected Improvement | Risk |

|------------|----------|------------------|---------------------|------|

| Early Fusion | ⭐⭐⭐ | -50% | 10-15% | Low |

| FPN Segmentation | ⭐⭐⭐ | +10% | 15-25% (seg) | Low |

| Cross-Modal Attention | ⭐⭐ | +10% | 5-10% | Medium |

| Gated Fusion | ⭐⭐ | +5% | 3-7% | Low |

| Multi-Scale Fusion | ⭐ | +20% | 5-10% | High (overfitting) |

| SE Blocks | ⭐ | +1% | 2-5% | Low |

**Recommendation**: Start with Early Fusion + FPN Segmentation. These two alone could give 25-35% improvement with lower overfitting risk than current architecture.