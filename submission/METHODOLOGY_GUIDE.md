# Methodology Document - Usage Guide

## Overview
The `METHODOLOGY.md` file contains a formal, publication-ready methodology section for your research paper on multi-modal calorie estimation.

## Key Features of the Academic Style

### 1. **Formal Language**
- Past tense for completed work ("We conducted", "We employed")
- Passive voice where appropriate
- Technical precision with mathematical notation
- No casual language or emojis

### 2. **Mathematical Notation**
The document uses LaTeX-style math for equations:
- Loss function: $\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$
- Volume estimation: $V = \sum_{i \in \mathcal{F}} d_i \cdot A_{\text{pixel}}$
- Learning rate schedule with piecewise functions

### 3. **Structured Presentation**
- Section 3.1: Dataset description
- Section 3.2: Preprocessing and augmentation
- Section 3.3: Network architectures (ResNet, InceptionV3)
- Section 3.4: Fusion strategies (early, middle, late)
- Section 3.5: Training procedures
- Section 3.6: Evaluation metrics
- Section 3.7: Experimental design with Table 1
- Section 3.8: Implementation details

### 4. **Research Paper Conventions**
- Clear subsection numbering (3.1, 3.2, etc.)
- Table 1 summarizing all experiments
- Mathematical definitions for all key operations
- Justification for design choices
- Focus on reproducibility

## How to Use in Your Paper

1. **Direct Integration**: Copy the entire content into your paper's methodology section
2. **Adapt Numbering**: If methodology is not Section 3, update all subsection numbers
3. **Add References**: Insert citations where indicated (e.g., "Following the baseline established in the original Nutrition5K work [citation]")
4. **Figures**: Consider adding architecture diagrams referenced from your notebooks

## Suggested Additions

### If space allows, consider adding:
1. **Figure 1**: Dual-stream architecture diagram showing RGB/Depth encoders and fusion
2. **Figure 2**: Comparison of fusion strategies (early/middle/late) as visual diagrams
3. **Algorithm box**: Pseudocode for the training procedure
4. **Extended table**: Add results preview to Table 1 if this is a results section

## Notes
- All mathematical notation is LaTeX-compatible
- The document balances technical depth with readability
- Maintains focus on "why" not just "what"
- Suitable for computer vision conferences (CVPR, ICCV, ECCV style) or journals

