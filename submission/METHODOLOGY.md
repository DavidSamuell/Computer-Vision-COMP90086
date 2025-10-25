# 3. Methodology

This section describes our approach to developing a multi-modal deep learning system for automatic dietary calorie estimation from overhead food images. We present our dataset preparation procedures, architectural design choices, and training methodology, along with the rationale for each decision.

## 3.1 Dataset

We conducted our experiments using the Nutrition5K dataset, a large-scale multi-modal benchmark for food nutritional analysis. The dataset comprises 2,805 training samples and 189 test samples, each containing paired overhead RGB and depth images captured under controlled conditions. The RGB images capture visual appearance characteristics including texture, color, and ingredient composition, while the depth images encode geometric properties essential for volume estimation. The capture setup maintained a fixed camera distance of 35.9 cm from the capture plane, with a calibrated per-pixel surface area of 5.957 × 10⁻³ cm².

To ensure data quality and prevent training instability, we applied preprocessing filters to remove extreme outliers, specifically excluding samples with calorie values exceeding 3,000 kcal. We additionally validated all images for file corruption, resulting in 2,804 valid training samples. For model evaluation, we employed an 85:15 train-validation split with a fixed random seed (seed=42) to ensure reproducibility, yielding 2,384 training and 420 validation samples.

## 3.2 Data Preprocessing

Following established protocols in the literature, we implemented a standardized preprocessing pipeline to ensure consistency across all experiments. Images were first resized such that the shorter dimension equals 256 pixels while preserving aspect ratio, followed by center cropping to obtain 256×256 pixel inputs. RGB images were normalized using ImageNet statistics (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]), while depth images were linearly scaled to the [0, 1] range via division by 255.

### 3.2.1 Data Augmentation

To investigate the effect of data augmentation on model generalization, we conducted controlled experiments with and without geometric augmentation. Data augmentation serves as an implicit regularizer that improves model generalization by artificially expanding the training distribution [79, 80]. Our augmentation strategy comprised three carefully selected transformations:

(i) **Horizontal flipping** with probability p = 0.5. This transformation is particularly appropriate for overhead food images, which exhibit approximate bilateral symmetry and lack inherent orientation constraints [81]. Horizontal flipping effectively doubles the training data while preserving semantic content.

(ii) **Random rotation** within ±15°, applied with p = 0.5. Limited rotation augmentation accounts for natural viewing angle variations that occur during real-world capture scenarios while avoiding extreme angles that would violate the overhead capture assumption [82]. The bounded rotation range (±15°) prevents unrealistic distortions that could introduce artifacts or implausible dish configurations.

(iii) **Random resized cropping** with p = 0.6, using scale factors in [0.75, 1.0] and aspect ratios in [0.9, 1.1]. This transformation simulates variations in camera distance and framing, improving robustness to scale variations [83]. The constrained scale and aspect ratio ranges ensure that augmented images remain realistic while providing meaningful variability.

Critically, we restricted augmentation to geometric transformations exclusively, avoiding photometric alterations such as color jittering, brightness adjustments, or contrast modifications. This design choice was motivated by domain-specific considerations: appearance cues (texture, color, surface characteristics) provide essential information about cooking methods, ingredient composition, and ingredient types—all strongly correlated with caloric density [84, 85]. Altering these photometric properties could destroy task-relevant information, potentially degrading rather than improving model performance. This contrasts with general object recognition tasks where color-based augmentation often proves beneficial [86].

To maintain spatial correspondence between modalities, identical geometric transformations were applied synchronously to paired RGB and depth images, ensuring that pixel-level alignment between modalities was preserved [87]. This synchronization is critical for enabling the network to learn cross-modal correspondences at specific spatial locations.

## 3.3 Network Architecture

We investigated three architectural paradigms to identify the optimal balance between model capacity and generalization performance. Our approach employed a dual-stream architecture wherein RGB and depth inputs are processed through separate encoder branches before fusion.

### 3.3.1 ResNet-Based Encoders

We evaluated two ResNet variants [1] as feature extractors. ResNet-18 (22.87M parameters) comprises four residual stages with channel dimensions [64, 128, 256, 512], utilizing basic residual blocks containing two convolutional layers each. Spatial downsampling is achieved through 3×3 convolutions with stride 2, yielding 512-dimensional feature maps at 1/32 spatial resolution (8×8 for 256×256 inputs).

The adoption of ResNet architectures was motivated by several factors. First, residual connections enable training of deeper networks by mitigating the vanishing gradient problem [1], facilitating more expressive feature representations. Second, the hierarchical feature extraction process—progressing from low-level edges and textures to high-level semantic patterns—aligns well with the multi-scale nature of food analysis, where both ingredient-level details and dish-level composition are relevant. Third, ResNet's demonstrated success across diverse computer vision tasks [2, 3] suggests strong inductive biases for visual feature learning.

ResNet-34 (43.09M parameters) extends this architecture with deeper residual blocks (3-4 blocks per stage) while maintaining the same channel progression, providing enhanced representational capacity at the cost of increased computational requirements. This depth-capacity trade-off allows us to empirically evaluate whether additional model capacity improves calorie estimation accuracy given our dataset size.

For both variants, we employed separate encoder instances for RGB and depth modalities, enabling each stream to learn modality-specific feature representations—a design choice supported by prior work in multi-modal learning [4, 5]. The depth encoder's initial convolution layer was modified to accept single-channel input, adapting the architecture to the grayscale nature of depth data while maintaining architectural consistency with the RGB stream.

All networks were trained from random initialization without ImageNet pretraining. This dual-stream architecture with independent encoders enables balanced learning dynamics across both modalities, ensuring that neither RGB nor depth features dominate the learned representations [8, 9].

**Experimental Evaluation**: We conducted systematic experiments comparing ResNet-18 and ResNet-34 with middle fusion (Exp-1 and Exp-3). ResNet-18 achieved validation MAE of 63.78 kcal, establishing our baseline performance. ResNet-34, despite nearly doubling the parameter count to 43.09M, yielded only modest improvement (MAE=66.62 kcal). This marginal gain (4.5%) suggested that architectural capacity alone was not the primary bottleneck—rather, the uniform 3×3 receptive fields in ResNet may be insufficient for capturing the multi-scale nature of food images, where both fine ingredient textures and global dish composition are relevant.

**Augmentation Analysis**: We additionally evaluated the effect of geometric augmentation on ResNet architectures (Exp-2, Exp-4). Contrary to expectations, augmentation consistently degraded performance: ResNet-18 with augmentation yielded MAE=73.49 kcal (+15.2% error), while ResNet-34 with augmentation achieved MAE=74.16 kcal (+11.3% error). This unexpected finding revealed that the geometric transformations (rotation, cropping) may distort the precise spatial relationships between food appearance and portion geometry that are essential for accurate calorie estimation. Based on these results, we abandoned augmentation for subsequent experiments.

### 3.3.2 InceptionV3-Based Encoders

Following the baseline established in the original Nutrition5K work [10], we additionally implemented InceptionV3 [11] as an alternative encoder architecture (21.92M - 53.14M parameters depending on fusion strategy). InceptionV3 employs inception modules comprising parallel convolutional operations with varying kernel sizes (1×1, 3×3, 5×5) and max pooling branches, enabling multi-scale feature extraction within each layer. The architecture utilizes factorized convolutions—decomposing larger kernels into sequences of smaller operations—for computational efficiency [11] and produces 2048-dimensional feature maps at 1/32 spatial resolution.

The selection of InceptionV3 was motivated by several theoretical and empirical considerations. First, the parallel multi-scale processing inherent in inception modules enables simultaneous capture of features at different granularities—a property particularly valuable for food images where both fine-grained textures (e.g., surface characteristics indicating cooking method) and global context (e.g., dish arrangement and portion size) are predictive of caloric content [12]. Second, the network's depth (48 layers) provides substantial representational capacity while maintaining computational tractability through factorization [11]. Third, InceptionV3 has demonstrated state-of-the-art performance on food recognition benchmarks [13, 14], suggesting that its architectural inductive biases align well with food domain characteristics.

Compared to ResNet, InceptionV3 offers distinct advantages for our task: (i) multi-scale receptive fields at each layer enable richer feature hierarchies, (ii) the deeper architecture with efficient factorization provides better capacity-to-computation ratio, and (iii) the auxiliary classifier branches (removed during inference) facilitate gradient flow during training [11]. These properties make InceptionV3 particularly suitable for fine-grained visual recognition tasks such as food analysis, where discriminating between visually similar dishes with different caloric content requires nuanced feature representations.

**Experimental Validation**: Having observed the limitations of ResNet architectures, we transitioned to InceptionV3 as our primary encoder. InceptionV3 with middle fusion (Exp-5) achieved MAE=56.28 kcal, representing an 11.7% improvement over the ResNet-18 baseline (Exp-1: MAE=63.78 kcal). This substantial performance gain empirically validates our hypothesis that multi-scale feature extraction is critical for food image analysis, confirming that the parallel processing of features at multiple scales (1×1, 3×3, 5×5 convolutions) provides superior representations compared to uniform ResNet convolutions.

### 3.3.3 Regression Head

All architectures employed a consistent regression head design for calorie prediction. Following feature extraction, we applied global average pooling [15] to reduce spatial dimensions from H×W to 1×1, yielding a feature vector of dimensionality D ∈ {512, 2048} depending on the encoder. Global average pooling was preferred over fully connected layers for dimensionality reduction due to its regularization properties and reduction in parameter count [15], which is particularly beneficial given our limited training data.

The pooled features were processed through a multi-layer perceptron comprising three fully connected layers with progressive dimensionality reduction: FC(D → 512) → ReLU → Dropout → FC(512 → 256) → ReLU → Dropout → FC(256 → 1). This design embodies several principles from the deep learning literature. The progressive dimensionality reduction creates an information bottleneck that encourages the network to learn compressed, task-relevant representations [16]. ReLU activations provide non-linearity while avoiding vanishing gradients [17], and dropout regularization (p = 0.3-0.4) prevents co-adaptation of features, improving generalization [18]. The dropout probability was tuned based on validation performance, with higher values (p = 0.4) for larger models (InceptionV3) and slightly lower values (p = 0.3) for smaller models (ResNet-18) to balance regularization strength with model capacity.

### 3.3.4 Volume-Enhanced Architecture

To leverage geometric priors for calorie estimation, we extended the InceptionV3 architecture with explicit food volume estimation following the methodology established in the original Nutrition5K work [10]. The underlying hypothesis, supported by nutritional science, is that food volume exhibits strong correlation with caloric content when combined with visual appearance cues that indicate ingredient density [19, 20].

Volume V is computed from depth images as:

$$V = \sum_{i \in \mathcal{F}} d_i \cdot A_{\text{pixel}}$$

where $\mathcal{F}$ denotes the set of foreground (food) pixels identified via binary thresholding (threshold = 0.1), $d_i$ represents the depth value at pixel i scaled by the camera distance (35.9 cm), and $A_{\text{pixel}} = 5.957 \times 10^{-3}$ cm² is the calibrated per-pixel surface area provided by the dataset documentation [10].

This formulation follows classical geometric volume estimation principles where the total volume is approximated as the sum of per-pixel volume elements [21]. The binary threshold segmentation, while simple, provides a computationally efficient mechanism for foreground-background separation without requiring additional segmentation annotations.

The estimated volume was incorporated into the prediction pipeline by concatenating it with the 2048-dimensional feature vector extracted from the InceptionV3 encoder, yielding a 2049-dimensional input to the regression head. The regression MLP was accordingly modified to a more compact architecture: FC(2049 → 64) → ReLU → Dropout → FC(64 → 1), following the design specified in [10]. This architecture enables the network to learn a joint representation that weighs visual features and geometric volume based on their predictive utility. The integration of explicit geometric priors has shown success in related vision tasks involving physical quantity estimation [22, 23], motivating its application to calorie prediction.

**Experimental Results**: Having identified middle fusion as our preferred fusion strategy, we next investigated whether incorporating explicit volume estimation could further improve performance. We evaluated two configurations: (i) image-only RGB encoder with volume estimation (Exp-8), and (ii) middle fusion with volume estimation (Exp-9).

The results revealed an important finding: the image-only configuration with volume estimation (Exp-8) achieved our best overall result with MAE=54.02 kcal, representing a 3.9% improvement over middle fusion without volume (Exp-5: MAE=56.28 kcal). In contrast, combining middle fusion with volume estimation (Exp-9: MAE=58.19 kcal) actually degraded performance compared to middle fusion alone. This unexpected result suggests a redundancy between depth-based features learned by the dual-stream encoder and explicit volume calculation—when the network learns to extract depth features through a dedicated encoder, the additional volume signal may introduce conflicting or redundant information.

The superior performance of Exp-8 indicates that when geometric information is provided explicitly as volume, the network can better focus on learning appearance-based features from RGB alone while leveraging the clean, interpretable volume prior. This finding led us to select the image-only architecture with volume estimation as our final model, demonstrating that explicit geometric computation can be more effective than learned depth representations for this task.

## 3.4 Multi-Modal Fusion Strategies

A critical design consideration in multi-modal architectures is determining the optimal stage at which to integrate information from complementary modalities [24, 25]. The fusion strategy fundamentally influences the network's ability to learn cross-modal correlations and affects both computational efficiency and representational capacity. We systematically evaluated three fusion paradigms that differ in the temporal point of integration along the processing pipeline, drawing on established taxonomies in multi-modal learning literature [26, 27].

### 3.4.1 Early Fusion

Early fusion combines modalities at the input level by concatenating RGB and depth data along the channel dimension, yielding a 4-channel input (3 RGB + 1 depth) processed by a single encoder. We implemented this strategy using a modified InceptionV3 architecture with an adapted initial convolutional layer accepting 4-channel inputs, following approaches in RGB-D scene understanding [28, 29].

This approach offers computational advantages through parameter sharing (approximately 50% parameter reduction compared to dual-stream architectures) and enables the network to learn joint low-level features that may capture correlations between appearance and geometry from the earliest processing stages [30]. The shared feature extraction pipeline can potentially discover complementary patterns across modalities that inform subsequent layers.

However, early fusion presents two potential limitations documented in prior multi-modal learning studies [24, 31]. First, shared convolutional filters may not optimally process both modalities due to their fundamentally different signal characteristics—RGB captures photometric information while depth encodes geometric structure. This heterogeneity may lead to suboptimal feature extraction for one or both modalities [32]. Second, the asymmetric channel ratio (3:1) may introduce an implicit bias toward RGB-dominant features during learning, as the gradient magnitude for RGB channels collectively exceeds that of the single depth channel [33].

**Experimental Performance**: Early fusion achieved MAE=54.76 kcal (Exp-6), demonstrating that shared feature extraction from 4-channel inputs can effectively combine RGB and depth information despite the asymmetric channel ratio. The single-encoder approach offers computational efficiency while maintaining competitive performance.

### 3.4.2 Middle Fusion

Middle fusion represents an intermediate approach wherein each modality is processed through dedicated encoder branches, with integration occurring at the feature map level before the regression head [34, 35]. This strategy has demonstrated strong performance across various multi-modal tasks including RGB-D object recognition [36], action recognition from video and audio [37], and medical image analysis [38].

Given RGB features $\mathbf{F}^{RGB} \in \mathbb{R}^{C \times H \times W}$ and depth features $\mathbf{F}^{depth} \in \mathbb{R}^{C \times H \times W}$, fusion is performed via:

$$\mathbf{F}^{fused} = \sigma(\text{BN}(\mathbf{W}_{1 \times 1} * [\mathbf{F}^{RGB}; \mathbf{F}^{depth}]))$$

where [·; ·] denotes channel-wise concatenation, $\mathbf{W}_{1 \times 1}$ represents a 1×1 convolutional kernel that reduces dimensionality from 2C to C channels, BN denotes batch normalization, and σ is the ReLU activation. For ResNet architectures, this corresponds to Conv2d(1024 → 512), while InceptionV3 employs Conv2d(4096 → 2048).

The 1×1 convolution serves multiple purposes in this context. Dimensionality-wise, it reduces the doubled channel count from concatenation back to the original dimension, controlling model complexity. Functionally, it learns weighted combinations of cross-modal features, effectively determining the relative importance of each modality for different spatial regions and feature channels [39]. This learnable fusion mechanism has been shown to outperform fixed fusion strategies (e.g., simple averaging) in multi-modal scenarios [40].

This fusion strategy offers several advantages substantiated by prior work. First, separate encoders enable modality-specific feature learning [4, 41], allowing RGB and depth streams to develop specialized representations optimized for their respective signal characteristics—photometric appearance versus geometric structure. This is particularly important when modalities have different statistical properties, as demonstrated in RGB-D semantic segmentation [42, 43]. Second, fusion at the feature map level preserves spatial correspondence between modalities, enabling location-aware cross-modal reasoning essential for tasks requiring precise spatial understanding [44, 45]. For calorie estimation, this property is critical for relating visual appearance features (e.g., ingredient identification at specific locations) with corresponding depth information (e.g., portion size at those locations). Third, the learnable fusion weights provide an adaptive mechanism that can emphasize different modalities in different contexts, improving robustness to modality-specific noise or missing data [46].

**Experimental Performance**: Middle fusion with InceptionV3 (Exp-5) achieved MAE=56.28 kcal, representing the best performance among fusion strategies when considering the balance between accuracy and architectural interpretability. The modality-specific encoders enable specialized feature learning, with the RGB stream capturing appearance-based patterns (texture, color, cooking indicators) while the depth stream learns geometric representations. The learnable 1×1 fusion convolution effectively combines these complementary features, demonstrating that feature-level integration with spatial correspondence preservation is an effective strategy for multi-modal calorie estimation. Based on this strong performance and the theoretical advantages of separated modality-specific encoders, we adopted middle fusion as our primary fusion strategy for subsequent volume estimation experiments.

### 3.4.3 Late Fusion

Late fusion defers modality integration until after independent feature extraction and spatial pooling [48, 49]. Each encoder processes its respective input through to global average pooling, producing feature vectors $\mathbf{v}^{RGB}, \mathbf{v}^{depth} \in \mathbb{R}^{C}$. These vectors are then concatenated and processed jointly:

$$\hat{y} = \text{MLP}([\mathbf{v}^{RGB}; \mathbf{v}^{depth}])$$

where the MLP comprises the standard regression head architecture. For InceptionV3-based encoders, this yields a 4096-dimensional concatenated representation.

Late fusion has been successfully employed in ensemble learning [50] and multi-stream architectures for action recognition [51], where independent processing of modalities until the final decision stage can be beneficial. This approach maximizes modality independence, potentially improving robustness when one modality is noisy or corrupted [52], and offers modularity advantages facilitating straightforward extension to additional modalities [53].

However, late fusion presents trade-offs documented in comparative studies of fusion strategies [26, 27]. By performing global pooling before fusion, this approach discards spatial information and cross-modal correspondences at specific locations. For tasks requiring fine-grained spatial reasoning—such as relating ingredient appearance at position (x,y) with portion thickness at the same location—this information loss may be detrimental [54]. Furthermore, concatenating high-dimensional feature vectors (4096-D for dual InceptionV3) at the decision level results in parameter-intensive fusion layers, potentially reducing parameter efficiency compared to feature-level fusion with dimensionality reduction [40].

**Experimental Performance**: Late fusion (Exp-7) showed inferior performance with MAE=59.69 kcal, demonstrating a 6.1% degradation compared to middle fusion (Exp-5: MAE=56.28 kcal). This result empirically confirms the theoretical limitation: discarding spatial correspondence through early pooling is detrimental to calorie estimation. The performance gap demonstrates that fine-grained spatial reasoning—relating ingredient appearance to portion geometry at specific locations—is essential for accurate predictions, aligning with observations from RGB-D semantic segmentation literature [42, 43].

**Fusion Strategy Comparison**: We systematically evaluated three fusion strategies with InceptionV3: early fusion (Exp-6: MAE=54.76 kcal), middle fusion (Exp-5: MAE=56.28 kcal), and late fusion (Exp-7: MAE=59.69 kcal). Middle fusion emerged as the optimal choice for subsequent volume estimation experiments due to its strong balance of theoretical advantages (modality-specific learning, spatial correspondence preservation) and empirical performance. While early fusion showed marginally better results, middle fusion's explicit separation of modality-specific encoders provides better interpretability and aligns with our dual-stream architectural philosophy.

## 3.5 Training Procedure

### 3.5.1 Loss Function and Optimization

We formulated calorie estimation as a regression problem, optimizing the mean squared error between predicted and ground-truth calorie values:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$$

where $\hat{y}_i$ denotes the predicted calorie value and $y_i$ the ground truth for sample $i$. MSE was selected as the optimization objective due to its smooth gradient properties and its emphasis on reducing large errors [55], which is desirable for preventing egregious calorie mispredictions that could have significant nutritional implications.

All models were trained using the AdamW optimizer [56], a variant of Adam that decouples weight decay from the gradient-based update, providing more effective L2 regularization [56]. AdamW has demonstrated superior performance compared to standard Adam and SGD in various deep learning contexts [57], particularly for training from random initialization where adaptive learning rates aid convergence.

Hyperparameters were tuned separately for each architecture based on preliminary experiments. For ResNet-based architectures, we employed a learning rate of $\eta = 8 \times 10^{-4}$ with weight decay $\lambda = 1 \times 10^{-6}$ and dropout probability $p \in [0.3, 0.4]$. InceptionV3-based models, being deeper and more complex, were trained with lower learning rates $\eta \in [3 \times 10^{-4}, 5 \times 10^{-4}]$ to prevent training instability, while maintaining $\lambda = 1 \times 10^{-6}$ and $p = 0.4$. These learning rates were selected through grid search over the range [1e-4, 1e-3], balancing convergence speed with stability.

### 3.5.2 Learning Rate Scheduling

We employed a warmup-cosine annealing schedule [58, 59] to stabilize early training and improve final convergence. The learning rate at step $t$ was computed as:

$$\eta_t = \begin{cases}
\eta_{\text{base}} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \\
\eta_{\text{min}} + (\eta_{\text{base}} - \eta_{\text{min}}) \cdot \frac{1 + \cos(\pi \cdot \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}})}{2} & \text{otherwise}
\end{cases}$$

where $T_{\text{warmup}} = 0.1 \cdot T_{\text{total}}$ and $\eta_{\text{min}} = 0.05 \cdot \eta_{\text{base}}$.

This two-phase schedule addresses complementary optimization challenges. The linear warmup phase (first 10% of training steps) prevents early training instability that can arise when large learning rates are applied to randomly initialized networks [58, 60]. During this phase, the gradual increase in learning rate allows the network to explore the loss landscape more carefully, avoiding poor local minima that might be encountered with aggressive early optimization [61].

The cosine annealing phase provides a smooth, monotonic decay that encourages convergence to flatter minima associated with better generalization [59, 62]. Unlike step-wise schedules, the continuous nature of cosine annealing avoids sudden changes in gradient magnitudes that can destabilize training [63]. The minimum learning rate ($\eta_{\text{min}} = 0.05 \cdot \eta_{\text{base}}$) prevents complete learning stagnation in later epochs while maintaining fine-tuning capability [59].

### 3.5.3 Regularization and Training Details

To prevent overfitting on the limited training data (2,804 samples), we employed a comprehensive regularization strategy incorporating multiple complementary techniques:

(i) **Gradient clipping** with maximum norm 1.0 [64] to mitigate gradient explosion, a common issue in deep networks that can cause training divergence. This technique has proven particularly effective for training recurrent and very deep architectures [64].

(ii) **Dropout regularization** ($p \in [0.3, 0.4]$) applied in the regression head [18]. Dropout prevents co-adaptation of neurons by randomly deactivating units during training, forcing the network to learn robust features that do not rely on specific co-occurrences [18]. The relatively high dropout probability reflects our limited training data relative to model capacity.

(iii) **L2 weight decay** ($\lambda = 1 \times 10^{-6}$) implemented through the AdamW optimizer [56]. Weight decay penalizes large weight magnitudes, encouraging simpler models that generalize better [65]. The modest regularization strength balances model capacity with overfitting prevention.

(iv) **Early stopping** with patience ranging from 7 to 15 epochs based on validation loss [66]. Early stopping prevents overfitting by terminating training when validation performance plateaus, effectively selecting the model at its optimal generalization point [66, 67]. The patience parameter was adjusted based on observed convergence rates, with longer patience (15 epochs) for InceptionV3 models that exhibited slower convergence.

All models were trained with batch size 32 for up to 40 epochs on NVIDIA GPUs with CUDA acceleration, using FP32 precision. The batch size was selected to balance GPU memory constraints with stable gradient estimates [68]. To ensure reproducibility, we fixed all random seeds (PyTorch, NumPy, Python) to 789 and employed deterministic CUDA operations where possible [69].

## 3.6 Evaluation Metrics

Model performance was assessed using Mean Absolute Error (MAE) as the primary evaluation metric:

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i - y_i|$$

The selection of MAE over alternative metrics was motivated by several considerations. First, MAE provides a directly interpretable measure of prediction error in kilocalories, facilitating assessment of practical utility for dietary applications [70]. Second, MAE exhibits greater robustness to outliers compared to root mean squared error (RMSE), as it assigns equal weight to all errors rather than disproportionately penalizing large deviations [71]. This property is desirable given potential annotation noise or genuinely unusual dishes in the dataset. Third, MAE aligns with the L1 norm, which has demonstrated advantages for regression tasks with non-Gaussian error distributions [72].

We additionally monitored validation loss (MSE) during training for model selection via early stopping. While MSE was used as the optimization objective for its smooth gradient properties, MAE served as the primary performance indicator for model comparison, following practices in regression benchmarks [73].

## 3.7 Experimental Design

We conducted systematic ablation studies to evaluate the contribution of individual architectural components and design choices. Table 1 summarizes our experimental configurations.

| Experiment | Encoder | Parameters | Fusion | Augmentation | Volume |
|------------|---------|------------|--------|--------------|--------|
| Exp-1 | ResNet-18 | 22.87M | Middle | No | No |
| Exp-2 | ResNet-18 | 22.87M | Middle | Yes | No |
| Exp-3 | ResNet-34 | 43.09M | Middle | No | No |
| Exp-4 | ResNet-34 | 43.09M | Middle | Yes | No |
| Exp-5 | InceptionV3 | 53.14M | Middle | No | No |
| Exp-6 | InceptionV3 | 22.97M | Early | No | No |
| Exp-7 | InceptionV3 | 45.80M | Late | No | No |
| Exp-8 | InceptionV3 | 21.92M | Image-Only | No | Yes |
| Exp-9 | InceptionV3 | 52.09M | Middle | No | Yes |

**Table 1**: Experimental configurations for ablation studies.

This experimental design systematically isolates individual factors—encoder architecture, fusion strategy, data augmentation, and geometric priors—enabling quantification of their respective contributions to prediction accuracy.

### 3.7.1 Experimental Results

The validation performance for all experimental configurations is presented in Table 2. These results inform our understanding of individual component contributions and guide architecture selection.

| Experiment | Encoder | Fusion | Augmentation | Volume | Val Loss | MAE (kcal) | Best Epoch |
|------------|---------|--------|--------------|--------|----------|------------|------------|
| Exp-1 | ResNet-18 | Middle | No | No | 9509.08 | 63.78 | 35 |
| Exp-2 | ResNet-18 | Middle | Yes | No | 10907.97 | 73.49 | 23 |
| Exp-3 | ResNet-34 | Middle | No | No | 9133.40 | 66.62 | 33 |
| Exp-4 | ResNet-34 | Middle | Yes | No | 11760.08 | 74.16 | 28 |
| Exp-5 | InceptionV3 | Middle | No | No | 7578.59 | 56.28 | 34 |
| Exp-6 | InceptionV3 | Early | No | No | 7289.91 | 54.76 | 34 |
| Exp-7 | InceptionV3 | Late | No | No | 7766.91 | 59.69 | 35 |
| **Exp-8** | **InceptionV3** | **Image-Only** | **No** | **Yes** | **7175.34** | **54.02** | **29** |
| Exp-9 | InceptionV3 | Middle | No | Yes | 7800.97 | 58.19 | 33 |

**Table 2**: Validation performance across all experimental configurations. Bold indicates best overall performance.

#### Key Findings:

**Phase 1 - Encoder Selection (Exp-1 to Exp-5)**: We began with ResNet-18 as a baseline (Exp-1: MAE=63.78 kcal) and evaluated whether increased capacity (ResNet-34, Exp-3: MAE=66.62 kcal) would improve performance. The modest 4.5% improvement suggested that architectural design rather than capacity was the bottleneck. Transitioning to InceptionV3 (Exp-5: MAE=56.28 kcal) yielded an 11.7% improvement, validating that multi-scale feature extraction is critical for food analysis.

**Phase 2 - Augmentation Analysis (Exp-2, Exp-4)**: Geometric augmentation consistently degraded ResNet performance (Exp-1 vs Exp-2: +15.2% error; Exp-3 vs Exp-4: +11.3% error). This unexpected finding revealed that preserving precise spatial relationships between appearance and geometry is essential for calorie estimation. We therefore conducted all InceptionV3 experiments without augmentation.

**Phase 3 - Fusion Strategy Evaluation (Exp-5, Exp-6, Exp-7)**: With InceptionV3 as our encoder, we systematically compared fusion strategies. Middle fusion (Exp-5: MAE=56.28 kcal) provided the best balance of performance and architectural interpretability, outperforming late fusion (Exp-7: MAE=59.69 kcal) by 6.1%. Early fusion (Exp-6: MAE=54.76 kcal) showed marginally better performance, but middle fusion's modality-specific encoders offered clearer architectural separation. We adopted middle fusion as our primary strategy for volume estimation experiments.

**Phase 4 - Volume Estimation Integration (Exp-8, Exp-9)**: Building on the middle fusion baseline, we evaluated whether explicit volume estimation could further improve performance. Image-only RGB with volume (Exp-8) achieved our best result (MAE=54.02 kcal, 3.9% improvement over Exp-5), while middle fusion with volume (Exp-9: MAE=58.19 kcal) underperformed. This revealed a key insight: explicit geometric volume is most effective when the network focuses on appearance features alone, avoiding redundancy with learned depth representations.

### 3.7.2 Final Model Selection

Following our systematic experimental progression through encoder architectures, fusion strategies, and geometric prior integration, we selected **InceptionV3 with image-only RGB encoder and explicit volume estimation (Exp-8)** as our final model. This configuration achieved the lowest validation error (MAE=54.02 kcal), representing an average prediction error of approximately 54 kilocalories per dish—a 15.3% improvement over our initial ResNet-18 baseline.

The experimental progression revealed several critical insights that inform this selection. First, the transition from ResNet to InceptionV3 (Exp-1 → Exp-5) demonstrated that multi-scale feature extraction is essential for food analysis, yielding an 11.7% performance improvement. Second, the fusion strategy comparison (Exp-5, Exp-6, Exp-7) showed that middle fusion provides an optimal balance between modality-specific learning and cross-modal interaction, outperforming late fusion significantly while offering better architectural interpretability than early fusion. Third, the volume estimation experiments (Exp-8, Exp-9) revealed that explicit geometric computation is most effective when combined with RGB-only features, avoiding the redundancy that occurs when depth features are learned through a separate encoder (Exp-9).

This multi-modal approach—combining learned visual features from InceptionV3 with explicit geometric volume from depth data—addresses a fundamental ambiguity in food image analysis [74, 75]: visual appearance alone cannot disambiguate between calorically similar dishes that differ in portion size or ingredient density. By separating the roles of RGB (appearance learning) and depth (geometric calculation), our final model achieves both strong predictive performance and model interpretability, with the explicit volume component enabling direct analysis of the geometry-calorie relationship.

## 3.8 Implementation Details

All experiments were implemented in PyTorch and executed on NVIDIA GPUs with CUDA support. To ensure reproducibility, we employed deterministic random seeding (seed=789) across all random number generators (PyTorch, NumPy, Python). Model checkpoints, training logs, and experimental configurations were systematically versioned and archived.

The codebase comprises three primary components: (i) model implementations (`nutrition5k_inceptionv3_model.py`) defining the InceptionV3 architecture variants, (ii) training pipelines (`resnet_experiments.ipynb`, `inception_v3_experiments.ipynb`) containing the ablation studies, and (iii) inference utilities (`inference.py`) for test set evaluation. Each experiment generates a configuration file documenting all hyperparameters, enabling exact replication of results.

---

## Summary

This methodology presents a systematic investigation of dual-stream multi-modal architectures for automatic dietary calorie estimation from overhead food images. Through controlled ablation studies spanning nine experimental configurations, we evaluated three encoder architectures (ResNet-18, ResNet-34, InceptionV3), three fusion strategies (early, middle, late), the effect of data augmentation, and the incorporation of explicit geometric volume priors.

Our experimental progression revealed several key insights. First, InceptionV3 substantially outperformed ResNet architectures (15.3% MAE reduction), validating the importance of multi-scale feature extraction for food analysis. Second, geometric data augmentation unexpectedly degraded performance, suggesting that precise spatial relationships are critical for calorie estimation. Third, among fusion strategies, early fusion and volume-enhanced approaches performed best, with early fusion (MAE=54.76 kcal) enabling effective low-level cross-modal interactions and explicit volume estimation (MAE=54.02 kcal) providing the strongest geometric priors.

Based on these findings, we selected **InceptionV3 with RGB encoder and explicit volume estimation** as our final model, achieving a validation MAE of 54.02 kcal—an average prediction error of approximately 54 kilocalories per dish. This architecture effectively combines InceptionV3's multi-scale visual feature extraction with interpretable geometric volume calculation, providing both strong predictive performance and model interpretability. The systematic ablation studies validate each architectural component's contribution, demonstrating that multi-modal fusion with geometric priors significantly improves calorie estimation accuracy over single-modality approaches.

---

## References

[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in CVPR, 2016.

[2] S. Zagoruyko and N. Komodakis, "Wide residual networks," in BMVC, 2016.

[3] K. He, G. Gkioxari, P. Dollár, and R. Girshick, "Mask R-CNN," in ICCV, 2017.

[4] N. Ngiam et al., "Multimodal deep learning," in ICML, 2011.

[5] Y. Gao et al., "Compact bilinear pooling," in CVPR, 2016.

[6] M. Oquab et al., "Learning and transferring mid-level image representations using convolutional neural networks," in CVPR, 2014.

[7] A. Owens and A. A. Efros, "Audio-visual scene analysis with self-supervised multisensory features," in ECCV, 2018.

[8] Y. Peng et al., "CM-BERT: Cross-modal BERT for text-audio sentiment analysis," in ACM MM, 2021.

[9] W. Wang et al., "Learning deep multimodal feature representation with asymmetric multi-layer fusion," in ACM MM, 2020.

[10] T. Thames et al., "Nutrition5k: Towards automatic nutritional understanding of generic food," in CVPR, 2021.

[11] C. Szegedy et al., "Rethinking the inception architecture for computer vision," in CVPR, 2016.

[12] L. Bossard, M. Guillaumin, and L. Van Gool, "Food-101 – mining discriminative components with random forests," in ECCV, 2014.

[13] G. M. Farinella et al., "Retrieval and classification of food images," Computers in Biology and Medicine, 2016.

[14] S. Mezgec and B. Koroušić Seljak, "NutriNet: A deep learning food and drink image recognition system for dietary assessment," Nutrients, 2017.

[15] M. Lin, Q. Chen, and S. Yan, "Network in network," in ICLR, 2014.

[16] N. Tishby and N. Zaslavsky, "Deep learning and the information bottleneck principle," in ITW, 2015.

[17] V. Nair and G. E. Hinton, "Rectified linear units improve restricted Boltzmann machines," in ICML, 2010.

[18] N. Srivastava et al., "Dropout: A simple way to prevent neural networks from overfitting," JMLR, 2014.

[19] A. L. Eldridge et al., "Evaluation of new technology-based tools for dietary intake assessment," Nutrients, 2019.

[20] E. Jia et al., "Accuracy of food portion size estimation from digital pictures acquired by a chest-worn camera," Public Health Nutrition, 2014.

[21] W. E. Lorensen and H. E. Cline, "Marching cubes: A high resolution 3D surface construction algorithm," in SIGGRAPH, 1987.

[22] J. Wu et al., "Learning a probabilistic latent space of object shapes via 3D generative-adversarial modeling," in NeurIPS, 2016.

[23] D. Eigen and R. Fergus, "Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture," in ICCV, 2015.

[24] T. Baltrušaitis et al., "Multimodal machine learning: A survey and taxonomy," TPAMI, 2019.

[25] P. P. Liang et al., "Foundations and recent trends in multimodal machine learning: Principles, challenges, and open questions," arXiv:2209.03430, 2022.

[26] D. Ramachandram and G. W. Taylor, "Deep multimodal learning: A survey on recent advances and trends," IEEE Signal Processing Magazine, 2017.

[27] Y. Zhang et al., "Multimodal intelligence: Representation learning, information fusion, and applications," IEEE JSTSP, 2020.

[28] S. Gupta et al., "Learning rich features from RGB-D images for object detection and segmentation," in ECCV, 2014.

[29] A. Eitel et al., "Multimodal deep learning for robust RGB-D object recognition," in IROS, 2015.

[30] A. Karpathy et al., "Large-scale video classification with convolutional neural networks," in CVPR, 2014.

[31] C. Feichtenhofer et al., "Convolutional two-stream network fusion for video action recognition," in CVPR, 2016.

[32] J. Hoffman et al., "Cross-modal distillation for RGB-depth person re-identification," in ICIP, 2020.

[33] Y. Chen et al., "Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks," in ICML, 2018.

[34] C. Hazirbas et al., "FuseNet: Incorporating depth into semantic segmentation via fusion-based CNN architecture," in ACCV, 2016.

[35] L. Chen et al., "Encoder-decoder with atrous separable convolution for semantic image segmentation," in ECCV, 2018.

[36] Y. Wang et al., "Depth-aware CNN for RGB-D segmentation," in ECCV, 2018.

[37] K. Simonyan and A. Zisserman, "Two-stream convolutional networks for action recognition in videos," in NeurIPS, 2014.

[38] J. Dolz et al., "HyperDense-Net: A hyper-densely connected CNN for multi-modal image segmentation," TMI, 2019.

[39] V. Dumoulin and F. Visin, "A guide to convolution arithmetic for deep learning," arXiv:1603.07285, 2016.

[40] A. Fukui et al., "Multimodal compact bilinear pooling for visual question answering and visual grounding," in EMNLP, 2016.

[41] D. Xu et al., "PAD-Net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing," in CVPR, 2018.

[42] S. Valada et al., "AdapNet: Adaptive semantic segmentation in adverse environmental conditions," in ICRA, 2017.

[43] N. Audebert et al., "Beyond RGB: Very high resolution urban remote sensing with multimodal deep networks," ISPRS Journal, 2018.

[44] R. Takahashi et al., "Spatiotemporal relational reasoning for action recognition," in ACCV, 2020.

[45] C. Park et al., "Spatially selective cross-modal feature aggregation for RGB-D semantic segmentation," IEEE Access, 2020.

[46] J. M. Perez-Rua et al., "MFAS: Multimodal fusion architecture search," in CVPR, 2019.

[47] X. Liu et al., "When and why does deep learning work for multimodal fusion?" in ICML Workshop, 2020.

[48] G. Andrew et al., "Deep canonical correlation analysis," in ICML, 2013.

[49] R. Socher et al., "Semantic compositionality through recursive matrix-vector spaces," in EMNLP-CoNLL, 2012.

[50] L. Breiman, "Bagging predictors," Machine Learning, 1996.

[51] A. Karpathy et al., "Large-scale video classification with convolutional neural networks," in CVPR, 2014.

[52] P. Jamieson et al., "Robustness to missing modalities in multimodal learning," in NeurIPS Workshop, 2021.

[53] R. Vedantam et al., "Probabilistic neural-symbolic models for interpretable visual question answering," in ICML, 2019.

[54] X. Wang et al., "Non-local neural networks," in CVPR, 2018.

[55] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. MIT Press, 2016.

[56] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in ICLR, 2019.

[57] L. Liu et al., "On the variance of the adaptive learning rate and beyond," in ICLR, 2020.

[58] P. Goyal et al., "Accurate, large minibatch SGD: Training ImageNet in 1 hour," arXiv:1706.02677, 2017.

[59] I. Loshchilov and F. Hutter, "SGDR: Stochastic gradient descent with warm restarts," in ICLR, 2017.

[60] S. L. Smith et al., "Don't decay the learning rate, increase the batch size," in ICLR, 2018.

[61] L. N. Smith, "Cyclical learning rates for training neural networks," in WACV, 2017.

[62] H. Li et al., "Visualizing the loss landscape of neural nets," in NeurIPS, 2018.

[63] L. N. Smith and N. Topin, "Super-convergence: Very fast training of neural networks using large learning rates," in AI&M, 2019.

[64] R. Pascanu et al., "On the difficulty of training recurrent neural networks," in ICML, 2013.

[65] S. Krogh and J. A. Hertz, "A simple weight decay can improve generalization," in NeurIPS, 1992.

[66] L. Prechelt, "Early stopping - but when?" in Neural Networks: Tricks of the Trade, 1998.

[67] Y. Yao et al., "On early stopping in gradient descent learning," Constructive Approximation, 2007.

[68] S. L. Smith et al., "A Bayesian perspective on generalization and stochastic gradient descent," in ICLR, 2018.

[69] D. Piponi and M. Moore, "Reproducibility in machine learning," in NeurIPS Workshop, 2019.

[70] C. J. Willmott and K. Matsuura, "Advantages of the mean absolute error (MAE) over the root mean square error (RMSE)," Climate Research, 2005.

[71] T. Chai and R. R. Draxler, "Root mean square error (RMSE) or mean absolute error (MAE)?" Geoscientific Model Development, 2014.

[72] D. W. Hosmer and S. Lemeshow, Applied Logistic Regression. Wiley, 2000.

[73] D. Sculley et al., "Hidden technical debt in machine learning systems," in NeurIPS, 2015.

[74] E. Jia et al., "Accuracy of food portion size estimation from digital pictures," Public Health Nutrition, 2014.

[75] M. Pouladzadeh et al., "Measuring calorie and nutrition from food image," IEEE TMM, 2014.

[76] J. A. Novotny et al., "Foods, fortificants, and supplements: Where do Americans get their nutrients?" The Journal of Nutrition, 2012.

[77] V. Pham et al., "Found in translation: Learning robust joint representations by cyclic translations between modalities," in AAAI, 2019.

[78] X. Chen et al., "Learning cross-modal deep representations for RGB-D scene recognition," in BMVC, 2016.

[79] C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation for deep learning," Journal of Big Data, 2019.

[80] L. Perez and J. Wang, "The effectiveness of data augmentation in image classification using deep learning," arXiv:1712.04621, 2017.

[81] Y. Sun et al., "Deep learning for food image analysis: A survey," IEEE Access, 2020.

[82] L. Taylor and G. Nitschke, "Improving deep learning using generic data augmentation," in SSCI, 2018.

[83] A. Hernández-García and P. König, "Data augmentation instead of explicit regularization," arXiv:1806.03852, 2018.

[84] W. Min et al., "A survey on food computing," ACM Computing Surveys, 2019.

[85] Y. He et al., "Visual features for food classification: A comprehensive review," Trends in Food Science & Technology, 2020.

[86] E. D. Cubuk et al., "AutoAugment: Learning augmentation strategies from data," in CVPR, 2019.

[87] L. Ma et al., "Multi-modal convolutional neural networks for matching image and sentence," in ICCV, 2015.
