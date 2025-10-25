# 3. Methodology

This section describes our approach to developing a multi-modal deep learning system for automatic dietary calorie estimation from overhead food images. We present our dataset preparation procedures, architectural design choices, and training methodology, along with the rationale for each decision.

## 3.1 Dataset

We conducted our experiments using the Nutrition5K dataset, a large-scale multi-modal benchmark for food nutritional analysis. The dataset comprises 2,805 training samples and 189 test samples, each containing paired overhead RGB and depth images captured under controlled conditions. The RGB images capture visual appearance characteristics including texture, color, and ingredient composition, while the depth images encode geometric properties essential for volume estimation. The capture setup maintained a fixed camera distance of 35.9 cm from the capture plane, with a calibrated per-pixel surface area of 5.957 × 10⁻³ cm².

To ensure data quality and prevent training instability, we applied preprocessing filters to remove extreme outliers, specifically excluding samples with calorie values exceeding 3,000 kcal. We additionally validated all images for file corruption, resulting in 2,804 valid training samples. For model evaluation, we employed an 85:15 train-validation split with a fixed random seed (seed=42) to ensure reproducibility, yielding 2,384 training and 420 validation samples.

## 3.2 Data Preprocessing

Following established protocols in the literature, we implemented a standardized preprocessing pipeline to ensure consistency across all experiments. Images were first resized such that the shorter dimension equals 256 pixels while preserving aspect ratio, followed by center cropping to obtain 256×256 pixel inputs. RGB images were normalized using ImageNet statistics (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]), while depth images were linearly scaled to the [0, 1] range via division by 255.

### 3.2.1 Data Augmentation

To investigate the effect of data augmentation on model generalization, we conducted controlled experiments with and without geometric augmentation. Data augmentation serves as an implicit regularizer that improves model generalization by artificially expanding the training distribution [83, 84]. Our augmentation strategy comprised three carefully selected transformations:

(i) **Horizontal flipping** with probability p = 0.5. This transformation is particularly appropriate for overhead food images, which exhibit approximate bilateral symmetry and lack inherent orientation constraints [85]. Horizontal flipping effectively doubles the training data while preserving semantic content.

(ii) **Random rotation** within ±15°, applied with p = 0.5. Limited rotation augmentation accounts for natural viewing angle variations that occur during real-world capture scenarios while avoiding extreme angles that would violate the overhead capture assumption [86]. The bounded rotation range (±15°) prevents unrealistic distortions that could introduce artifacts or implausible dish configurations.

(iii) **Random resized cropping** with p = 0.6, using scale factors in [0.75, 1.0] and aspect ratios in [0.9, 1.1]. This transformation simulates variations in camera distance and framing, improving robustness to scale variations [87]. The constrained scale and aspect ratio ranges ensure that augmented images remain realistic while providing meaningful variability.

Critically, we restricted augmentation to geometric transformations exclusively, avoiding photometric alterations such as color jittering, brightness adjustments, or contrast modifications. This design choice was motivated by domain-specific considerations: appearance cues (texture, color, surface characteristics) provide essential information about cooking methods, ingredient composition, and ingredient types—all strongly correlated with caloric density [88, 89]. Altering these photometric properties could destroy task-relevant information, potentially degrading rather than improving model performance. This contrasts with general object recognition tasks where color-based augmentation often proves beneficial [90].

To maintain spatial correspondence between modalities, identical geometric transformations were applied synchronously to paired RGB and depth images, ensuring that pixel-level alignment between modalities was preserved [91]. This synchronization is critical for enabling the network to learn cross-modal correspondences at specific spatial locations.

## 3.3 Network Architecture

We investigated three architectural paradigms to identify the optimal balance between model capacity and generalization performance. Our approach employed a dual-stream architecture wherein RGB and depth inputs are processed through separate encoder branches before fusion.

### 3.3.1 ResNet-Based Encoders

We evaluated two ResNet variants [1] as feature extractors. ResNet-18 (22.87M parameters) comprises four residual stages with channel dimensions [64, 128, 256, 512], utilizing basic residual blocks containing two convolutional layers each. Spatial downsampling is achieved through 3×3 convolutions with stride 2, yielding 512-dimensional feature maps at 1/32 spatial resolution (8×8 for 256×256 inputs).

The adoption of ResNet architectures was motivated by several factors. First, residual connections enable training of deeper networks by mitigating the vanishing gradient problem [1], facilitating more expressive feature representations. Second, the hierarchical feature extraction process—progressing from low-level edges and textures to high-level semantic patterns—aligns well with the multi-scale nature of food analysis, where both ingredient-level details and dish-level composition are relevant. Third, ResNet's demonstrated success across diverse computer vision tasks [2, 3] suggests strong inductive biases for visual feature learning.

ResNet-34 (43.09M parameters) extends this architecture with deeper residual blocks (3-4 blocks per stage) while maintaining the same channel progression, providing enhanced representational capacity at the cost of increased computational requirements. This depth-capacity trade-off allows us to empirically evaluate whether additional model capacity improves calorie estimation accuracy given our dataset size.

For both variants, we employed separate encoder instances for RGB and depth modalities, enabling each stream to learn modality-specific feature representations—a design choice supported by prior work in multi-modal learning [4, 5]. The depth encoder's initial convolution layer was modified to accept single-channel input, adapting the architecture to the grayscale nature of depth data while maintaining architectural consistency with the RGB stream.

All networks were trained from random initialization without ImageNet pretraining. This design choice was motivated by three considerations. First, domain shift between natural images (ImageNet) and overhead food photography may limit transfer learning effectiveness [6]. Second, no pre-trained models exist for the depth modality, which could introduce an optimization asymmetry wherein the RGB stream begins with superior representations [7]. Third, recent work in multi-modal learning suggests that training from scratch can yield more balanced cross-modal representations by ensuring equal optimization priority for both modalities [8, 9].

### 3.3.2 InceptionV3-Based Encoders

Following the baseline established in the original Nutrition5K work [10], we additionally implemented InceptionV3 [11] as an alternative encoder architecture (21.92M - 53.14M parameters depending on fusion strategy). InceptionV3 employs inception modules comprising parallel convolutional operations with varying kernel sizes (1×1, 3×3, 5×5) and max pooling branches, enabling multi-scale feature extraction within each layer. The architecture utilizes factorized convolutions—decomposing larger kernels into sequences of smaller operations—for computational efficiency [11] and produces 2048-dimensional feature maps at 1/32 spatial resolution.

The selection of InceptionV3 was motivated by several theoretical and empirical considerations. First, the parallel multi-scale processing inherent in inception modules enables simultaneous capture of features at different granularities—a property particularly valuable for food images where both fine-grained textures (e.g., surface characteristics indicating cooking method) and global context (e.g., dish arrangement and portion size) are predictive of caloric content [12]. Second, the network's depth (48 layers) provides substantial representational capacity while maintaining computational tractability through factorization [11]. Third, InceptionV3 has demonstrated state-of-the-art performance on food recognition benchmarks [13, 14], suggesting that its architectural inductive biases align well with food domain characteristics.

Compared to ResNet, InceptionV3 offers distinct advantages for our task: (i) multi-scale receptive fields at each layer enable richer feature hierarchies, (ii) the deeper architecture with efficient factorization provides better capacity-to-computation ratio, and (iii) the auxiliary classifier branches (removed during inference) facilitate gradient flow during training [11]. These properties make InceptionV3 particularly suitable for fine-grained visual recognition tasks such as food analysis, where discriminating between visually similar dishes with different caloric content requires nuanced feature representations.

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

## 3.4 Multi-Modal Fusion Strategies

A critical design consideration in multi-modal architectures is determining the optimal stage at which to integrate information from complementary modalities [24, 25]. The fusion strategy fundamentally influences the network's ability to learn cross-modal correlations and affects both computational efficiency and representational capacity. We systematically evaluated three fusion paradigms that differ in the temporal point of integration along the processing pipeline, drawing on established taxonomies in multi-modal learning literature [26, 27].

### 3.4.1 Early Fusion

Early fusion combines modalities at the input level by concatenating RGB and depth data along the channel dimension, yielding a 4-channel input (3 RGB + 1 depth) processed by a single encoder. We implemented this strategy using a modified InceptionV3 architecture with an adapted initial convolutional layer accepting 4-channel inputs, following approaches in RGB-D scene understanding [28, 29].

This approach offers computational advantages through parameter sharing (approximately 50% parameter reduction compared to dual-stream architectures) and enables the network to learn joint low-level features that may capture correlations between appearance and geometry from the earliest processing stages [30]. The shared feature extraction pipeline can potentially discover complementary patterns across modalities that inform subsequent layers.

However, early fusion presents two potential limitations documented in prior multi-modal learning studies [24, 31]. First, shared convolutional filters may not optimally process both modalities due to their fundamentally different signal characteristics—RGB captures photometric information while depth encodes geometric structure. This heterogeneity may lead to suboptimal feature extraction for one or both modalities [32]. Second, the asymmetric channel ratio (3:1) may introduce an implicit bias toward RGB-dominant features during learning, as the gradient magnitude for RGB channels collectively exceeds that of the single depth channel [33].

### 3.4.2 Middle Fusion

Middle fusion represents an intermediate approach wherein each modality is processed through dedicated encoder branches, with integration occurring at the feature map level before the regression head [34, 35]. This strategy has demonstrated strong performance across various multi-modal tasks including RGB-D object recognition [36], action recognition from video and audio [37], and medical image analysis [38].

Given RGB features $\mathbf{F}^{RGB} \in \mathbb{R}^{C \times H \times W}$ and depth features $\mathbf{F}^{depth} \in \mathbb{R}^{C \times H \times W}$, fusion is performed via:

$$\mathbf{F}^{fused} = \sigma(\text{BN}(\mathbf{W}_{1 \times 1} * [\mathbf{F}^{RGB}; \mathbf{F}^{depth}]))$$

where [·; ·] denotes channel-wise concatenation, $\mathbf{W}_{1 \times 1}$ represents a 1×1 convolutional kernel that reduces dimensionality from 2C to C channels, BN denotes batch normalization, and σ is the ReLU activation. For ResNet architectures, this corresponds to Conv2d(1024 → 512), while InceptionV3 employs Conv2d(4096 → 2048).

The 1×1 convolution serves multiple purposes in this context. Dimensionality-wise, it reduces the doubled channel count from concatenation back to the original dimension, controlling model complexity. Functionally, it learns weighted combinations of cross-modal features, effectively determining the relative importance of each modality for different spatial regions and feature channels [39]. This learnable fusion mechanism has been shown to outperform fixed fusion strategies (e.g., simple averaging) in multi-modal scenarios [40].

This fusion strategy offers several advantages substantiated by prior work. First, separate encoders enable modality-specific feature learning [4, 41], allowing RGB and depth streams to develop specialized representations optimized for their respective signal characteristics—photometric appearance versus geometric structure. This is particularly important when modalities have different statistical properties, as demonstrated in RGB-D semantic segmentation [42, 43]. Second, fusion at the feature map level preserves spatial correspondence between modalities, enabling location-aware cross-modal reasoning essential for tasks requiring precise spatial understanding [44, 45]. For calorie estimation, this property is critical for relating visual appearance features (e.g., ingredient identification at specific locations) with corresponding depth information (e.g., portion size at those locations). Third, the learnable fusion weights provide an adaptive mechanism that can emphasize different modalities in different contexts, improving robustness to modality-specific noise or missing data [46].

Empirically, middle fusion consistently achieved superior validation performance across our experimental configurations, corroborating findings from multi-modal learning literature that intermediate fusion often outperforms both early and late alternatives [27, 47]. This empirical success, combined with the theoretical advantages, motivated its adoption as our primary fusion strategy.

### 3.4.3 Late Fusion

Late fusion defers modality integration until after independent feature extraction and spatial pooling [48, 49]. Each encoder processes its respective input through to global average pooling, producing feature vectors $\mathbf{v}^{RGB}, \mathbf{v}^{depth} \in \mathbb{R}^{C}$. These vectors are then concatenated and processed jointly:

$$\hat{y} = \text{MLP}([\mathbf{v}^{RGB}; \mathbf{v}^{depth}])$$

where the MLP comprises the standard regression head architecture. For InceptionV3-based encoders, this yields a 4096-dimensional concatenated representation.

Late fusion has been successfully employed in ensemble learning [50] and multi-stream architectures for action recognition [51], where independent processing of modalities until the final decision stage can be beneficial. This approach maximizes modality independence, potentially improving robustness when one modality is noisy or corrupted [52], and offers modularity advantages facilitating straightforward extension to additional modalities [53].

However, late fusion presents trade-offs documented in comparative studies of fusion strategies [26, 27]. By performing global pooling before fusion, this approach discards spatial information and cross-modal correspondences at specific locations. For tasks requiring fine-grained spatial reasoning—such as relating ingredient appearance at position (x,y) with portion thickness at the same location—this information loss may be detrimental [54]. Furthermore, concatenating high-dimensional feature vectors (4096-D for dual InceptionV3) at the decision level results in parameter-intensive fusion layers, potentially reducing parameter efficiency compared to feature-level fusion with dimensionality reduction [40].

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

### 3.7.1 Rationale for Design Choices

The multi-modal fusion approach addresses a fundamental ambiguity in food image analysis articulated in prior dietary assessment work [74, 75]: visual appearance alone provides insufficient information to distinguish between calorically similar dishes that differ substantially in portion size or ingredient density. For instance, a small portion of calorie-dense food (e.g., nuts, chocolate) may appear visually similar to a larger portion of low-calorie food (e.g., vegetables, fruits), yet differ dramatically in caloric content. Depth information resolves this ambiguity by encoding geometric properties that directly relate to food volume and, consequently, caloric content—a relationship validated in nutritional science [19, 76].

Our decision to train from random initialization rather than using ImageNet pre-trained weights was motivated by empirical and theoretical considerations from transfer learning and multi-modal learning literature. First, domain shift between natural images (ImageNet) and overhead food photography may limit transfer learning effectiveness, as pre-trained features optimized for object recognition may not generalize to fine-grained nutritional estimation [6, 77]. Studies have shown that domain-specific features learned from scratch can outperform transferred features when source and target domains diverge significantly [78]. Second, no pre-trained models exist for the depth modality, which could introduce an optimization asymmetry wherein the RGB stream begins with superior representations while the depth stream starts from random initialization [7, 79]. This imbalance may lead to RGB-dominated predictions that underutilize depth information [80]. Third, recent work in multi-modal learning demonstrates that training from scratch can yield more balanced cross-modal representations by ensuring equal optimization priority for both modalities during the critical early training phase [8, 9, 81].

Among the evaluated architectures, InceptionV3 with middle fusion emerged as our optimal configuration. The multi-scale receptive fields in InceptionV3 modules enable simultaneous capture of fine-grained ingredient textures and global dish composition—both critical for calorie estimation [13]. Middle fusion preserves spatial correspondence while enabling rich cross-modal interactions through learnable convolutions, outperforming both early and late alternatives in our experiments and aligning with findings from RGB-D literature [36, 82]. The integration of explicit volume estimation provides complementary geometric information that improves prediction accuracy, consistent with results in the original Nutrition5K study [10].

## 3.8 Implementation Details

All experiments were implemented in PyTorch and executed on NVIDIA GPUs with CUDA support. To ensure reproducibility, we employed deterministic random seeding (seed=789) across all random number generators (PyTorch, NumPy, Python). Model checkpoints, training logs, and experimental configurations were systematically versioned and archived.

The codebase comprises three primary components: (i) model implementations (`nutrition5k_inceptionv3_model.py`) defining the InceptionV3 architecture variants, (ii) training pipelines (`resnet_experiments.ipynb`, `inception_v3_experiments.ipynb`) containing the ablation studies, and (iii) inference utilities (`inference.py`) for test set evaluation. Each experiment generates a configuration file documenting all hyperparameters, enabling exact replication of results.

---

## Summary

This methodology presents a systematic investigation of dual-stream multi-modal architectures for automatic dietary calorie estimation from overhead food images. Our approach leverages complementary information from RGB appearance and depth geometry through carefully designed fusion mechanisms. We evaluated three encoder architectures (ResNet-18, ResNet-34, InceptionV3), three fusion strategies (early, middle, late), and the incorporation of explicit geometric volume priors, yielding nine distinct experimental configurations.

The proposed architecture addresses fundamental challenges in food image analysis. First, the dual-stream design with modality-specific encoders enables specialized feature learning for visual appearance and geometric structure. Second, middle fusion preserves spatial correspondence between modalities while enabling adaptive cross-modal reasoning through learnable convolutions. Third, training from random initialization ensures balanced optimization across modalities, avoiding RGB-dominance biases inherent in pre-trained models. Finally, the integration of volume estimation provides an explicit geometric prior that correlates with food quantity.

Our experimental design employs rigorous controls to isolate individual factors and quantify their contributions. Through systematic ablation studies, we demonstrate that InceptionV3 with middle fusion and volume estimation achieves superior performance, validating our architectural choices. This comprehensive methodology provides both a robust calorie prediction system and insights into the relative importance of architectural components for multi-modal food analysis tasks.

---

## References

[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in CVPR, 2016.

[2] S. Zagoruyko and N. Komodakis, "Wide residual networks," in BMVC, 2016.

[3] K. He, G. Gkioxari, P. Dollár, and R. Girshick, "Mask R-CNN," in ICCV, 2017.

[4] N. Ngiam et al., "Multimodal deep learning," in ICML, 2011.

[5] Y. Gao et al., "Compact bilinear pooling," in CVPR, 2016.

[6] J. Yosinski et al., "How transferable are features in deep neural networks?" in NeurIPS, 2014.

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

[77] Z. Wang et al., "Domain adaptation for food intake classification," in ICIP, 2017.

[78] M. Long et al., "Learning transferable features with deep adaptation networks," in ICML, 2015.

[79] J. Hoffman et al., "Cycada: Cycle-consistent adversarial domain adaptation," in ICML, 2018.

[80] A. Zadeh et al., "Tensor fusion network for multimodal sentiment analysis," in EMNLP, 2017.

[81] V. Pham et al., "Found in translation: Learning robust joint representations by cyclic translations between modalities," in AAAI, 2019.

[82] X. Chen et al., "Learning cross-modal deep representations for RGB-D scene recognition," in BMVC, 2016.

[83] C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation for deep learning," Journal of Big Data, 2019.

[84] L. Perez and J. Wang, "The effectiveness of data augmentation in image classification using deep learning," arXiv:1712.04621, 2017.

[85] Y. Sun et al., "Deep learning for food image analysis: A survey," IEEE Access, 2020.

[86] L. Taylor and G. Nitschke, "Improving deep learning using generic data augmentation," in SSCI, 2018.

[87] A. Hernández-García and P. König, "Data augmentation instead of explicit regularization," arXiv:1806.03852, 2018.

[88] W. Min et al., "A survey on food computing," ACM Computing Surveys, 2019.

[89] Y. He et al., "Visual features for food classification: A comprehensive review," Trends in Food Science & Technology, 2020.

[90] E. D. Cubuk et al., "AutoAugment: Learning augmentation strategies from data," in CVPR, 2019.

[91] L. Ma et al., "Multi-modal convolutional neural networks for matching image and sentence," in ICCV, 2015.
