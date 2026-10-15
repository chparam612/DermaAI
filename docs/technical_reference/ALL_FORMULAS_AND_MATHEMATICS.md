# Complete Mathematical Formulations & Equation Reference

**Author**: Senior Machine Learning Engineering & Research Science Reference  
**Project**: SkinCancerTracker / DermaAI  
**Target Audience**: B.Tech CSE / AI Researchers, Faculty Evaluators, Technical Interviewers  
**Specification Version**: Phase 1 & Phase 2 Technical Baseline  

---

## Structure and Taxonomy of Mathematical Concepts

To ensure scientific integrity and prevent conflating implemented code with theoretical concepts, this document strictly divides all mathematics into three mutually exclusive categories:
- **CATEGORY 1**: Directly implemented in our Python source code.
- **CATEGORY 2**: Implemented internally by underlying deep learning libraries (`torch`, `timm`, `sklearn`), but central to understanding model mechanics.
- **CATEGORY 3**: Theoretical formulations referenced in research literature or foundation model backbones, but not directly executed in our active pipeline.

---

# CATEGORY 1: Directly Implemented in Our Code

---

### Formula 1.1: Label-Smoothing Cross-Entropy Loss

#### 1. Mathematical Notation
$$\mathcal{L}_{\text{LS}}(z, y) = (1 - \epsilon) \cdot \left[ -\log \frac{\exp(z_y)}{\sum_{j=1}^K \exp(z_j)} \right] + \epsilon \cdot \left[ -\frac{1}{K} \sum_{k=1}^K \log \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)} \right]$$

Equivalently written in terms of negative log-likelihood ($L_{NLL}$) and uniform smooth loss ($L_{smooth}$):
$$\mathcal{L}_{\text{LS}} = (1 - \epsilon) L_{NLL} + \epsilon L_{smooth}$$
where:
$$L_{NLL} = -\log p_y, \quad L_{smooth} = -\frac{1}{K} \sum_{k=1}^K \log p_k, \quad p_k = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}$$

#### 2. Exact Implementation Location
- **File**: [`src/losses/classification.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/losses/classification.py#L11-L32)
- **Class**: `LabelSmoothingCrossEntropy(nn.Module)`
- **Lines 27–31**:
  ```python
  logprobs = F.log_softmax(x, dim=-1)
  nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
  smooth_loss = -logprobs.mean(dim=-1)
  loss = self.confidence * nll_loss + self.smoothing * smooth_loss
  return loss.mean()
  ```

#### 3. Explanation of Symbols and Variables
- $z \in \mathbb{R}^K$: Vector of raw, unnormalized model output logits across $K=11$ classes.
- $y \in \{0, 1, \dots, K-1\}$: Ground-truth integer class index.
- $\epsilon \in [0, 1)$: Label smoothing hyperparameter ($\epsilon = 0.10$ in `configs/experiments/exp2_panderm.yaml`).
- $1 - \epsilon$: Confidence assigned to the ground-truth target ($0.90$).
- $p_k$: Softmax probability assigned by the model to class $k$.
- $K$: Total number of diagnostic classes ($K=11$).

#### 4. What the Formula Calculates
Instead of forcing the model to fit a degenerate Dirac delta one-hot distribution ($q_y = 1, q_{k \ne y} = 0$), label smoothing creates a soft target distribution:
$$q_k = \begin{cases} 1 - \epsilon + \frac{\epsilon}{K}, & \text{if } k = y \\ \frac{\epsilon}{K}, & \text{if } k \ne y \end{cases}$$
The formula computes the Kullback-Leibler divergence / cross-entropy between this smoothed target distribution $q$ and the predicted distribution $p$.

#### 5. Why It Is Used in SkinCancerTracker
Dermatology lesion boundaries are inherently subjective. A lesion diagnosed as Melanoma (`MEL`) or Dysplastic Nevus (`NV`) frequently has diagnostic disagreement among board-certified dermatologists. Standard cross-entropy forces logits toward $\pm \infty$, resulting in overconfident misclassifications and poor clinical calibration. Label smoothing prevents logit explosion and improves generalization on difficult ambiguous lesions.

#### 6. Numerical Step-by-Step Example
Let $K = 3$ classes (0: BCC, 1: NV, 2: MEL). True class $y = 2$ (MEL). Smoothing $\epsilon = 0.1$.
Raw model logits: $z = [1.0, 2.0, 3.0]$.
1. Exponentials: $\exp(z) = [e^1, e^2, e^3] = [2.718, 7.389, 20.086]$.
2. Denominator sum: $\sum \exp(z_j) = 30.193$.
3. Probabilities: $p = [0.0900, 0.2447, 0.6653]$.
4. Log-probabilities $\log p = [-2.408, -1.408, -0.408]$.
5. Negative log-likelihood: $L_{NLL} = -(-0.408) = 0.408$.
6. Smooth loss: $L_{smooth} = -\frac{1}{3} (-2.408 - 1.408 - 0.408) = -\frac{1}{3}(-4.224) = 1.408$.
7. Final Loss:
   $$\mathcal{L} = (1 - 0.1)(0.408) + (0.1)(1.408) = (0.9)(0.408) + (0.1)(1.408) = 0.3672 + 0.1408 = 0.508$$
   *(Compare with unregularized CE: $\mathcal{L}_{CE} = 0.408$)*.

#### 7. Interview & Viva Explanation
> *"We implement label smoothing cross-entropy with $\epsilon=0.1$ to regularize the Vision Transformer during fine-tuning. Because dermatology datasets exhibit inter-observer diagnostic variability, pushing the neural network to output extreme probability values ($1.0$) harms calibration. Label smoothing caps target confidence at $1 - \epsilon + \frac{\epsilon}{K} = 0.909$, effectively penalizing overconfident predictions and improving out-of-distribution robustness."*

#### 8. Common Mistakes
- **Mistake**: Believing label smoothing changes the class prediction at test time.
- **Correction**: Smoothing only modifies the loss function and gradient during training; at inference, the argmax of logits $\arg\max_k z_k$ remains unchanged.

---

### Formula 1.2: Class-Frequency Balanced Sampler Probability

#### 1. Mathematical Notation
$$w_c = \frac{1}{N_c}, \quad \forall c \in \{0, 1, \dots, K-1\}$$
$$P(\text{Sample } i \text{ is selected}) = \frac{w_{y_i}}{\sum_{j=1}^N w_{y_j}}$$

#### 2. Exact Implementation Location
- **File**: [`src/data/samplers.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/data/samplers.py#L23-L33)
- **Function**: `create_balanced_class_sampler(labels: Sequence[int])`
- **Lines 23–32**:
  ```python
  labels_arr = np.array(labels)
  unique_classes, counts = np.unique(labels_arr, return_counts=True)
  class_weights = {cls: 1.0 / count for cls, count in zip(unique_classes, counts)}
  sample_weights = [class_weights[label] for label in labels_arr]
  sampler = WeightedRandomSampler(
      weights=torch.as_tensor(sample_weights, dtype=torch.double),
      num_samples=len(sample_weights),
      replacement=True,
  )
  ```

#### 3. Explanation of Symbols and Variables
- $N_c$: Number of training images belonging to class $c$ (support of class $c$).
- $w_c$: Sampling weight assigned to any individual image belonging to class $c$.
- $y_i$: Ground-truth integer class label of sample $i$.
- $N$: Total number of samples in the training set ($N = 7,336$).

#### 4. What the Formula Calculates
Calculates an inverse-frequency sampling probability distribution over the dataset. Under this distribution, the expected number of samples drawn from class $c$ in an epoch of size $N$ is equal across all classes:
$$\mathbb{E}[N_c^{\text{sampled}}] = N \cdot \frac{N_c \cdot \frac{1}{N_c}}{\sum_{k=1}^K N_k \cdot \frac{1}{N_k}} = N \cdot \frac{1}{K} = \frac{N}{K}$$

#### 5. Why It Is Used in SkinCancerTracker
Our training set exhibits a $280.2\times$ class imbalance ($2,522$ BCC lesions vs $9$ MAL_OTH lesions). Without balanced sampling, a random mini-batch of 32 images will almost never contain a rare malignant lesion (`MAL_OTH` or `BEN_OTH`). `WeightedRandomSampler` ensures every mini-batch contains a diverse mix of rare and common pathologies.

#### 6. Numerical Step-by-Step Example
Dataset with $N=10$ samples across 2 classes: Class 0 (9 samples), Class 1 (1 sample).
1. Class counts: $N_0 = 9, N_1 = 1$.
2. Weights: $w_0 = 1/9 \approx 0.111, w_1 = 1/1 = 1.0$.
3. Total sum of sample weights:
   $$\sum_{j=1}^{10} w_{y_j} = 9 \times \frac{1}{9} + 1 \times 1 = 1.0 + 1.0 = 2.0$$
4. Sampling probability for each Class 0 sample: $P(i \in C_0) = \frac{1/9}{2.0} = \frac{1}{18} \approx 0.0555$.
5. Sampling probability for the single Class 1 sample: $P(i \in C_1) = \frac{1.0}{2.0} = 0.5000$.
6. Collective probability of drawing Class 0: $9 \times \frac{1}{18} = 0.50$.
7. Collective probability of drawing Class 1: $1 \times 0.50 = 0.50$. Perfect $50/50$ balance achieved.

#### 7. Interview & Viva Explanation
> *"In highly imbalanced medical imaging, uniform random sampling starves minority gradient updates. We use inverse-frequency weighted sampling ($w_c = 1 / N_c$) with replacement via PyTorch's `WeightedRandomSampler`. This ensures that in expectation, each of the 11 diagnostic classes contributes equally to the loss gradient per epoch, without having to discard majority-class data."*

---

### Formula 1.3: Two-Stage Relative Group Partitioning Ratio

#### 1. Mathematical Notation
$$r_{\text{val\_rel}} = \frac{r_{\text{val}}}{r_{\text{train}} + r_{\text{val}}}$$

#### 2. Exact Implementation Location
- **File**: [`src/data/splitting.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/data/splitting.py#L115)
- **Function**: `create_lesion_level_splits()`
- **Line 115**:
  ```python
  val_rel_ratio = val_ratio / (train_ratio + val_ratio)
  ```

#### 3. Explanation of Symbols and Variables
- $r_{\text{train}}$: Target global proportion of lesions in train partition ($0.70$).
- $r_{\text{val}}$: Target global proportion of lesions in validation partition ($0.15$).
- $r_{\text{test}}$: Target global proportion of lesions in test partition ($0.15$).
- $r_{\text{val\_rel}}$: Conditional split proportion for the second stage ($0.17647$).

#### 4. What the Formula Calculates
When performing a 3-way split using a 2-way splitting operator (`GroupShuffleSplit`), Stage 1 extracts the test partition of size $r_{\text{test}} = 0.15$, leaving a remaining pool of proportion $1 - 0.15 = 0.85$. The formula computes the exact conditional fraction of this remaining pool that must be allocated to validation to yield exactly $15\%$ of the total dataset:
$$0.85 \times r_{\text{val\_rel}} = 0.85 \times \frac{0.15}{0.85} = 0.15$$

---

### Formula 1.4: Subgroup Demographic Performance Disparity

#### 1. Mathematical Notation
$$\Delta_{\text{metric}} = \max_{g \in \mathcal{G}^*} \text{Metric}(g) - \min_{g \in \mathcal{G}^*} \text{Metric}(g)$$
where:
$$\mathcal{G}^* = \{g \in \mathcal{G} \mid N_g \ge N_{\text{min}}\}, \quad N_{\text{min}} = 10$$

#### 2. Exact Implementation Location
- **File**: [`src/eval/metrics.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/eval/metrics.py#L136-L142)
- **Function**: `calculate_subgroup_metrics()`
- **Lines 136–142**:
  ```python
  if n_samples >= 10:
      subgroup_accs.append(sub_acc)
      subgroup_bal_accs.append(sub_bal_acc)

  acc_disparity = float(max(subgroup_accs) - min(subgroup_accs)) if len(subgroup_accs) > 1 else 0.0
  bal_acc_disparity = float(max(subgroup_bal_accs) - min(subgroup_bal_accs)) if len(subgroup_bal_accs) > 1 else 0.0
  ```

#### 3. Explanation of Symbols and Variables
- $\mathcal{G}$: The set of all skin tone demographic categories ($\{\text{Tone } 0, \text{Tone } 1, \dots, \text{Tone } 5\}$).
- $\mathcal{G}^*$: Filtered subset of skin tone groups satisfying minimum sample support ($N_g \ge 10$).
- $\text{Metric}(g)$: Performance metric (e.g., accuracy or balanced accuracy) computed exclusively on group $g$.
- $\Delta_{\text{metric}}$: Empirical disparity range across demographic subgroups.

#### 4. Clinical Significance
A low aggregate loss can mask severe diagnostic disparities. For example, if a model achieves $90\%$ accuracy on skin tone 3 but only $45\%$ accuracy on skin tone 5, $\Delta_{\text{Acc}} = 0.45$. In medical imaging, large disparity ranges represent severe demographic inequity and clinical safety hazards.

---

### Formula 1.5: Gradient $L_2$ Norm Threshold Clipping

#### 1. Mathematical Notation
$$\|g\|_2 = \sqrt{\sum_{p \in \Theta} \sum_j \left( \frac{\partial \mathcal{L}}{\partial \theta_{p, j}} \right)^2}$$
$$g \leftarrow \begin{cases} g, & \text{if } \|g\|_2 \le \tau \\ g \cdot \frac{\tau}{\|g\|_2}, & \text{if } \|g\|_2 > \tau \end{cases}$$

#### 2. Exact Implementation Location
- **File**: [`src/train/engine.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/train/engine.py#L61)
- **Function**: `train_one_epoch()`
- **Line 61**:
  ```python
  if clip_grad is not None and clip_grad > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
  ```
- **Configured Parameter**: `clip_grad: 1.0` in both `configs/experiments/exp1_baseline.yaml` and `configs/experiments/exp2_panderm.yaml`.

#### 3. What the Formula Calculates
Calculates the global Euclidean ($L_2$) norm across all parameter gradients concatenated into a single vector. If this norm exceeds threshold $\tau = 1.0$, the entire gradient vector is scaled down proportionally without altering its directional orientation.

#### 4. Why It Is Used in SkinCancerTracker
Vision Transformers (ViTs) are prone to gradient explosion during early fine-tuning epochs when large loss gradients from the randomly initialized linear classification head propagate back through deep self-attention layers. Norm clipping stabilizes backpropagation.

---

### Formula 1.6: Channel-Wise Z-Score Image Normalization

#### 1. Mathematical Notation
$$x_{\text{norm}, c}(i, j) = \frac{x_c(i, j) - \mu_c}{\sigma_c}, \quad \forall c \in \{R, G, B\}$$

#### 2. Exact Implementation Location
- **File**: [`src/data/transforms.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/data/transforms.py#L14-L20)
- **Values**:
  - ImageNet (Exp-1): $\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$
  - PanDerm (Exp-2): $\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.228, 0.224, 0.225]$

#### 3. What the Formula Calculates
Zero-centers pixel distributions around mean $0.0$ with unit variance $1.0$. This prevents saturation in activation functions and ensures uniform conditioning of the loss surface Hessian matrix.

---

### Formula 1.7: Softmax Probability Distribution & Decision Rule

#### 1. Mathematical Notation
$$\hat{p}_k = \sigma(z)_k = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}, \quad \hat{y} = \arg\max_{k \in \{0, \dots, K-1\}} \hat{p}_k$$

#### 2. Exact Implementation Location
- **File**: [`src/eval/predictions.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/eval/predictions.py#L59-L60)
- **Lines 59–60**:
  ```python
  probs = torch.softmax(logits, dim=-1).cpu().numpy()
  preds = np.argmax(probs, axis=1)
  ```

---

# CATEGORY 2: Implemented Internally by External Libraries

---

### Formula 2.1: AdamW (Decoupled Weight Decay) Optimization

#### 1. Mathematical Notation
$$\theta_t = \theta_{t-1} - \eta_t \lambda \theta_{t-1} - \frac{\eta_t}{\sqrt{\hat{v}_t} + \epsilon_{\text{opt}}} \hat{m}_t$$
where moments are updated as:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

#### 2. Where Invoked
- **Files**: [`src/train/train_experiment.py:255`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/train/train_experiment.py#L255), [`external/panderm/classification/furnace/optim_factory.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/external/panderm/classification/furnace/optim_factory.py)
- **Configured Values**:
  - Exp-1: $\eta = 10^{-4}, \lambda = 0.01, \beta_1 = 0.9, \beta_2 = 0.999, \epsilon_{\text{opt}} = 10^{-8}$
  - Exp-2: $\eta = 5 \times 10^{-4}, \lambda = 0.05, \beta_1 = 0.9, \beta_2 = 0.999, \epsilon_{\text{opt}} = 10^{-8}$

#### 3. Mathematical Distinction: Adam vs AdamW
In standard $L_2$ regularization with Adam, weight decay is added directly to the gradient ($g_t \leftarrow g_t + \lambda \theta_t$). This causes weights with large second moments to decay *less* than weights with small second moments. AdamW decouples weight decay by subtracting $\eta_t \lambda \theta_{t-1}$ directly from the parameter update, ensuring true exponential weight shrinkage across all parameters.

---

### Formula 2.2: Vision Transformer Scaled Dot-Product Multi-Head Self-Attention with Relative Position Bias

#### 1. Mathematical Notation
$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + B_{\text{rel}} \right) V$$
$$\text{MultiHead}(X) = \left[ \text{head}_1 \,\|\, \text{head}_2 \,\|\, \dots \,\|\, \text{head}_h \right] W^O$$
$$\text{where } \text{head}_i = \text{Attention}(X W_i^Q, X W_i^K, X W_i^V)$$

#### 2. Exact Implementation Location
- **File**: [`external/panderm/classification/models/modeling_finetune.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/external/panderm/classification/models/modeling_finetune.py#L116-L199)
- **Class**: `Attention(nn.Module)`
- **Lines 179–196**:
  ```python
  q = q * self.scale
  attn = (q @ k.transpose(-2, -1))
  if self.relative_position_bias_table is not None:
      attn = attn + relative_position_bias.unsqueeze(0)
  attn = attn.softmax(dim=-1)
  attn = self.attn_drop(attn)
  x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
  ```

#### 3. Explanation of Symbols
- $Q, K, V \in \mathbb{R}^{B \times h \times N \times d_k}$: Queries, Keys, Values projected from token embeddings ($N = 197$ tokens for $14 \times 14$ patches plus CLS token).
- $d_k = \frac{d}{h} = \frac{768}{12} = 64$: Dimension of each individual attention head.
- $\frac{1}{\sqrt{d_k}} = \frac{1}{\sqrt{64}} = 0.125$: Scaling factor preventing inner products from growing excessively large, which pushes softmax into vanishing gradient saturation regions.
- $B_{\text{rel}} \in \mathbb{R}^{N \times N}$: Learnable relative position bias matrix indexed by 2D spatial coordinate offsets $(\Delta x, \Delta y)$ between patch tokens.

---

### Formula 2.3: 2D Sinusoidal Position Embedding

#### 1. Mathematical Notation
For patch token at grid coordinate $(x, y)$ where $x \in \{0, \dots, W-1\}, y \in \{0, \dots, H-1\}$ and channel index $i \in \{0, \dots, d/4 - 1\}$:
$$\omega_i = \frac{1}{10000^{4i / d}}$$
$$\text{PE}(x, y) = \Big[ \sin(\omega_i x), \; \cos(\omega_i x), \; \sin(\omega_i y), \; \cos(\omega_i y) \Big] \in \mathbb{R}^d$$

#### 2. Exact Implementation Location
- **File**: [`external/panderm/classification/models/modeling_finetune.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/external/panderm/classification/models/modeling_finetune.py#L447-L463)
- **Function**: `build_2d_sincos_position_embedding()`
- **Lines 454–458**:
  ```python
  omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
  omega = 1. / (temperature ** omega)
  out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
  out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
  pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)
  ```

---

### Formula 2.4: LayerScale in Transformer Blocks

#### 1. Mathematical Notation
$$x_{l}' = x_{l-1} + \text{DropPath}\Big( \text{diag}(\gamma_{1, l}) \cdot \text{Attention}(\text{LN}(x_{l-1})) \Big)$$
$$x_l = x_{l}' + \text{DropPath}\Big( \text{diag}(\gamma_{2, l}) \cdot \text{MLP}(\text{LN}(x_{l}')) \Big)$$
where $\gamma_{1, l}, \gamma_{2, l} \in \mathbb{R}^d$ are learnable diagonal vectors initialized to:
$$\gamma_{1, l}^{(0)} = \gamma_{2, l}^{(0)} = 0.1 \quad (\text{for Base ViT}), \quad 10^{-5} \quad (\text{for Large ViT})$$

#### 2. Exact Implementation Location
- **File**: [`external/panderm/classification/models/modeling_finetune.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/external/panderm/classification/models/modeling_finetune.py#L280-L292)
- **Class**: `Block(nn.Module)`
- **Lines 289–291**:
  ```python
  x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
  x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
  ```

#### 3. Mathematical Purpose
In deep Vision Transformers (12 to 24 layers), identity residual paths accumulate unbounded variance. Initializing $\gamma$ to small values ($0.1$ or $10^{-5}$) scales down initial block contributions, allowing unimpeded early gradient flow and preventing exploding activation dynamics.

---

### Formula 2.5: Batch Normalization in ResNet-50

#### 1. Mathematical Notation
During training across mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$:
$$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_{\mathcal{B}})^2$$
$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon_{\text{BN}}}}, \quad y_i = \gamma \hat{x}_i + \beta$$
Running statistics tracked via exponential moving average:
$$\mu_{\text{run}} \leftarrow (1 - \rho) \mu_{\text{run}} + \rho \mu_{\mathcal{B}}, \quad \sigma_{\text{run}}^2 \leftarrow (1 - \rho) \sigma_{\text{run}}^2 + \rho \sigma_{\mathcal{B}}^2$$

#### 2. Where Used
- Used across all 49 convolutional layers of the ResNet-50 baseline in [`src/models/baseline.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/models/baseline.py).

---

### Formula 2.6: Layer-Wise Learning Rate Decay (LLRD)

#### 1. Mathematical Notation
For layer $l \in \{0, 1, \dots, L\}$ where $L$ is the total depth and decay rate $\alpha = 0.9$:
$$\eta_l = \eta_{\text{base}} \cdot \alpha^{L - l}$$

#### 2. Exact Implementation Location
- **File**: [`external/panderm/classification/furnace/optim_factory.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/external/panderm/classification/furnace/optim_factory.py#L38-L46)
- **Class**: `LayerDecayValueAssigner`
- **Equation**:
  $$\text{values} = \Big[ \alpha^{L + 1 - i} \;\text{for } i \in \{0, \dots, L+1\} \Big]$$

#### 3. Mathematical Rationale
Lower ViT layers extract generic spatial features (edges, color gradients, texture) that are already well-optimized during PanDerm pretraining. Upper layers extract high-level semantic pathology. Decaying the learning rate for earlier layers preserves foundational dermatological representations while allowing upper layers to specialize in 11-class diagnostic classification.

---

# CATEGORY 3: Theoretical Formulations for Background

---

### Formula 3.1: Focal Loss (Lin et al.)

#### 1. Mathematical Notation
$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
where:
$$p_t = \begin{cases} p, & \text{if } y = 1 \\ 1 - p, & \text{otherwise} \end{cases}$$

#### 2. Scientific Status in SkinCancerTracker
- **Status**: Theoretical candidate for future extreme imbalance experiments.
- **Comparison with Implemented Loss**: The active repository implements `LabelSmoothingCrossEntropy` and inverse class-frequency weighted sampling rather than Focal Loss.

---

### Formula 3.2: Sparse Autoencoder (SAE) Mechanistic Interpretability Loss

#### 1. Mathematical Notation
$$\mathcal{L}_{\text{SAE}}(x) = \|x - \hat{x}\|_2^2 + \lambda \|f(x)\|_1$$
where:
$$f(x) = \text{ReLU}\Big( W_{\text{enc}} (x - b_{\text{tied}}) + b_{\text{enc}} \Big)$$
$$\hat{x} = W_{\text{dec}} f(x) + b_{\text{tied}}$$

#### 2. Implementation Location in Repository
- **File**: [`external/dermfm-zero/automated-concept-discovery/sae/sparse_autoencoder/autoencoder/model.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/external/dermfm-zero/automated-concept-discovery/sae/sparse_autoencoder/autoencoder/model.py#L26-L100)
- **Purpose**: Used in the upstream DermFM-Zero submodule for automated clinical concept discovery by extracting monosemantic feature directions from dense transformer activations.
