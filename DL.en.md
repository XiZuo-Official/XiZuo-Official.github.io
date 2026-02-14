# 1. Tensor Creation
### 1. Basic Tensor Creation
    ```python
    import torch
    # torch.tensor: create a tensor from explicit data
    # torch.Tensor: create by shape
    # torch.IntTensor | torch.FloatTensor: create with a specified dtype
    ```
# 2. Tensor Operations
### 1. Common Operation Functions
- Square root
    ```python
    print(data.sqrt())
    ```
- Exponential
    ```python
    print(data.exp())
    ```
- Logarithm
    ```python
    print(data.log())
    print(data.log2())
    print(data.log10())
    ```

# 3. Activation Functions
### 1. Role of Activation Functions
- Add a nonlinear rule to the weighted-sum output so neural networks are no longer purely linear models.

### 2. Sigmoid

\[\sigma(x) = \frac{1}{1 + e^{-x}}\]

<div style="text-align:center;">            
<img src="pic/sigmoid.png" width="50%">
</div>

- Domain: \((-\infty, +\infty)\)
- Range: \((0, 1)\)
- Monotonically increasing
- Input values need to be roughly in \((-3, 3)\) for visible output differences; otherwise outputs are close to 0 or 1.

* **Sigmoid derivative:**
$$
\sigma'(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr)
$$ 
<div style="text-align:center;">  
<img src="pic/sigmoid_derivative.png" width="50%">
</div>


- When \(x = 0\), \(\sigma'(x)_{\max} = 0.25\)
- Range: \((0, 0.25)\)
- When \(|x|\) is large, the derivative approaches 0, so gradients are hard to backpropagate in deep networks.
- Therefore, Sigmoid is typically used only in the output layer of binary classification.
>  **Why Sigmoid is not suitable for hidden layers:**
>  **1. Vanishing gradients**
>   Its maximum derivative is only 0.25. By chain rule, gradients for early-layer weights are:
>   \[\frac{\partial \mathcal L}{\partial \omega_1} = \frac{\partial \mathcal L}{\partial a_2}
>      \cdot
>      \frac{\partial a_2}{\partial z_2}
>      \cdot
>      \frac{\partial z_2}{\partial a_1}
>      \cdot
>      \frac{\partial a_1}{\partial z_1}
>      \cdot
>      \frac{\partial z_1}{\partial \omega_1}\]
>     where \(z\) is linear transform (weighted sum), and \(a\) is activation.
>   \(\dfrac{\partial a}{\partial z}\) is Sigmoid derivative, capped at 0.25.
>      After many layers, \(\dfrac{\partial \mathcal L}{\partial \omega_1}\) tends to 0, causing vanishing gradients.<br>
>      **2. Wide saturation region**
>      If \(|x| > 6\), outputs become insensitive to input changes and stay near 0 or 1.
>      **Neuron saturation:** in this region Sigmoid derivative is near 0, so backprop cannot effectively update weights.<br>
>      **3. Non-zero-centered output**
>      Sigmoid outputs are in (0,1), not centered at 0. This can bias update directions and cause zig-zag optimization with slower convergence.

### 3. Tanh

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$
<div style="text-align:center;">  
<img src="pic/tanh.png" width="50%">
</div>
- Domain: \((-\infty, +\infty)\)
- Range: \((-1, 1)\)
- Monotonically increasing
- Most sensitive in \((-3, 3)\); outputs gradually saturate outside this range.

* **Tanh derivative:**
$$
\tanh'(x) = 1 - \tanh^2(x)
$$
<div style="text-align:center;"> 
<img src="pic/tanh_derivative.png" width="50%">
</div>
- When \(x = 0\), \(\tanh'(x)_{\max} = 1\)
- Range: \((0, 1]\)
- When \(|x|\) is large, derivative approaches 0, so vanishing gradients still occur. Usually better for shallow networks.
- Because Tanh is zero-centered, it often converges faster than Sigmoid as a hidden activation.

### 4. ReLU

$$
\mathrm{ReLU}(x) = \max(0, x)
$$
<div style="text-align:center;"> 
<img src="pic/relu.png" width="50%">
</div>
- Domain: \((-\infty, +\infty)\)
- Range: \([0, +\infty)\)
- Piecewise function; non-strictly increasing (constant 0 when \(x<0\))
- For \(x>0\), output is linear in input

* **ReLU derivative (gradient):**
$$
\mathrm{ReLU}'(x) =
\begin{cases}
0, & x < 0 \\
1, & x > 0
\end{cases}
$$

<div style="text-align:center;"> 
<img src="pic/relu_derivative.png" width="50%">
</div>

- For \(x > 0\), gradient is 1, helping avoid vanishing gradients
- For \(x < 0\), gradient is 0
- Not differentiable at \(x = 0\) (practical implementations typically choose 0 or 1)

> **Why ReLU is suitable for hidden layers:**<br>
>   **1. Simple and efficient computation**  
>   ReLU only needs comparison and selection, so it is computationally cheap.<br>
>   **2. Mitigates vanishing gradients**  
>   In the positive region, derivative is 1, so gradients are less likely to vanish through deep chains.<br>
>   **3. Sparse activation**  
>   When \(x < 0\), output is 0, making some neurons inactive and often improving representation efficiency.<br>
>   **4. Potential issue (Dying ReLU)**  
>   During training, many inputs may fall in \(x < 0\), making gradient always 0 and parameters stop updating.

>  **When Dying ReLU is likely to happen:**<br>
>    1. **Overly negative bias \(b\)**  
>     If 
>    \[b \ll 0 \Rightarrow z=w^\top x + b < 0\ \text{(for most }x\text{)} \] 
>    then 
>    \[\mathrm{ReLU}'(z)=0\], gradients are 0 and the neuron can stay inactive for a long time.<br>
>   2. **Learning rate \(\eta\) too large, causing one-step jump into negative region**  
>  Parameter update:
>  \[ \omega \leftarrow \omega-\eta\frac{\partial \mathcal L}{\partial \omega},\qquad b \leftarrow b-\eta\frac{\partial L}{\partial b}
>  \] 
>  If \(\eta\) is too large, \(\omega,b\) can jump to a state where
>  \[\omega^\top x + b < 0\] holds for most samples, and then \(\mathrm{ReLU}'(z)=0\) prevents recovery.<br>
>   3. **Input shift (overall negative input) or poor initialization**  
>   If at early training stage most samples satisfy
>   \[\omega^\top x + b < 0 \], the neuron is almost inactive from the start.

### 5. Softmax

$$
\mathrm{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$



- Input is a vector \(z = (z_1, z_2, \dots, z_K)\)
- Output is a probability distribution
- Each component is in \((0, 1)\)
- Sum of outputs equals 1:
\[
\sum_{i=1}^{K} \mathrm{Softmax}(z_i) = 1
\]

> **Why Softmax is usually used in the output layer:**<br>
>   **1. Clear probabilistic interpretation**  
>   Outputs are in \((0,1)\) and sum to 1, directly representing class probabilities.<br>
>   **2. Suitable for multiclass classification**  
>   Commonly paired with Cross Entropy loss.<br>
>   **3. Not suitable as hidden-layer activation**  
>   Outputs compete across classes and gradients are coupled, which is not ideal for intermediate feature learning.

# 4. Parameter Initialization
### 1. Why Initialization Matters
- **Prevent vanishing or exploding gradients**
    - If weights are too large, \( z = \omega x + b \) can become very large, pushing Sigmoid/Tanh into saturation and causing vanishing gradients.
- **Improve convergence speed**
    - A good starting point helps gradients propagate stably and speeds up learning.
- **Break symmetry**
    - If all parameters are initialized identically, neurons learn the same thing.

### 2. Initialization Methods
- **All-zero / all-one / constant initialization**
    ```python
    import torch
    import torch.nn as nn
    nn.init.zeros_()
    nn.init.ones_()
    nn.init.constant_()
    ```
- **Uniform initialization**   \( \omega \sim U(-a, a) \)
    ```python
    nn.init.uniform_()
    ```
- **Normal initialization** \( \omega \sim \mathcal{N}(0, \sigma^2) \)
    ```python
    nn.init.normal_()
    ```
- **He (Kaiming) normal initialization**
$$
\omega \sim \mathcal{N}\left(0,\ \frac{2}{fan_{in}}\right)
$$
    ```python
    nn.init.kaiming_normal_()
    ```
    - \(fan_{in}\): number of input connections per neuron.
    - Linear layer: \(fan_{in} =\) `in_features`
    - Conv2d layer: \(fan_{in} =\) in_channels × kernel_height × kernel_width

- **He (Kaiming) uniform initialization**
$$
\omega \sim \mathcal{U}\left(
-\sqrt{\frac{6}{fan_{in}}},
+\sqrt{\frac{6}{fan_{in}}}
\right)
$$
    ```python
    nn.init.kaiming_uniform_()
    ```

>  **Why Kaiming is specialized for ReLU:**
> - ReLU is asymmetric; roughly half the inputs are mapped to 0.
> - After ReLU, activation variance is roughly halved.
> - If this repeats layer by layer, signals/gradients decay.
> - Kaiming compensates by setting weight variance appropriately.

>  **Why variance matters:**
> - Variance controls activation spread (signal scale).
> - Too small: outputs become similar and information weakens.
> - Too large: unstable extremes.
> - If forward variance shrinks, backward gradients also tend to shrink.

> **How Kaiming ranges are derived:**
>   Target is variance preservation: \(\mathrm{Var}(a)\approx \mathrm{Var}(x)\)
>   \[
>       \mathrm{Var}(a) \approx \frac{1}{2}\mathrm{Var}(z)
>   = \frac{1}{2}\left({fan_{in}}\mathrm{Var}(\omega)\mathrm{Var}(x)\right)
>   \]
>   So set \(\mathrm{Var}(\omega)=\frac{2}{fan_{in}}\).

- **Xavier normal initialization**
$$
\omega \sim \mathcal{N}\left(
0,\ \frac{2}{fan_{in}+fan_{out}}
\right)
$$
    ```python
    nn.init.xavier_normal_()
    ```
    - \(fan_{in}\): number of input connections per neuron.
    - \(fan_{out}\): number of output connections per neuron.

- **Xavier uniform initialization**
$$
\omega \sim \mathcal{U}\left(
-\sqrt{\frac{6}{fan_{in}+fan_{out}}},
+\sqrt{\frac{6}{fan_{in}+fan_{out}}}
\right)
$$
    ```python
    nn.init.xavier_uniform_()
    ```

# 5. Loss Functions
### 1. Multiclass Cross-Entropy Loss
- Mean loss over all samples:
    $$
        \mathcal{L}
        =
        -\frac{1}{N}
        \sum_{n=1}^{N}
        \sum_{i=1}^{C}
        y_{n,i}\,\log(p_{n,i})
    $$
    - \(\mathcal{L}\): average multiclass loss
    - \(N\): number of samples, \(C\): number of classes
    - \(y_{n,i}\): one-hot label (usually 1 or 0)
    - \(p_{n,i}\): predicted probability for class \(i\)

- Simplified form (one true class per sample):
    $$
        \mathcal{L}
        =
        -\sum_{n=1}^{N}
        \log\big(p_i\big)
    $$
    - Only the probability of the true class contributes.
    - Intuitively, cross entropy measures how *unconfident* the model is about the true class.

- Single-sample loss shape:
    $$
        \mathcal{L}
        =
        -
        \log\big(p\big)
    $$
<div style="text-align:center;"> 
<img src="pic/Cross_Entropy.png" width="50%">
</div>

### 2. Binary Cross-Entropy Loss
\[
\mathcal{L} = - y \log(p) - (1-y)\log(1-p)
\]
- \(y \in \{0,1\}\): ground-truth label  
  \(p \in (0,1)\): predicted probability of positive class
<div style="text-align:center;"> 
<img src="pic/Binary_Entropy.png" width="50%">
</div>

### 3. MAE / L1 Loss
\[
\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
\]
- Uses absolute error magnitude.
<div style="text-align:center;"> 
<img src="pic/MAE_LOSS.png" width="50%">
</div>
- Pros: robust to outliers, intuitive meaning, stable values.
- Cons: gradient is non-smooth and magnitude is not error-sensitive, which can slow optimization.

### 4. MSE / L2 Loss
\[
\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}\bigl(y_i - \hat{y}_i\bigr)^2
\]
<div style="text-align:center;"> 
<img src="pic/MSE_LOSS.png" width="50%">
</div>
- Pros: smooth and differentiable; penalizes large errors strongly.
- Cons: sensitive to outliers; squared unit is less interpretable.

### 5. Huber Loss / Smooth L1
$$
L(y, y_i)=
\begin{cases}
\frac{1}{2}(y - y_i)^2, & \text{if } |y - y_i| \le \delta, \\
\delta |y - y_i| - \frac{1}{2}\delta^2, & \text{if } |y - y_i| > \delta .
\end{cases}
$$
<div style="text-align:center;">
<img src="pic/Huber_LOSS.png" width="50%">
</div>
- Small error region behaves like MSE.
- Large error region behaves like MAE.

# 6. Gradient-Based Optimization Algorithms
### 1. Basic Terms
- Epoch: one full pass through the training set.
- Batch size: number of samples used to compute one gradient update.
- Iteration: one parameter update step.

### 2. Limitations of Plain Gradient Descent
- Local minima in non-convex objectives
- Saddle-point stagnation
- Slow progress in flat regions
- Sensitivity to initialization

### 3. Exponential Moving Average (EMA)
- EMA is exponentially decayed averaging of historical values: newer values get larger weight.

\[
v_t = \beta v_{t-1} + (1-\beta)x_t
\]

### 4. Momentum
- Momentum stabilizes noisy mini-batch gradients via EMA.

\[\omega_t = \omega_{t-1} - \eta V_t\]

$$
V_t = \beta V_{t-1} + (1-\beta) \cdot \left.\frac{\partial \mathcal L_t}{\partial \omega} \right|_{\omega_{t-1}}
$$

### 5. AdaGrad
- AdaGrad uses per-parameter accumulated squared gradients:
\[G_{t,i} = \sum_{k=1}^{t} g_{k,i}^2\]
\[\omega_{t,i} = \omega_{t-1,i} - \frac{\eta}{ \sqrt{G_{t,i}} + \varepsilon} \, g_{t,i}\]
- Issue: learning rates monotonically decay and may become too small.

### 6. RMSProp
- RMSProp replaces full accumulation with EMA of squared gradients:
\[V_{t,i} = \beta V_{t-1,i} + (1-\beta)\, g_{t,i}^2\]
\[
\omega_{t,i} = \omega_{t-1,i} - \frac{\eta}{\sqrt{V_{t,i}} + \varepsilon}\,g_{t,i}
\]

### 7. Adam
- Adam combines Momentum (first moment) and RMSProp (second moment).
- First moment:
\[ m_{t,i} = \beta_1 m_{t-1,i}+(1-\beta_1)\, g_{t,i}\]
- Second moment:
\[
v_{t,i} = \beta_2 v_{t-1,i} + (1-\beta_2)\, g_{t,i}^2
\]
- Bias correction:
\[\hat m_{t,i} = \frac{m_{t,i}}{1-\beta_1^t},\qquad \hat v_{t,i}=\frac{v_{t,i}}{1-\beta_2^t}\]
- Update:
\[
\omega_{t,i} = \omega_{t-1,i} - \frac{\eta}{\sqrt{\hat v_{t,i}} + \varepsilon}\,\hat m_{t,i}
\]

# 7. Learning Rate Decay Methods
### 1. Step Decay
- Decays the learning rate by a constant factor every fixed number of steps/epochs.
- Pros: simple, practical.
- Cons: discontinuous jumps; schedule is manually tuned.

### 2. MultiStepLR
- Decays learning rate at specified milestones.

### 3. Exponential Decay
- Continuously decays as:
\[
\eta_t = \eta_0 \cdot \gamma^t
\]
- Smooth but can become too small too early.

### 4. Cosine Annealing
- Smoothly decays from \(\eta_{\max}\) to \(\eta_{\min}\):
\[
\eta_t = \eta_{\min} + \frac{1}{2} (\eta_{\max}-\eta_{\min}) \left(1 + \cos\left(\frac{\pi t}{T_{\max}}\right)\right)
\]

### 5. ReduceLROnPlateau (Metric-Based LR Decay)
- Instead of time-based schedule, monitors validation metrics (loss/accuracy).
- If no improvement is observed for a patience window, learning rate is reduced.

# 8. Regularization
### 1. Dropout
- Dropout is a regularization technique for neural networks to reduce overfitting and improve generalization.
- During training, it randomly drops a portion of neurons, preventing over-reliance on specific units and reducing co-adaptation.
- Training phase:
    - Randomly drop neurons
    - Each iteration behaves like a different subnetwork
    - Activations are expectation-scaled (inverted dropout)
- Inference phase:
    - Dropout is disabled; full network is used

### 2. Batch Normalization (BN)
- BN standardizes intermediate activations to stabilize distributions, accelerate training, and provide mild regularization.
- Typical flow:
\[
x \xrightarrow{\text{linear}} z \xrightarrow{\text{BN normalize}} \hat z \xrightarrow{\text{scale/shift}} y \xrightarrow{\text{activation}} h
\]
- Normalization:
\[
\hat z_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}
\]
- Affine transform:
\[
y_i = \gamma \hat z_i + \beta
\]
- BN is usually applied to pre-activation linear outputs rather than post-activation values.

# 9. CNN
### 1. Image Classification Basics
- Binary image: single channel, pixel values 0/1
- Grayscale image: single channel, pixel values in [0,255]
- RGB image: three channels (R, G, B)

### 2. CNN Overview
- CNN is a neural network containing convolutional layers.
- Core idea: exploit spatial locality and weight sharing to learn image features.
- Typical block: Convolution -> (optional BN) -> Activation (ReLU) -> Pooling.

### 3. Convolution Layer
- Kernel: \(K \in \mathbb{R}^{k \times k \times C}\)
- In RGB input, kernel channel count should match input channel count.
- Convolution slides over spatial dimensions (H/W), not channel dimension.
- One kernel produces one feature map.

- Main properties:
    - Feature extraction
    - Weight sharing
    - Local connectivity (local receptive field)
    - Translation equivariance

- Stride:
    - Step size of kernel movement
    - Controls sampling density / retained spatial detail

- Padding:
    - Adds border values (usually zeros) to preserve edge information and control output size

- Output spatial size:
\[N = \frac{W-F+2P}{S}+1\]

### 4. Pooling Layer
- Downsamples feature maps without learnable parameters.
- Reduces spatial size and computation while preserving salient information.
- Max pooling: take maximum in each window.
- Average pooling: take mean in each window.
- Pooling does not change channel count; each channel is pooled independently.

# 10. RNN
### 1. Core Idea
- Sequence data has temporal/order dependency.
- RNN propagates hidden state across time steps.
- Current output depends on current input and historical hidden state.

### 2. Recurrent Cell
- At time step \(t\):
    - Input: \(x_t\)
    - Hidden update combines \(x_t\) and \(h_{t-1}\), then applies activation (often Tanh)
    - Output is generated by a linear layer (optionally with activation)

### 3. Word Embedding
- Build an embedding matrix: each row is a token vector.
- Typical workflow:
    - Tokenization
    - Token-to-id mapping
    - Embedding lookup by id
- Embedding converts discrete tokens into dense vectors and captures semantic similarity.

### 4. RNN Layer
\[
h_t = \phi(\omega_{xh} x_t + \omega_{hh} h_{t-1} + b_h)
\]
- \(h_t\): current hidden state
- \(\omega_{xh}\): input-to-hidden weights
- \(\omega_{hh}\): hidden-to-hidden (recurrent) weights

### 5. Output Layer
- Regression:
\[
\hat{y}_t = \omega_{hy}h_t + b_y
\]
- Classification:
\[
\hat{y}_t = Softmax(\omega_{hy}h_t + b_y)
\]
\[
\hat{y}_t = \sigma (\omega_{hy}h_t + b_y)
\]

### 6. Characteristics of RNN
- Parameter sharing over time
- Order sensitivity
- Memory via hidden state

### 7. Limitations of RNN
- Long-range dependency is difficult (vanishing/exploding gradients)
- Hard to parallelize due to sequential dependence
- Information may decay over time
- Weaker expressive power than gated/attention architectures

# API
### 1. DataLoader
- `DataLoader` in PyTorch handles batching, shuffling, and parallel data loading.

### 2. Convolution Layer API
- `torch.nn.Conv2d(...)`

### 3. Pooling Layer API
- `torch.nn.MaxPool2d(...)`
- `torch.nn.AvgPool2d(...)`

### 4. Embedding Layer API
- `torch.nn.Embedding(...)`

# To be updated...
