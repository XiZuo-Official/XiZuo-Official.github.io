# 一、テンソル作成
### 1. テンソルの基本作成
    ```python
    import torch
    # torch.tensor: 明示データからテンソルを作成
    # torch.Tensor: 形状指定で作成
    # torch.IntTensor | torch.FloatTensor: 型を指定して作成
    ```
# 二、テンソル演算
### 1. 演算関数
- 平方根
    ```python
    print(data.sqrt())
    ```
- 指数
    ```python
    print(data.exp())
    ```
- 対数
    ```python
    print(data.log())
    print(data.log2())
    print(data.log10())
    ```

# 三、活性化関数
### 1. 活性化関数の役割
- 「重み付き和の出力」に非線形ルールを与え、ニューラルネットワークを単なる線形モデルにしない。

### 2. Sigmoid

\[\sigma(x) = \frac{1}{1 + e^{-x}}\]

<div style="text-align:center;">            
<img src="pic/sigmoid.png" width="50%">
</div>

- 定義域: \((-\infty, +\infty)\)
- 値域: \((0, 1)\)
- 単調増加
- 入力は概ね \((-3, 3)\) にあると出力差が出やすい。範囲外では 0 か 1 に近づく。

* **Sigmoid の導関数:**
$$
\sigma'(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr)
$$ 
<div style="text-align:center;">  
<img src="pic/sigmoid_derivative.png" width="50%">
</div>


- \(x = 0\) のとき、\(\sigma'(x)_{\max} = 0.25\)
- 値域: \((0, 0.25)\)
- \(|x|\) が大きいと導関数は 0 に近づき、深層で勾配が伝播しにくい
- そのため Sigmoid は通常、2値分類の出力層で使う
>  **Sigmoid が隠れ層に不向きな理由:**
>  **1. 勾配消失**
>   導関数の最大値が 0.25 と小さく、連鎖律により初段重みに対する勾配が縮む:
>   \[\frac{\partial \mathcal L}{\partial \omega_1} = \frac{\partial \mathcal L}{\partial a_2}
>      \cdot
>      \frac{\partial a_2}{\partial z_2}
>      \cdot
>      \frac{\partial z_2}{\partial a_1}
>      \cdot
>      \frac{\partial a_1}{\partial z_1}
>      \cdot
>      \frac{\partial z_1}{\partial \omega_1}\]
>     ここで \(z\) は線形変換（重み付き和）、\(a\) は活性化。
>   \(\dfrac{\partial a}{\partial z}\) は Sigmoid 導関数で、最大でも 0.25。
>      多層で掛け合わされると \(\dfrac{\partial \mathcal L}{\partial \omega_1}\) は 0 に近づく。<br>
>      **2. 飽和領域が広い**
>      \(|x| > 6\) では出力が入力変化に鈍感になり、0/1 付近で飽和する。
>      **ニューロン飽和:** 飽和域では導関数がほぼ 0 となり、逆伝播で重み更新が進みにくい。<br>
>      **3. 非ゼロ中心**
>      Sigmoid の値域は (0,1) で 0 中心ではない。更新方向が偏りやすく、zig-zag 的になって収束が遅くなる。

### 3. Tanh

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$
<div style="text-align:center;">  
<img src="pic/tanh.png" width="50%">
</div>
- 定義域: \((-\infty, +\infty)\)
- 値域: \((-1, 1)\)
- 単調増加
- 入力が \((-3, 3)\) 付近で最も変化し、外側では徐々に飽和する

* **Tanh の導関数:**
$$
\tanh'(x) = 1 - \tanh^2(x)
$$
<div style="text-align:center;"> 
<img src="pic/tanh_derivative.png" width="50%">
</div>
- \(x = 0\) のとき、\(\tanh'(x)_{\max} = 1\)
- 値域: \((0, 1]\)
- \(|x|\) が大きいと導関数は 0 に近づくため、勾配消失は依然として起こる（浅いネット向き）
- Tanh は 0 中心なので、Sigmoid より隠れ層で収束しやすい

### 4. ReLU

$$
\mathrm{ReLU}(x) = \max(0, x)
$$
<div style="text-align:center;"> 
<img src="pic/relu.png" width="50%">
</div>
- 定義域: \((-\infty, +\infty)\)
- 値域: \([0, +\infty)\)
- 区分関数。\(x<0\) では常に 0 のため厳密単調増加ではない
- \(x>0\) では入力と出力が線形関係

* **ReLU の導関数（勾配）:**
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

- \(x > 0\) では勾配が 1 で、勾配消失を起こしにくい
- \(x < 0\) では勾配が 0
- \(x = 0\) は微分不可能（実装では 0 または 1 を採用）

> **ReLU が隠れ層に適する理由:**<br>
>   **1. 計算が簡単で高速**  
>   比較と代入のみで計算でき、学習が速い。<br>
>   **2. 勾配消失を緩和**  
>   正領域では導関数が 1 のため、深層でも勾配が減衰しにくい。<br>
>   **3. 疎な活性化**  
>   \(x < 0\) で出力 0 となり、一部ニューロンが非活性化。表現効率向上に寄与。<br>
>   **4. 潜在的問題（Dying ReLU）**  
>   学習中に入力が \(x < 0\) に偏ると勾配が常に 0 になり、更新できない。

>  **Dying ReLU が起きやすい条件:**<br>
>    1. **バイアス \(b\) が過度に負**  
>     もし 
>    \[b \ll 0 \Rightarrow z=w^\top x + b < 0\ \text{(多数の }x\text{で)} \] 
>    なら 
>    \[\mathrm{ReLU}'(z)=0\] となり、長期的に非活性化しやすい。<br>
>   2. **学習率 \(\eta\) が大きすぎる**  
>  更新式:
>  \[ \omega \leftarrow \omega-\eta\frac{\partial \mathcal L}{\partial \omega},\qquad b \leftarrow b-\eta\frac{\partial L}{\partial b}
>  \] 
>  \(\eta\) が大きいと 1 ステップで
>  \[\omega^\top x + b < 0\] が多数サンプルで成立し、以後 \(\mathrm{ReLU}'(z)=0\) で戻りにくい。<br>
>   3. **入力分布の負側シフト / 初期化不良**  
>   学習初期から
>   \[\omega^\top x + b < 0 \] が多いと、ほぼ最初から死んだ状態になる。

### 5. Softmax

$$
\mathrm{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$



- 入力はベクトル \(z = (z_1, z_2, \dots, z_K)\)
- 出力は確率分布
- 各成分の範囲は \((0, 1)\)
- 全出力の和は 1:
\[
\sum_{i=1}^{K} \mathrm{Softmax}(z_i) = 1
\]

> **Softmax が通常出力層で使われる理由:**<br>
>   **1. 確率として解釈しやすい**  
>   出力は \((0,1)\) かつ総和 1 のため、クラス確率を直接表せる。<br>
>   **2. 多クラス分類に適合**  
>   交差エントロピー損失（Cross Entropy）と組み合わせて使うことが多い。<br>
>   **3. 隠れ層活性化には不向き**  
>   クラス間競合が生じ、勾配が相互依存するため中間表現学習に不利。

# 四、パラメータ初期化
### 1. 初期化の目的
- **勾配消失/爆発の抑制**
    - 重みが大きすぎると \( z = \omega x + b \) が過大になり、Sigmoid/Tanh が飽和して勾配消失しやすい。
- **収束速度の改善**
    - 適切な初期値は勾配伝播を安定化し、学習を速める。
- **対称性の破壊**
    - 全パラメータを同一値で初期化すると、全ニューロンが同じ特徴を学習する。

### 2. 初期化手法
- **全0 / 全1 / 定数初期化**
    ```python
    import torch
    import torch.nn as nn
    nn.init.zeros_()
    nn.init.ones_()
    nn.init.constant_()
    ```
- **一様分布初期化**   \( \omega \sim U(-a, a) \)
    ```python
    nn.init.uniform_()
    ```
- **正規分布初期化** \( \omega \sim \mathcal{N}(0, \sigma^2) \)
    ```python
    nn.init.normal_()
    ```
- **He（Kaiming）正規初期化**
$$
\omega \sim \mathcal{N}\left(0,\ \frac{2}{fan_{in}}\right)
$$
    ```python
    nn.init.kaiming_normal_()
    ```
    - \(fan_{in}\): ニューロン1個あたりの入力結合数

- **He（Kaiming）一様初期化**
$$
\omega \sim \mathcal{U}\left(
-\sqrt{\frac{6}{fan_{in}}},
+\sqrt{\frac{6}{fan_{in}}}
\right)
$$
    ```python
    nn.init.kaiming_uniform_()
    ```

- **Xavier 正規初期化**
$$
\omega \sim \mathcal{N}\left(
0,\ \frac{2}{fan_{in}+fan_{out}}
\right)
$$
    ```python
    nn.init.xavier_normal_()
    ```

- **Xavier 一様初期化**
$$
\omega \sim \mathcal{U}\left(
-\sqrt{\frac{6}{fan_{in}+fan_{out}}},
+\sqrt{\frac{6}{fan_{in}+fan_{out}}}
\right)
$$
    ```python
    nn.init.xavier_uniform_()
    ```

# 五、損失関数
### 1. 多クラス交差エントロピー損失
- 全サンプル平均の式:
    $$
        \mathcal{L}
        =
        -\frac{1}{N}
        \sum_{n=1}^{N}
        \sum_{i=1}^{C}
        y_{n,i}\,\log(p_{n,i})
    $$

### 2. 二値交差エントロピー損失
\[
\mathcal{L} = - y \log(p) - (1-y)\log(1-p)
\]

### 3. MAE損失 / L1損失
\[
\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
\]
- 外れ値に比較的頑健だが、最適化が遅くなりやすい。

### 4. MSE損失 / L2損失
\[
\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}\bigl(y_i - \hat{y}_i\bigr)^2
\]
- 平滑で微分可能。大誤差に強いペナルティ。
- 外れ値に敏感。

### 5. Huber Loss / Smooth L1
$$
L(y, y_i)=
\begin{cases}
\frac{1}{2}(y - y_i)^2, & \text{if } |y - y_i| \le \delta, \\
\delta |y - y_i| - \frac{1}{2}\delta^2, & \text{if } |y - y_i| > \delta .
\end{cases}
$$
- 小誤差ではMSE、大誤差ではMAE的に振る舞う。

# 六、勾配降下系最適化アルゴリズム
### 1. 基本用語
- Epoch: 訓練データ全体を1回学習
- Batch size: 1回の更新で使うサンプル数
- Iteration: パラメータ更新1回

### 2. 勾配降下の限界
- 局所最適
- 鞍点停滞
- 平坦領域で遅い
- 初期値依存

### 3. 指数移動平均（EMA）
\[
v_t = \beta v_{t-1} + (1-\beta)x_t
\]

### 4. Momentum
\[\omega_t = \omega_{t-1} - \eta V_t\]
$$
V_t = \beta V_{t-1} + (1-\beta) \cdot \left.\frac{\partial \mathcal L_t}{\partial \omega} \right|_{\omega_{t-1}}
$$

### 5. AdaGrad
\[G_{t,i} = \sum_{k=1}^{t} g_{k,i}^2\]
\[\omega_{t,i} = \omega_{t-1,i} - \frac{\eta}{ \sqrt{G_{t,i}} + \varepsilon} \, g_{t,i}\]
- 学習率が単調減少し、早期に小さくなりすぎることがある。

### 6. RMSProp
\[V_{t,i} = \beta V_{t-1,i} + (1-\beta)\, g_{t,i}^2\]
\[
\omega_{t,i} = \omega_{t-1,i} - \frac{\eta}{\sqrt{V_{t,i}} + \varepsilon}\,g_{t,i}
\]

### 7. Adam
- 一次モーメント:
\[ m_{t,i} = \beta_1 m_{t-1,i}+(1-\beta_1)\, g_{t,i}\]
- 二次モーメント:
\[
v_{t,i} = \beta_2 v_{t-1,i} + (1-\beta_2)\, g_{t,i}^2
\]
- バイアス補正:
\[\hat m_{t,i} = \frac{m_{t,i}}{1-\beta_1^t},\qquad \hat v_{t,i}=\frac{v_{t,i}}{1-\beta_2^t}\]
- 更新式:
\[
\omega_{t,i} = \omega_{t-1,i} - \frac{\eta}{\sqrt{\hat v_{t,i}} + \varepsilon}\,\hat m_{t,i}
\]

# 七、学習率減衰手法
### 1. 等間隔減衰
- 一定ステップ/epochごとに学習率を定率で下げる。

### 2. MultiStepLR（指定ステップ減衰）
- 指定した milestone に到達したときに学習率を減衰する。

### 3. 指数減衰
\[
\eta_t = \eta_0 \cdot \gamma^t
\]
- 連続的・平滑だが、早期に小さくなりすぎる場合がある。

### 4. 余弦アニーリング（Cosine Annealing）
\[
\eta_t = \eta_{\min} + \frac{1}{2} (\eta_{\max}-\eta_{\min}) \left(1 + \cos\left(\frac{\pi t}{T_{\max}}\right)\right)
\]

### 5. ReduceLROnPlateau（性能指標ベース減衰）
- step数ではなく検証指標（loss/accuracy）を監視し、一定期間改善しなければ学習率を下げる。

# 八、正則化
### 1. Dropout
- Dropout はニューラルネットワークの正則化手法で、過学習を抑え汎化性能を高める。
- 学習時に一部ニューロンをランダムに無効化し、特定ユニットへの過度な依存（共適応）を防ぐ。
- 学習フェーズ:
    - ニューロンをランダムにドロップ
    - 各 iteration が異なる部分ネットワークとして機能
    - 期待値が保たれるようにスケーリング
- 推論フェーズ:
    - Dropout を無効化し、完全ネットワークを使う

### 2. バッチ正規化（Batch Normalization）
- BN は中間活性を標準化して分布を安定化し、学習を高速化する。
- 処理フロー:
\[
x \xrightarrow{\text{線形変換}} z \xrightarrow{\text{BN正規化}} \hat z \xrightarrow{\text{Scale/Shift}} y \xrightarrow{\text{活性化}} h
\]
- 正規化:
\[
\hat z_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}
\]
- アフィン変換:
\[
y_i = \gamma \hat z_i + \beta
\]

# 九、CNN
### 1. 画像分類の基礎
- 2値画像: 1チャネル、画素は0/1
- グレースケール: 1チャネル、画素範囲[0,255]
- RGB画像: 3チャネル（R/G/B）

### 2. CNN 概要
- CNN は畳み込み層を含むネットワーク。
- 空間的局所性と重み共有を利用して特徴を学習する。
- 典型構成: 畳み込み -> （任意でBN）-> ReLU -> プーリング。

### 3. 畳み込み層
- カーネル: \(K \in \mathbb{R}^{k \times k \times C}\)
- RGB入力ではカーネルのチャネル数は入力と一致させる。
- スライドは空間方向（行・列）のみ。
- カーネル1個につき特徴マップ1枚。

- 主な性質:
    - 特徴抽出
    - 重み共有
    - 局所結合（受容野）
    - 平行移動等変性

- stride（ストライド）:
    - カーネル移動幅
    - サンプリング密度を決める

- Padding（パディング）:
    - 境界情報保持や出力サイズ制御のために周辺を補う（通常0埋め）

- 出力サイズ:
\[N = \frac{W-F+2P}{S}+1\]

### 4. プーリング層
- 学習パラメータなしで特徴マップをダウンサンプリング。
- 空間次元を縮小し計算量を削減。
- 最大プーリング: ウィンドウ内最大値
- 平均プーリング: ウィンドウ内平均値
- チャネル数は変えず、各チャネルを独立に処理。

# 十、RNN
### 1. 原理
- 系列データは時間/順序依存を持つ。
- RNN は隠れ状態を時系列方向に伝播する。
- 現在の出力は現在入力と過去状態に依存する。

### 2. Recurrent Cell
- 時刻 \(t\) で:
    - 入力: \(x_t\)
    - \(x_t\) と \(h_{t-1}\) を統合し活性化（通常 Tanh）
    - 線形層（必要なら活性化）で出力生成

### 3. 単語埋め込み
- 埋め込み行列を作り、各行を単語ベクトルとする。
- 手順:
    - トークナイズ
    - token -> id 変換
    - id で埋め込みベクトルを参照
- 離散トークンを低次元・密ベクトルへ写像し、意味類似性を表現。

### 4. RNN層
\[
h_t = \phi(\omega_{xh} x_t + \omega_{hh} h_{t-1} + b_h)
\]

### 5. 出力層
- 回帰:
\[
\hat{y}_t = \omega_{hy}h_t + b_y
\]
- 分類:
\[
\hat{y}_t = Softmax(\omega_{hy}h_t + b_y)
\]
\[
\hat{y}_t = \sigma (\omega_{hy}h_t + b_y)
\]

### 6. RNN の特徴
- 時間方向でのパラメータ共有
- 入力順序に敏感
- 隠れ状態による記憶

### 7. RNN の限界
- 長期依存が苦手（勾配消失/爆発）
- 逐次依存のため並列化しにくい
- 時間とともに情報が減衰しやすい
- ゲート/注意機構に比べ表現力が弱い

# API
### 1. DataLoader
- `DataLoader` はバッチ化・シャッフル・並列読み込みを担当する。

### 2. 畳み込み層API
- `torch.nn.Conv2d(...)`

### 3. プーリング層API
- `torch.nn.MaxPool2d(...)`
- `torch.nn.AvgPool2d(...)`

### 4. 埋め込み層API
- `torch.nn.Embedding(...)`

# 更新予定...
