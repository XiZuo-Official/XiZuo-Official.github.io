# 機械学習の実務ワークフロー

課題定義 - データ収集 - データクリーニング - 特徴量エンジニアリング - データ分割 - モデリング - 学習 - モデル評価 - ハイパーパラメータ調整 - 本番運用・監視

# データクリーニング / 前処理
### 1. 欠損値の検出
- 表全体で確認
    ```python
    df.info() # テーブル概要
    df.isna().any().any() # 全体に欠損があるか: BOOL
    np.where(df.isna()) # 欠損の座標を厳密に特定（欠損が少ない場合）
    df.isna().corr() # 欠損パターン間の Pearson 相関
    ```
- 列単位で確認
    ```python
    df.isna().any() # 各列に欠損があるか
    df.loc[:, df.isna().any()]# 欠損を含む列を抽出
    df.isna().sum() # 各列の欠損件数
    df.isna().mean() # 各列の欠損率
    ```
- 行単位で確認
    ```python
    df.isna().any(axis=1) # 各行に欠損があるか
    df[df.isna().any(axis=1)] # 欠損を含む行を抽出
    df.isna().sum(axis=1) # 各行の欠損件数
    df.isna().mean(axis=1) # 各行の欠損率
    ```

- 特殊プレースホルダを確認
    ```python
    df.isin([0, "Unknown", ""]).any().any() # 全表にプレースホルダがあるか
    df.isin([0, "Unknown", ""]).any() # どの列にあるか
    df.isin([0, "Unknown", ""]).any(axis=1) # どの行にあるか
    df[df.isin([0, "Unknown", ""]).any(axis=1)] # プレースホルダを含む行
    df.loc[:, df.isin([0, "Unknown", ""]).any()] # プレースホルダを含む列
    df.isin([0, "Unknown", ""]) # 同形状の真偽値表
    ``` 
- 時系列で欠損を確認
    ```python
    df[df['price'].isna()] # price 欠損行を返す（日時連続性を確認）
    df['price'].isna().astype(int) # 行ごとの欠損を1/0で表示（日付indexが必要）
    df.loc[:, ['date', 'price']].assign(miss=df['price'].isna().astype(int)) # indexが日付でない場合
    df['price'].isna().resample('M').sum() # 月次の欠損件数（日付index必須）
    ```
- その他のテクニック
    ```python
    import numpy as np
    import pandas as pd
    df.shape # データ形状
    df.shape[0] # 行数（サンプル数）
    df.shape[1] # 列数（特徴量数）
    np.unique(df) # 重複なしの値
    len(np.unique(df)) # ユニーク値の個数
    ```

### 2. 欠損値処理
- **削除**
    - 欠損率が低く、かつランダム欠損なら有効
    - 時系列構造を壊しやすい
    ```python
    df.dropna() # 欠損を含む行を削除
    df.dropna(axis=1) # 欠損を含む列を削除
    ```
- **既定値で補完**
    - カテゴリ特徴は業務的に意味のある Unknown などで補完可
    - 時系列の数値欠損は安易に埋めない場合も多い
    ```python
    df['gender'] = df['gender'].fillna('Male')

    df = df.fillna({
    'income': 0,
    'gender': 'Unknown'
    }) # dictで複数列を一括補完
    ``` 
- **統計値補完**
    ```python
    df['income'] = df['income'].fillna(df['income'].mean()) # 平均補完
    df['income'] = df['income'].fillna(df['income'].median()) # 中央値補完
    df['gender'] = df['gender'].fillna(df['gender'].mode()[0]) # 最頻値補完

    df = df.fillna({
        'income': df['income'].median(),
        'age': df['age'].median(),
        'gender': df['gender'].mode()[0]
    }) # 複数列を一度に補完
    ```
    - 最頻値: カテゴリ特徴・カテゴリ数が少なく主要カテゴリが強い場合
    - 中央値: 歪んだ数値分布に有効（分散が縮む傾向）
    - 平均値: ほぼ対称分布・外れ値が少ない数値特徴に有効（分散が縮む傾向）

- **欠損インジケータ変数**
    - 手順:
        - 1. 欠損有無を示す新しい特徴（BOOL）を作成
        - 2. 元の特徴を補完
        - 3. 補完後の**元特徴**と**インジケータ**を両方使って学習

    - 非ランダム欠損（例: 資産非開示）に有効
    - 完全ランダム欠損ではノイズになりうる
    - カテゴリ列はインジケータより Unknown を使うことが多い
    ```python
    df['income_missing'] = df['income'].isna().astype(int) # インジケータ作成
    df['income'] = df['income'].fillna(df['income'].median()) 
    ```
- **グループ別補完**
    ```python
    df['salary'] = (
        df.groupby('industry')['salary'] # 業界でグループ化
        .transform(lambda x: x.fillna(x.median())) # 各グループで補完
    )
    ```
    - 集団差が明確な場合に有効
    - グループ後のサンプルが少ない、またはグループキー自体が欠損する場合は不向き

- **補間法**
    ```python
    df['price'].interpolate() # 線形補間
    df['price'].ffill() # 前方補完
    df['price'].bfill() # 後方補完
    df['price'].interpolate(method='time') # 時間間隔補間（時系列index必須）
    ```
- **多項式補間**
    - 原理: n点を通る n-1 次多項式が存在
    - 1次は2点間の線形補間
    - 高次では Runge 現象（大きな振動）が起きやすい
    - 1点の変化が全体曲線に影響しやすい
    ```python
    df['price'].interpolate(method='polynomial', order=2) # orderは次数
    ```
- **spline補間**

    - 手順:
        - 1. 複数の多項式をつないで滑らかな曲線を作る
        - 2. 次数nを決め、近傍n+1点でn区間に分割
        - 3. 接続点で値と1～(n-1)階導関数を連続にする
        - 4. 連立方程式を解いて各多項式を求める

    - 時系列で連続・平滑な変化、欠損が疎、非線形、トレンド重視のケースに適用。
    - 長い連続欠損、構造変化（政策/危機）、ジャンプ変数、テールリスク、非時系列データには不向き。

    - **実務で cubic spline がよく使われる理由**
        - cubic spline は3次スプライン補間
        - 関数値・1階導関数・2階導関数が連続
        - 1階連続で角が立たず、2階連続で曲率が自然
        - 高次はより滑らかだがノイズに敏感で数値不安定
        - 3次は局所性が高く遠方点の影響を受けにくい
    ```python
    # pandas
    df['price'].interpolate(method='spline', order=3)  
    ```
    ```python
    # scipy低レベル
    from scipy.interpolate import CubicSpline
    x = df.index.astype('int64') # indexは数値必須
    y = df['price'] # 補完対象特徴

    cs = CubicSpline(x[~y.isna()], y.dropna())
    df.loc[y.isna(), 'price'] = cs(x[y.isna()])
    ```
- **モデル補完**
    - 線形回帰、KNN、決定木、ランダムフォレスト、LightGBM、勾配ブースティング、深層学習などを利用可能
    - 実務ではランダムフォレスト + 欠損インジケータがよく使われる
    - 変数間相関が強く、特徴量が多く非線形な場合に適用

    ```python
    # ランダムフォレスト補完の例
    # この例はデータリークを含む可能性あり
    # 実務では後続学習のtrain/test分割と整合したデータで補完する
    from sklearn.ensemble import RandomForestRegressor

    target = 'price'
    features = ['volume', 'market_return', 'volatility', 'industry_code']

    train_df = df[df[target].notna()]
    pred_df  = df[df[target].isna()]

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )

    rf.fit(train_df[features], train_df[target])

    df.loc[df[target].isna(), target] = rf.predict(pred_df[features])
    ```
### 3. 外れ値の検出
- **記述統計**
    ```python
    df.describe()
    ```
    - min/max が不自然に大きい・小さいか
    - std が過大か
    - mean と median の乖離が大きいか

- **分布の可視化**

    - **ヒストグラム**
        ```python
    import numpy as np
    np.log1p(df['price']).hist(bins=50)
        ```
        - 全体形状が正規的か、左/右に歪んでいるか
        - ロングテール/ヘビーテールか（外れ値は尾部に出やすい）
            > ロングテールは右側が長く伸び、平均 $>>$ 中央値になりやすい
            > ヘビーテールは極端値の出現頻度が高い
            > 乗法的蓄積、資源偏在、稀なイベントが原因になりやすい
            > 平均・分散や勾配が不安定になり、モデルが極端サンプルを過重視しやすい
        - 尖り（ピーク）が強いか
            > プレースホルダ、下限打ち切り（閾値以下を0扱い）、業務ルール（無料取引など）
            > 分布の不連続が偽シグナルを生み、平均・分散を歪める
        - 多峰性があるか
            > 異なる母集団や生成メカニズムが混在（混合分布）
            > 単一ルールを学習できず、平均の代表性が低下
    <div style="text-align:center;">            
    <img src="pic/Hist.png" width="70%">
    </div>
   
    - **対数スケールヒストグラム**
        ```python
    import numpy as np
    np.log1p(df['price']).hist(bins=50)
        ```
        - 金額・出来高などはロングテールになりやすく、$\log(1+x)$ 変換が有効
        - 変換後に正規に近づけば、元データがロングテールだった可能性が高い
        - 変換後にも孤立極端点が残るなら、真の異常値の可能性

    - **箱ひげ図**
        ```python
    import matplotlib.pyplot as plt
    df.boxplot(column='price')
    plt.show()
        ```
        - $Q1$: 25パーセンタイル  
            $Q3$: 75パーセンタイル  
            $IQR = Q3 - Q1$

        - 外れ値定義:
        \[x < Q1 - 1.5 \times IQR \quad \text{or} \quad x > Q3 + 1.5 \times IQR\]

        <div style="text-align:center;">
        <img src="pic/Box_Plot.png" width="50%">
        </div>

        - 箱中央線: 中央値
          箱の上下辺: $Q1 / Q3$
          ひげ: $1.5\times IQR$ の範囲
          ひげ外の点: 外れ値

        - 中規模サンプルの単変量スクリーニングに有効（正規性仮定不要）
          ただし極端なロングテールでは外れ値が多発しやすい

        - 箱ひげ規則で手動抽出
            ```python
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df['price'] < lower) | (df['price'] > upper)]
            ```

    - **$Q-Q$ 図**
        - サンプル分布と理論分布（通常は正規分布）の一致度を確認
        - 横軸: 理論分位点
        縦軸: サンプル分位点
        - 各分位が近い -> 分布が近い
        尾部の乖離が大きい -> 尾部挙動が異なる
        <div style="text-align:center;">
        <img src="pic/QQ.png" width="70%">
        </div>
   
- **Z-score法**
    - Z-score（標準得点）は、データ点が平均から何標準偏差離れているかを示す指標。

    \[z = \frac{x - \mu}{\sigma}\]

    - \(x\): サンプル値  
    \(\mu\): サンプル平均  
     \(\sigma\): サンプル標準偏差

    - \(|z| > 2\) は外れ値候補
      \(|z| > 3\) は強い外れ値
      正規分布では約99.7%が $\pm3\sigma$ に入る

    - Z-score は正規分布に適する。歪み・ロングテール・ヘビーテール・多峰には不向き。
    外れ値自身に敏感で、平均・標準偏差が引っ張られる。小標本では不安定。

- **MAD（中央値絶対偏差）法**
    - MAD は外れ値に頑健なばらつき指標。
    標準偏差と違い、中心に平均でなく中央値を使うため極端値の影響が小さい。
        - 1. 各サンプルの中央値からの絶対偏差を計算  
        \[|x_i - \text{median}|\]
        - 2. 絶対偏差の中央値を取る  
        \[\text{MAD} = \text{median}(|x_i - \text{median}|)\]
        - MAD は「典型的な点」が中央値からどれだけ離れるかを表す

    - MAD Robust Z-Score
        - 正規分布では $MAD = \Phi^{-1}(0.75) = 0.6745 \sigma$。
          $3\sigma$ 基準に合わせるため、通常は次でスケーリング:

        \[z_{robust} = \frac{x - \text{median}}{1.4826 \times \text{MAD}}\]
        - 1.4826 は正規分布下の補正係数で、MADと標準偏差を比較可能にする
          よく使う閾値: \(|z_{robust}| > 3\) または \(> 3.5\)

    - MAD は正規性仮定に依存せず、ロングテール/ヘビーテールや外れ値の多い実データに有効。
      金融価格、取引金額、ユーザー行動データなどで頑健性が高い。

### 4. 外れ値処理
- **削除**
    - 明らかな誤記・異常レコードは削除

- **Winsorization（外れ値切り詰め）**
    - 分位点外（例: 両端1%）を境界値で置換し、サンプルを保持
        ```python
    df["income"] = df["income"].clip(
        lower=df["income"].quantile(0.01),
        upper=df["income"].quantile(0.99)
    )
        ```
- **フラグ変数追加**
    - 極端値をフラグ化し、「極端かどうか」を新特徴として利用
        ```python
    df["is_high_income"] = (df["income"] > threshold).astype(int)
        ```
- **非線形変換**
    > 非線形変換は外れ値の影響（レバレッジ）を圧縮し、歪み/ロングテールを緩和してモデル安定性を高める

    - **対数変換**
        - 右に歪んだロングテール（金額/収入/出来高/カウント）に有効
        非負データが必要
        ```python
    import numpy as np
    df["x_log"] = np.log(df["x"])
    df["x_log1p"] = np.log1p(df["x"]) # 0を含む場合
        ```
    - **平方根変換**
        - 軽～中程度の右歪み・ポアソン的データに有効（logより穏やか）
        非負データが必要
        ```python
    import numpy as np
    df['x_sqrt'] = np.sqrt(df['x'])
    df['x_sqrt'] = np.sqrt(df['x'] + 1e-6) # 0対策
        ```
    - **Box-Cox 変換**
        - パラメトリックなべき変換。\(\lambda\) により圧縮強度が変化。\(\lambda=0\) で log と等価。
        - 適切な \(\lambda\) を選んで正規性に近づける。
            > 目的: $x^{(\lambda)} \sim \mathcal{N}(\mu, \sigma^2)$ を満たすよう MLE で \(\lambda\) 推定
            > $$ \ell (\lambda) = -\frac{n}{2} \log{\hat{\sigma}^2_\lambda} + (\lambda - 1)\sum_{i=1}^{n} \log{x_i}$$
            
        - 変換後は解釈性が下がる
        - 入力は正値（0不可）。式:
            $$ x^{(\lambda)} =
            \begin{cases}
            \dfrac{x^\lambda - 1}{\lambda}, & \lambda \neq 0 \\
            \log(x), & \lambda = 0
            \end{cases}$$
        - scipy 例
            ```python
    from scipy.stats import boxcox

    amount_bc, lambda_hat = boxcox(df['amount'])
    df['amount_boxcox'] = amount_bc
    print("Optimal lambda:", lambda_hat)
            ```
        - sklearn 例
            ```python
    from sklearn.preprocessing import PowerTransformer

    pt = PowerTransformer(method='box-cox', standardize=True)

    df['amount_boxcox'] = pt.fit_transform(df[['amount']])
            ```
    - **Yeo–Johnson 変換**
        - こちらもべき変換で、歪み・ロングテール・外れ値影響を緩和
        - 0や負値を扱える
        - \(x \ge 0\) のとき:
        \[x^{(\lambda)} =
        \begin{cases}
        \frac{(x+1)^\lambda - 1}{\lambda}, & \lambda \neq 0 \\
        \log(x+1), & \lambda = 0
        \end{cases}\]
          \(x < 0\) のとき:
        \[x^{(\lambda)} =
        \begin{cases}
        -\frac{(1-x)^{2-\lambda} - 1}{2-\lambda}, & \lambda \neq 2 \\
        -\log(1-x), & \lambda = 2
        \end{cases}\]
            > \(x+1\) により x=0 を扱え、\(\lambda=0\) でも滑らかに遷移。\(-1\) は \(\lambda=1\) を恒等写像に近づけるため。
        - 歪み分布、ロングテール、線形モデルでの外れ値レバレッジ、正規性違反に有効
        - scipy 例
            ```python
    from scipy.stats import yeojohnson
    amount_yj, lambda_hat = yeojohnson(df['amount']) 

    df['amount_yeojohnson'] = amount_yj
    print("Optimal lambda:", lambda_hat)
            ```
        - sklearn 例
            ```python
    from sklearn.preprocessing import PowerTransformer

    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    X_train_yj = pt.fit_transform(X_train[["x"]])
    X_test_yj  = pt.transform(X_test[["x"]])
            ```

### 5. 標準化
- 定義
    - 標準化は特徴量を **平均0・標準偏差1** に変換する処理。代表は **Z-score 標準化**:
        > 標準化は線形変換であり、分布形状自体は変えない

    \[x_{\text{std}} = \frac{x - \mu}{\sigma}\]

        - $\mu$: サンプル平均
        $\sigma$: サンプル標準偏差

- 特徴量全体の標準化コード
    ```python
    # データリーク防止: trainでfit、testはtransformのみ
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train_std = scaler.fit_transform(X_train) 
    X_test_std  = scaler.transform(X_test)
    ```
- 指定列のみ標準化
    ```python
    from sklearn.preprocessing import StandardScaler

    cols_to_scale = ["age", "income"]
    scaler = StandardScaler()
    
    X_train_std = X_train.copy()
    X_test_std  = X_test.copy()

    X_train_std[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_std[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])
    ```


- 標準化の効果
    - 単位差の除去: 大スケール特徴の支配を防ぐ
    - 距離の比較可能化: 距離/内積ベース手法で重要
    - 収束高速化: 勾配降下の数値安定性を改善
        > 特徴量スケール差により $\frac{\partial L}{\partial w_j} = x_j (\hat{y} - y)$ の大きさが不揃いになる
    - 正則化の公平化: L1/L2罰則を特徴間で公平に適用
        > 正則化は重みを罰するため、スケール小の特徴は相対的に不利になりやすい
- 適用モデル

    | モデリング機構 | モデル | 標準化が必要な理由 |
    |---------|----------|------------------|
    | 距離ベース | KNN、K-means、階層クラスタリング | 距離比較にはスケール整合が必要 |
    | 内積ベース | SVM、PCA、線形モデル | 内積はスケールに依存し方向が歪む |
    | 勾配最適化 | Linear、Logistic、ニューラルネット | 勾配スケール不一致で収束が遅い/不安定 |
    | 正則化項 | Ridge、Lasso、Elastic Net | 尺度差でL1/L2罰則が不公平 |
- 木モデルが標準化を不要とする理由:
   - 木は値の大小順と分割閾値にのみ依存し、距離・内積・勾配スケールを使わない。標準化は単調変換なので分割結果が変わらない。

### 6. 正規化（Normalization）

- 定義
    - 正規化は特徴量を固定区間（通常 $[0,1]$）へ線形写像する処理。代表は **Min-Max 正規化**:

    \[x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}\]

        - $x_{\max} / x_{\min}$: サンプル最大値 / 最小値

- 特徴量全体の正規化コード
    ```python
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler() 

    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm  = scaler.transform(X_test)
    ```

- 指定列のみ正規化
    ```python
    from sklearn.preprocessing import MinMaxScaler

    cols_to_scale = ["age", "income"]  
    scaler = MinMaxScaler()

    X_train_norm = X_train.copy()
    X_test_norm  = X_test.copy()

    X_train_norm[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_norm[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])
    ```

- 正規化の効果
    - 値域の統一: 単位の異なる特徴を同一範囲へ
    - 数値安定性向上: 過大値によるオーバーフローを抑制
    - 有界特徴に適合: 比率・確率・画素値など
    - 距離ベース手法で有効（特徴が同等の意味尺度を持つ場合）
        > 正規化は skewness を解消せず、線形スケーリングのみ行う

- 正規化が有効なモデル
    | モデリング機構 | モデル | 正規化が必要な理由 |
    |---------|------|------------------|
    | 距離ベース | KNN、K-means | 距離は値域の影響を直接受ける |
    | NN + 有界活性 | NN（Sigmoid / Tanh） | 入力を活性の感度領域に保ちやすい |
    | 画像データ | CNN | 画素値は本来上下限を持つ |
    | 自然な境界を持つ特徴 | 比率・確率 | 区間意味を維持できる |

- 正規化の限界
    - 外れ値に非常に敏感: 極端値が他サンプルを圧縮
    - 分布形状は不変: 歪み・多峰性は残る
    - データセット間で不安定: テスト値が訓練範囲外なら >1 や <0 になる

- 木モデルが正規化を不要とする理由:
    - 木は特徴量の順序と分割閾値に依存
    - 正規化は単調線形変換で順序を変えない
    - 左右分割集合が不変で、情報利得 / Gini / MSE 低下量も不変

# 決定木
### 1. 情報理論

- **情報量**
    - 事象確率を $p(x)$ とすると情報量は:
        $$
        I(x) = - \log_2 p(x)
        $$
    <div style="text-align:center;">
    <img src="pic/Info.png" width="40%">
    </div>

       - 情報とは「事前予測をどれだけ裏切るか」の度合い
       - **起こりにくい事象ほど情報量が大きい**
       - 符号化の観点では、低確率事象ほど長い符号長が必要。$\frac{1}{2}, \frac{1}{4},\frac{1}{8}$ は最適符号でそれぞれ1,2,3 bit
    <br>
- **エントロピー Entropy**

    - エントロピーは不確実性の尺度。不確実性が高いほどエントロピーは大きい
    - 離散確率変数 $Y$ のクラス確率が $p_1, p_2, ..., p_k$ のとき:
        \[H(Y) = - \sum_{i=1}^{k} p_i \log_2 p_i\]
      すなわち平均情報量:
        $$H(Y) = \mathbb{E}[-\log_2 p(Y)]$$
    - 連続変数では微分エントロピー:

        $$
        H(Y) = - \int_{-\infty}^{\infty} f(y) \log f(y)\, dx
        $$
        - 微分エントロピーは底が $e$ なので単位は bit でなく nat

    - あるクラスの確率が1なら不確実性は0で、エントロピーは0
    - 多クラスでは一様分布で最大: $H_{\max} = \log_2 k$

- **条件付きエントロピー**
    - $X$ を特徴、$Y$ を目的とすると、$H(Y|X)$ は「X既知でも残るYの不確実性」。
    - 各X状態でのYエントロピーの加重平均:
    \[H(Y \mid X) = \sum_{x} p(x)\, H(Y \mid X=x)\] 
      ここで:
    \[H(Y \mid X=x) = -\sum_{y} p(y \mid x)\log p(y \mid x)\]

- **情報利得**
    - 特徴が不確実性をどれだけ減らすか（相互情報量）
    - 式は「元エントロピー - 条件付きエントロピー」:
        \[
        IG(Y, X) = H(Y) - H(Y\mid X)
        \]
        - $X$: 特徴、$Y$: 目的変数
        - \(IG\) が大きいほど、\(X\) の説明力が高い
        - \(IG=0\) は独立（情報を与えない）

- **ジニ指数**
    - データ集合の不純度（impurity）を測る指標
    - ランダムに2サンプルを引いたとき、異なるクラスである確率で表せる:
    $$
    Gini = 1 - \sum_{i=1}^{k} p_i^2
    $$
    - Gini=0 は最純。クラス一様分布で最大

### 2. ID3 決定木
- 分類アルゴリズム。分割を繰り返してノード純度を高め、不確実性を下げる。分割評価は情報利得で、IG が大きいほど良い。

- ID3 の流れ:
    - 1. データ集合のエントロピーを計算
    - 2. 各特徴で分割した条件付きエントロピーを計算
    - 3. 情報利得（エントロピー - 条件付きエントロピー）を求め、最大IGを選択
    - 4. 特徴が尽きる / エントロピー0 / サンプル0 まで再帰分割

- ID3 の限界:
    - カテゴリ数が多い特徴ほど情報利得が大きくなりやすい
    - 離散特徴のみ扱える
    - 剪定機構がなく過学習しやすい
        > ID3 が過学習しやすい理由:
        >   1. 純化するまで深く分割し続ける
        >   2. 情報利得の貪欲法で局所最適になりやすい
        >   3. 貪欲法自体は汎化性能を直接評価しない
        >   4. 高カテゴリ特徴に偏り、極端には User_ID で丸暗記になる

### 3. C4.5 決定木
- 基本思想はID3と同じだが、指標に情報利得率を使い、高カテゴリ特徴への偏りを抑える。

- C4.5 の流れ:
    - 1. データ集合のエントロピーを計算
    - 2. 各特徴分割の条件付きエントロピーを計算
    - 3. 各特徴の情報利得を計算
    - 4. 分割情報を計算
    - 5. 情報利得率を計算し最大値で分割
    - 6. 停止条件まで再帰分割
    - 7. 剪定

- 情報利得率:
    - 特徴の情報利得を特徴エントロピーで割る:
    \[
    GainRatio(D,A)=\frac{IG(D,A)}{H_A(D)}
    \]
        - D: データ集合
        - A: 分割特徴
        - $H_A(D)$: 特徴Aの分割情報（経験分布に基づくエントロピー）
            > 情報理論の $H(X)$ に対応。ただし真分布ではなくD上の経験分布

- C4.5 の限界:
    - 貪欲法の局所最適
    - 計算コストが高い
    - 木が深くなりノイズに敏感になりやすい

### 4. CART 決定木
- CART は2分割を繰り返し、子ノードをより純粋/安定にする。
- CART は分類・回帰の両方に対応。
  分類はジニ指数最小、回帰は MSE/SSE 最小を選択。

- CART の流れ:
    - 1. 全特徴について全ての2分割候補を列挙
    - 2. 各分割のジニ指数または MSE を計算
    - 3. 最適分割を選択
    - 4. 停止条件なら終了、そうでなければ再帰
    - 5. 代償複雑度剪定

- **CART 分類木:**
    - 離散特徴の場合:
        - 1. 2分割を全列挙。$k$カテゴリなら $2^{k-1}-1$ 通り
        - 2. 分割後の加重ジニ指数を計算:
            \[Gini_{split}(X_j, t) = \frac{|D_L|}{|D|} Gini(D_L) + \frac{|D_R|}{|D|} Gini(D_R)\]
            > $\frac{|D_L|}{|D|}, \frac{|D_R|}{|D|}$ は左右ノードのサンプル比率
            > $Gini(D_L), Gini(D_R)$ は左右ノードの不純度
            > 特徴は分割のみに使われ、ジニ計算はノード内サンプルで行う
        - 3. 最小ジニの分割を採用

    - 連続特徴の場合:
        - 1. 値を昇順に並べる
        - 2. 隣接する異なる値の中点を分割候補閾値にする
        - 3. 各候補で加重ジニ指数を計算
        - 4. 最小ジニの閾値を採用

- **CART 回帰木:**
    - 連続値を予測するため、特徴空間を分割し各領域に定数を割り当てる。
        - 1. 全特徴の全分割候補を列挙
        - 2. 分割後損失を計算:
            \[Loss_{split}(X_j, t) = Loss(D_L) + Loss(D_R)\]
            > $Loss(D_L), Loss(D_R)$ は左右ノードの回帰損失
            > 典型的には SSE:
            >    \[SSE(D)=\sum_{i\in D}\left(y_i-\bar{y}_D\right)^2\]
            > 特徴は分割にのみ使い、損失計算はノード内サンプルで行う
        - 3. SSE 最小の分割を採用

    - 予測出力（葉ノード）:
        - 葉ノードの目的変数平均を出力:
            \[\hat{y}(x)=\bar{y}_{D_{\text{leaf}}}\]
            > CART回帰木は通常SSE（総和）で分割評価するため、左右サンプル数の追加重み付けは不要。MSE評価なら重み付けが必要。
### 5. 剪定 Pruning
- 目的は過学習を抑え、汎化性能を高めること。

- **事前剪定 / 早期停止**
    - 木の成長中に、分割前に性能改善を評価して停止させる。
    - 代表的な制約:
        -   最大深さ（max_depth）
        -   最小分割サンプル数（min_samples_split）
        -   最小葉サンプル数（min_samples_leaf）
        -   最小不純度減少（min_impurity_decrease）

    - 例
        ```python
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=30,
        min_samples_leaf=10,
        min_impurity_decrease=1e-4,
        max_leaf_nodes=50,
        random_state=42
    )
    tree.fit(X_train, y_train)
        ```

- **事後剪定**
    - まず完全木を作り、下から各部分木を評価。残すより切る方が誤差が小さければ剪定する。

- **悲観的誤差剪定**
    - C4.5 の事後剪定で使う。学習誤差は $\frac{e}{n}$ だが、悲観的推定では $\frac{e+0.5}{n}$ として小標本葉へのペナルティを与える。
    - 学習データのみで評価し、統計的補正により将来誤差を見積もる。
    - 推定原理:
        - 1. 葉ノードでは正誤の2値なので、誤り数 $e$ は二項分布 $E \sim Bino(n,p)$
        - 2. $\hat{p}=\frac{e}{n}$ は楽観的点推定。上側信頼境界 $e_{up}$ を使い、$P(E \leqslant e_{up}) = 1 - \alpha$
        - 3. 標本十分大（np>5 かつ nq>5）なら二項分布は正規近似可能: $E \approx N(np,npq)$
        - 4. 標準化:
        $$Z = \frac{E - np}{\sqrt{npq}} \approx N(0,1)$$ 
        - 5. 正規は連続、二項は離散なので連続性補正を行う:
        $$P(E \leqslant e_{up}) = P(X \leqslant e_{up} + 0.5)$$
        - 6. よって C4.5 の $\frac{e+0.5}{n}$ は、正規近似 + 連続性補正に基づく保守的誤差率推定。

- **代償複雑度剪定（Cost-Complexity Pruning）**
    - CART の事後剪定。最大木を作成後、剪定で部分木系列を作り、検証/交差検証で最良木を選ぶ。
    - 評価指標は正則化付き経験リスク $R_\alpha(T)$。

    - 手順:
        - 1. 最大木を生成
        - 2. 全内部ノードについて、剪定前後の経験リスクと葉数を計算
        - 3. 各ノードの臨界係数を計算し、最小 $\alpha_t$ ノードから剪定
        - 4. weakest-link 戦略で繰り返し、入れ子部分木系列を得る
        - 5. 交差検証で誤差最小の部分木を選択
        
    - CART の剪定目的関数:
    \[
    R_\alpha(T) = R(T) + \alpha |T|
    \]
        - $R(T)$ は経験リスク。分類では誤分類率（場合により Gini/交差エントロピー）、回帰では SSE を使うことが多い。
        - $|T|$ は葉ノード数で、複雑度ペナルティ

    - 臨界ペナルティ係数
        \[
        \alpha_t = \frac{R(t)-R(T_t)}{|T_t|-1}
        \]
        - 分子 \(R(t)-R(T_t)\): 部分木を1葉に置換したとき増える経験リスク
        - 分母 \(|T_t|-1\): 剪定で減る葉数
        - したがって \(\alpha_t\) は「葉1つ削減あたりの誤差増加コスト」
        - 設定した $\alpha$ がこの閾値より小さいなら、ペナルティが弱く部分木を残す方が有利

# 更新予定...
