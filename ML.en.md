# Machine Learning Industrial Workflow

Problem definition - Data collection - Data cleaning - Feature engineering - Dataset split - Modeling - Training - Model evaluation - Hyperparameter tuning - Deployment and monitoring

# Data Cleaning / Preprocessing
### 1. Finding Missing Values
- Query at table level
    ```python
    df.info() # table overview
    df.isna().any().any() # whether the whole table has missing values: BOOL
    np.where(df.isna()) # locate exact coordinates of missing values (for very sparse missingness)
    df.isna().corr() # Pearson correlation matrix of missingness across features
    ```
- Query by column
    ```python
    df.isna().any() # whether each column has missing values
    df.loc[:, df.isna().any()]# columns with missing values
    df.isna().sum() # number of missing values per column
    df.isna().mean() # missing ratio per column
    ```
- Query by row
    ```python
    df.isna().any(axis=1) # whether each row has missing values
    df[df.isna().any(axis=1)] # rows with missing values
    df.isna().sum(axis=1) # number of missing values per row
    df.isna().mean(axis=1) # missing ratio per row
    ```

- Check special placeholders
    ```python
    df.isin([0, "Unknown", ""]).any().any() # whether placeholders appear in full table
    df.isin([0, "Unknown", ""]).any() # which columns contain placeholders
    df.isin([0, "Unknown", ""]).any(axis=1) # which rows contain placeholders
    df[df.isin([0, "Unknown", ""]).any(axis=1)] # rows with placeholders
    df.loc[:, df.isin([0, "Unknown", ""]).any()] # columns with placeholders
    df.isin([0, "Unknown", ""]) # boolean table with same shape
    ``` 
- Time-series missing value query
    ```python
    df[df['price'].isna()] # full rows where price is missing; inspect date continuity
    df['price'].isna().astype(int) # 1/0 marker for missing price in each row (date index required)
    df.loc[:, ['date', 'price']].assign(miss=df['price'].isna().astype(int)) # same as above when index is not date
    df['price'].isna().resample('M').sum() # monthly missing count for price (date index required)
    ```
- Other tricks
    ```python
    import numpy as np
    import pandas as pd
    df.shape # dataset shape
    df.shape[0] # number of rows (sample size)
    df.shape[1] # number of columns (feature count)
    np.unique(df) # non-duplicate values in the table
    len(np.unique(df)) # number of unique values
    ```

### 2. Missing Value Handling
- **Deletion**
    - Useful when missing ratio is small and missingness is random
    - Can easily break time-series structure
    ```python
    df.dropna() # drop rows with any missing value
    df.dropna(axis=1) # drop columns with any missing value
    ```
- **Default-value imputation**
    - For categorical features, use business-meaningful values like Unknown
    - For time-series missing numerical values, often avoid hard filling
    ```python
    df['gender'] = df['gender'].fillna('Male')

    df = df.fillna({
    'income': 0,
    'gender': 'Unknown'
    }) # fill multiple columns by dict
    ``` 
- **Statistical imputation**
    ```python
    df['income'] = df['income'].fillna(df['income'].mean()) # mean imputation
    df['income'] = df['income'].fillna(df['income'].median()) # median imputation
    df['gender'] = df['gender'].fillna(df['gender'].mode()[0]) # mode imputation, [0] takes the first mode

    df = df.fillna({
        'income': df['income'].median(),
        'age': df['age'].median(),
        'gender': df['gender'].mode()[0]
    }) # fill multiple columns at once
    ```
    - Mode: suitable for categorical features with few classes and dominant category
    - Median: suitable for skewed numeric features; tends to shrink variance
    - Mean: suitable for near-symmetric numeric features with few outliers; also shrinks variance

- **Missing indicator variable**
    - Steps:
        - 1. Create an indicator variable as a new feature to mark missingness (BOOL).
        - 2. Impute the original feature.
        - 3. Use both the imputed **original feature** and the **indicator variable** for modeling.

    - Suitable for non-random missingness, e.g., respondents hiding assets
    - Can introduce noise when data are MCAR
    - For categorical columns, Unknown is often used instead of an indicator
    ```python
    df['income_missing'] = df['income'].isna().astype(int) # create indicator
    df['income'] = df['income'].fillna(df['income'].median()) 
    ```
- **Group-wise imputation**
    ```python
    df['salary'] = (
        df.groupby('industry')['salary'] # group data by industry, then take salary within each group
        .transform(lambda x: x.fillna(x.median())) # transform each group and keep same length
    ) # x in lambda is the salary series of each group
    ```
    - Suitable when group differences are obvious
    - Not suitable when groups are too small or grouping key itself is missing

- **Interpolation**
    ```python
    df['price'].interpolate() # linear interpolation
    df['price'].ffill() # forward fill
    df['price'].bfill() # backward fill
    df['price'].interpolate(method='time') # time interpolation, requires ordered time index
    ```
- **Polynomial interpolation**
    - Principle: there exists a polynomial of degree n-1 passing through n points
    - Degree 1 equals linear interpolation between two points
    - Prone to Runge phenomenon: high-order curves oscillate heavily
    - A single point can affect the whole curve globally
    ```python
    df['price'].interpolate(method='polynomial', order=2) # order = polynomial degree
    ```
- **Spline interpolation**

    - Steps:
        - 1. Stitch several polynomials into a smooth curve
        - 2. Given degree n, use nearest n+1 points and split into n segments
        - 3. Ensure equal value and continuous 1st to (n-1)-th derivatives at junctions
        - 4. Solve the equation system to obtain those polynomials

    - Suitable for time-series with smooth continuous changes, sparse missing points, nonlinearity, and trend focus.
    - Not suitable for long missing intervals, structural breaks (policy/crisis), jump variables, tail-risk regimes, or unordered data.

    - **Why cubic spline is widely used in industry**
        - Cubic spline means third-order spline interpolation
        - It keeps value, first derivative, and second derivative continuous
        - First-derivative continuity avoids corners; second-derivative continuity makes curvature natural
        - Higher order can be smoother but more noise-sensitive and numerically unstable
        - Cubic polynomial has stronger locality and is less affected by far-away points
    ```python
    # pandas
    df['price'].interpolate(method='spline', order=3)  
    ```
    ```python
    # scipy low-level
    from scipy.interpolate import CubicSpline
    x = df.index.astype('int64') # index must be numeric
    y = df['price'] # target feature to impute

    cs = CubicSpline(x[~y.isna()], y.dropna()) # cs = CubicSpline(x_known, y_known)
    df.loc[y.isna(), 'price'] = cs(x[y.isna()]) # fill missing rows with spline output
    ```
- **Model-based imputation**
    - Methods include linear regression, KNN, decision tree, random forest, LightGBM, GBDT, deep learning, etc.
    - In industry, random forest + missing indicator is commonly used.
    - Suitable when variables are strongly related and features are many/nonlinear.

    ```python
    # example: random-forest imputation
    # this sample has data leakage: train data used for imputation may overlap future test set
    # in practice, imputation training samples should align with downstream model split
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
### 3. Detecting Outliers
- **Descriptive statistics**
    ```python
    df.describe()
    ```
    - Check whether min/max are implausible
    - Check whether std is too large
    - Check whether mean and median differ greatly

- **Distribution visualization**

    - **Histogram**
        ```python
    import numpy as np
    np.log1p(df['price']).hist(bins=50)
        ```
        - Check overall shape: normal or skewed
        - Whether it is long-tail or heavy-tail; outliers often appear in tails.
            > A long tail usually means the right side stretches far; mean $>>$ median
            > A heavy tail means extreme values occur more frequently
            > Multiplicative accumulation effect, resource imbalance, and rare events can cause it
            > Mean/variance become unstable, gradients become unstable, and models over-focus on extreme samples
        - Whether it is sharply peaked
            > Placeholders, lower-bound truncation (values below threshold set to 0), business rules (e.g., free transactions)
            > Discontinuous distribution leads to fake signals and distorted mean/variance
        - Whether it is multimodal
            > Mixed populations or mechanisms (mixture distributions)
            > Model cannot learn one consistent rule; mean loses representativeness
    <div style="text-align:center;">            
    <img src="pic/Hist.png" width="70%">
    </div>
   
    - **Log-scale histogram**
        ```python
    import numpy as np
    np.log1p(df['price']).hist(bins=50)
        ```
        - Monetary amount / volume data are often long-tailed; use $\log(1+x)$ transform.
        - If data look more normal after log transform, original data were long-tailed.
        - If isolated extreme points remain after log transform, true extreme anomalies may exist.

    - **Box plot**
        ```python
    import matplotlib.pyplot as plt
    df.boxplot(column='price')
    plt.show()
        ```
        - $Q1$: 25th percentile  
            $Q3$: 75th percentile  
            $IQR = Q3 - Q1$

        - Outlier definition:
        \[x < Q1 - 1.5 \times IQR \quad \text{or} \quad x > Q3 + 1.5 \times IQR\]

        <div style="text-align:center;">
        <img src="pic/Box_Plot.png" width="50%">
        </div>

        - Center line in box: median
          Top/bottom of box: $Q1 / Q3$
          Upper/lower whiskers: range within $1.5\times IQR$
          Points outside whiskers: outliers

        - Good for medium sample size and fast univariate scan; no normality assumption needed.
          Not ideal for extremely long-tailed data (too many flagged points).

        - Manual outlier filtering via box-plot rule
            ```python
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df['price'] < lower) | (df['price'] > upper)]
            ```

    - **$Q-Q$ Plot**
        - Compares sample distribution with a theoretical distribution (usually normal)
        - X-axis: theoretical quantiles
        Y-axis: sample quantiles
        - Similar at each percentile -> similar distributions
        Large difference in tails -> different tail behavior
        <div style="text-align:center;">
        <img src="pic/QQ.png" width="70%">
        </div>
   
- **Z-score Method**
    - Z-score (standard score) measures how far a point is from the mean in standard-deviation units.

    \[z = \frac{x - \mu}{\sigma}\]

    - \(x\): sample value  
    \(\mu\): sample mean  
     \(\sigma\): sample standard deviation

    - When \(|z| > 2\), it is a potential outlier.
      When \(|z| > 3\), it is a strong outlier.
      In a normal distribution, about 99.7% points lie within $\pm3\sigma$.

    - Z-score is suitable for near-normal data; not suitable for skewed, long-tail, heavy-tail, or multimodal data.
    It is sensitive to outliers themselves, since mean/std are easily pulled; unstable on small samples.

- **MAD (Median Absolute Deviation) Method**
    - MAD is a robust dispersion measure against outliers.
    Unlike standard deviation, MAD uses median as center and is less affected by extremes.
        - 1. Compute absolute deviation from median for each sample  
        \[|x_i - \text{median}|\]
        - 2. Take median of absolute deviations  
        \[\text{MAD} = \text{median}(|x_i - \text{median}|)\]
        - MAD indicates how far a typical point is from median

    - MAD Robust Z-Score
        - Under normality, $MAD = \Phi^{-1}(0.75) = 0.6745 \sigma$.
          To align with standard deviation scale ($3\sigma$ rule), MAD is usually rescaled:

        \[z_{robust} = \frac{x - \text{median}}{1.4826 \times \text{MAD}}\]
        - 1.4826 is the normal-consistency factor, making MAD comparable to std.
          Common thresholds: \(|z_{robust}| > 3\) or \(> 3.5\)

    - MAD does not rely on normality assumptions; suitable for long-tail/heavy-tail data with many outliers,
      such as financial prices, transaction amounts, and user behavior data.
      Its median-centered criterion is robust and less “contaminated” by outliers.

### 4. Outlier Treatment
- **Deletion**
    - Directly remove clearly erroneous records

- **Winsorization**
    - Replace values beyond percentile cutoffs (e.g., two-sided 1%) with boundary values while keeping samples
        ```python
    df["income"] = df["income"].clip(
        lower=df["income"].quantile(0.01),
        upper=df["income"].quantile(0.99)
    )
        ```
- **Add indicator variable**
    - Mark extreme values and use “is extreme” as an additional feature
        ```python
    df["is_high_income"] = (df["income"] > threshold).astype(int)
        ```
- **Nonlinear transform**
    > Nonlinear transforms compress outlier impact on models. Goal: reduce leverage of extreme values, improve skew/long-tail behavior, and increase stability.

    - **Log transform**
        - Suitable for right-skewed and long-tailed data, e.g., amounts/income/volume/counts
        Data must be non-negative
        ```python
    import numpy as np
    df["x_log"] = np.log(df["x"])
    df["x_log1p"] = np.log1p(df["x"]) # safe when x contains 0
        ```
    - **Square-root transform**
        - Suitable for mild to moderate right skew and count-like data; milder than $\log$
        Data must be non-negative
        ```python
    import numpy as np
    df['x_sqrt'] = np.sqrt(df['x'])
    df['x_sqrt'] = np.sqrt(df['x'] + 1e-6) # safe near 0
        ```
    - **Box-Cox Transform**
        - Parameterized power transform that compresses large values; different \(\lambda\) gives different compression. Equivalent to $\log$ when \(\lambda = 0\).
        - Principle: choose \(\lambda\) so transformed data are closer to normal.
            > Target: $x^{(\lambda)} \sim \mathcal{N}(\mu, \sigma^2)$, estimate \(\lambda\) by MLE under normality
            > $$ \ell (\lambda) = -\frac{n}{2} \log{\hat{\sigma}^2_\lambda} + (\lambda - 1)\sum_{i=1}^{n} \log{x_i}$$
            
        - Interpretability decreases after transform
        - Input must be strictly positive; transform:
            $$ x^{(\lambda)} =
            \begin{cases}
            \dfrac{x^\lambda - 1}{\lambda}, & \lambda \neq 0 \\
            \log(x), & \lambda = 0
            \end{cases}$$
        - scipy code
            ```python
    from scipy.stats import boxcox

    amount_bc, lambda_hat = boxcox(df['amount']) # Box-Cox transform
    df['amount_boxcox'] = amount_bc # save result
    print("Optimal lambda:", lambda_hat)
            ```
        - sklearn code
            ```python
    from sklearn.preprocessing import PowerTransformer

    pt = PowerTransformer(method='box-cox', standardize=True)

    df['amount_boxcox'] = pt.fit_transform(df[['amount']])
            ```
    - **Yeo–Johnson Transform**
        - Also a power transform for skew/long-tail/outlier impact
        - Allows zero and negative values
        - For \(x \ge 0\):
        \[x^{(\lambda)} =
        \begin{cases}
        \frac{(x+1)^\lambda - 1}{\lambda}, & \lambda \neq 0 \\
        \log(x+1), & \lambda = 0
        \end{cases}\]
          For \(x < 0\):
        \[x^{(\lambda)} =
        \begin{cases}
        -\frac{(1-x)^{2-\lambda} - 1}{2-\lambda}, & \lambda \neq 2 \\
        -\log(1-x), & \lambda = 2
        \end{cases}\]
            > \(x+1\) allows x=0 and smooth transition at \(\lambda=0\). Subtracting 1 anchors the transform so \(\lambda=1\) is identity-like.
        - Suitable for skewed data, long tails, outlier leverage in linear models, and normality violations
        - scipy code
            ```python
    from scipy.stats import yeojohnson
    amount_yj, lambda_hat = yeojohnson(df['amount']) 

    df['amount_yeojohnson'] = amount_yj
    print("Optimal lambda:", lambda_hat)
            ```
        - sklearn code
            ```python
    from sklearn.preprocessing import PowerTransformer

    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    X_train_yj = pt.fit_transform(X_train[["x"]])
    X_test_yj  = pt.transform(X_test[["x"]])
            ```

### 5. Standardization
- Definition
    - Standardization transforms features to **mean 0, std 1**, usually via **Z-score standardization**:
        > Standardization is linear; it does not change distribution shape.

    \[x_{\text{std}} = \frac{x - \mu}{\sigma}\]

        - $\mu$: sample mean
        $\sigma$: sample standard deviation

- Standardize full feature set
    ```python
    # avoid data leakage: fit on train, transform on test
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # X_train: all numeric columns
    X_train_std = scaler.fit_transform(X_train) 
    X_test_std  = scaler.transform(X_test)
    ```
- Standardize selected columns
    ```python
    from sklearn.preprocessing import StandardScaler

    cols_to_scale = ["age", "income"]
    scaler = StandardScaler()
    
    # use copies for in-place assignment
    X_train_std = X_train.copy()
    X_test_std  = X_test.copy()

    X_train_std[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_std[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])
    ```


- Why standardization helps
    - Remove unit-scale effects so large-scale features do not dominate
    - Make distances/comparisons meaningful for distance/dot-product models
    - Speed up convergence and improve optimization stability
        > In gradient descent, scale mismatch causes different gradient magnitudes: $\frac{\partial L}{\partial w_j} = x_j (\hat{y} - y)$
    - Fair regularization across features under L1/L2
        > Regularization penalizes weights; small-scale features naturally need larger coefficients
- Applicable model families

    | Modeling mechanism | Models | Why standardization is needed |
    |---------|----------|------------------|
    | Distance-based | KNN, K-means, hierarchical clustering | Distance must be comparable; otherwise large-scale features dominate |
    | Dot-product-based | SVM, PCA, linear models | Dot products are scale-sensitive; directions get distorted |
    | Gradient-based optimization | Linear, Logistic, neural networks | Inconsistent gradient scale slows/unstabilizes convergence |
    | Regularization term | Ridge, Lasso, Elastic Net | L1/L2 penalty becomes unfair across scales |
- Why tree models do not need standardization:
   - Trees do not rely on units, distances, dot products, or gradient scale. They rely on ordering and split thresholds. Standardization preserves ordering, so split quality (IG/Gini/MSE reduction) is unchanged.

### 6. Normalization

- Definition
    - Normalization maps features linearly into a fixed range (usually $[0, 1]$), typically via **Min-Max normalization**:

    \[x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}\]

        - $x_{\max} / x_{\min}$: max / min of sample

- Normalize full feature set
    ```python
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler() 

    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm  = scaler.transform(X_test)
    ```

- Normalize selected columns
    ```python
    from sklearn.preprocessing import MinMaxScaler

    cols_to_scale = ["age", "income"]  
    scaler = MinMaxScaler()

    X_train_norm = X_train.copy()
    X_test_norm  = X_test.copy()

    X_train_norm[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_norm[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])
    ```

- Why normalization helps
    - Align value ranges across features with different units
    - Prevent overly large numeric values and improve numerical stability
    - Useful for naturally bounded features (ratios, probabilities, pixels)
    - Helpful for distance-based models when features share comparable semantic scales
        > Normalization does not remove skewness; it only rescales linearly

- Applicable model families
    | Modeling mechanism | Models | Why normalization is needed |
    |---------|------|------------------|
    | Distance-based | KNN, K-means | Distances are directly affected by feature ranges |
    | Neural nets + bounded activations | NN (Sigmoid / Tanh) | Inputs stay in sensitive activation ranges |
    | Image data | CNN | Pixel values have natural bounds |
    | Naturally bounded features | ratios, probabilities | Preserve interval semantics |

- Limitations
    - Extremely sensitive to outliers: extreme values compress the rest
    - Does not change distribution shape: skewness/multimodality remains
    - Unstable across datasets: test values can exceed [0,1] if outside train range

- Why tree models do not need normalization:
    - Trees depend only on ordering and split thresholds
    - Normalization is monotonic linear mapping and does not change sample order
    - Left/right split subsets remain identical; IG/Gini/MSE reduction remains unchanged

# Decision Trees
### 1. Information Theory

- **Self-information**
    - If event probability is $p(x)$, self-information is:
        $$
        I(x) = - \log_2 p(x)
        $$
    <div style="text-align:center;">
    <img src="pic/Info.png" width="40%">
    </div>

       - Information measures how much an event breaks prior expectation
       - **Lower-probability events carry more information**
       - In coding, rarer events require longer code lengths. Under optimal coding, equiprobable events with probabilities $\frac{1}{2}, \frac{1}{4},\frac{1}{8}$ need 1, 2, 3 bits respectively
    <br>
- **Entropy**

    - Entropy measures uncertainty. More uncertainty -> higher entropy
    - For a discrete random variable $Y$ with classes and probabilities $p_1, p_2, ..., p_k$, entropy is:
        \[H(Y) = - \sum_{i=1}^{k} p_i \log_2 p_i\]
      i.e., average information:
        $$H(Y) = \mathbb{E}[-\log_2 p(Y)]$$
    - For continuous random variables, differential entropy is:

        $$
        H(Y) = - \int_{-\infty}^{\infty} f(y) \log f(y)\, dx
        $$
        - Differential entropy uses base $e$, so unit is nat (not bit)

    - If one class has probability 1, uncertainty is zero and entropy is 0
    - In multiclass setting, entropy is maximized by uniform distribution: $H_{\max} = \log_2 k$

- **Conditional entropy**
    - Let $X$ be a feature and $Y$ the target. $H(Y|X)$ is residual uncertainty of $Y$ given $X$.
    - Computed as weighted average entropy over all $X$ states:
    \[H(Y \mid X) = \sum_{x} p(x)\, H(Y \mid X=x)\] 
      where:
    \[H(Y \mid X=x) = -\sum_{y} p(y \mid x)\log p(y \mid x)\]

- **Information gain**
    - Measures uncertainty reduction by a feature (mutual information)
    - Formula: original entropy minus conditional entropy
        \[
        IG(Y, X) = H(Y) - H(Y\mid X)
        \]
        - $X$ is feature, $Y$ is target
        - Larger \(IG\) means stronger explanatory power of \(X\) for \(Y\)
        - \(IG=0\) means \(X\) provides no information about \(Y\) (independence)

- **Gini index**
    - Measures impurity of a dataset
    - Can be interpreted as probability that two randomly drawn samples have different classes:
    $$
    Gini = 1 - \sum_{i=1}^{k} p_i^2
    $$
    - Gini = 0 means pure set; under uniform class distribution Gini is maximal

### 2. ID3 Decision Tree
- A classification algorithm. Core idea: split data to make nodes purer and labels more certain. Split quality uses information gain; larger IG is better.

- ID3 procedure:
    - 1. Compute dataset entropy
    - 2. Compute conditional entropy for each feature split
    - 3. Compute IG = entropy - conditional entropy for each feature, pick max IG
    - 4. Repeat recursively until no feature remains / entropy is 0 / sample count is 0

- Limitations of ID3:
    - Features with many categories tend to get larger information gain
    - Handles discrete features only
    - Prone to overfitting; no pruning in original form
        > Why ID3 overfits:
        >   1. Tree grows too deep until near-pure leaves
        >   2. Greedy information-gain split gives local optimum
        >   3. Greedy criterion does not evaluate generalization
        >   4. Bias toward high-cardinality features (e.g., User_ID) can memorize data directly

### 3. C4.5 Decision Tree
- Same core idea as ID3, but uses gain ratio to mitigate high-cardinality feature bias.

- C4.5 procedure:
    - 1. Compute dataset entropy
    - 2. Compute conditional entropy of each feature split
    - 3. Compute information gain
    - 4. Compute split information
    - 5. Compute gain ratio and split by maximum value
    - 6. Repeat recursively until stopping conditions are met
    - 7. Prune

- Gain ratio:
    - Information gain divided by feature entropy:
    \[
    GainRatio(D,A)=\frac{IG(D,A)}{H_A(D)}
    \]
        - D: dataset
        - A: splitting feature
        - $H_A(D)$: split information (feature entropy under empirical distribution in D)
            > Equivalent to entropy $H(X)$ in information theory, but empirical on D rather than true distribution

- Limitations of C4.5:
    - Greedy local optimum
    - Computationally heavier
    - Trees may still be deep and noise-sensitive

### 4. CART Decision Tree
- CART repeatedly performs binary splits to make child nodes purer/more stable.
- CART handles both classification and regression.
  For classification, use Gini-based split; choose minimum Gini.
  For regression, use MSE/SSE-based split; choose minimum loss.

- CART procedure:
    - 1. Traverse all features and enumerate all possible binary splits
    - 2. Compute Gini or MSE for each split
    - 3. Select best split node
    - 4. Stop if stopping rule is met; otherwise build subtrees recursively
    - 5. Cost-complexity pruning

- **CART Classification Tree:**
    - For discrete features:
        - 1. Enumerate all binary partitions; with $k$ categories there are $2^{k-1}-1$ partitions
        - 2. Compute weighted split Gini:
            \[Gini_{split}(X_j, t) = \frac{|D_L|}{|D|} Gini(D_L) + \frac{|D_R|}{|D|} Gini(D_R)\]
            > $\frac{|D_L|}{|D|}, \frac{|D_R|}{|D|}$ are left/right sample ratios
            > $Gini(D_L), Gini(D_R)$ are child-node impurities
            > Feature only defines split; impurity is computed on node sample labels
        - 3. Pick split with minimum Gini

    - For continuous features:
        - 1. Sort values ascending
        - 2. Use midpoints of adjacent distinct values as candidate thresholds
        - 3. Compute weighted split Gini for each threshold
        - 4. Pick minimum Gini threshold

- **CART Regression Tree:**
    - Regression tree predicts continuous targets by partitioning feature space and assigning a constant to each region.
        - 1. Enumerate candidate split points on all features
        - 2. Compute split loss:
            \[Loss_{split}(X_j, t) = Loss(D_L) + Loss(D_R)\]
            > $Loss(D_L), Loss(D_R)$ are regression losses of child nodes
            > Often measured by SSE:
            >    \[SSE(D)=\sum_{i\in D}\left(y_i-\bar{y}_D\right)^2\]
            > Feature only controls partition; loss is computed from node samples
        - 3. Choose split with minimum SSE

    - Prediction at leaf:
        - Output mean target value of the leaf node:
            \[\hat{y}(x)=\bar{y}_{D_{\text{leaf}}}\]
            > CART regression typically uses SSE (sum), so no extra weighting by child sample size is needed. If using MSE, weighting is required.
### 5. Pruning
- Core goal: reduce overfitting and improve generalization.

- **Pre-pruning / Early stopping**
    - Stop tree growth during training by checking gain before each split.
    - Common controls:
        -   max_depth
        -   min_samples_split
        -   min_samples_leaf
        -   min_gain / min_impurity_decrease

    - Example
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

- **Post-pruning**
    - Grow a full tree first, then check subtrees bottom-up. If keeping a subtree yields larger error than replacing it, prune it.

- **Pessimistic Error Pruning**
    - Used in C4.5 post-pruning. Training error is $\frac{e}{n}$; pessimistic estimate inflates it to $\frac{e+0.5}{n}$, penalizing small-sample leaves likely to overfit.
    - It evaluates using training data only, using statistical correction to estimate future error.
    - Rationale:
        - 1. In a leaf, each prediction is correct/incorrect, so error count follows binomial $E \sim Bino(n,p)$.
        - 2. Point estimate $\hat{p}=\frac{e}{n}$ is optimistic. Use an upper confidence bound $e_{up}$ such that $P(E \leqslant e_{up}) = 1 - \alpha$.
        - 3. With large sample (np>5 and nq>5), binomial approximates normal: $E \approx N(np,npq)$.
        - 4. Standardization:
        $$Z = \frac{E - np}{\sqrt{npq}} \approx N(0,1)$$ 
        - 5. Normal is continuous while binomial is discrete; use continuity correction:
        $$P(E \leqslant e_{up}) = P(X \leqslant e_{up} + 0.5)$$
        - 6. Therefore C4.5’s $\frac{e+0.5}{n}$ is a conservative future-error estimate based on normal approximation + continuity correction.

- **Cost-Complexity Pruning**
    - Used in CART post-pruning. Build maximal tree first, then generate nested subtrees and select the best by validation/cross-validation.
    - Criterion is cost complexity $R_\alpha(T)$, i.e., empirical risk with regularization.

    - Procedure:
        - 1. Build maximal tree
        - 2. For each internal node, compute pre/post-pruning empirical risk and leaf count
        - 3. Compute critical penalty coefficient for each node, prune the node with smallest $\alpha_t$
        - 4. Repeat on new tree via weakest-link strategy to get nested subtree sequence
        - 5. Choose subtree with minimum CV error
        
    - CART objective for pruning:
    \[
    R_\alpha(T) = R(T) + \alpha |T|
    \]
        - $R(T)$ is empirical risk. For classification usually error rate (sometimes Gini/cross-entropy); for regression usually SSE.
        - $|T|$ is number of leaves, used as complexity penalty

    - Critical penalty coefficient
        \[
        \alpha_t = \frac{R(t)-R(T_t)}{|T_t|-1}
        \]
        - Numerator \(R(t)-R(T_t)\): increase in empirical risk if subtree is replaced by one leaf
        - Denominator \(|T_t|-1\): number of leaves removed by pruning
        - So \(\alpha_t\) is error increase cost per one removed leaf
        - If chosen $\alpha$ is below this threshold, penalty is weak and keeping subtree is better

# To be updated...
