---
layout: post
title: What is Multicollinearity?
tags: models
---

<!-- more -->

---
Table of content:

{: class="table-of-content"}
* TOC
{:toc}

---
---

### 1. What is Multicollinearity? Please explain with examples and discuss which models are suitable for this situation and why.

**Multicollinearity** refers to a phenomenon in multiple regression models where two or more independent variables (predictors) are highly linearly correlated with each other. This high correlation makes it difficult for the model to distinguish the individual effects of each predictor on the dependent variable. As a result, the Ordinary Least Squares (OLS) coefficient estimates become unstable, with inflated variances, large standard errors, and sometimes coefficients with incorrect signs that contradict economic or theoretical expectations.

#### Mathematical Definition
Consider the linear regression model:
\[
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon$$
\]
When predictors $$\(X_i\) and \(X_j\)$$ are highly correlated (e.g., correlation coefficient > 0.8 or 0.9), the matrix \(X^T X\) becomes near-singular. This causes the variance-covariance matrix of the coefficient estimates:
$$\[
\text{Var}(\hat{\beta}) = \sigma^2 (X^T X)^{-1}
\]$$
to have greatly inflated diagonal elements, leading to large standard errors and insignificant t-tests, even when the overall model has a high \(R^2\).

#### Example
Suppose we want to predict **house prices** (\(Y\)) using the following predictors:
- \(X_1\): Floor area (square meters)
- \(X_2\): Number of bedrooms
- \(X_3\): Living room area

In real data, floor area and number of bedrooms are usually highly positively correlated (larger houses tend to have more bedrooms), with a correlation often exceeding 0.85. This creates multicollinearity:
- The estimated coefficient \(\hat{\beta_1}\) (effect of floor area) becomes unstable and may even change sign across different samples.
- Even if the model has strong explanatory power (Adjusted \(R^2\) close to 0.9), individual p-values may all exceed 0.05, making it impossible to determine whether floor area or number of bedrooms truly drives the price.
- Predictions for new houses become highly sensitive to small changes in the coefficients.

The core issue is that while the model may predict well overall, the **interpretation and stability** of individual coefficients are severely compromised.

#### Suitable Models for Handling Multicollinearity and Why
When multicollinearity is present, traditional OLS regression is not recommended. Instead, consider the following models:

1. **Ridge Regression** — The most commonly recommended solution  
   \[
   \hat{\beta}^{\text{Ridge}} = \arg\min_{\beta} \left( \sum (y_i - \hat{y}_i)^2 + \lambda \sum \beta_j^2 \right)
   \]
   - **Reason**: The L2 penalty (\(\lambda \sum \beta_j^2\)) shrinks coefficients toward zero, reducing variance and stabilizing estimates even when \(X^T X\) is ill-conditioned. It improves prediction accuracy with moderate bias.
   - Best for: Moderate number of predictors when you still want to keep all variables for interpretation.

2. **Lasso Regression**  
   \[
   \hat{\beta}^{\text{Lasso}} = \arg\min_{\beta} \left( \sum (y_i - \hat{y}_i)^2 + \lambda \sum |\beta_j| \right)
   \]
   - **Reason**: The L1 penalty not only shrinks coefficients but can also set some to exactly zero, performing automatic feature selection and removing redundant correlated variables.
   - Best for: High-dimensional data where variable selection is desired.

3. **Elastic Net**  
   Combines Ridge and Lasso penalties. It offers both shrinkage and selection, performing especially well when predictors are grouped in correlated clusters.

4. **Principal Component Regression (PCR) or Partial Least Squares (PLS)**  
   - **Reason**: These methods first transform the correlated predictors into uncorrelated principal components via PCA, then perform regression on the new components. This completely eliminates multicollinearity, though at the cost of interpretability.
   - Best for: Very high-dimensional data where prediction is the main goal.

5. **Tree-based Models (e.g., Random Forest, XGBoost)**  
   - **Reason**: These non-parametric models do not rely on matrix inversion and do not assume independence among predictors, making them naturally robust to multicollinearity.
   - Best for: Situations with strong non-linear relationships.

**Quick Diagnosis & Prevention**:  
- Use Variance Inflation Factor (VIF). VIF > 5 (conservative) or > 10 indicates severe multicollinearity.  
- Check correlation matrices and condition numbers beforehand.  
- Consider removing one of the highly correlated variables or applying regularization.

**Summary**: Ridge Regression is usually the best starting point for multicollinearity because it stabilizes coefficients without excessive bias. Use Lasso or Elastic Net if variable selection is also needed. The choice depends on whether your priority is prediction accuracy or coefficient interpretability.

---

### 2. Explain the VIF Calculation Method

**VIF (Variance Inflation Factor)** is one of the most widely used diagnostics for detecting multicollinearity in multiple linear regression. It measures how much the variance of a regression coefficient is inflated due to linear correlations with other predictors.

#### Mathematical Formula
For the \( i \)-th predictor \( X_i \), the VIF is:
\[
VIF_i = \frac{1}{1 - R_i^2}
\]
where \( R_i^2 \) is the coefficient of determination from the **auxiliary regression** — a regression where \( X_i \) is temporarily treated as the dependent variable and regressed on all the other independent variables in the model.

The **Tolerance** is simply the reciprocal:
\[
\text{Tolerance}_i = 1 - R_i^2 = \frac{1}{VIF_i}
\]

#### Calculation Steps (Example with 3 predictors)
Assume the model is: \( Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \epsilon \)

1. Perform auxiliary regressions for each predictor:
   - Regress \( X_1 \) on \( X_2 \) and \( X_3 \) → obtain \( R_1^2 \)
   - Regress \( X_2 \) on \( X_1 \) and \( X_3 \) → obtain \( R_2^2 \)
   - Regress \( X_3 \) on \( X_1 \) and \( X_2 \) → obtain \( R_3^2 \)

2. Compute VIF for each:
   - \( VIF_1 = 1 / (1 - R_1^2) \)
   - \( VIF_2 = 1 / (1 - R_2^2) \)
   - \( VIF_3 = 1 / (1 - R_3^2) \)

In practice, statistical software (e.g., Python’s statsmodels, R’s car package) computes VIF automatically.

#### Interpretation Guidelines
- **VIF = 1**: No multicollinearity.
- **1 < VIF ≤ 5**: Moderate multicollinearity, often acceptable.
- **VIF > 5** (strict) or **VIF > 10** (lenient): Severe multicollinearity. Coefficients become unstable with large standard errors.

#### Numerical Example
- Auxiliary \( R_1^2 = 0.80 \) → \( VIF_1 = 5 \)
- Auxiliary \( R_2^2 = 0.90 \) → \( VIF_2 = 10 \)
- Auxiliary \( R_3^2 = 0.60 \) → \( VIF_3 = 2.5 \)

Here, \( X_2 \) shows severe multicollinearity (VIF=10).

VIF directly links to the variance of OLS coefficients:
\[
\text{Var}(\hat{\beta}_i) \propto \frac{1}{1 - R_i^2}
\]
Higher \( R_i^2 \) means higher VIF and more inflated variance.

---

### 3. Introduce the Condition Number Diagnostic Method

The **Condition Number** (also called Condition Index) is another important diagnostic tool for multicollinearity. It assesses the numerical stability of the entire design matrix from a global perspective and complements VIF (which is variable-specific).

#### Concept
In the model \( Y = X\beta + \epsilon \), let \( X \) be the \( n \times (p+1) \) design matrix (including intercept). The condition number \(\kappa\) is defined as the ratio of the largest to smallest singular value of \( X \) (or \( X^T X \)):

\[
\kappa(X) = \frac{\sigma_{\max}}{\sigma_{\min}}
\]

where \(\sigma_{\max}\) and \(\sigma_{\min}\) are the maximum and minimum singular values. Equivalently, it can be expressed using eigenvalues \(\lambda\) of \( X^T X \):
\[
\kappa = \sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}}
\]

- \(\kappa \approx 1\): The matrix is well-conditioned (orthogonal predictors, no multicollinearity).
- Larger \(\kappa\): The matrix is ill-conditioned (near singular), meaning small changes in data can cause large swings in coefficient estimates.

#### Calculation Steps
1. Standardize the predictors (recommended to reduce scale sensitivity).
2. Perform Singular Value Decomposition (SVD) or eigenvalue decomposition on \( X \) or \( X^T X \).
3. Compute the ratio of the largest to smallest singular value/eigenvalue.
4. For advanced diagnosis (Belsley’s method): Examine each **Condition Index** and the associated **Variance Decomposition Proportions (VDP)**. A condition index > 30 combined with two or more variables having VDP > 0.5 indicates severe multicollinearity involving those variables.

Most software outputs the condition number directly (e.g., in statsmodels `summary()`, R’s `kappa()`, or SPSS collinearity diagnostics).

#### Interpretation Guidelines
- **< 10**: Little to no multicollinearity.
- **10 – 30**: Moderate multicollinearity.
- **> 30**: Severe multicollinearity (coefficients unstable).
- Some sources use higher thresholds: < 100 (acceptable), 100–1000 (moderate to severe), > 1000 (extreme).

**Note**: Condition number is sensitive to variable scaling, so standardization or using the correlation matrix is strongly advised.

#### Comparison with VIF
- **VIF**: Identifies *which specific variable* is affected.
- **Condition Number**: Provides an *overall measure* of multicollinearity severity and can detect higher-order dependencies that pairwise VIFs might miss.
- **Recommendation**: Use both together.

#### Practical Advice
When the condition number is high:
- Remove one of the highly correlated variables.
- Use Principal Component Analysis (PCA).
- Switch to regularized models such as **Ridge Regression** (particularly robust against ill-conditioned matrices).
- Collect more data or create composite variables.

The condition number is a standard tool from numerical linear algebra for detecting ill-conditioned problems and forms a complete multicollinearity diagnostic toolkit when combined with VIF.

---

### 1. 何謂 多重共線性（Multicollinearity）? 請舉例說明並解釋何種模型適合用於此情形以及其原因？

多重共線性（Multicollinearity）是指在多變量迴歸模型（Multiple Regression Model）中，兩個或多個自變數（獨立變數，Independent Variables）之間存在高度線性相關（High Linear Correlation）的現象。這會導致模型無法清楚區分各個自變數對依變數（Dependent Variable）的個別影響，使得普通最小平方法（Ordinary Least Squares, OLS）的係數估計變得不穩定、變異數（Variance）大幅膨脹，甚至係數符號可能與實際經濟意義相反。

#### 簡單數學定義
假設線性迴歸模型為：
\[
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon
\]
當自變數 \(X_i\) 與 \(X_j\) 高度相關（例如相關係數 \(r > 0.8\) 或 \(0.9\)），則設計矩陣 \(X^T X\) 接近奇異（Near Singular），導致係數 \(\hat{\beta}\) 的變異數–共變異數矩陣：
\[
\text{Var}(\hat{\beta}) = \sigma^2 (X^T X)^{-1}
\]
中的對角元素大幅增加，係數估計的標準誤（Standard Error）暴增，\(t\)-檢定變得不顯著，即使整體模型 \(R^2\) 很高也一樣。

#### 舉例說明
假設我們想用多元線性迴歸預測**房屋價格**（\(Y\)），自變數包含：
- \(X_1\): 房屋坪數（平方公尺）
- \(X_2\): 臥室數量
- \(X_3\): 客廳面積

在真實資料中，「坪數」與「臥室數量」通常高度正相關（坪數越大，臥室通常越多），相關係數可能高達 0.85 以上。此時模型會出現多重共線性：
- OLS 估計出的 \(\hat{\beta_1}\)（坪數的影響）可能變得不穩定，甚至在不同樣本中正負號翻轉。
- 即使整體模型解釋力很高（Adjusted \(R^2\) 接近 0.9），個別變數的 \(p\)-value 卻都 > 0.05，無法判斷「到底是坪數還是臥室數影響價格」。
- 預測新房屋時，係數小小變動就會導致價格預測值劇烈波動。

這就是典型的多重共線性問題：模型整體預測能力不差，但**係數解釋**與**穩定性**嚴重受損。

#### 適合處理多重共線性的模型及其原因
當資料存在多重共線性時，**不建議直接使用傳統 OLS 線性迴歸**，而應改用以下模型（依適用情境排序）：

1. **Ridge Regression（嶺迴歸）** —— 最常用且最直接的解決方案  
   模型形式：
   \[
   \hat{\beta}^{\text{Ridge}} = \arg\min_{\beta} \left( \sum_{i=1}^n (y_i - \beta_0 - \sum \beta_j x_{ij})^2 + \lambda \sum_{j=1}^p \beta_j^2 \right)
   \]
   - **原因**：加入 L2 正則化項（\(\lambda \sum \beta_j^2\)），強迫係數向 0 收縮（Shrinkage），有效降低係數的變異數。當 \(\lambda\) 適中時，即使 \(X^T X\) 接近奇異，模型仍能穩定估計係數，且預測誤差通常比 OLS 低。  
   - 適合情境：自變數數量中等、需要保留所有變數做解釋時（係數不會被完全消除）。

2. **Lasso Regression（Lasso 迴歸）**  
   模型形式：
   \[
   \hat{\beta}^{\text{Lasso}} = \arg\min_{\beta} \left( \sum (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\beta_j| \right)
   \]
   - **原因**：L1 正則化不僅縮小係數，還能將高度共線的自變數其中之一的係數直接壓縮為 0，達到**自動變數選擇**（Feature Selection）的效果，同時解決共線性。  
   - 適合情境：自變數很多、希望模型自動挑選最重要變數時。

3. **Elastic Net（彈性網）**  
   結合 Ridge + Lasso：
   \[
   \text{Loss} = \text{RSS} + \lambda_1 \sum |\beta_j| + \lambda_2 \sum \beta_j^2
   \]
   - **原因**：同時擁有 L1 的變數選擇與 L2 的穩定性，尤其在自變數群高度分組共線（Grouped Multicollinearity）時表現最佳。

4. **主成分迴歸（Principal Component Regression, PCR）或偏最小平方法（Partial Least Squares, PLS）**  
   - **原因**：先對自變數做主成分分析（PCA），將高度相關的自變數轉換成互不相關的主成分（Principal Components），再用這些新變數做迴歸。完全消除多重共線性，但會犧牲部分可解釋性（主成分不易解釋）。  
   - 適合情境：自變數維度很高（High-Dimensional Data），且預測為主要目標而非係數解釋。

5. **樹系模型（Tree-based Models）如 Random Forest、XGBoost**  
   - **原因**：這些模型不依賴矩陣求逆（Non-parametric），也不假設自變數彼此獨立，因此天生對多重共線性較不敏感。它們會透過特徵重要性（Feature Importance）來處理相關變數，不會出現係數不穩定的問題。  
   - 適合情境：資料非線性關係強烈、想同時處理共線性與非線性時（但解釋性較差）。

#### 快速診斷與預防
- **診斷**：計算 Variance Inflation Factor (VIF)，VIF > 5（保守）或 > 10 即表示有嚴重共線性。
- **預防**：事先檢查 Pearson 相關矩陣、條件數（Condition Number），或直接移除其中一個高度相關變數（但可能損失資訊）。

總結：**Ridge Regression 是處理多重共線性最推薦的起點**，因為它在不犧牲太多偏誤的情況下大幅提升穩定性；若還需要變數選擇，則改用 Lasso 或 Elastic Net。選擇模型時要依據你的目標是「預測準確」還是「係數可解釋」來決定。

### 2. 解釋VIF計算方法

**VIF（Variance Inflation Factor，變異數膨脹因子）** 是檢測多元線性迴歸模型中**多重共線性（Multicollinearity）** 嚴重程度的最常用指標。它量化了某一個自變數的係數估計變異數（Variance），因為其他自變數之間的線性相關而被「膨脹」了多少倍。

#### VIF (Variance Inflation Factor) 的數學公式
對於第 \( i \) 個自變數 \( X_i \)，其 VIF 計算公式為：

\[
VIF_i = \frac{1}{1 - R_i^2}
\]

其中：
- \( R_i^2 \) 是**輔助回歸（Auxiliary Regression）** 的決定係數（Coefficient of Determination）。
- 輔助回歸是指：將 \( X_i \) 當作**暫時的依變數**，用模型中**其餘所有自變數**作為解釋變數，對它進行一次普通的線性迴歸，所得到的 \( R^2 \) 值。

**容忍度（Tolerance）** 是 VIF 的倒數：
\[
\text{Tolerance}_i = 1 - R_i^2 = \frac{1}{VIF_i}
\]

VIF = 1 表示完全無共線性；VIF 越大，表示該變數與其他變數的線性相關越強，其係數估計的變異數被膨脹得越嚴重。

#### VIF 的計算步驟（以 3 個自變數為例）
假設模型為：\( Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \epsilon \)

1. **對每個自變數逐一做輔助回歸**：
   - 第一次：以 \( X_1 \) 為依變數，用 \( X_2, X_3 \) 做迴歸 → 得到 \( R_1^2 \)
   - 第二次：以 \( X_2 \) 為依變數，用 \( X_1, X_3 \) 做迴歸 → 得到 \( R_2^2 \)
   - 第三次：以 \( X_3 \) 為依變數，用 \( X_1, X_2 \) 做迴歸 → 得到 \( R_3^2 \)

2. **套入公式計算 VIF**：
   - \( VIF_1 = \frac{1}{1 - R_1^2} \)
   - \( VIF_2 = \frac{1}{1 - R_2^2} \)
   - \( VIF_3 = \frac{1}{1 - R_3^2} \)

在實際操作中，通常使用統計軟體（如 Python 的 statsmodels、R 的 car 套件、SPSS、Stata）直接計算，無需手動跑多次迴歸。

#### 判斷標準（常見經驗法則）
- **VIF = 1**：該變數與其他自變數完全無線性相關，無共線性問題。
- **1 < VIF ≤ 5**：輕度或中度共線性，通常還可以接受。
- **VIF > 5**（較嚴格）或 **VIF > 10**（較寬鬆）：表示存在嚴重多重共線性，係數估計會變得不穩定，標準誤變大，\( t \)-檢定容易不顯著。
- 有些文獻也看 Tolerance < 0.2（或 < 0.1）即有問題。

**注意**：即使整體模型的 \( R^2 \) 很高，只要個別 VIF 很高，係數的解釋力和穩定性就會受影響。

#### 簡單數值範例
假設有三個自變數，計算後得到：
- 對 \( X_1 \) 的輔助迴歸 \( R_1^2 = 0.80 \) → \( VIF_1 = 1 / (1 - 0.80) = 5 \)
- 對 \( X_2 \) 的輔助迴歸 \( R_2^2 = 0.90 \) → \( VIF_2 = 1 / (1 - 0.90) = 10 \)
- 對 \( X_3 \) 的輔助迴歸 \( R_3^2 = 0.60 \) → \( VIF_3 = 1 / (1 - 0.60) = 2.5 \)

此時 \( X_2 \) 的 VIF = 10，表示嚴重共線性，建議考慮移除 \( X_2 \)、合併變數、或改用 Ridge Regression 等方法處理。

#### 為什麼 VIF 與多重共線性有關？
回想 OLS 係數的變異數公式：
\[
\text{Var}(\hat{\beta}_i) = \sigma^2 \cdot \frac{1}{SST_i (1 - R_i^2)}
\]
其中 \( 1 - R_i^2 \) 越小（即 \( R_i^2 \) 越大），變異數就越大，VIF 正是把這個膨脹效果量化成倍數。

VIF 是診斷多重共線性的標準工具，計算簡單且直觀。當你發現多個 VIF 值偏高時，通常需要進一步處理（如變數選擇、正則化迴歸等），才能讓模型的係數估計更可靠。 


### 3. 介紹條件數診斷方法

**條件數（Condition Number）** 是診斷多元線性迴歸模型中**多重共線性（Multicollinearity）** 嚴重程度的另一個重要指標。它從整體（Global）角度評估設計矩陣（Design Matrix）的穩定性，與 VIF（針對單一變數）互補使用。

#### 條件數的基本概念 Condition Number
在多元線性迴歸中，模型為 \( Y = X\beta + \epsilon \)，其中 \( X \) 是 \( n \times (p+1) \) 的設計矩陣（包含截距項）。  
條件數 \(\kappa\)（或稱 Condition Index）定義為設計矩陣 \( X \)（或其相關矩陣、\( X^T X \)）的**最大奇異值（Singular Value）** 與**最小奇異值** 的比率：

\[
\kappa(X) = \frac{\sigma_{\max}}{\sigma_{\min}}
\]

其中 \(\sigma_{\max}\) 和 \(\sigma_{\min}\) 分別是 \( X \) 的最大與最小奇異值（等同於 \( X^T X \) 特徵值 \(\lambda\) 的平方根之比：\(\kappa = \sqrt{\lambda_{\max} / \lambda_{\min}}\)）。

- \(\kappa = 1\)：矩陣完全正交（Orthogonal），無共線性，數值穩定性最佳。
- \(\kappa\) 越大：矩陣越接近奇異（Near Singular），表示自變數之間存在高度線性相依，小小的資料擾動就會導致係數估計 \(\hat{\beta}\) 劇烈變化（不穩定）。

這正是多重共線性的核心問題：設計矩陣 \( X^T X \) 接近不可逆，導致 OLS 估計的變異數大幅膨脹。

#### 條件數的計算步驟
1. **準備矩陣**：通常對自變數進行**標準化（Standardized）** 後，計算設計矩陣 \( X \)（或相關矩陣）。
2. **奇異值分解（SVD）** 或特徵值分解：求出所有奇異值（或特徵值）。
3. **計算條件數**：取最大值與最小值之比。
4. **進階診斷（Belsley 方法）**：不僅看最大條件數，還看每個**條件指數（Condition Index）** 及其對應的**變異分解比例（Variance Decomposition Proportion, VDP）**。當某個條件指數 > 30，且有兩個以上變數的 VDP > 0.5 時，表示這些變數共同造成嚴重共線性。

在實務上，統計軟體會自動輸出：
- Python（statsmodels）：`model.summary()` 中會顯示 Condition Number。
- R：`kappa()` 函數或 `lm()` 搭配 colldiag。
- SPSS：在迴歸的「共線性診斷」選項中，會輸出「特徵值（Eigenvalue）」與「條件指數（Condition Index）」。

#### 判斷標準（經驗法則）
不同文獻的門檻略有差異，常見如下：

- **< 10**：無或極弱共線性，模型穩定。
- **10 ~ 30**：存在中度多重共線性，需注意。
- **> 30**：嚴重多重共線性，係數估計不穩定，可能出現符號錯誤或 t-檢定不顯著（即使整體 \( R^2 \) 很高）。
- 有些保守標準：
  - < 100：問題不大
  - 100 ~ 1000：中等到嚴重
  - > 1000：極端嚴重（特別是對相關矩陣計算時）

**注意**：條件數對資料尺度（Scale）很敏感，因此建議使用**標準化後的自變數** 或相關矩陣來計算，以獲得更可靠的結果。

#### 條件數與 VIF 的比較
- **VIF**：針對**個別自變數**，告訴你「哪一個變數」受到共線性影響（輔助迴歸 \( R_i^2 \) 越高，VIF 越大）。
- **條件數**：提供**整體模型**的共線性嚴重程度摘要，能偵測「高階共線性」（多個變數共同相關，而非兩兩相關），這是 VIF 有時會遺漏的部分。
- **建議**：兩者搭配使用。先看條件數判斷整體問題嚴重與否，再用 VIF 或條件指數 + VDP 找出具體涉及哪些變數。

#### 簡單範例
假設房屋價格模型的自變數包括：坪數、臥室數、總樓層面積。三者高度相關（坪數大 → 臥室多 → 總面積大）。  
計算後若條件數 = 45（>30），表示嚴重共線性。此時即使模型預測力不錯，個別係數（如「坪數的邊際影響」）會因小樣本變動而大幅波動，解釋時不可靠。

#### 優缺點
- **優點**：單一數值就能反映整體問題；能處理多變數間的複雜相依關係。
- **缺點**：無法直接指出「哪些變數」是問題來源（需搭配 VDP）；對尺度敏感；門檻值帶有主觀性。

#### 實務建議
當條件數偏高時，處理方式與 VIF 類似：
- 移除高度相關變數之一
- 進行主成分分析（PCA）或因素分析
- 使用正則化模型（Ridge Regression 最推薦，因為它對條件數高的矩陣特別穩健）
- 增加樣本量或收集更多獨立資訊

條件數是數值線性代數中衡量「病態問題（Ill-conditioned）」的標準工具，在迴歸診斷中與 VIF 共同構成多重共線性檢測的完整圖像。

