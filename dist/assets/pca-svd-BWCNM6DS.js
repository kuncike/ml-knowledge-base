const n=`# PCA / SVD

## 主成分分析 (PCA)

### 核心思想

PCA 寻找数据中**方差最大**的方向（主成分），将高维数据投影到这些方向上，实现降维的同时保留尽可能多的信息。

### 数学推导

给定数据中心化后的矩阵 $\\mathbf{X} \\in \\mathbb{R}^{n \\times d}$，找到投影方向 $\\mathbf{w}$ 使得投影后方差最大：

$$\\max_{\\|\\mathbf{w}\\|=1} \\frac{1}{n} \\sum_{i=1}^{n} (\\mathbf{x}_i^T \\mathbf{w})^2 = \\max_{\\|\\mathbf{w}\\|=1} \\mathbf{w}^T \\left(\\frac{1}{n}\\mathbf{X}^T \\mathbf{X}\\right) \\mathbf{w}$$

即求解协方差矩阵 $\\mathbf{C} = \\frac{1}{n}\\mathbf{X}^T \\mathbf{X}$ 的最大特征值对应的特征向量。

### 算法流程

1. 数据中心化：$\\mathbf{X} \\leftarrow \\mathbf{X} - \\mathbf{\\mu}$
2. 计算协方差矩阵 $\\mathbf{C} = \\frac{1}{n-1}\\mathbf{X}^T \\mathbf{X}$
3. 对 $\\mathbf{C}$ 进行特征值分解
4. 取前 $k$ 大特征值对应的特征向量构成投影矩阵 $\\mathbf{W}_k$
5. 降维结果：$\\mathbf{Z} = \\mathbf{X} \\mathbf{W}_k$

### 通过 SVD 实现 PCA

不仅对 $\\mathbf{X}^T \\mathbf{X}$ 做特征分解，还可以直接对 $\\mathbf{X}$ 做 SVD：

$$\\mathbf{X} = \\mathbf{U} \\boldsymbol{\\Sigma} \\mathbf{V}^T$$

- $\\mathbf{U}$：左奇异向量，维度 $n \\times n$
- $\\boldsymbol{\\Sigma}$：奇异值对角矩阵
- $\\mathbf{V}$：右奇异向量（即主成分方向！），维度 $d \\times d$

降维结果：$\\mathbf{Z} = \\mathbf{U}_k \\boldsymbol{\\Sigma}_k = \\mathbf{X} \\mathbf{V}_k$

## 方差解释率

第 $j$ 个主成分的方差解释比例：

$$\\frac{\\lambda_j}{\\sum_{i=1}^{d} \\lambda_i}$$

累加前 $k$ 个即可衡量保留了多少信息。

## Python 实现

\`\`\`python
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.k = n_components

    def fit_transform(self, X):
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # SVD 方式
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[:self.k]
        self.explained_variance_ratio_ = (S[:self.k] ** 2) / (S ** 2).sum()
        return X_centered @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_
\`\`\`

## SVD 的广泛用途

| 应用 | 描述 |
|------|------|
| PCA | 右奇异向量 = 主成分方向 |
| 矩阵压缩 | 保留前 $k$ 个奇异值近似原矩阵 |
| 推荐系统 | 隐语义模型 (LFM) |
| 伪逆 | $\\mathbf{X}^+ = \\mathbf{V}\\boldsymbol{\\Sigma}^{-1}\\mathbf{U}^T$ |
| 潜在语义分析 (LSA) | 词-文档矩阵分解 |
`;export{n as default};
