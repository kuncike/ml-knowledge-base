const n=`# ICA / LDA

## 独立成分分析 (ICA)

### 核心思想

与 PCA（寻找互不相关的方向）不同，ICA 寻找**统计独立**的成分。经典的**鸡尾酒会问题**：从混合音频中分离出独立的说话人声音。

### 数学模型

假设观测信号 $\\mathbf{X} \\in \\mathbb{R}^{n \\times d}$ 由独立源信号 $\\mathbf{S}$ 线性混合而成：

$$\\mathbf{X} = \\mathbf{A} \\mathbf{S}$$

目标是找到解混矩阵 $\\mathbf{W} = \\mathbf{A}^{-1}$，使得：

$$\\hat{\\mathbf{S}} = \\mathbf{W} \\mathbf{X}$$

### FastICA 算法

基于**非高斯最大化**原则（中心极限定理：独立源混合后更接近高斯分布）：

1. 白化数据（PCA + 归一化方差）
2. 寻找投影方向 $\\mathbf{w}$ 最大化投影的非高斯性

负熵近似（非高斯度量）：

$$J(y) \\propto [E\\{G(y)\\} - E\\{G(\\nu)\\}]^2$$

其中 $\\nu \\sim \\mathcal{N}(0, 1)$，$G$ 是非二次函数（如 $\\log\\cosh$）。

## PCA vs ICA

| | PCA | ICA |
|------|-----|-----|
| 准则 | 最大方差 | 统计独立 |
| 约束 | 成分正交 | 成分独立 |
| 分布假设 | 仅用二阶矩 | 非高斯性 |
| 典型应用 | 降维/去噪 | 盲源分离 |

---

## 线性判别分析 (LDA)

### 核心思想

LDA 是一种**有监督**降维方法，目标是找到投影方向使得**类间散度最大、类内散度最小**，最大化类别可分性。

### 数学推导

**类内散度矩阵** $\\mathbf{S}_W$：

$$\\mathbf{S}_W = \\sum_{k=1}^{K} \\sum_{x_i \\in C_k} (x_i - \\mu_k)(x_i - \\mu_k)^T$$

**类间散度矩阵** $\\mathbf{S}_B$：

$$\\mathbf{S}_B = \\sum_{k=1}^{K} n_k (\\mu_k - \\mu)(\\mu_k - \\mu)^T$$

目标（广义 Rayleigh 商）：

$$\\max_{\\mathbf{W}} \\frac{\\mathbf{W}^T \\mathbf{S}_B \\mathbf{W}}{\\mathbf{W}^T \\mathbf{S}_W \\mathbf{W}}$$

解是 $\\mathbf{S}_W^{-1} \\mathbf{S}_B$ 的前 $K-1$ 个特征向量（LDA 最多降到 $K-1$ 维）。

## Python 实现

\`\`\`python
import numpy as np

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X, y):
        n, d = X.shape
        classes = np.unique(y)
        mean_overall = X.mean(axis=0)

        S_W = np.zeros((d, d))
        S_B = np.zeros((d, d))
        for c in classes:
            X_c = X[y == c]
            mean_c = X_c.mean(axis=0)
            S_W += (X_c - mean_c).T @ (X_c - mean_c)
            n_c = len(X_c)
            mean_diff = (mean_c - mean_overall).reshape(-1, 1)
            S_B += n_c * mean_diff @ mean_diff.T

        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_W) @ S_B)
        idx = np.argsort(-np.abs(eigvals))
        self.components_ = eigvecs[:, idx[:self.n_components]].real
        return X @ self.components_
\`\`\`

## 应用场景

- **ICA**：EEG/MEG 信号分离，金融时间序列分析，图像去噪
- **LDA**：人脸识别 (Fisherfaces)，模式分类的特征提取
`;export{n as default};
