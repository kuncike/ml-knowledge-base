const n=`# 线性回归 (Linear Regression)

## 核心思想

线性回归是最基础的监督学习算法，目标是拟合一条直线（或超平面）来最小化预测值与真实值之间的误差。

## 数学模型

假设有 $n$ 个样本，每个样本有 $d$ 个特征：

$$y = \\mathbf{w}^T \\mathbf{x} + b = \\sum_{i=1}^{d} w_i x_i + b$$

损失函数采用均方误差 (MSE)：

$$J(\\mathbf{w}, b) = \\frac{1}{2n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 = \\frac{1}{2n} \\|\\mathbf{y} - \\mathbf{X}\\mathbf{w} - b\\|^2$$

## 求解方法

### 1. 正规方程 (Normal Equation)

当 $\\mathbf{X}^T \\mathbf{X}$ 可逆时，有闭式解：

$$\\hat{\\mathbf{w}} = (\\mathbf{X}^T \\mathbf{X})^{-1} \\mathbf{X}^T \\mathbf{y}$$

**复杂度**：$O(d^3 + nd^2)$，特征数 $d$ 很大时不可行。

### 2. 梯度下降 (Gradient Descent)

$$\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta \\cdot \\frac{1}{n} \\mathbf{X}^T (\\mathbf{X}\\mathbf{w} - \\mathbf{y})$$

## Python 实现

\`\`\`python
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0
        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            dw = (1 / n) * X.T @ (y_pred - y)
            db = (1 / n) * np.sum(y_pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return X @ self.w + self.b
\`\`\`

## 关键假设

| 假设 | 说明 |
|------|------|
| 线性 | 因变量与自变量线性相关 |
| 独立性 | 观测值相互独立 |
| 同方差性 | 误差项方差恒定 |
| 正态性 | 误差项服从正态分布 |
| 无多重共线性 | 特征间不高度相关 |

## 优缺点

- **优点**：可解释性强，计算简单，有闭式解
- **缺点**：无法拟合非线性关系，对异常值敏感
`;export{n as default};
