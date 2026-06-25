const n=`# Ridge / Lasso / Elastic Net

## 为什么需要正则化

线性回归的 MSE 损失函数对异常值和多重共线性很敏感。正则化通过给损失函数添加惩罚项来约束模型复杂度。

## 三种正则化对比

| 方法 | 惩罚项 | 效果 |
|------|--------|------|
| Ridge | $\\lambda \\sum w_j^2$ | 收缩系数但不稀疏 |
| Lasso | $\\lambda \\sum \\|w_j\\|$ | 产生稀疏解（特征选择） |
| Elastic Net | $\\lambda(\\alpha \\sum \\|w_j\\| + (1-\\alpha)\\sum w_j^2)$ | 结合两者优点 |

## Ridge 回归

$$J(\\mathbf{w}) = \\frac{1}{2n} \\|\\mathbf{y} - \\mathbf{X}\\mathbf{w}\\|^2 + \\lambda \\|\\mathbf{w}\\|_2^2$$

闭式解：

$$\\hat{\\mathbf{w}} = (\\mathbf{X}^T \\mathbf{X} + n\\lambda \\mathbf{I})^{-1} \\mathbf{X}^T \\mathbf{y}$$

加入 $\\lambda \\mathbf{I}$ 保证了矩阵可逆，解决多重共线性问题。

## Lasso 回归

$$J(\\mathbf{w}) = \\frac{1}{2n} \\|\\mathbf{y} - \\mathbf{X}\\mathbf{w}\\|^2 + \\lambda \\|\\mathbf{w}\\|_1$$

$L_1$ 惩罚的几何特性使得最优解落在坐标轴上，产生稀疏性。

### 为什么 Lasso 产生稀疏解而 Ridge 不产生？

$L_1$ 球的尖角使得等高线更可能落在坐标轴上。

## Elastic Net

$$J(\\mathbf{w}) = \\frac{1}{2n} \\|\\mathbf{y} - \\mathbf{X}\\mathbf{w}\\|^2 + \\lambda \\left( \\alpha \\|\\mathbf{w}\\|_1 + (1-\\alpha) \\|\\mathbf{w}\\|_2^2 \\right)$$

当 $\\alpha=1$ 退化为 Lasso，$\\alpha=0$ 退化为 Ridge。

## Python 实现

\`\`\`python
import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0, lr=0.01, epochs=1000):
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0
        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            # L2 惩罚的梯度：添加 alpha * w
            dw = (1 / n) * X.T @ (y_pred - y) + self.alpha * self.w
            db = (1 / n) * np.sum(y_pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

class LassoRegression:
    def __init__(self, alpha=1.0, lr=0.01, epochs=1000):
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0
        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            dw = (1 / n) * X.T @ (y_pred - y)
            # 次梯度方法处理 L1
            dw += self.alpha * np.sign(self.w)
            db = (1 / n) * np.sum(y_pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
\`\`\`

## 选型建议

- **Ridge**：大多数特征都有用时
- **Lasso**：希望自动特征选择时
- **Elastic Net**：特征数 > 样本数，或特征高度相关时
`;export{n as default};
