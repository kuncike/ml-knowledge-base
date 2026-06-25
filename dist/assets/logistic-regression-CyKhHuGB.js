const n=`# Logistic Regression (逻辑回归)

## 核心思想

名字带"回归"但实为**分类模型**。逻辑回归在线性组合 $\\mathbf{w}^T\\mathbf{x} + b$ 之上套一层 Sigmoid，将任意实数输出压缩到 $(0, 1)$ 区间——解释为"属于正类的概率"。它使用交叉熵损失而非 MSE，因为 MSE + Sigmoid 会导致梯度消失。

---

## 数学定义与原理解析

### 模型定义

$$
P(y=1 \\mid \\mathbf{x}) = \\hat{y} = \\sigma(\\mathbf{w}^T \\mathbf{x} + b) = \\frac{1}{1 + e^{-(\\mathbf{w}^T \\mathbf{x} + b)}}
$$

决策边界：当 $\\hat{y} \\geq 0.5$ 预测为正类，即 $\\mathbf{w}^T \\mathbf{x} + b \\geq 0$ —— 是一个**线性超平面**。

### 为什么不用 MSE？

Sigmoid + MSE 的梯度为：

$$
\\frac{\\partial L}{\\partial w} = 2(\\hat{y} - y) \\cdot \\hat{y}(1-\\hat{y}) \\cdot x
$$

当 $\\hat{y} \\to 0$ 或 $\\hat{y} \\to 1$（即预测非常确信时），$\\hat{y}(1-\\hat{y}) \\to 0$ → **梯度消失**。即使模型完全预测错误（$y=1, \\hat{y}\\approx0$），梯度也接近于零——模型无法从错误中学习。

### 交叉熵损失

$$
J(\\mathbf{w}) = -\\frac{1}{n} \\sum_{i=1}^{n} \\left[ y_i \\log(\\hat{y}_i) + (1-y_i) \\log(1-\\hat{y}_i) \\right]
$$

Sigmoid + 交叉熵的梯度经化简后**消除**了 Sigmoid 的导数项：

$$
\\frac{\\partial J}{\\partial \\mathbf{w}} = \\frac{1}{n} \\mathbf{X}^T (\\hat{\\mathbf{y}} - \\mathbf{y})
$$

梯度 = 预测值 - 真实值——与线性回归的 MSE 梯度形式一致！这是逻辑回归使用交叉熵的数学原因。

### 多分类扩展：Softmax 回归

$$
\\hat{y}_k = \\frac{e^{\\mathbf{w}_k^T \\mathbf{x}}}{\\sum_{j=1}^{K} e^{\\mathbf{w}_j^T \\mathbf{x}}}
$$

训练 $K$ 组权重向量，通过 Softmax 归一化为 $K$ 类概率分布。

---

## 可视化展示

### Sigmoid 函数与决策边界

\`\`\`echarts
return {
  xAxis: { type: 'value', min: -10, max: 10, name: 'z = wᵗx + b' },
  yAxis: { type: 'value', min: 0, max: 1, name: 'P(y=1|x)' },
  legend: { data: ['Sigmoid σ(z)', '决策边界 (z=0)'] },
  series: [
    {
      name: 'Sigmoid σ(z)', type: 'line', smooth: true,
      lineStyle: { color: '#2c3e50', width: 2.5 },
      data: (function() { const d = []; for (let i = -10; i <= 10; i += 0.05) d.push([i, 1/(1+Math.exp(-i))]); return d; })()
    },
    {
      name: '决策边界 (z=0)', type: 'line',
      markLine: { data: [{ xAxis: 0, label: { formatter: 'z=0 → p=0.5' } }], silent: true,
        lineStyle: { color: '#d35400', type: 'dashed', width: 2 } },
      data: []
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

### MSE vs 交叉熵的梯度对比

\`\`\`echarts
return {
  xAxis: { type: 'value', min: 0.001, max: 0.999, name: '预测概率 p̂ (当 y=1)' },
  yAxis: { type: 'value', min: -10, max: 10, name: '∂L/∂z (梯度)' },
  legend: { data: ['交叉熵 (∂L/∂z = p̂ - y)', 'MSE (∂L/∂z = 2(p̂-y)·p̂(1-p̂))'] },
  series: [
    {
      name: '交叉熵 (∂L/∂z = p̂ - y)', type: 'line', smooth: true,
      lineStyle: { color: '#16a085', width: 2.5 },
      data: (function() { const d = []; for (let p = 0.001; p <= 0.999; p += 0.002) d.push([p, p - 1]); return d; })()
    },
    {
      name: 'MSE (∂L/∂z = 2(p̂-y)·p̂(1-p̂))', type: 'line', smooth: true,
      lineStyle: { color: '#c0392b', width: 2 },
      data: (function() { const d = []; for (let p = 0.001; p <= 0.999; p += 0.002) d.push([p, 2*(p-1)*p*(1-p)]); return d; })()
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

MSE 在 $p \\to 0$ 时梯度趋近于零（梯度消失），而交叉熵的梯度保持线性——越错梯度越大，这是交叉熵收敛更快的原因。

---

## 核心代码实现

### Scikit-learn

\`\`\`python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    penalty='l2',     # L2 正则化
    C=1.0,            # 正则化强度的倒数
    solver='lbfgs',   # 优化器
    max_iter=1000
)
clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)[:, 1]  # 获取正类概率
\`\`\`

### NumPy 手写

\`\`\`python
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0
        for _ in range(self.epochs):
            z = X @ self.w + self.b
            y_pred = self.sigmoid(z)
            dw = (1 / n) * X.T @ (y_pred - y)
            db = (1 / n) * np.sum(y_pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
\`\`\`

---

## 优缺点

| 优点 | 缺点 |
|------|------|
| 可解释性强：权重直接反映特征影响 | 只能处理线性可分数据 |
| 输出天然是概率 | 对特征缩放敏感 |
| 计算高效，内存占用小 | 无法处理非线性边界 |
| 自带 L1/L2 正则化（sklearn） | 对多重共线性敏感 |
`;export{n as default};
