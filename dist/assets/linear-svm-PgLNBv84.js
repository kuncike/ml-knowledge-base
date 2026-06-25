const n=`# Linear SVM (线性支持向量机)

## 核心思想

SVM 不关心"所有样本都分对"，而是追求"离最近样本尽可能远"——它在各类样本之间画一条"最宽的高速公路"，路的两侧边界由少数关键样本点（**支持向量**）撑起。这种最大化间隔的策略赋予了 SVM 极强的泛化能力。

---

## 数学定义与原理解析

### 硬间隔 SVM（线性可分）

$$
\\min_{\\mathbf{w}, b} \\frac{1}{2} \\|\\mathbf{w}\\|^2
$$

$$
\\text{s.t.} \\quad y_i (\\mathbf{w}^T \\mathbf{x}_i + b) \\geq 1, \\quad \\forall i
$$

**几何间隔**：$\\gamma = \\frac{2}{\\|\\mathbf{w}\\|}$——两个类之间"路"的宽度。最小化 $\\|\\mathbf{w}\\|^2$ 等价于最大化间隔。

### 软间隔 SVM（线性不可分）

引入松弛变量 $\\xi_i \\geq 0$，允许部分样本越过边界：

$$
\\min_{\\mathbf{w}, b, \\xi} \\frac{1}{2} \\|\\mathbf{w}\\|^2 + C \\sum_{i=1}^{n} \\xi_i
$$

$$
\\text{s.t.} \\quad y_i (\\mathbf{w}^T \\mathbf{x}_i + b) \\geq 1 - \\xi_i, \\quad \\xi_i \\geq 0
$$

- $C \\to \\infty$：硬间隔，零容忍错误
- $C \\to 0$：软间隔，允许更多错误，间隔更大
- $C$ 是偏差-方差权衡的核心旋钮

### 对偶问题与核技巧的入口

通过拉格朗日对偶性转换：

$$
\\max_{\\alpha} \\sum_{i=1}^{n} \\alpha_i - \\frac{1}{2} \\sum_{i=1}^{n} \\sum_{j=1}^{n} \\alpha_i \\alpha_j y_i y_j (\\mathbf{x}_i^T \\mathbf{x}_j)
$$

$$
\\text{s.t.} \\quad 0 \\leq \\alpha_i \\leq C, \\quad \\sum_{i=1}^{n} \\alpha_i y_i = 0
$$

**关键观察**：
- 只有 $\\alpha_i > 0$ 的点才是**支持向量**（通常仅占训练集的 10%-20%）
- 内积 $\\mathbf{x}_i^T \\mathbf{x}_j$ 的位置可以直接替换为核函数 $K(\\mathbf{x}_i, \\mathbf{x}_j)$——这是非线性 SVM 的核心技巧

### Hinge Loss 视角（SVM 的现代理解）

SVM 等价于 Hinge Loss + L2 正则化：

$$
J(\\mathbf{w}) = \\frac{1}{2} \\|\\mathbf{w}\\|^2 + C \\sum_{i=1}^{n} \\max(0, 1 - y_i(\\mathbf{w}^T \\mathbf{x}_i + b))
$$

$$
\\ell_{\\text{hinge}}(y, \\hat{y}) = \\max(0, 1 - y\\hat{y})
$$

当 $y\\hat{y} \\geq 1$（样本在间隔外）时，loss = 0；当 $y\\hat{y} < 1$ 时产生惩罚。

---

## 可视化展示

### Hinge Loss vs 其他分类损失

\`\`\`echarts
return {
  xAxis: { type: 'value', min: -2, max: 3, name: 'y·ŷ (margin)' },
  yAxis: { type: 'value', min: 0, max: 5, name: 'Loss' },
  legend: { data: ['Hinge Loss', 'Logistic (CE)', '0-1 Loss'] },
  series: [
    {
      name: 'Hinge Loss', type: 'line',
      lineStyle: { color: '#2c3e50', width: 2.5 },
      data: (function() { const d = []; for (let i = -2; i <= 3; i += 0.02) d.push([i, Math.max(0, 1 - i)]); return d; })()
    },
    {
      name: 'Logistic (CE)', type: 'line', smooth: true,
      lineStyle: { color: '#2980b9', width: 2 },
      data: (function() { const d = []; for (let i = -2; i <= 3; i += 0.02) d.push([i, Math.log(1 + Math.exp(-i))]); return d; })()
    },
    {
      name: '0-1 Loss', type: 'line',
      lineStyle: { color: '#c0392b', width: 1.5, type: 'dashed' },
      data: (function() { const d = []; for (let i = -2; i <= 3; i += 0.02) d.push([i, i >= 0 ? 0 : 1]); return d; })()
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

Hinge Loss 在 $y\\hat{y} \\geq 1$ 后直接归零——SVM 只关心"是否正确且足够确信"，不追完美概率校准。

### 参数 $C$ 对决策边界的影响

$C$ 越小 → 间隔越大 → 允许更多误分类 → 模型越简单（偏差增大）。

\`\`\`echarts
return {
  xAxis: { type: 'value', min: 0.01, max: 100, name: 'C (log scale)' },
  yAxis: { type: 'value', min: 0, max: 1, name: '正则化强度' },
  series: [{
    type: 'line', smooth: true,
    data: (function() { const d = []; for (let c = 0.01; c <= 100; c *= 1.1) d.push([Math.log10(c), 1 / (1 + c)]); return d; })(),
    lineStyle: { color: '#2c3e50', width: 2 },
    areaStyle: { color: 'rgba(44, 62, 80, 0.1)' }
  }],
  tooltip: { trigger: 'axis', formatter: 'C = {b}<br/>正则化强度 ≈ {c}' },
  grid: { left: 60, right: 20, top: 20, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### Scikit-learn

\`\`\`python
from sklearn.svm import LinearSVC, SVC

# 线性 SVM（快速，基于 LIBLINEAR）
clf = LinearSVC(C=1.0, loss='hinge', max_iter=5000)

# 通用 SVM（支持核函数，基于 LIBSVM）
clf = SVC(C=1.0, kernel='linear')
clf.fit(X_train, y_train)
\`\`\`

### NumPy 手写（Hinge Loss 梯度下降版）

\`\`\`python
import numpy as np

class LinearSVM:
    def __init__(self, C=1.0, lr=0.001, epochs=1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0
        for _ in range(self.epochs):
            margins = y * (X @ self.w + self.b)
            mask = margins < 1                    # 仅更新 margin < 1 的样本
            dw = self.w - self.C * (X[mask].T @ y[mask])
            db = -self.C * np.sum(y[mask])
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.sign(X @ self.w + self.b)
\`\`\`
`;export{n as default};
