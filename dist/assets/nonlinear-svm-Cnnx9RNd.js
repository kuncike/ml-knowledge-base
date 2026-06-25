const n=`# 非线性 SVM (核方法)

## 核心思想

当数据在原始空间中线性不可分时，SVM 通过**核函数 (Kernel Function)** 隐式地将数据映射到高维空间，在高维空间中寻找线性超平面。

## 核技巧 (Kernel Trick)

在对偶问题中，特征只以内积 $\\mathbf{x}_i^T \\mathbf{x}_j$ 的形式出现。核函数直接计算高维空间的内积，无需显式做特征映射：

$$K(\\mathbf{x}_i, \\mathbf{x}_j) = \\langle \\phi(\\mathbf{x}_i), \\phi(\\mathbf{x}_j) \\rangle$$

## 常见核函数

### RBF (高斯核) — 最常用

$$K(\\mathbf{x}_i, \\mathbf{x}_j) = \\exp\\left(-\\gamma \\|\\mathbf{x}_i - \\mathbf{x}_j\\|^2\\right)$$

- $\\gamma$ 控制单个样本的影响范围
- $\\gamma$ 大 → 高方差（过拟合风险）
- $\\gamma$ 小 → 高偏差（欠拟合风险）

### 多项式核

$$K(\\mathbf{x}_i, \\mathbf{x}_j) = (\\gamma \\mathbf{x}_i^T \\mathbf{x}_j + r)^d$$

- $d$ 是多项式的次数
- $d=1$ 退化为线性核

### Sigmoid 核

$$K(\\mathbf{x}_i, \\mathbf{x}_j) = \\tanh(\\gamma \\mathbf{x}_i^T \\mathbf{x}_j + r)$$

与神经网络的激活函数有相似之处。

## RBF 核的直觉

RBF 核可以理解为：在样本 $\\mathbf{x}_i$ 周围放置一个高斯"影响球"，距离越近的点相似度越接近 1，越远越接近 0。

## 超参数调优

| 参数 | 作用 | 过大 | 过小 |
|------|------|------|------|
| $C$ | 正则化强度 | 过拟合 | 欠拟合 |
| $\\gamma$ (RBF) | 单样本影响半径 | 过拟合 | 欠拟合 |

## scikit-learn 示例

\`\`\`python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm = SVC(kernel='rbf')
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
}
grid = GridSearchCV(svm, param_grid, cv=5)
grid.fit(X_train, y_train)
\`\`\`

## 数学性质

根据 **Mercer 定理**，只要核函数对应的 Gram 矩阵是半正定的，就隐式存在某个特征空间使核函数有效。

## 优缺点

- **优点**：可以拟合极其复杂的决策边界，全局最优解（凸优化）
- **缺点**：$O(n^2)$ 空间和 $O(n^3)$ 时间的复杂度使其不适合大规模数据
`;export{n as default};
