const n=`# 孤立森林 / LOF / One-Class SVM

## 孤立森林 (Isolation Forest)

### 核心思想

异常点**少且不同**——它们更容易被随机划分"孤立"出来。孤立森林通过随机构建多棵二叉树，用点被孤立的平均深度作为异常分数。

### 算法原理

1. 随机选取一个特征和一个分割值
2. 递归划分直到每个点被隔离或达到深度限制
3. 点的**路径长度**越短，越可能是异常

异常分数（归一化）：

$$s(x, n) = 2^{-\\frac{E[h(x)]}{c(n)}}$$

其中 $c(n) = 2H(n-1) - 2(n-1)/n$ 是二叉搜索树的平均路径长度，$H(\\cdot)$ 是调和数。

- $s \\approx 1$：异常
- $s \\approx 0.5$：正常
- $s \\ll 0.5$：太"正常"（可能）

### 优势

- 不需要正常数据建模
- 线性时间复杂度 $O(n)$
- 对高维数据的子采样有效

---

## 局部异常因子 (LOF)

### 核心思想

LOF 基于**局部密度**：如果一个点的密度明显低于其邻居的密度，则为异常。这使其能检测**局部异常**（在全局看来正常但在局部邻域中异常）。

### 计算方法

对点 $p$，设其 $k$-距离邻域为 $N_k(p)$：

**局部可达密度**：

$$\\text{lrd}_k(p) = \\frac{|N_k(p)|}{\\sum_{o \\in N_k(p)} \\text{reach-dist}_k(p, o)}$$

**局部异常因子**：

$$\\text{LOF}_k(p) = \\frac{\\sum_{o \\in N_k(p)} \\frac{\\text{lrd}_k(o)}{\\text{lrd}_k(p)}}{|N_k(p)|}$$

- LOF $\\approx 1$：密度与邻居相当，正常
- LOF $> 1$：密度低于邻居，可能异常

---

## One-Class SVM

### 核心思想

One-Class SVM 在特征空间中找到一个超平面（或超球面）将数据与原点（或外部）最大程度地分开。

$$\\min_{\\mathbf{w}, \\rho, \\xi} \\frac{1}{2}\\|\\mathbf{w}\\|^2 + \\frac{1}{\\nu n}\\sum_i \\xi_i - \\rho$$

其中 $\\nu \\in (0, 1]$ 是异常比例的上界。

## Python 示例

\`\`\`python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# 孤立森林
iso = IsolationForest(contamination=0.1)
iso.fit(X)

# LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
labels = lof.fit_predict(X)

# One-Class SVM
svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
svm.fit(X)
\`\`\`

## 对比

| 方法 | 适用场景 | 速度 |
|------|----------|------|
| Isolation Forest | 大规模数据，高维 | 极快 |
| LOF | 局部异常检测 | 中等 |
| One-Class SVM | 小样本，核方法 | 慢 |
`;export{n as default};
