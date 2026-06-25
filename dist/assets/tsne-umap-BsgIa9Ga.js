const n=`# t-SNE / UMAP

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

### 核心思想

t-SNE 是一种**非线性降维**方法，专门用于**可视化**高维数据。它在低维空间中保持高维空间中点的**局部邻域关系**，使用 t 分布避免低维空间的"拥挤问题"。

### 算法原理

**Step 1 — 高维空间的条件概率**（高斯核）：

$$p_{j|i} = \\frac{\\exp(-\\|x_i - x_j\\|^2 / 2\\sigma_i^2)}{\\sum_{k \\neq i} \\exp(-\\|x_i - x_k\\|^2 / 2\\sigma_i^2)}$$

对称化：$p_{ij} = \\frac{p_{j|i} + p_{i|j}}{2n}$

**Step 2 — 低维空间的相似度**（t 分布，1 个自由度）：

$$q_{ij} = \\frac{(1 + \\|y_i - y_j\\|^2)^{-1}}{\\sum_{k \\neq l} (1 + \\|y_k - y_l\\|^2)^{-1}}$$

**Step 3 — 最小化 KL 散度**：

$$\\text{KL}(P \\| Q) = \\sum_{i \\neq j} p_{ij} \\log \\frac{p_{ij}}{q_{ij}}$$

梯度下降更新低维嵌入：

$$\\frac{\\partial C}{\\partial y_i} = 4 \\sum_j (p_{ij} - q_{ij}) (y_i - y_j)(1 + \\|y_i - y_j\\|^2)^{-1}$$

### 为什么用 t 分布？

t 分布的重尾特性使得高维空间中中远距离的点在低维中被拉开，解决了"拥挤问题"（高斯核把所有点挤在一起）。

### 困惑度 (Perplexity)

控制 $\\sigma_i$ 的参数，典型值 5~50：

$$\\text{Perp}(P_i) = 2^{H(P_i)}, \\quad H(P_i) = -\\sum_j p_{j|i} \\log_2 p_{j|i}$$

---

## UMAP (Uniform Manifold Approximation and Projection)

### 核心思想

UMAP 基于**黎曼几何和拓扑数据分析**，比 t-SNE 更快且更好地保留了全局结构。

### 关键步骤

1. **构建模糊单纯形集**（fuzzy simplicial set）表示高维拓扑结构
2. **优化低维嵌入**使得低维模糊单纯形集尽可能匹配高维

UMAP 使用**交叉熵**而非 KL 散度作为损失：

$$L = \\sum_{i,j} \\left[ p_{ij} \\log\\frac{p_{ij}}{q_{ij}} + (1-p_{ij})\\log\\frac{1-p_{ij}}{1-q_{ij}} \\right]$$

这在保留局部结构的同时更好地保留了全局结构。

## t-SNE vs UMAP

| | t-SNE | UMAP |
|------|-------|------|
| 速度 | 慢 ($O(n^2)$) | 快 |
| 全局结构 | 差 | 较好 |
| 可扩展性 | 受限 | 支持新点嵌入 |
| 超参数 | 困惑度 | n_neighbors + min_dist |
| 理论基础 | 概率 | 拓扑 |

## 使用建议

\`\`\`python
# t-SNE: 可视化，小数据
from sklearn.manifold import TSNE
embedding = TSNE(perplexity=30).fit_transform(X)

# UMAP: 大数据，需保留全局结构
import umap
embedding = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X)
\`\`\`

> **注意**：两者都是随机算法，多次运行结果可能不同。且它们主要用于可视化，**不适合**作为其他 ML 流程中的降维步骤。
`;export{n as default};
