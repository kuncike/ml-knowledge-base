const n=`# 谱聚类 (Spectral Clustering) / GMM

## 谱聚类核心思想

将聚类问题转化为**图的划分问题**。样本点构成图的节点，边的权重表示相似度。使用图的**拉普拉斯矩阵的特征向量**将数据投影到低维流形，再在投影空间使用 K-Means。

## 算法流程

1. 构建相似度矩阵 $\\mathbf{W}$（常用 RBF 核）

$$W_{ij} = \\exp\\left(-\\frac{\\|x_i - x_j\\|^2}{2\\sigma^2}\\right)$$

2. 构建度矩阵 $\\mathbf{D}$（对角矩阵，$D_{ii} = \\sum_j W_{ij}$）
3. 计算**非归一化拉普拉斯矩阵**：

$$\\mathbf{L} = \\mathbf{D} - \\mathbf{W}$$

或**归一化拉普拉斯矩阵**：

$$\\mathbf{L}_{sym} = \\mathbf{D}^{-1/2} \\mathbf{L} \\mathbf{D}^{-1/2} = \\mathbf{I} - \\mathbf{D}^{-1/2} \\mathbf{W} \\mathbf{D}^{-1/2}$$

4. 计算 $\\mathbf{L}_{sym}$ 的前 $K$ 个最小非零特征值对应的特征向量，构成矩阵 $\\mathbf{U} \\in \\mathbb{R}^{n \\times K}$
5. 将 $\\mathbf{U}$ 的行归一化后使用 K-Means

## 谱聚类 vs 传统聚类

谱聚类可以发现**任意形状的簇**，因为它基于图连通性而非距离度量。关键是在正确的**流形**上进行聚类。

---

## 高斯混合模型 (GMM)

### 核心思想

假设数据由 $K$ 个高斯分布混合生成：

$$p(x) = \\sum_{k=1}^{K} \\pi_k \\cdot \\mathcal{N}(x \\mid \\mu_k, \\Sigma_k)$$

其中 $\\sum_k \\pi_k = 1$，$\\pi_k$ 是第 $k$ 个成分的权重。

### EM 算法求解

GMM 使用 **EM 算法**（而非 K-Means 的硬分配）做**软聚类**，每个点属于各簇的概率（责任度）：

**E 步**：

$$\\gamma_{ik} = \\frac{\\pi_k \\cdot \\mathcal{N}(x_i \\mid \\mu_k, \\Sigma_k)}{\\sum_{j=1}^{K} \\pi_j \\cdot \\mathcal{N}(x_i \\mid \\mu_j, \\Sigma_j)}$$

**M 步**：

$$\\mu_k = \\frac{\\sum_i \\gamma_{ik} x_i}{\\sum_i \\gamma_{ik}}$$

$$\\Sigma_k = \\frac{\\sum_i \\gamma_{ik} (x_i - \\mu_k)(x_i - \\mu_k)^T}{\\sum_i \\gamma_{ik}}$$

$$\\pi_k = \\frac{\\sum_i \\gamma_{ik}}{n}$$

### GMM vs K-Means

| | GMM | K-Means |
|------|-----|---------|
| 分配方式 | 软（概率）| 硬（0/1）|
| 簇形状 | 椭圆（任意协方差）| 圆/球 |
| 参数 | $\\pi_k, \\mu_k, \\Sigma_k$ | $\\mu_k$ |
| 收敛 | EM 算法 | Lloyd 算法 |

K-Means 是 GMM 当所有协方差矩阵为 $\\epsilon\\mathbf{I}$ 且 $\\epsilon \\to 0$ 时的极限情况。

## 优缺点

**谱聚类**：可发现任意形状簇，理论优美；但计算量大 ($O(n^3)$)，大 $n$ 下不实用

**GMM**：软聚类提供不确定性信息，可拟合椭圆簇；但对初始化敏感，需要预设 K
`;export{n as default};
