const n=`# K-Means / K-Medoids

## K-Means 核心思想

将 $n$ 个样本划分到 $K$ 个簇，使得簇内样本到簇中心（centroid）的距离平方和最小。

## 目标函数

$$\\text{WCSS} = \\sum_{k=1}^{K} \\sum_{x_i \\in C_k} \\|x_i - \\mu_k\\|^2$$

其中 $\\mu_k = \\frac{1}{|C_k|} \\sum_{x_i \\in C_k} x_i$ 是簇 $C_k$ 的均值。

## Lloyd 算法

1. **初始化**：随机选择 $K$ 个点作为初始质心（或使用 K-Means++）
2. **分配步**：将每个点分配给最近的质心
3. **更新步**：将质心更新为所属点的均值
4. 重复 2-3 直到收敛

收敛性：算法保证每步 WCSS 不增，收敛到局部最优。

## K-Means++ 初始化

用概率分布改善初始点选择：

$$P(x_i) = \\frac{D(x_i)^2}{\\sum_{x_j} D(x_j)^2}$$

其中 $D(x_i)$ 是 $x_i$ 到最近已选质心的距离。保证 $O(\\log K)$ 近似比。

## Python 实现

\`\`\`python
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4):
        self.K = n_clusters
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        n, d = X.shape
        # K-Means++ 初始化
        centroids = [X[np.random.randint(n)]]
        for _ in range(1, self.K):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dists / dists.sum()
            centroids.append(X[np.random.choice(n, p=probs)])
        centroids = np.array(centroids)

        for _ in range(self.max_iters):
            dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids]).T
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.K)])
            if np.allclose(centroids, new_centroids, atol=self.tol):
                break
            centroids = new_centroids

        self.centroids_, self.labels_ = centroids, labels
\`\`\`

## 选择 K 值

### 肘部法则 (Elbow Method)

画出 WCSS 随 K 的变化曲线，选择"拐点"。

### 轮廓系数 (Silhouette Score)

$$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$$

- $a(i)$：样本 $i$ 到同簇其他点的平均距离
- $b(i)$：样本 $i$ 到最近异簇点的平均距离
- 取值范围 $[-1, 1]$，越大越好

## K-Medoids (PAM)

用实际数据点作为簇中心（medoid），对异常值更鲁棒：

$$\\text{Medoid} = \\arg\\min_{x \\in C_k} \\sum_{x_j \\in C_k} \\|x - x_j\\|$$

## 优缺点

- **优点**：简单高效，$O(nKd)$ 每轮，可扩展性好
- **缺点**：需预设 K，对初始值敏感，不适合非球形簇，受异常值影响大
`;export{n as default};
