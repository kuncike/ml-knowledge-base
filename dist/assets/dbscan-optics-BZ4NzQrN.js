const n=`# DBSCAN / OPTICS

## DBSCAN 核心思想

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 基于密度定义簇：**簇是密度相连的点的最大集合**。它可以发现任意形状的簇并自动识别噪声。

## 两个核心参数

- **$\\epsilon$ (eps)**：邻域半径
- **MinPts**：成为核心点所需的最小邻域点数

## 三种点类型

- **核心点**：$\\epsilon$-邻域内点数 $\\geq$ MinPts
- **边界点**：自身不是核心点，但在某个核心点的 $\\epsilon$-邻域内
- **噪声点**：既不是核心点也不是边界点

## 密度可达与密度相连

- **直接密度可达**：$q$ 在 $p$ 的 $\\epsilon$-邻域内，且 $p$ 是核心点
- **密度可达**：存在一条核心点链连接
- **密度相连**：存在点 $o$，从 $o$ 出发对两个点都密度可达

**簇 = 所有密度相连的点的集合。**

## 算法流程

\`\`\`python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def dbscan(X, eps, min_pts):
    n = X.shape[0]
    labels = -np.ones(n, dtype=int)  # -1 = unvisited
    nn = NearestNeighbors(radius=eps).fit(X)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        neighbors = nn.radius_neighbors([X[i]], return_distance=False)[0]
        if len(neighbors) < min_pts:
            labels[i] = -2  # noise
            continue

        labels[i] = cluster_id
        seeds = list(neighbors[neighbors != i])
        j = 0
        while j < len(seeds):
            p = seeds[j]
            if labels[p] == -2:
                labels[p] = cluster_id
            if labels[p] != -1:
                j += 1
                continue
            labels[p] = cluster_id
            p_neighbors = nn.radius_neighbors([X[p]], return_distance=False)[0]
            if len(p_neighbors) >= min_pts:
                for nbr in p_neighbors:
                    if labels[nbr] == -1:
                        seeds.append(nbr)
            j += 1
        cluster_id += 1
    return labels
\`\`\`

## OPTICS — 消除 $\\epsilon$ 的敏感度

OPTICS 不需要指定 $\\epsilon$，而是生成一个**可达距离图**：

- **核心距离**：点到其第 MinPts 近邻的距离
- **可达距离**：$\\max(\\text{核心距离}(p), d(p, q))$

从可达距离图中提取簇结构，可同时发现不同密度的簇。

## DBSCAN vs K-Means

| | DBSCAN | K-Means |
|------|--------|---------|
| 簇形状 | 任意形状 | 球形 |
| 噪声处理 | 自动识别 | 强制分配 |
| 参数 | eps + MinPts | K |
| 密度差异 | 单密度（OPTICS 多密度）| 不支持 |

## 优缺点

- **优点**：不需要预设 K，可发现任意形状簇，自动识别噪声
- **缺点**：对 eps 敏感，密度差异大时效果差，高维数据稀疏导致效果下降
`;export{n as default};
