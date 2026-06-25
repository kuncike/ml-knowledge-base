const n=`# 层次聚类 (Hierarchical Clustering)

## 核心思想

层次聚类构建一个**树状结构 (Dendrogram)** 来表示数据的嵌套分组，无需预设簇数 $K$。

## 两种策略

### 凝聚式 (Agglomerative) — 自底向上

1. 每个点自成一簇
2. 找到最近的两个簇合并
3. 重复直到只剩一个簇

复杂度 $O(n^3)$，用优先队列可降到 $O(n^2 \\log n)$。

### 分裂式 (Divisive) — 自顶向下

1. 所有点为一个簇
2. 选择一个簇按某种准则分裂
3. 重复直到每个点一簇

## 簇间距离度量 (Linkage)

| 方法 | 定义 | 特点 |
|------|------|------|
| Single Link | $\\min_{a \\in A, b \\in B} d(a,b)$ | 链式，易产生细长簇 |
| Complete Link | $\\max_{a \\in A, b \\in B} d(a,b)$ | 倾向于紧凑簇 |
| Average Link | $\\frac{1}{\\|A\\|\\|B\\|} \\sum_{a \\in A}\\sum_{b \\in B} d(a,b)$ | 折中方案 |
| Ward's Method | 最小化合并后方差增量 | 倾向于等大小簇 |

## Python 示例

\`\`\`python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# 凝聚聚类
Z = linkage(X, method='ward')

# 从树状图中切出 K 个簇
labels = fcluster(Z, t=3, criterion='maxclust')

# 绘制树状图
dendrogram(Z)
plt.show()
\`\`\`

## Ward's 方法详解

合并簇 $A$ 和 $B$ 后的方差增量：

$$\\Delta(A, B) = \\frac{|A||B|}{|A| + |B|} \\|\\mu_A - \\mu_B\\|^2$$

每次合并选择使 $\\Delta$ 最小的两个簇。

## 优缺点

- **优点**：无需预设簇数，树状图提供丰富可视化，链接方式灵活
- **缺点**：$O(n^3)$ 或 $O(n^2 \\log n)$ 的复杂度不适合大规模数据，一旦合并/分裂不能撤销
`;export{n as default};
