const n=`# SOM (自组织映射 / Kohonen 网络)

## 核心思想

SOM 将高维数据映射到 2D 网格上，同时**保持拓扑结构**——相似的输入映射到网格上相邻的神经元。这是无监督的：没有标签，只有数据点之间的欧氏距离。SOM 的关键机制是**竞争学习 + 邻域协作**：获胜神经元（BMU, Best Matching Unit）不仅更新自己，还拉着邻居一起靠近输入。

典型应用：高维数据可视化、客户细分、故障诊断。在 t-SNE/UMAP 兴起前，SOM 是数据探索的主力工具。

---

## 数学定义与原理解析

### 竞争阶段：找胜者 (BMU)

给定输入 $\\mathbf{x} \\in \\mathbb{R}^d$，网格上每个神经元 $j$ 有权重向量 $\\mathbf{w}_j \\in \\mathbb{R}^d$：

$$
b = \\arg\\min_j \\| \\mathbf{x} - \\mathbf{w}_j \\|^2
$$

### 协作阶段：邻域更新

$$
\\mathbf{w}_j(t+1) = \\mathbf{w}_j(t) + \\alpha(t) \\cdot h_{b,j}(t) \\cdot (\\mathbf{x} - \\mathbf{w}_j(t))
$$

其中：
- $\\alpha(t)$ **学习率**，随时间衰减
- $h_{b,j}(t)$ **邻域函数**（Gaussian）：

$$
h_{b,j}(t) = \\exp\\left(-\\frac{\\| \\mathbf{r}_b - \\mathbf{r}_j \\|^2}{2\\sigma^2(t)}\\right)
$$

- $\\mathbf{r}_b, \\mathbf{r}_j$ 是网格上的 2D 坐标
- $\\sigma(t)$ 是邻域半径，同样随时间衰减

### 衰减策略

$$
\\alpha(t) = \\alpha_0 \\cdot \\exp(-t / \\tau_\\alpha)
$$

$$
\\sigma(t) = \\sigma_0 \\cdot \\exp(-t / \\tau_\\sigma)
$$

典型参数：$\\alpha_0 = 0.1$，$\\sigma_0 = \\text{grid\\_size}/2$，$\\tau = \\text{epochs}$。

### U-Matrix（统一距离矩阵）

SOM 训练后的可视化工具——每个神经元与其邻居的权重距离由颜色编码，距离大的地方形成"山脊"（聚类边界）。

---

## 可视化展示

### SOM 拓扑映射

\`\`\`mermaid
graph TD
    subgraph Input["输入空间 (ℝᵈ)"]
        X["数据点 x"]
    end
    subgraph Grid["输出网格 (2D)"]
        N1["神经元₁"] --- N2["神经元₂"]
        N1 --- N3["神经元₃"]
        N2 --- N4["神经元₄"]
        N3 --- N4
    end
    X -->|"最近邻匹配"| BMU["🏆 BMU"]
    BMU -.->|"拉邻居一起更新"| N1 & N2 & N3 & N4
\`\`\`

### 邻域半径衰减

\`\`\`echarts
return {
  title: { text: 'SOM 训练过程中邻域半径的收缩', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'value', name: '迭代次数' },
  yAxis: { type: 'value', name: '邻域半径 σ(t)', min: 0, max: 5 },
  series: [{
    type: 'line', smooth: true,
    data: (function() {
      const d = [];
      for (let t = 0; t <= 100; t++) d.push([t, 5 * Math.exp(-t/30)]);
      return d;
    })(),
    lineStyle: { color: '#2c3e50', width: 2.5 }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

初期邻域大，全局拓扑成形；后期邻域小，局部细节收敛。

---

## 核心代码实现

### NumPy — SOM

\`\`\`python
import numpy as np

class SOM:
    def __init__(self, grid_h, grid_w, input_dim, lr=0.1, sigma=None):
        self.grid_h, self.grid_w = grid_h, grid_w
        self.n_neurons = grid_h * grid_w
        self.lr0 = lr
        self.sigma0 = sigma or max(grid_h, grid_w) / 2

        # 权重初始化 + 2D 网格坐标
        self.W = np.random.randn(grid_h, grid_w, input_dim) * 0.1
        y, x = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
        self.coords = np.stack([y, x], axis=-1)  # [H, W, 2]

    def _decay(self, t, max_t):
        return self.lr0 * np.exp(-t / max_t), \\
               self.sigma0 * np.exp(-t / max_t)

    def train(self, X, epochs=100):
        n = X.shape[0]
        for epoch in range(epochs):
            lr, sigma = self._decay(epoch, epochs)
            for i in np.random.permutation(n):
                # 找 BMU
                diff = self.W - X[i]  # [H, W, d]
                dist = np.sum(diff ** 2, axis=-1)  # [H, W]
                by, bx = np.unravel_index(np.argmin(dist), (self.grid_h, self.grid_w))

                # 邻域函数
                coord_diff = self.coords - np.array([by, bx])
                d2 = np.sum(coord_diff ** 2, axis=-1)  # [H, W]
                h = np.exp(-d2 / (2 * sigma ** 2))

                # 批量更新所有权重
                self.W += lr * h[..., np.newaxis] * (X[i] - self.W)

    def map(self, X):
        """返回每个输入的 BMU 坐标"""
        result = []
        for x in X:
            dist = np.sum((self.W - x) ** 2, axis=-1)
            result.append(np.unravel_index(np.argmin(dist), (self.grid_h, self.grid_w)))
        return np.array(result)
\`\`\`
`;export{n as default};
