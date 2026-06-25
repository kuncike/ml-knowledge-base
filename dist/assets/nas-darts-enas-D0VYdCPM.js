const n=`# NAS / DARTS / ENAS (神经架构搜索)

## 核心思想

手工设计网络架构需要大量专家经验和试错。NAS 的目标是**让算法自动发现最优架构**。早期 NAS 用强化学习或进化算法在离散搜索空间中采样——计算成本极高（成千上万 GPU 小时）。DARTS 的革命性贡献：**将离散的架构选择松弛为连续权重**，使得整个搜索过程可微分——用梯度下降同时优化架构参数和网络权重。ENAS 进一步加速：让所有候选架构**共享参数**，彻底避免了从零训练每个候选。

---

## 数学定义与原理解析

### DARTS — 可微分架构搜索

搜索空间定义为一个**有向无环图**（DAG），每个节点 $x^{(j)}$ 是其所有前驱节点 $x^{(i)}$ 的变换的加权和：

$$
x^{(j)} = \\sum_{i < j} \\sum_{o \\in \\mathcal{O}} \\frac{\\exp(\\alpha_{o}^{(i,j)})}{\\sum_{o' \\in \\mathcal{O}} \\exp(\\alpha_{o'}^{(i,j)})} \\cdot o(x^{(i)})
$$

- $\\mathcal{O}$：候选操作集合（卷积 3×3、卷积 5×5、池化 3×3、跳跃连接、None）
- $\\alpha_{o}^{(i,j)}$：架构参数（操作 $o$ 在边 $(i,j)$ 上的权重）
- Softmax 将离散选择松弛为连续混合

### 双层优化

$$
\\min_\\alpha \\mathcal{L}_{val}(w^*(\\alpha), \\alpha)
$$

$$
\\text{s.t.} \\quad w^*(\\alpha) = \\arg\\min_w \\mathcal{L}_{train}(w, \\alpha)
$$

- **外层**：在验证集上优化架构参数 $\\alpha$
- **内层**：在训练集上优化网络权重 $w$

实际用一阶近似（交替梯度下降，不等到 $w^*$ 收敛）：

$$
\\nabla_\\alpha \\mathcal{L}_{val}(w - \\xi \\nabla_w \\mathcal{L}_{train}(w, \\alpha), \\alpha)
$$

### ENAS — 参数共享

所有候选子网络**共享同一套权重**，由 RNN controller 选择子网络。训练流程：

1. Controller 采样一个子网络（选择每个节点的操作和连接）
2. 用该子网络做前向+反向（使用共享权重）
3. 更新共享权重 + Controller 参数

相比 NAS：30000 GPU 小时 → **0.5 GPU 小时**（约 60000× 加速）。

### 最终架构生成

搜索完成后，对每条边 $(i,j)$ 选 $\\arg\\max_o \\alpha_{o}^{(i,j)}$ 作为最终操作。

---

## 可视化展示

### DARTS 搜索空间

\`\`\`mermaid
graph TD
    C0["c_{k-2}"] --> N0["节点 0"]
    C1["c_{k-1}"] --> N0
    C0 --> N1["节点 1"]
    C1 --> N1
    N0 --> N1

    N0 --> N2["节点 2"]
    N1 --> N2
    N0 --> N3["节点 3"]
    N1 --> N3
    N2 --> N3

    N2 --> OUT["c_k (concat)"]
    N3 --> OUT

    N0 -.-|"候选操作: 3×3 Conv, 5×5 Conv, \\n 3×3 Pool, Identity, Zero"| N0
\`\`\`

### NAS 效率对比

\`\`\`echarts
return {
  title: { text: 'NAS 方法 GPU 开销对比', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['NAS-RL', 'NAS-EA', 'DARTS', 'ENAS', 'OFA'] },
  yAxis: { type: 'log', name: 'GPU Hours (log scale)', min: 0, max: 5 },
  series: [{
    type: 'bar',
    data: [31500, 3150, 4, 0.5, 0.08],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top',
      formatter: (p) => p.value > 100 ? (p.value/1000).toFixed(1)+'K' : p.value }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — DARTS Cell (搜索阶段)

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
    'none': lambda C, stride: Zero(stride),
    'skip_connect': lambda C, stride: Identity(),
    'conv_3x3': lambda C, stride: nn.Sequential(
        nn.ReLU(), nn.Conv2d(C, C, 3, stride, 1, bias=False), nn.BatchNorm2d(C)),
    'conv_5x5': lambda C, stride: nn.Sequential(
        nn.ReLU(), nn.Conv2d(C, C, 5, stride, 2, bias=False), nn.BatchNorm2d(C)),
    'avg_pool_3x3': lambda C, stride: nn.AvgPool2d(3, stride, 1),
}

class MixedOp(nn.Module):
    """DARTS 的混合操作——所有候选操作的加权和"""
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList([OP(name)(C, stride) for name in OPS])

    def forward(self, x, weights):
        # weights: [n_ops] — softmax(alpha)
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class DARTSCell(nn.Module):
    def __init__(self, C, n_nodes=4):
        super().__init__()
        self.n_nodes = n_nodes
        self.edges = nn.ModuleList()
        for i in range(n_nodes):
            for j in range(i + 2):  # 输入: c_{k-2}, c_{k-1}, node_0, ..., node_{i-1}
                stride = 1
                self.edges.append(MixedOp(C, stride))

    def forward(self, x, weights):
        # x: [c_{k-2}, c_{k-1}]
        states = [x[0], x[1]]
        offset = 0
        for i in range(self.n_nodes):
            s = 0
            for j, h in enumerate(states):
                s += self.edges[offset + j](h, weights[offset + j])
            offset += len(states)
            states.append(s)
        return torch.cat(states[2:], dim=1)  # concat 所有中间节点


class DARTSNetwork(nn.Module):
    def __init__(self, C=16, n_layers=8, n_nodes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1), nn.BatchNorm2d(C))

        self.cells = nn.ModuleList()
        for _ in range(n_layers):
            self.cells.append(DARTSCell(C, n_nodes))

        # 架构参数 alpha: 每个边, 每个操作一个参数
        n_edges = sum(i + 2 for i in range(n_nodes))
        self.alpha = nn.Parameter(torch.randn(n_edges, len(OPS)))

    def forward(self, x):
        x = self.stem(x)
        weights = F.softmax(self.alpha, dim=-1)
        s0 = s1 = x
        for cell in self.cells:
            s0, s1 = s1, cell([s0, s1], weights)
        return s1
\`\`\`
`;export{n as default};
