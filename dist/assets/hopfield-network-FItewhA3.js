const n=`# Hopfield Network (霍普菲尔德网络)

## 核心思想

Hopfield 网络是**内容寻址存储器**——你给它一个部分损坏的记忆片段，它沿着能量下降的方向自动恢复到最相似的完整记忆。每个存储的模式（pattern）都是能量景观中的一个局部极小值。现代关联：**现代 Hopfield 网络**（Dense Associative Memory）用指数存储容量重新定义了这一范式，直接催生了 Transformer 的注意力机制（Krotov & Hopfield, 2016）。

---

## 数学定义与原理解析

### 能量函数

对于 $N$ 个二元神经元 $\\mathbf{s} \\in \\{-1, +1\\}^N$：

$$
E(\\mathbf{s}) = -\\frac12 \\sum_{i,j} w_{ij} s_i s_j - \\sum_i b_i s_i
$$

无自连接：$w_{ii} = 0$。权重对称：$w_{ij} = w_{ji}$，保证能量单调下降（Lyapunov 函数）。

### Hebbian 学习规则

存储 $P$ 个模式 $\\{\\xi^\\mu\\}_{\\mu=1}^{P}$：

$$
w_{ij} = \\frac{1}{N} \\sum_{\\mu=1}^{P} \\xi_i^\\mu \\xi_j^\\mu, \\quad w_{ii} = 0
$$

### 异步更新

每次随机选一个神经元 $i$，计算局部场：

$$
h_i = \\sum_j w_{ij} s_j + b_i
$$

更新：$s_i \\leftarrow \\text{sign}(h_i)$。能量必定下降或不变：

$$
\\Delta E_i = -2 s_i h_i \\leq 0
$$

### 存储容量

Hebbian 学习下的理论容量约为 $P_{\\max} \\approx 0.14N$（$N$ 为神经元数量）。超过此值，记忆之间产生串扰，能量曲面出现虚假的"混合态"极小值。

### 现代 Hopfield 网络（连续状态）

$$
E(\\mathbf{x}) = -\\frac{1}{\\beta} \\log \\sum_{\\mu=1}^{P} \\exp(\\beta \\mathbf{x}^T \\xi^\\mu) + \\frac12 \\mathbf{x}^T \\mathbf{x}
$$

更新规则与 Transformer 注意力惊人相似：
$$
\\mathbf{x}^{\\text{new}} = \\sum_{\\mu} \\text{softmax}(\\beta \\mathbf{x}^T \\Xi)_{\\mu} \\cdot \\xi^\\mu
$$

---

## 可视化展示

### Hopfield 网络结构（全连接 + 对称权重）

\`\`\`mermaid
graph TD
    S1["s₁"] <-->|"w₁₂=w₂₁"| S2["s₂"]
    S1 <-->|"w₁₃=w₃₁"| S3["s₃"]
    S1 <-->|"w₁₄=w₄₁"| S4["s₄"]
    S2 <-->|"w₂₃=w₃₂"| S3
    S2 <-->|"w₂₄=w₄₂"| S4
    S3 <-->|"w₃₄=w₄₃"| S4
\`\`\`

### 能量下降过程

\`\`\`echarts
return {
  title: { text: '异步更新中能量单调下降', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'value', name: '更新步数' },
  yAxis: { type: 'value', name: '能量 E(s)', min: -60, max: -20 },
  series: [{
    type: 'line', smooth: true,
    data: (function() {
      const d = [];
      for (let i = 0; i <= 20; i++) d.push([i, -55 + 35 * Math.exp(-i/4)]);
      return d;
    })(),
    lineStyle: { color: '#2c3e50', width: 2.5 },
    areaStyle: { color: 'rgba(44, 62, 80, 0.1)' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

每一步异步更新保证 $\\Delta E \\leq 0$——网络在能量景观中"滚动"到最近的局部极小值。

---

## 核心代码实现

### NumPy — Hopfield 网络

\`\`\`python
import numpy as np

class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))

    def store(self, patterns):
        """用 Hebbian 规则存储模式"""
        for p in patterns:
            self.W += np.outer(p, p)
        self.W /= self.n
        np.fill_diagonal(self.W, 0)

    def energy(self, s):
        return -0.5 * s @ self.W @ s

    def recall(self, query, steps=100):
        """异步更新恢复记忆"""
        s = query.copy()
        for _ in range(steps):
            i = np.random.randint(self.n)
            h = self.W[i] @ s
            s[i] = np.sign(h)
        return s

    def capacity_check(self, patterns):
        """测试存储容量：存储后能否正确检索"""
        self.store(patterns)
        success = 0
        for p in patterns:
            retrieved = self.recall(p.copy())
            if np.array_equal(retrieved, p):
                success += 1
        return success / len(patterns)
\`\`\`
`;export{n as default};
