const n=`# Self-Attention (自注意力)

## 核心思想

自注意力的本质是**让序列中每个位置"看到"所有其他位置，并自主决定重点关注哪些位置**。它不是通过距离（如 CNN）或时序递推（如 RNN），而是通过"查询-键-值"（Q-K-V）的相似度匹配来实现全局信息交互——这是 Transformer 架构的核心。

---

## 数学定义与原理解析

### 从输入到 Q, K, V

给定输入序列 $\\mathbf{X} \\in \\mathbb{R}^{n \\times d}$，通过三个独立的线性投影得到 Query、Key、Value：

$$
\\mathbf{Q} = \\mathbf{X} \\mathbf{W}^Q, \\quad \\mathbf{K} = \\mathbf{X} \\mathbf{W}^K, \\quad \\mathbf{V} = \\mathbf{X} \\mathbf{W}^V
$$

其中 $\\mathbf{W}^Q, \\mathbf{W}^K \\in \\mathbb{R}^{d \\times d_k}$，$\\mathbf{W}^V \\in \\mathbb{R}^{d \\times d_v}$。

**直觉**：
- **Query**："我（当前位置）在寻找什么样的信息？"
- **Key**："我（每个位置）包含什么样的信息？"
- **Value**："我（每个位置）的实际内容是什么？"

### Scaled Dot-Product Attention

$$
\\text{Attention}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{softmax}\\left(\\frac{\\mathbf{Q} \\mathbf{K}^T}{\\sqrt{d_k}}\\right) \\mathbf{V}
$$

计算流程：
1. $\\mathbf{Q}\\mathbf{K}^T$：计算所有位置对的**相似度得分**（$n \\times n$ 矩阵）
2. $/\\sqrt{d_k}$：缩放，防止点积过大导致 Softmax 梯度消失
3. $\\text{softmax}$：将得分归一化为**注意力权重**（每行和为 1）
4. $\\times \\mathbf{V}$：按权重加权求和——输出是 Value 的加权平均

### 为什么除以 $\\sqrt{d_k}$？

假设 $\\mathbf{q}, \\mathbf{k}$ 的每个分量独立，均值为 0，方差为 1：

$$
\\text{Var}(q \\cdot k) = \\sum_{i=1}^{d_k} \\text{Var}(q_i k_i) = d_k \\cdot 1 \\cdot 1 = d_k
$$

当 $d_k$ 很大时（如 64 或 128），点积方差也很大 → Softmax 输入分布极端 → 梯度趋近于 0。除以 $\\sqrt{d_k}$ 将方差压回 1：

$$
\\text{Var}\\left(\\frac{q \\cdot k}{\\sqrt{d_k}}\\right) = 1
$$

---

## 可视化展示

### Self-Attention 计算流程

\`\`\`mermaid
graph TD
    X["输入 X (n×d)"] --> WQ["W^Q"] --> Q["Q (n×d_k)"]
    X --> WK["W^K"] --> K["K (n×d_k)"]
    X --> WV["W^V"] --> V["V (n×d_v)"]
    Q --> MM1["Q·K^T"]
    K --> MM1
    MM1 --> SCALE["÷ √d_k"]
    SCALE --> SM["Softmax (行归一化)"]
    SM --> ATT["注意力权重矩阵 A (n×n)"]
    ATT --> MM2["A·V"]
    V --> MM2
    MM2 --> OUT["输出 (n×d_v)"]
\`\`\`

### Softmax 温度效应：不同 $d_k$ 下的注意力分布

\`\`\`echarts
return {
  xAxis: { type: 'category', data: ['词1', '词2', '词3', '词4', '词5'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '注意力权重' },
  legend: { data: ['d_k=4 (未缩放)', 'd_k=64 (缩放后)'] },
  series: [
    {
      name: 'd_k=4 (未缩放)', type: 'bar',
      data: [0.02, 0.03, 0.88, 0.04, 0.03],
      itemStyle: { color: '#c0392b' }
    },
    {
      name: 'd_k=64 (缩放后)', type: 'bar',
      data: [0.12, 0.18, 0.40, 0.17, 0.13],
      itemStyle: { color: '#2980b9' }
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 60 }
}
\`\`\`

不缩放时注意力过于集中（Softmax 饱和）；缩放后分布更平滑，梯度流动更健康。

---

## 核心代码实现

### PyTorch 完整实现

\`\`\`python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k=None):
        super().__init__()
        d_k = d_k or d_model
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
        self.scale = math.sqrt(d_k)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = Q @ K.transpose(-2, -1) / self.scale
        attn = torch.softmax(scores, dim=-1)
        return attn @ V
\`\`\`

### NumPy 手写

\`\`\`python
import numpy as np

def self_attention(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    attn_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))  # 数值稳定 softmax
    attn_weights /= attn_weights.sum(axis=-1, keepdims=True)
    return attn_weights @ V
\`\`\`

---

## 复杂度分析

| 指标 | 复杂度 | 瓶颈 |
|------|--------|------|
| 时间 | $O(n^2 d)$ | $n$ 为序列长度 |
| 空间 | $O(n^2)$ | 注意力矩阵 $n \\times n$ |
| 参数量 | $3 \\cdot d \\cdot d_k$ | 不含序列长度 |

**长序列的根本瓶颈**：$n^2$ 的注意力矩阵。当 $n=4096$ 时，单个注意力矩阵占用 $4096^2 \\times 4\\text{bytes} \\approx 67\\text{MB}$。这催生了 FlashAttention、Sparse Attention、Linformer 等优化。

## 自注意力 vs RNN vs CNN

| 特性 | Self-Attention | RNN | CNN |
|------|---------------|-----|-----|
| 全局感受野 | ✅ $O(1)$ 步 | ❌ $O(n)$ 步 | 受限（kernel 大小） |
| 并行计算 | ✅ 完全并行 | ❌ 串行 | ✅ 并行 |
| 序列长度敏感 | $O(n^2)$ | $O(n)$ | $O(n)$ |
| 位置信息 | 需显式编码 | 隐式 | 隐式 |
`;export{n as default};
