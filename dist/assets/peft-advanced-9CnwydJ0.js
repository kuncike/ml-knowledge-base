const n=`# PEFT Advanced (AdaLoRA / DoRA / IA3)

## 核心思想

LoRA 对所有层用相同的秩 $r$，但不同层/矩阵对任务的重要性不同——有的需要大改动，有的只需微调。AdaLoRA 自动为重要矩阵分配更高秩（基于 SVD 奇异值大小），DoRA 将权重分解为**幅值 + 方向**（分别微调），IA3 极端到每层只学 3 个缩放向量。共同方向：**在不牺牲效果的前提下，进一步缩减可训练参数和显存**。

---

## 数学定义与原理解析

### AdaLoRA — 自适应秩分配

将 $\\Delta W$ 写为 SVD 形式：

$$
\\Delta W = P \\Lambda Q, \\quad \\Lambda = \\text{diag}(\\sigma_1, \\ldots, \\sigma_r)
$$

训练时根据 $\\sigma_i$ 的重要性（由损失对 $\\sigma_i$ 的敏感度衡量）动态调整秩——不重要的奇异值被剪掉，节省的"秩预算"分配给其他层。

重要性度量：

$$
I(\\sigma_i) = \\left| \\frac{\\partial \\mathcal{L}}{\\partial \\sigma_i} \\cdot \\sigma_i \\right|
$$

### DoRA — 权重分解

将预训练权重 $W_0$ 分解为幅值 $m$ 和方向 $V$：

$$
W = m \\cdot \\frac{V}{\\|V\\|_c}
$$

其中 $\\|V\\|_c$ 是逐列的 L2 范数。然后对方向部分施加 LoRA：

$$
W' = \\underbrace{\\|W_0\\|_c \\cdot (1 + \\Delta m)}_{\\text{幅值 Adapter}} \\cdot \\underbrace{\\frac{W_0 + B A}{\\|W_0 + B A\\|_c}}_{\\text{方向 LoRA}}
$$

$\\Delta m$ 和 $B, A$ 都是可训练参数。幅值和方向分开学习——方向变化更大时幅值也不一定需要变。

### IA3 — 极简 Adapter

只添加 3 个可学习向量——Key/value 缩放 $\\ell_v \\in \\mathbb{R}^{d_k}$、Query 缩放 $\\ell_q \\in \\mathbb{R}^{d_k}$、FFN 缩放 $\\ell_f \\in \\mathbb{R}^{d_{ff}}$：

$$
\\text{Attention}: Q' = \\ell_q \\odot Q, \\; K' = \\ell_k \\odot K, \\; V' = \\ell_v \\odot V
$$

$$
\\text{FFN}: x' = \\ell_f \\odot \\text{FFN}(x)
$$

可训练参数量：$L \\times (3d_k + d_{ff})$，对 T0-3B 仅需 0.24M 参数（<0.01%）。

### PEFT 方法对比

| 方法 | 可训练参数 | 额外推理开销 | 特点 |
|------|----------|------------|------|
| LoRA | ~0.5% | 0 (可合并) | 低秩分解 |
| AdaLoRA | ~0.5% | 0 | 自适应秩 |
| DoRA | ~0.6% | 0 | 幅值+方向分解 |
| IA3 | <0.01% | 极低 | 仅缩放向量 |

---

## 可视化展示

### LoRA → DoRA 的演化

\`\`\`mermaid
graph TD
    W0["W₀ (冻结)"] --> LORA["LoRA<br/>W = W₀ + BA<br/>(秩 r 分解)"]
    LORA --> ADALORA["AdaLoRA<br/>W = W₀ + PΛQ<br/>(自适应秩分配)"]
    ADALORA --> DORA["DoRA<br/>W = m · (W₀+BA)/‖W₀+BA‖<br/>(幅值+方向分开学)"]
\`\`\`

### 参数效率对比

\`\`\`echarts
return {
  title: { text: 'PEFT 方法可训练参数 vs 性能 (LLaMA-7B)', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'value', name: '可训练参数 (%)', min: 0, max: 1 },
  yAxis: { type: 'value', name: 'MT-Bench Score', min: 5, max: 7 },
  series: [
    { type: 'scatter', symbolSize: 16,
      data: [[0.5, 6.5], [0.6, 6.6], [0.06, 6.3], [0.01, 6.1]],
      label: { show: true, formatter: (p) => ['LoRA','DoRA','AdaLoRA','IA3'][p.dataIndex], position: 'top' }
    }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — IA3

\`\`\`python
import torch
import torch.nn as nn

class IA3Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear  # 冻结
        self.linear.weight.requires_grad = False
        # 可学习的缩放向量
        self.l = nn.Parameter(torch.ones(linear.out_features))

    def forward(self, x):
        return self.l * self.linear(x)


class IA3Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

        # 冻结原始权重
        for p in [self.W_q, self.W_k, self.W_v]:
            p.weight.requires_grad = False

        # IA3 缩放向量
        self.l_q = nn.Parameter(torch.ones(dim))
        self.l_k = nn.Parameter(torch.ones(dim))
        self.l_v = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        Q = self.l_q * self.W_q(x)
        K = self.l_k * self.W_k(x)
        V = self.l_v * self.W_v(x)
        # ... 后续标准 attention 操作
        return Q, K, V
\`\`\`

### PyTorch — DoRA (权重分解)

\`\`\`python
class DoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        in_f, out_f = linear.in_features, linear.out_features

        # LoRA 低秩矩阵 (仅在方向部分)
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        # 幅值适配器
        self.magnitude = nn.Parameter(torch.ones(out_f))
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        W0 = self.linear.weight  # [out, in]
        # 方向: W0 + LoRA 更新, 然后规范化
        delta = (self.lora_B @ self.lora_A) * (self.alpha / self.rank)
        W_dir = W0 + delta
        W_dir_norm = W_dir / W_dir.norm(dim=1, keepdim=True)
        # 幅值: 预训练幅值 × (1 + 适配幅值)
        init_mag = W0.norm(dim=1, keepdim=True)
        W = init_mag * self.magnitude.unsqueeze(1) * W_dir_norm
        return x @ W.T + self.linear.bias
\`\`\`
`;export{n as default};
