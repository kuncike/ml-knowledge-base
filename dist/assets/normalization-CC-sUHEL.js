const n=`# BN / LN / IN / GN (归一化方法)

## 核心思想

归一化层的本质是**消除内部协变量偏移（Internal Covariate Shift）**——让每一层的输入分布保持稳定，从而允许使用更大的学习率，加速收敛。四种方法的区别仅在于 **"在哪个维度上算均值和方差"**：BN 跨 batch，LN 跨特征，IN 跨空间，GN 折中分组。

---

## 数学定义与原理解析

所有归一化方法共享同一范式：

$$
\\hat{x}_i = \\frac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}, \\quad y_i = \\gamma \\hat{x}_i + \\beta
$$

其中 $\\mu, \\sigma^2$ 在**不同的维度集合**上计算，$\\gamma, \\beta$ 是可学习的仿射参数（恢复表达能力），$\\epsilon$ 是数值稳定常数（如 $10^{-5}$）。

### Batch Normalization (BN)

在 **(N, H, W)** 三个维度上计算统计量（每个通道独立）：

$$
\\mu_c = \\frac{1}{N \\cdot H \\cdot W} \\sum_{n, h, w} x_{nchw}
$$

$$
\\sigma_c^2 = \\frac{1}{N \\cdot H \\cdot W} \\sum_{n, h, w} (x_{nchw} - \\mu_c)^2
$$

**优势**：大 batch 下统计量准确，收敛快；有轻微正则化效果（batch 统计量的随机噪声）。

**致命缺陷**：
- 小 batch（$N < 8$）下统计量极不稳定
- 训练/测试行为不一致（需维护 running mean/var）
- 不适用于 RNN（序列长度变化，无法维护固定维度的 running stats）

### Layer Normalization (LN)

在 **(C, H, W)** 或 **(C)** 维度上计算——沿**特征轴**，而非 batch 轴：

$$
\\mu_n = \\frac{1}{C} \\sum_{c=1}^{C} x_{nc}, \\quad \\sigma_n^2 = \\frac{1}{C} \\sum_{c=1}^{C} (x_{nc} - \\mu_n)^2
$$

**优势**：与 batch 大小完全无关；训练和推理行为一致；**Transformer 的标配**。

$$
\\text{LN}(\\mathbf{x}) = \\gamma \\odot \\frac{\\mathbf{x} - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta
$$

### Instance Normalization (IN)

在 **(H, W)** 维度上计算——每个样本、每个通道独立：

$$
\\mu_{nc} = \\frac{1}{HW} \\sum_{h, w} x_{nchw}
$$

主要用于风格迁移：消除每个实例特有的对比度信息，保留内容结构。

### Group Normalization (GN)

BN 与 LN 的折中：将通道分成 $G$ 组，每组内做归一化：

$$
\\mu_{ng} = \\frac{1}{(C/G) \\cdot HW} \\sum_{c \\in \\text{group}_g,\\ h, w} x_{nchw}
$$

- $G = 1$：退化为 LayerNorm
- $G = C$：退化为 InstanceNorm
- 常用 $G = 32$

---

## 可视化展示

### 四种归一化的计算维度示意

\`\`\`mermaid
graph LR
    subgraph "BN: 跨 Batch"
        A1["(N, H, W) → μ_c, σ_c"]
    end
    subgraph "LN: 跨 Feature"
        A2["(C) → μ_n, σ_n"]
    end
    subgraph "IN: 跨 Spatial"
        A3["(H, W) → μ_nc, σ_nc"]
    end
    subgraph "GN: 跨 Group"
        A4["(C/G, H, W) → μ_ng, σ_ng"]
    end
\`\`\`

### BN 的 batch size 效应

不同 batch size 下 BN 统计量的准确性对比：

\`\`\`echarts
return {
  xAxis: { type: 'category', data: ['BN(batch=2)', 'BN(batch=8)', 'BN(batch=32)', 'LN', 'GN(G=32)'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '相对稳定性' },
  legend: { data: ['训练稳定性', '小batch适应性'] },
  series: [
    {
      name: '训练稳定性', type: 'bar',
      data: [0.3, 0.65, 0.95, 0.85, 0.8],
      itemStyle: { color: '#2c3e50' }
    },
    {
      name: '小batch适应性', type: 'bar',
      data: [0.1, 0.3, 0.5, 0.95, 0.9],
      itemStyle: { color: '#16a085' }
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 70 }
}
\`\`\`

---

## 对比总结

| 方法 | 归一化维度 | 参数量 | 最佳场景 | 典型模型 |
|------|-----------|--------|----------|----------|
| BN | $N \\times H \\times W$ | $2C$ | 大 batch CNN | ResNet, VGG |
| LN | $C$ | $2C$ | Transformer, RNN | BERT, GPT, LLaMA |
| IN | $H \\times W$ | $2C$ | 风格迁移 | CycleGAN, AdaIN |
| GN | $\\frac{C}{G} \\times H \\times W$ | $2C$ | 小 batch 检测/分割 | Mask R-CNN |

## 核心代码实现

\`\`\`python
import torch
import torch.nn as nn

# BatchNorm — CNN 首选
nn.BatchNorm2d(num_features=64)   # 2D 卷积
nn.BatchNorm1d(num_features=256)  # 1D / 全连接

# LayerNorm — Transformer 标配
nn.LayerNorm(normalized_shape=512)           # 对最后一维
nn.LayerNorm(normalized_shape=[H, W, C])     # 指定归一化形状

# InstanceNorm — 风格迁移
nn.InstanceNorm2d(num_features=64)

# GroupNorm — 小 batch 检测/分割
nn.GroupNorm(num_groups=32, num_channels=128)
\`\`\`

### NumPy 手写 LN（Transformer 核心）

\`\`\`python
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta
\`\`\`
`;export{n as default};
