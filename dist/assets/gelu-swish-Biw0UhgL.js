const n=`# GELU / Swish / Mish

## 核心思想

这是 Transformer 时代的激活函数三剑客。它们抛弃了 ReLU 的"硬阈值"设计，改用**光滑、非单调**的曲线。直觉上：ReLU 是"一刀切"，GELU/Swish 是"S 曲线调节"——输入越不确定，输出越趋近于零，自带一种类似 Dropout 的随机正则化效果。

---

## 数学定义与原理解析

### GELU (Gaussian Error Linear Unit)

BERT、GPT、ViT 等主流 Transformer 的默认激活函数。

**精确形式**（基于标准正态分布的 CDF）：

$$
\\text{GELU}(x) = x \\cdot \\Phi(x) = x \\cdot \\frac{1}{2} \\left[1 + \\text{erf}\\left(\\frac{x}{\\sqrt{2}}\\right)\\right]
$$

**工程近似**（Tanh 近似，速度快，误差极小）：

$$
\\text{GELU}(x) \\approx 0.5x \\left(1 + \\tanh\\left[\\sqrt{\\frac{2}{\\pi}}(x + 0.044715 x^3)\\right]\\right)
$$

**直觉**：$\\Phi(x)$ 是标准正态的累积分布函数，表示 "$x$ 相对于随机噪声有多大"。GELU = 输入 × 被激活的概率。这赋予了它自适应的"软门控"特性。

### Swish / SiLU

Google 通过自动架构搜索（NAS）发现，被 EfficientNet 采纳。

$$
\\text{Swish}(x) = x \\cdot \\sigma(x) = \\frac{x}{1 + e^{-x}}
$$

带可学习参数 $\\beta$ 的版本：

$$
\\text{Swish}_\\beta(x) = x \\cdot \\sigma(\\beta x)
$$

- $\\beta \\to \\infty$：退化为 ReLU
- $\\beta \\to 0$：退化为线性函数 $y = 0.5x$
- $\\beta = 1$：标准 Swish = SiLU（PyTorch 中两者等价）

### Mish

YOLOv4 中引入，据称在部分任务上优于 Swish 和 ReLU。

$$
\\text{Mish}(x) = x \\cdot \\tanh(\\text{softplus}(x)) = x \\cdot \\tanh(\\ln(1 + e^x))
$$

- 下方**无界**：允许大的负输出（正则化效果）
- 上方**有界**：避免梯度爆炸
- 光滑且非单调

---

## 可视化展示

### GELU / Swish / Mish / ReLU 对比

\`\`\`echarts
return {
  xAxis: { type: 'value', min: -4, max: 4, name: 'x' },
  yAxis: { type: 'value', min: -0.8, max: 4, name: 'f(x)' },
  legend: { data: ['ReLU', 'GELU', 'Swish', 'Mish'] },
  series: [
    {
      name: 'ReLU', type: 'line',
      lineStyle: { color: '#95a5a6', width: 1.5, type: 'dashed' },
      data: (function() { const d = []; for (let i = -4; i <= 4; i += 0.03) d.push([i, Math.max(0, i)]); return d; })()
    },
    {
      name: 'GELU', type: 'line', smooth: true,
      lineStyle: { color: '#2c3e50', width: 2.5 },
      data: (function() {
        const d = [];
        for (let i = -4; i <= 4; i += 0.03) {
          const x = i;
          const gelu = 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
          d.push([i, gelu]);
        }
        return d;
      })()
    },
    {
      name: 'Swish', type: 'line', smooth: true,
      lineStyle: { color: '#2980b9', width: 2 },
      data: (function() {
        const d = [];
        for (let i = -4; i <= 4; i += 0.03) d.push([i, i / (1 + Math.exp(-i))]);
        return d;
      })()
    },
    {
      name: 'Mish', type: 'line', smooth: true,
      lineStyle: { color: '#d35400', width: 2 },
      data: (function() {
        const d = [];
        for (let i = -4; i <= 4; i += 0.03) {
          const sp = Math.log(1 + Math.exp(i));
          d.push([i, i * Math.tanh(sp)]);
        }
        return d;
      })()
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

### 负区间放大对比（$x \\in [-4, 0]$）

注意 GELU/Mish 在负区间的差异——这影响梯度流动和正则化行为：

\`\`\`echarts
return {
  xAxis: { type: 'value', min: -4, max: 0, name: 'x' },
  yAxis: { type: 'value', min: -0.6, max: 0.4, name: 'f(x)' },
  legend: { data: ['GELU', 'Swish', 'Mish'] },
  series: [
    {
      name: 'GELU', type: 'line', smooth: true,
      lineStyle: { color: '#2c3e50', width: 2 },
      data: (function() {
        const d = [];
        for (let i = -4; i <= 0; i += 0.02) {
          const x = i;
          const gelu = 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
          d.push([i, gelu]);
        }
        return d;
      })()
    },
    {
      name: 'Swish', type: 'line', smooth: true,
      lineStyle: { color: '#2980b9', width: 2 },
      data: (function() { const d = []; for (let i = -4; i <= 0; i += 0.02) d.push([i, i / (1 + Math.exp(-i))]); return d; })()
    },
    {
      name: 'Mish', type: 'line', smooth: true,
      lineStyle: { color: '#d35400', width: 2 },
      data: (function() {
        const d = [];
        for (let i = -4; i <= 0; i += 0.02) {
          d.push([i, i * Math.tanh(Math.log(1 + Math.exp(i)))]);
        }
        return d;
      })()
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

# GELU — Transformer 标配
nn.GELU()               # 精确版（erf）
nn.GELU(approximate='tanh')  # 快速近似版

# SiLU (Swish) — PyTorch 内置
nn.SiLU()               # x * sigmoid(x)
F.silu(x)

# Mish — 需手动实现或使用 timm
def mish(x):
    return x * torch.tanh(F.softplus(x))
\`\`\`

### NumPy 手写

\`\`\`python
import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(
        np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    ))

def swish(x, beta=1.0):
    return x / (1 + np.exp(-beta * x))

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))
\`\`\`

---

## 选型建议

| 函数 | 推荐场景 | 说明 |
|------|----------|------|
| GELU | Transformer 隐藏层首选 | BERT/GPT/ViT 的默认选择，久经考验 |
| Swish/SiLU | 轻量级 CNN、EfficientNet 类架构 | 比 GELU 稍快，效果相当 |
| Mish | 目标检测（YOLOv4） | 在部分视觉任务上可能优于 Swish |
| ReLU | 简单 CNN，对推理延迟敏感的部署 | 计算开销最小 |
`;export{n as default};
