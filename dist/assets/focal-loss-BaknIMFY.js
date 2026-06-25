const n=`# Focal Loss / Triplet Loss / Contrastive Loss

## 核心思想

标准交叉熵对所有样本"一视同仁"。Focal Loss 的直觉是**"让模型专注于它还不擅长的样本"**——如果一个样本已经被预测得很准（$p_t \\approx 0.9$），它的 loss 应该被大幅降权；如果模型还在纠结（$p_t \\approx 0.1$），loss 几乎不变。这在目标检测的极端类别不平衡（99% 是背景）中至关重要。

---

## 数学定义与原理解析

### Focal Loss

$$
\\text{FL}(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)
$$

其中：

$$
p_t = \\begin{cases} \\hat{y} & \\text{if } y = 1 \\\\ 1 - \\hat{y} & \\text{otherwise} \\end{cases}
$$

$p_t$ 是"模型对正确类别的预测概率"——越接近 1 越好。

- **$(1 - p_t)^\\gamma$**：**调制因子**（核心创新），降低易分类样本的权重
- **$\\gamma \\geq 0$**：聚焦参数，$\\gamma = 0$ 退化为标准交叉熵，$\\gamma = 2$ 效果最好
- **$\\alpha_t$**：类别权重，平衡正负样本（可选）

**数值示例**（$\\gamma = 2$）：

| $p_t$ | 调制因子 $(1-p_t)^2$ | Loss 缩放倍数 |
|-------|---------------------|--------------|
| 0.9（易） | $(0.1)^2 = 0.01$ | **缩小 100×** |
| 0.5（中） | $(0.5)^2 = 0.25$ | 缩小 4× |
| 0.1（难） | $(0.9)^2 = 0.81$ | 几乎不变 |

### Triplet Loss

用于度量学习（人脸识别、图像检索）：

$$
L = \\max(0, \\|f(a) - f(p)\\|^2 - \\|f(a) - f(n)\\|^2 + \\alpha)
$$

- $a$（Anchor）：锚点样本
- $p$（Positive）：同类样本
- $n$（Negative）：异类样本
- $\\alpha$（Margin）：最小间隔

目标：锚点到正样本的距离 < 锚点到负样本的距离，至少差 $\\alpha$。

### InfoNCE Loss（对比学习标配）

SimCLR、MoCo 等对比学习方法使用：

$$
L = -\\log \\frac{\\exp(q \\cdot k_+ / \\tau)}{\\exp(q \\cdot k_+ / \\tau) + \\sum_{k_-} \\exp(q \\cdot k_- / \\tau)}
$$

直觉：$K+1$ 个候选中，拉近正样本对 $(q, k_+)$，推远所有负样本 $(q, k_-)$。$\\tau$ 是温度参数。

---

## 可视化展示

### Focal Loss 的调制因子效应

\`\`\`echarts
return {
  xAxis: { type: 'value', min: 0, max: 1, name: 'pₜ (正确类别的预测概率)' },
  yAxis: { type: 'value', min: 0, max: 5, name: 'Loss' },
  legend: { data: ['CE (γ=0)', 'Focal γ=1', 'Focal γ=2', 'Focal γ=5'] },
  series: [
    {
      name: 'CE (γ=0)', type: 'line', smooth: true,
      lineStyle: { color: '#95a5a6', width: 2 },
      data: (function() { const d = []; for (let p = 0.001; p <= 1; p += 0.002) d.push([p, -Math.log(p)]); return d; })()
    },
    {
      name: 'Focal γ=1', type: 'line', smooth: true,
      lineStyle: { color: '#2980b9', width: 2 },
      data: (function() { const d = []; for (let p = 0.001; p <= 1; p += 0.002) d.push([p, -(1-p)*Math.log(p)]); return d; })()
    },
    {
      name: 'Focal γ=2', type: 'line', smooth: true,
      lineStyle: { color: '#2c3e50', width: 2.5 },
      data: (function() { const d = []; for (let p = 0.001; p <= 1; p += 0.002) d.push([p, -(1-p)*(1-p)*Math.log(p)]); return d; })()
    },
    {
      name: 'Focal γ=5', type: 'line', smooth: true,
      lineStyle: { color: '#c0392b', width: 2 },
      data: (function() { const d = []; for (let p = 0.001; p <= 1; p += 0.002) d.push([p, -Math.pow(1-p,5)*Math.log(p)]); return d; })()
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

$\\gamma$ 越大，易分类样本（$p_t$ 接近 1）的 loss 被压缩得越狠。当 $\\gamma=0$（标准 CE），$p_t=0.9$ 的 loss 仍有 0.1；而 $\\gamma=2$ 时仅 0.001。

### 不同 $\\gamma$ 下 Loss 的"压缩比"

\`\`\`echarts
return {
  xAxis: { type: 'category', data: ['pₜ=0.5', 'pₜ=0.7', 'pₜ=0.9', 'pₜ=0.99'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '相对 CE 的 Loss 比例' },
  legend: { data: ['CE (γ=0)', 'γ=1', 'γ=2'] },
  series: [
    { name: 'CE (γ=0)', type: 'bar', data: [1, 1, 1, 1], itemStyle: { color: '#95a5a6' } },
    { name: 'γ=1', type: 'bar', data: [0.5, 0.3, 0.1, 0.01], itemStyle: { color: '#2980b9' } },
    { name: 'γ=2', type: 'bar', data: [0.25, 0.09, 0.01, 0.0001], itemStyle: { color: '#2c3e50' } }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — Focal Loss

\`\`\`python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    p_t = torch.exp(-ce_loss)               # p_t = exp(-CE)
    focal_weight = (1 - p_t) ** gamma       # 调制因子
    return (alpha * focal_weight * ce_loss).mean()
\`\`\`

### PyTorch — Triplet Loss

\`\`\`python
import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(dim=1)
        neg_dist = (anchor - negative).pow(2).sum(dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)
        return loss.mean()
\`\`\`
`;export{n as default};
