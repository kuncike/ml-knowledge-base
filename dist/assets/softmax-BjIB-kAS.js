const n=`# Softmax

## 核心思想

Softmax 将任意实数向量 $\\mathbf{z} \\in \\mathbb{R}^K$ 映射为概率分布——每个分量在 $(0, 1)$ 之间，且所有分量之和为 1。可以理解为 $\\arg\\max$ 的"可微近似"：通过指数函数放大差异，再归一化。温度参数 $\\tau$ 控制分布的尖锐程度，是知识蒸馏的核心工具。

---

## 数学定义与原理解析

### 标准 Softmax

$$
\\text{softmax}(\\mathbf{z})_i = \\frac{e^{z_i}}{\\sum_{j=1}^{K} e^{z_j}}
$$

**关键性质**：

- **平移不变性**：$\\text{softmax}(\\mathbf{z} + c) = \\text{softmax}(\\mathbf{z})$，推导中分子分母的 $e^c$ 相消
- **输出和为 1**：天然构成合法概率分布
- **非饱和性**：不同于 Sigmoid，Softmax 始终有梯度流动

### 带温度的 Softmax

$$
\\text{softmax}(\\mathbf{z}/\\tau)_i = \\frac{e^{z_i/\\tau}}{\\sum_{j=1}^{K} e^{z_j/\\tau}}
$$

| $\\tau$ 值 | 分布形态 | 用途 |
|-----------|----------|------|
| $\\tau \\to 0$ | 趋近 one-hot（argmax） | 确定性推理 |
| $\\tau = 1$ | 标准 Softmax | 常规训练 |
| $\\tau > 1$ | 平滑、均匀 | 知识蒸馏（捕获"暗知识"） |

### 数值稳定性

直接计算 $e^{z_i}$ 当 $z_i$ 很大时会导致上溢。标准做法是减去最大值：

$$
\\text{softmax}(\\mathbf{z})_i = \\frac{e^{z_i - \\max(\\mathbf{z})}}{\\sum_j e^{z_j - \\max(\\mathbf{z})}}
$$

这等价于令 $c = -\\max(\\mathbf{z})$，不改变结果但保证指数最大为 $e^0 = 1$。

### Softmax + 交叉熵的优雅梯度

这是 Softmax 如此流行的深层原因——与交叉熵结合后，梯度形式极其简洁：

$$
\\frac{\\partial L}{\\partial z_i} = \\hat{y}_i - y_i
$$

其中 $\\hat{y}_i$ 是 Softmax 输出，$y_i$ 是 one-hot 标签。"预测值 - 真实值"——形式与线性回归的 MSE 梯度一致。

---

## 可视化展示

### 温度参数对 Softmax 分布的影响

对一个三维 logits $(3, 1, 0)$ 应用不同温度的 Softmax：

\`\`\`echarts
return {
  xAxis: { type: 'category', data: ['类别 A (logit=3)', '类别 B (logit=1)', '类别 C (logit=0)'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '概率' },
  legend: { data: ['τ=0.2', 'τ=0.5', 'τ=1.0', 'τ=2.0', 'τ=5.0'] },
  series: (function() {
    const logits = [3, 1, 0];
    const taus = [0.2, 0.5, 1.0, 2.0, 5.0];
    const colors = ['#c0392b', '#d35400', '#2c3e50', '#16a085', '#2980b9'];
    return taus.map(function(tau, idx) {
      const scaled = logits.map(function(z) { return Math.exp(z / tau); });
      const sum = scaled.reduce(function(a, b) { return a + b; }, 0);
      const probs = scaled.map(function(s) { return +(s / sum).toFixed(4); });
      return {
        name: 'τ=' + tau, type: 'bar',
        data: probs,
        itemStyle: { color: colors[idx] }
      };
    });
  })(),
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 60 }
}
\`\`\`

### Softmax 函数曲面（二维输入 → 二维概率）

\`\`\`echarts
return {
  xAxis: { type: 'value', min: -5, max: 5, name: 'z₁ - z₂' },
  yAxis: { type: 'value', min: 0, max: 1, name: 'P(类别1)' },
  legend: { data: ['τ=0.5', 'τ=1.0', 'τ=2.0'] },
  series: [
    {
      name: 'τ=0.5', type: 'line', smooth: true,
      lineStyle: { color: '#d35400', width: 2 },
      data: (function() {
        const d = [];
        for (let i = -5; i <= 5; i += 0.05) {
          const s1 = Math.exp(i / 0.5), s2 = Math.exp(0);
          d.push([i, s1 / (s1 + s2)]);
        }
        return d;
      })()
    },
    {
      name: 'τ=1.0', type: 'line', smooth: true,
      lineStyle: { color: '#2c3e50', width: 2.5 },
      data: (function() {
        const d = [];
        for (let i = -5; i <= 5; i += 0.05) {
          const s1 = Math.exp(i), s2 = Math.exp(0);
          d.push([i, s1 / (s1 + s2)]);
        }
        return d;
      })()
    },
    {
      name: 'τ=2.0', type: 'line', smooth: true,
      lineStyle: { color: '#2980b9', width: 2 },
      data: (function() {
        const d = [];
        for (let i = -5; i <= 5; i += 0.05) {
          const s1 = Math.exp(i / 2), s2 = Math.exp(0);
          d.push([i, s1 / (s1 + s2)]);
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

### PyTorch 实现

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Softmax 层
softmax = nn.Softmax(dim=-1)

# 数值稳定版本：直接使用 CrossEntropyLoss（内置 log_softmax + NLL）
loss_fn = nn.CrossEntropyLoss()

# 带温度的 Softmax
def temperature_softmax(logits, T=1.0):
    return F.softmax(logits / T, dim=-1)
\`\`\`

### NumPy 手写

\`\`\`python
import numpy as np

def softmax(z, axis=-1):
    z_max = np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

def temperature_softmax(z, T=1.0):
    z_max = np.max(z / T, axis=-1, keepdims=True)
    exp_z = np.exp(z / T - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
\`\`\`

---

## 应用场景总览

| 场景 | Softmax 的角色 |
|------|---------------|
| 多分类输出层 | 将 logits 转为 $K$ 类概率分布 |
| Self-Attention | $\\text{softmax}(QK^T/\\sqrt{d_k})$ 归一化注意力权重 |
| 知识蒸馏 | 高温度 $\\tau$ 软化教师输出，暴露"暗知识" |
| 强化学习 | 动作选择的概率分布（Policy Gradient） |
| 对比学习 | InfoNCE Loss：$\\text{softmax}(\\text{sim}(q, k_+)/\\tau)$ |
`;export{n as default};
