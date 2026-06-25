const n=`# Sigmoid / Tanh

## 核心思想

Sigmoid 和 Tanh 是最经典的 S 型激活函数。Sigmoid 将任意实数"挤压"到 $(0, 1)$ 区间，天然适合表示概率；Tanh 是 Sigmoid 的零均值升级版，将输出映射到 $(-1, 1)$。它们的共同致命弱点是**梯度消失**——当输入远离零点时，梯度趋近于零，导致深层网络训练停滞。

---

## 数学定义与原理解析

### Sigmoid

$$
\\sigma(x) = \\frac{1}{1 + e^{-x}}
$$

**导数**（具备自引用特性，计算高效）：

$$
\\sigma'(x) = \\sigma(x)(1 - \\sigma(x))
$$

- 输出范围：$(0, 1)$
- 适合二分类输出层（配合 BCE Loss）
- **梯度消失**：当 $|x| > 5$ 时，梯度 $< 0.007$
- **非零均值**：输出恒为正，导致反向传播时梯度全部同号（参数更新呈 zig-zag 路径）

### Tanh

$$
\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\\sigma(2x) - 1
$$

**导数**：

$$
\\tanh'(x) = 1 - \\tanh^2(x)
$$

- 输出范围：$(-1, 1)$，**零均值**——这是相比 Sigmoid 的关键优势
- 当 $|x| > 5$ 时，梯度仍 $< 0.0009$，梯度消失问题依旧存在

### 梯度消失的链式效应

深层网络中，梯度通过链式法则逐层回传：

$$
\\frac{\\partial L}{\\partial w^{(1)}} = \\prod_{l=2}^{L} \\sigma'(z^{(l)}) \\cdot \\cdots
$$

$L$ 层中每层的梯度因子 $\\sigma'(z) \\leq 0.25$（Sigmoid 的最大导数值），连乘后呈指数衰减 → 浅层权重几乎停止更新。

---

## 可视化展示

### Sigmoid 与 Tanh 函数曲线

\`\`\`echarts
return {
  xAxis: { type: 'value', min: -8, max: 8, name: 'x' },
  yAxis: { type: 'value', min: -1.2, max: 1.2, name: 'σ(x) / tanh(x)' },
  legend: { data: ['Sigmoid', 'Tanh', 'Sigmoid 导数', 'Tanh 导数'] },
  series: [
    {
      name: 'Sigmoid',
      type: 'line',
      smooth: true,
      lineStyle: { color: '#2980b9', width: 2 },
      data: (function() {
        const data = [];
        for (let i = -8; i <= 8; i += 0.05) data.push([i, 1 / (1 + Math.exp(-i))]);
        return data;
      })()
    },
    {
      name: 'Tanh',
      type: 'line',
      smooth: true,
      lineStyle: { color: '#c0392b', width: 2 },
      data: (function() {
        const data = [];
        for (let i = -8; i <= 8; i += 0.05) data.push([i, Math.tanh(i)]);
        return data;
      })()
    },
    {
      name: 'Sigmoid 导数',
      type: 'line',
      smooth: true,
      lineStyle: { color: '#2980b9', width: 1.5, type: 'dashed' },
      data: (function() {
        const data = [];
        for (let i = -8; i <= 8; i += 0.05) {
          const s = 1 / (1 + Math.exp(-i));
          data.push([i, s * (1 - s)]);
        }
        return data;
      })()
    },
    {
      name: 'Tanh 导数',
      type: 'line',
      smooth: true,
      lineStyle: { color: '#c0392b', width: 1.5, type: 'dashed' },
      data: (function() {
        const data = [];
        for (let i = -8; i <= 8; i += 0.05) {
          const t = Math.tanh(i);
          data.push([i, 1 - t * t]);
        }
        return data;
      })()
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

### 梯度消失示意：$L$ 层网络中的梯度衰减

\`\`\`echarts
return {
  xAxis: { type: 'value', name: '网络层数 L', min: 0, max: 20 },
  yAxis: { type: 'value', name: '梯度缩放因子', min: 0, max: 1 },
  series: [{
    type: 'line',
    smooth: false,
    data: (function() {
      const data = [];
      for (let L = 0; L <= 20; L++) {
        data.push([L, Math.pow(0.25, L)]);
      }
      return data;
    })(),
    areaStyle: { color: 'rgba(192, 57, 43, 0.15)' },
    lineStyle: { color: '#c0392b', width: 2 }
  }],
  tooltip: { trigger: 'axis', formatter: '层数 {b}<br/>梯度因子: {c}' },
  grid: { left: 60, right: 20, top: 20, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### NumPy 手写实现

\`\`\`python
import numpy as np

def sigmoid(x):
    """数值稳定的 Sigmoid"""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    pos = np.exp(x)
    neg = np.exp(-x)
    return (pos - neg) / (pos + neg)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t * t
\`\`\`

### PyTorch 调用

\`\`\`python
import torch
import torch.nn as nn

# 内置激活
sigmoid = torch.sigmoid          # 函数式
tanh = torch.tanh

# 或作为层使用
nn.Sigmoid()
nn.Tanh()
\`\`\`

---

## 使用场景

现代深度网络中，Sigmoid/Tanh 仅在特定位置使用：

| 函数 | 使用场景 | 原因 |
|------|----------|------|
| Sigmoid | 二分类输出层 | 输出天然是概率 |
| Sigmoid | LSTM/GRU 的遗忘门、输入门、输出门 | 门控需要 $(0,1)$ 范围 |
| Tanh | LSTM/GRU 的候选记忆 $\\tilde{C}_t$ | 零均值有利于记忆更新 |
| Tanh | RNN 隐藏状态激活 | 零均值防止偏移累积 |

**隐藏层几乎不再使用它们**——ReLU 家族已全面替代。
`;export{n as default};
