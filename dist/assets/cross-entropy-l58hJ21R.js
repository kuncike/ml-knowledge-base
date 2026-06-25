const n=`# Cross-Entropy Loss (交叉熵损失)

## 核心思想

交叉熵衡量"如果你用分布 $q$ 来编码来自分布 $p$ 的信息，平均需要多少比特"。在机器学习中，$p$ 是真实标签分布（one-hot），$q$ 是模型预测分布——最小化交叉熵等价于让预测分布尽可能接近真实分布。

---

## 数学定义与原理解析

### 信息论定义

$$
H(p, q) = -\\sum_{k} p_k \\log q_k
$$

其中 $p$ 是真实分布，$q$ 是预测分布。与 KL 散度的关系：

$$
\\text{KL}(p \\parallel q) = H(p, q) - H(p)
$$

对于 one-hot 标签 $H(p) = 0$，因此 $\\text{KL} = \\text{CE}$——最小化交叉熵等价于最小化 KL 散度。

### 二分类交叉熵 (Binary Cross-Entropy)

$$
L = -\\frac{1}{n} \\sum_{i=1}^{n} \\left[ y_i \\log(\\hat{y}_i) + (1-y_i) \\log(1-\\hat{y}_i) \\right]
$$

其中 $\\hat{y}_i = \\sigma(z_i)$ 是 Sigmoid 输出。当 $y=1$ 时，$L = -\\log(\\hat{y})$（惩罚低预测概率）；当 $y=0$ 时，$L = -\\log(1-\\hat{y})$（惩罚高预测概率）。

### 多分类交叉熵 (Categorical Cross-Entropy)

$$
L = -\\frac{1}{n} \\sum_{i=1}^{n} \\sum_{k=1}^{K} y_{ik} \\log(\\hat{y}_{ik})
$$

对于 one-hot 标签，简化为只惩罚正确类别的预测概率：

$$
L = -\\frac{1}{n} \\sum_{i=1}^{n} \\log(\\hat{y}_{i, k_i^*})
$$

### Softmax + Cross-Entropy 优雅梯度

这是 Softmax + CE 组合被广泛使用的深层原因：

$$
\\frac{\\partial L}{\\partial z_{ik}} = \\hat{y}_{ik} - y_{ik}
$$

梯度 = 预测值 - 真实值——形式与线性回归的 MSE 梯度一致，极其简洁。

### 标签平滑 (Label Smoothing)

防止模型对训练标签过于自信：

$$
y_{ik}^{LS} = (1 - \\epsilon) y_{ik} + \\frac{\\epsilon}{K}
$$

等价于在交叉熵基础上增加模型输出与均匀分布的 KL 散度惩罚，缓解过拟合，提升泛化能力。

---

## 可视化展示

### BCE Loss 曲线：预测概率 vs 损失值

\`\`\`echarts
return {
  xAxis: { type: 'value', min: 0.001, max: 0.999, name: '预测概率 p̂' },
  yAxis: { type: 'value', min: 0, max: 8, name: 'Loss' },
  legend: { data: ['y=1: -log(p̂)', 'y=0: -log(1-p̂)'] },
  series: [
    {
      name: 'y=1: -log(p̂)', type: 'line', smooth: true,
      lineStyle: { color: '#2980b9', width: 2 },
      data: (function() { const d = []; for (let i = 0.001; i <= 0.999; i += 0.001) d.push([i, -Math.log(i)]); return d; })()
    },
    {
      name: 'y=0: -log(1-p̂)', type: 'line', smooth: true,
      lineStyle: { color: '#c0392b', width: 2 },
      data: (function() { const d = []; for (let i = 0.001; i <= 0.999; i += 0.001) d.push([i, -Math.log(1 - i)]); return d; })()
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

当 $\\hat{y} \\to 0$ 而 $y=1$ 时，loss $\\to \\infty$——交叉熵对错误预测的惩罚非常严厉，这是它相对 MSE 收敛更快的核心原因。

### 标签平滑效果对比

\`\`\`echarts
return {
  xAxis: { type: 'category', data: ['无标签平滑', 'ε=0.05', 'ε=0.1', 'ε=0.2'] },
  yAxis: { type: 'value', min: 0, max: 1, name: 'Top-1 Accuracy' },
  legend: { data: ['训练集', '验证集', '校准误差'] },
  series: [
    { name: '训练集', type: 'bar', data: [0.95, 0.93, 0.91, 0.88], itemStyle: { color: '#2c3e50' } },
    { name: '验证集', type: 'bar', data: [0.83, 0.85, 0.86, 0.84], itemStyle: { color: '#16a085' } },
    { name: '校准误差', type: 'bar', data: [0.12, 0.06, 0.04, 0.05], itemStyle: { color: '#d35400' } }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch

\`\`\`python
import torch.nn as nn

# 多分类 — 内置 LogSoftmax + NLLLoss（数值稳定）
ce = nn.CrossEntropyLoss(label_smoothing=0.1)

# 二分类 — 内置 Sigmoid
bce = nn.BCEWithLogitsLoss()

# 多标签分类 — 每个类别独立做二分类
mlbce = nn.BCEWithLogitsLoss()

# 不要这么做（数值不稳定）：
# loss = nn.NLLLoss()(torch.log(nn.Softmax(dim=1)(logits)), targets)
\`\`\`

### NumPy 手写

\`\`\`python
import numpy as np

def cross_entropy(y_true, y_pred, eps=1e-12):
    """y_true: one-hot, y_pred: probabilities"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def binary_cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
\`\`\`
`;export{n as default};
