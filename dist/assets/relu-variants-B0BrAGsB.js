const n=`# ReLU / LeakyReLU / PReLU / ELU

## 核心思想

ReLU 用"一刀切"的方式解决了 Sigmoid/Tanh 的梯度消失问题：正区间梯度恒为 1，计算仅需一次 \`max(0, x)\`。但它也引入了新问题——**Dying ReLU**（神经元永久失活）。后续的 LeakyReLU、PReLU、ELU 都是在这个基础上的"修补"，核心目标是在保持 ReLU 优点的同时，让负区间也有一线生机。

---

## 数学定义与原理解析

### ReLU (Rectified Linear Unit)

$$
\\text{ReLU}(x) = \\max(0, x)
$$

$$
\\text{ReLU}'(x) = \\begin{cases} 1 & x > 0 \\\\ 0 & x \\leq 0 \\end{cases}
$$

**优势**：正区间梯度恒为 1，消除梯度消失；计算极其简单；产生稀疏激活（约 50% 神经元输出为零）。

**Dying ReLU 问题**：若某个神经元对所有样本都输出负值 → 梯度为 0 → 权重永不更新 → 该神经元"死亡"。常见诱因：学习率过大、负偏置过大。

### LeakyReLU

$$
\\text{LeakyReLU}(x) = \\begin{cases} x & x > 0 \\\\ \\alpha x & x \\leq 0 \\end{cases}, \\quad \\alpha = 0.01
$$

负区间不再是死路，保留微小梯度（$\\alpha = 0.01$），让"濒死"神经元有机会复活。

### PReLU (Parametric ReLU)

$$
\\text{PReLU}(x) = \\begin{cases} x & x > 0 \\\\ \\alpha x & x \\leq 0 \\end{cases}
$$

$\\alpha$ 变为**可学习参数**（每个通道独立），由反向传播自动调整——不依赖人工设置的超参数。

### ELU (Exponential Linear Unit)

$$
\\text{ELU}(x) = \\begin{cases} x & x > 0 \\\\ \\alpha(e^x - 1) & x \\leq 0 \\end{cases}
$$

- 负区间输出均值趋近于零（**加速收敛**）
- 处处可导（在 $x=0$ 处也是光滑的）
- 代价：涉及指数运算，计算稍慢

---

## 可视化展示

### 五种激活函数对比

\`\`\`echarts
return {
  xAxis: { type: 'value', min: -5, max: 5, name: 'x' },
  yAxis: { type: 'value', min: -1.5, max: 5, name: 'f(x)' },
  legend: { data: ['ReLU', 'LeakyReLU', 'PReLU(α=0.1)', 'ELU', 'GELU'] },
  series: [
    {
      name: 'ReLU', type: 'line', smooth: false,
      lineStyle: { color: '#2c3e50', width: 2.5 },
      data: (function() {
        const d = [];
        for (let i = -5; i <= 5; i += 0.05) d.push([i, Math.max(0, i)]);
        return d;
      })()
    },
    {
      name: 'LeakyReLU', type: 'line', smooth: false,
      lineStyle: { color: '#d35400', width: 2 },
      data: (function() {
        const d = [];
        for (let i = -5; i <= 5; i += 0.05) d.push([i, i > 0 ? i : 0.01 * i]);
        return d;
      })()
    },
    {
      name: 'PReLU(α=0.1)', type: 'line', smooth: false,
      lineStyle: { color: '#16a085', width: 1.5, type: 'dashed' },
      data: (function() {
        const d = [];
        for (let i = -5; i <= 5; i += 0.05) d.push([i, i > 0 ? i : 0.1 * i]);
        return d;
      })()
    },
    {
      name: 'ELU', type: 'line', smooth: true,
      lineStyle: { color: '#8e44ad', width: 2 },
      data: (function() {
        const d = [];
        for (let i = -5; i <= 5; i += 0.05) d.push([i, i > 0 ? i : Math.exp(i) - 1]);
        return d;
      })()
    },
    {
      name: 'GELU', type: 'line', smooth: true,
      lineStyle: { color: '#2980b9', width: 2 },
      data: (function() {
        const d = [];
        for (let i = -5; i <= 5; i += 0.05) {
          const x = i;
          const gelu = 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
          d.push([i, gelu]);
        }
        return d;
      })()
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

### Dying ReLU 现象可视化

当偏置 $b$ 为较大负值时，神经元对所有输入输出均为负 → 梯度为零 → 权重冻结。

\`\`\`echarts
return {
  xAxis: { type: 'value', min: -5, max: 5, name: '输入 x' },
  yAxis: { type: 'value', min: -4, max: 4, name: '输出' },
  legend: { data: ['健康神经元 (b=0)', '死亡神经元 (b=-2)'] },
  series: [
    {
      name: '健康神经元 (b=0)', type: 'line',
      lineStyle: { color: '#16a085', width: 2 },
      data: (function() { const d = []; for (let i = -5; i <= 5; i += 0.05) d.push([i, Math.max(0, i)]); return d; })()
    },
    {
      name: '死亡神经元 (b=-2)', type: 'line',
      lineStyle: { color: '#c0392b', width: 2 },
      data: (function() { const d = []; for (let i = -5; i <= 5; i += 0.05) d.push([i, Math.max(0, i - 2)]); return d; })()
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch 使用

\`\`\`python
import torch
import torch.nn as nn

# ReLU — 默认首选
nn.ReLU(inplace=True)  # inplace 节省显存

# LeakyReLU
nn.LeakyReLU(negative_slope=0.01)

# PReLU — 可学习的负斜率
nn.PReLU(num_parameters=1)  # 所有通道共享一个 α

# ELU
nn.ELU(alpha=1.0)
\`\`\`

### NumPy 手写

\`\`\`python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def relu_derivative(x):
    return (x > 0).astype(float)
\`\`\`

## 选择指南

| 激活函数 | 推荐场景 | 不推荐场景 |
|----------|----------|-----------|
| ReLU | CNN 隐藏层默认首选 | 需要负值信息的任务 |
| LeakyReLU | 轻微缓解 Dying ReLU | — |
| PReLU | 大规模数据集，追求极致精度 | 小数据集（α 学不好） |
| ELU | 追求训练速度，不怕计算开销 | 推理延迟敏感的场景 |
`;export{n as default};
