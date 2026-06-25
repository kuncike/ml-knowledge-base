const n=`# FFNN / MLP (多层感知机)

## 核心思想

多层感知机是深度学习的"Hello World"——线性层堆叠 + 非线性激活。单个线性层只能切一刀（线性分类器），但堆叠多层 + 非线性可以逼近任意连续函数（Universal Approximation Theorem）。XOR 问题是理解"为什么要非线性"的经典例子：单层感知机无法分开 XOR，加一个隐藏层即可。

---

## 数学定义与原理解析

### 前向传播

$$
\\mathbf{h}^{(l)} = f^{(l)}(\\mathbf{W}^{(l)} \\mathbf{h}^{(l-1)} + \\mathbf{b}^{(l)})
$$

其中 $\\mathbf{W}^{(l)} \\in \\mathbb{R}^{d_{out} \\times d_{in}}$，$\\mathbf{b}^{(l)} \\in \\mathbb{R}^{d_{out}}$，$f^{(l)}$ 为激活函数（ReLU/Tanh 等）。

### 反向传播（链式法则）

$$
\\frac{\\partial L}{\\partial \\mathbf{W}^{(l)}} = \\frac{\\partial L}{\\partial \\mathbf{h}^{(l)}} \\cdot \\frac{\\partial \\mathbf{h}^{(l)}}{\\partial \\mathbf{W}^{(l)}}
$$

关键是 $\\frac{\\partial L}{\\partial \\mathbf{h}^{(l)}}$ 从输出层递归计算：

$$
\\frac{\\partial L}{\\partial \\mathbf{h}^{(l-1)}} = (\\mathbf{W}^{(l)})^T \\frac{\\partial L}{\\partial \\mathbf{h}^{(l)}} \\odot f'(\\mathbf{W}^{(l)}\\mathbf{h}^{(l-1)} + \\mathbf{b}^{(l)})
$$

### 通用逼近定理（Universal Approximation Theorem）

具有单个隐藏层 + Sigmoid 激活 + 足够多神经元的 MLP 可以以任意精度逼近 $\\mathbb{R}^n$ 紧子集上的任意连续函数。注意：定理没说多少神经元才"足够"——实践中可能需要指数级数量。

### Xavier / He 初始化

- **Xavier**：$\\text{Var}(W) = \\frac{2}{d_{in} + d_{out}}$，配合 Sigmoid/Tanh
- **He**：$\\text{Var}(W) = \\frac{2}{d_{in}}$，配合 ReLU 族。ReLU 将一半输入置零 → 需要更大的权重方差来维持信号。

---

## 可视化展示

### MLP 典型架构

\`\`\`mermaid
graph LR
    subgraph Input["输入层 (d_in)"]
        x1["x₁"]
        x2["x₂"]
        x3["x₃"]
    end
    subgraph Hidden1["隐藏层1 (d₁)"]
        h11["h₁⁽¹⁾"]
        h12["h₂⁽¹⁾"]
        h13["h₃⁽¹⁾"]
        h14["h₄⁽¹⁾"]
    end
    subgraph Hidden2["隐藏层2 (d₂)"]
        h21["h₁⁽²⁾"]
        h22["h₂⁽²⁾"]
        h23["h₃⁽²⁾"]
    end
    subgraph Output["输出层 (d_out)"]
        y1["ŷ₁"]
        y2["ŷ₂"]
    end
    x1 --> h11 & h12 & h13 & h14
    x2 --> h11 & h12 & h13 & h14
    x3 --> h11 & h12 & h13 & h14
    h11 --> h21 & h22 & h23
    h12 --> h21 & h22 & h23
    h13 --> h21 & h22 & h23
    h14 --> h21 & h22 & h23
    h21 --> y1 & y2
    h22 --> y1 & y2
    h23 --> y1 & y2
\`\`\`

### 不同激活函数的决策边界 (XOR 问题)

\`\`\`echarts
return {
  title: { text: 'XOR 分类: 单隐藏层(4神经元) 不同激活函数', left: 'center', textStyle: { fontSize: 13 } },
  legend: { data: ['❌ 无激活(线性)', '✅ ReLU', '✅ Tanh'], bottom: 0 },
  xAxis: { type: 'value', min: -2, max: 2 },
  yAxis: { type: 'value', min: -2, max: 2 },
  series: [
    { name: '❌ 无激活(线性)', type: 'scatter', symbolSize: 14,
      data: [[0,0],[0,1],[1,0],[1,1]],
      itemStyle: { color: '#c0392b' },
      markArea: { silent: true, data: [[{xAxis:-2,yAxis:-2},{xAxis:2,yAxis:2}]] }
    },
    { name: '✅ ReLU', type: 'scatter', symbolSize: 14,
      data: [[0,0,'○'],[0,1,'×'],[1,0,'×'],[1,1,'○']],
      itemStyle: { color: '#16a085' }
    },
    { name: '✅ Tanh', type: 'scatter', symbolSize: 14,
      data: [[0,0,'○'],[0,1,'×'],[1,0,'×'],[1,1,'○']],
      itemStyle: { color: '#2980b9' }
    }
  ],
  grid: { left: 50, right: 20, top: 50, bottom: 50 }
}
\`\`\`

无激活的多层退化为单层线性变换（多乘几个矩阵＝一个矩阵），这正是 MLP 必须加非线性的数学原因。

---

## 核心代码实现

### PyTorch — MLP 分类器

\`\`\`python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256], output_dim=10, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout)
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, input_dim]
        return self.net(x)
\`\`\`

### NumPy — 手写前向 + 反向

\`\`\`python
import numpy as np

def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(float)
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# 2 层 MLP 前向
def forward(X, W1, b1, W2, b2):
    h = relu(X @ W1 + b1)
    y = softmax(h @ W2 + b2)
    return h, y

# 2 层 MLP 反向
def backward(X, y, y_true, h, W2):
    m = X.shape[0]
    dy = (y - y_true) / m            # softmax + CE 梯度
    dW2 = h.T @ dy
    db2 = dy.sum(axis=0)
    dh = dy @ W2.T * relu_grad(h)
    dW1 = X.T @ dh
    db1 = dh.sum(axis=0)
    return dW1, db1, dW2, db2
\`\`\`
`;export{n as default};
