const n=`# 全连接层 (Dense / Linear Layer)

## 核心定义

全连接层（又称线性层或稠密层）是神经网络最基础的组件。每个输入神经元与每个输出神经元都有连接权重。

## 数学表达

$$\\mathbf{y} = \\mathbf{W} \\mathbf{x} + \\mathbf{b}$$

其中 $\\mathbf{W} \\in \\mathbb{R}^{d_{out} \\times d_{in}}$，$\\mathbf{b} \\in \\mathbb{R}^{d_{out}}$。

对于批量数据 $\\mathbf{X} \\in \\mathbb{R}^{n \\times d_{in}}$：

$$\\mathbf{Y} = \\mathbf{X} \\mathbf{W}^T + \\mathbf{b}$$

## 参数量

$$d_{in} \\times d_{out} + d_{out}$$

全连接层是神经网络中参数最多的组件。例如，一个 512 → 1024 的全连接层有 \`512 × 1024 + 1024 = 525,312\` 个参数。

## PyTorch 实现

\`\`\`python
import torch
import torch.nn as nn

# 单层
linear = nn.Linear(in_features=512, out_features=256, bias=True)
x = torch.randn(32, 512)  # [batch, features]
y = linear(x)              # [32, 256]

# 等价于
w = linear.weight  # [256, 512]
b = linear.bias    # [256]
y_manual = x @ w.T + b
\`\`\`

## 反向传播

对损失 $L$ 的梯度：

$$\\frac{\\partial L}{\\partial \\mathbf{x}} = \\frac{\\partial L}{\\partial \\mathbf{y}} \\mathbf{W}$$

$$\\frac{\\partial L}{\\partial \\mathbf{W}} = \\frac{\\partial L}{\\partial \\mathbf{y}}^T \\mathbf{x}$$

$$\\frac{\\partial L}{\\partial \\mathbf{b}} = \\sum_i \\frac{\\partial L}{\\partial y_i}$$

## 在 Transformer 中的位置

全连接层是 Transformer 中 **FFN (Feed-Forward Network)** 的核心：

$$\\text{FFN}(x) = \\text{GELU}(x W_1 + b_1) W_2 + b_2$$

通常 $d_{ff} \\approx 4 \\times d_{model}$，这是 Transformer 中参数最主要的部分。

## 设计要点

- 用 **He 初始化**（配合 ReLU）或 **Xavier 初始化**（配合 Tanh/Sigmoid）
- 过大的全连接层容易过拟合，通常配合 Dropout
- 在 CNN 中全连接层用于最后的分类头；在 Transformer 中用于 FFN 子层
`;export{n as default};
