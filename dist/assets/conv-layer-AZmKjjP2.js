const n=`# 卷积层 (Convolutional Layers)

## 核心思想

卷积层通过**局部连接**和**权值共享**大幅减少参数，是 CNN 的基石。一个卷积核（filter）在输入上滑动，每次计算局部区域的加权和。

## 2D 卷积公式

$$y_{i,j} = \\sum_{u=0}^{k_h-1} \\sum_{v=0}^{k_w-1} \\sum_{c=0}^{C_{in}-1} x_{i+u, j+v, c} \\cdot w_{u, v, c} + b$$

## 核心参数

| 参数 | 含义 |
|------|------|
| kernel_size | 卷积核大小 $(k_h, k_w)$ |
| stride | 滑动步长 |
| padding | 边界填充量 |
| dilation | 空洞卷积的膨胀率 |

## 输出尺寸计算

$$H_{out} = \\left\\lfloor \\frac{H_{in} + 2P - D \\cdot (K-1) - 1}{S} + 1 \\right\\rfloor$$

## 卷积变体

### 空洞卷积 (Dilated / Atrous Convolution)

在卷积核元素间插入空洞，增大感受野而不增加参数：

\`\`\`
标准卷积 (dilation=1):   [a][b][c]
空洞卷积 (dilation=2):   [a] _ [b] _ [c]
\`\`\`

### 深度可分离卷积 (Depthwise Separable)

将标准卷积分解为两步：

1. **Depthwise**：每个通道单独卷积（$C_{in}$ 个 $k \\times k \\times 1$ 卷积核）
2. **Pointwise**：$1 \\times 1$ 卷积融合通道

参数量比：$\\frac{1}{C_{out}} + \\frac{1}{k^2}$，MobileNet 的核心。

### 转置卷积 (Transposed Convolution)

用于上采样（如在 GAN 或分割网络中）：

$$H_{out} = (H_{in} - 1) \\times S - 2P + K$$

## PyTorch 实现

\`\`\`python
import torch.nn as nn

# 标准卷积
conv = nn.Conv2d(in_channels=3, out_channels=64,
                 kernel_size=3, stride=1, padding=1)

# 深度可分离卷积
depthwise = nn.Conv2d(64, 64, 3, groups=64, padding=1)
pointwise = nn.Conv2d(64, 128, 1)

# 转置卷积
deconv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

# 空洞卷积
dilated = nn.Conv2d(64, 64, 3, dilation=2, padding=2)
\`\`\`

## 感受野 (Receptive Field)

第 $l$ 层的感受野计算公式：

$$RF_l = RF_{l-1} + (K_l - 1) \\times \\prod_{i=1}^{l-1} S_i$$

更大的感受野可以捕获更全局的上下文信息。
`;export{n as default};
