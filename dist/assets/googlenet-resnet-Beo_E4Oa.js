const n=`# GoogLeNet / ResNet / DenseNet

## GoogLeNet / Inception (2014)

### Inception 模块

并行使用多种卷积核大小，让网络自己"选择"合适的尺度：

\`\`\`
前一层 ──→ 1×1 Conv ──→ 3×3 Conv ──→ 5×5 Conv ──→ 3×3 MaxPool ──→ Concat
\`\`\`

**1×1 卷积（Bottleneck）** 用于降维，减少计算量。

### Inception v2/v3 改进

- 用两个 $3 \\times 3$ 卷积替代 $5 \\times 5$
- 引入 Batch Normalization
- 非对称卷积：$n \\times 1 + 1 \\times n$ 替代 $n \\times n$

### 辅助分类器

网络中间层添加额外的分类头，帮助梯度传播（类似于深度监督）。

---

## ResNet (2015)

### 残差学习

$$\\mathbf{y} = \\mathcal{F}(\\mathbf{x}, \\{W_i\\}) + \\mathbf{x}$$

学习残差 $\\mathcal{F}(\\mathbf{x})$ 而非直接学习映射 $\\mathcal{H}(\\mathbf{x})$。

### 为什么有效？

如果最优映射是恒等映射，残差块只需将 $\\mathcal{F}(\\mathbf{x})$ 推到 0（远比学习恒等映射容易）。

**梯度高速公路**：

$$\\frac{\\partial L}{\\partial \\mathbf{x}} = \\frac{\\partial L}{\\partial \\mathbf{y}} \\left(1 + \\frac{\\partial \\mathcal{F}}{\\partial \\mathbf{x}}\\right)$$

加法使得梯度可以不经变换地传播到浅层。

### Bottleneck 残差块

用 $1 \\times 1 \\to 3 \\times 3 \\to 1 \\times 1$ 代替两个 $3 \\times 3$：

\`\`\`python
class Bottleneck(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)
\`\`\`

### 经典配置

| 层数 | 架构 | 参数量 |
|------|------|--------|
| 18 | BasicBlock [2,2,2,2] | 11M |
| 34 | BasicBlock [3,4,6,3] | 21M |
| 50 | Bottleneck [3,4,6,3] | 25M |
| 101 | Bottleneck [3,4,23,3] | 44M |
| 152 | Bottleneck [3,8,36,3] | 60M |

---

## DenseNet (2017)

### 密集连接

每一层的输入是**前面所有层输出的拼接**：

$$\\mathbf{x}_l = H_l([\\mathbf{x}_0, \\mathbf{x}_1, \\ldots, \\mathbf{x}_{l-1}])$$

- 增长率 $k$：每层贡献 $k$ 个新特征图（通常 $k=32$）
- 第 $l$ 层输入通道数 = $k_0 + k \\times (l-1)$

### 优势

- 梯度流通更顺畅
- 参数效率更高（特征复用）
- 隐式的深度监督

### 劣势

- 通道数增长快，显存占用大
- 实现需要频繁的内存拼接操作
`;export{n as default};
