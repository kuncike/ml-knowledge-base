const n=`# FCN / U-Net

## FCN (Fully Convolutional Network)

### 核心思想

将分类网络的全连接层替换为**1×1 卷积**，并添加**转置卷积**进行上采样，使网络输出与输入同分辨率的像素级预测。

### 跳跃连接 (Skip Connections)

融合浅层细节和深层语义：

- FCN-32s：直接 32× 上采样（粗糙）
- FCN-16s：pool4 + 2× 上采样 + conv7 的 2× 上采样（较好）
- FCN-8s：pool3 + pool4 + conv7（最精细）

### 输出

$$H \\times W \\times C \\text{（每个像素的类别概率）}$$

---

## U-Net

### 架构：对称的 U 形结构

\`\`\`
编码器（下采样）          解码器（上采样）
    ↓                       ↑
  Conv×2                 Conv×2
    ↓  MaxPool              ↑  ConvTranspose
  Conv×2    ──concat──→   Conv×2  ←─
    ↓  MaxPool              ↑  ConvTranspose
  Conv×2    ──concat──→   Conv×2  ←─
    ↓  MaxPool              ↑  ConvTranspose
  Conv×2    ──concat──→   Conv×2  ←─
    ↓ (Bottleneck)
  Conv×2
\`\`\`

### U-Net 的优势

1. **跳跃连接保留空间细节**：上采样时拼接编码器对应层的特征图
2. **数据高效**：适合医学影像等小样本场景
3. **边缘清晰**：浅层的高频信息通过跳跃连接直接传递

### 损失函数

常用 Dice Loss 或加权交叉熵：

$$\\text{Dice}(y, \\hat{y}) = \\frac{2 |y \\cap \\hat{y}|}{|y| + |\\hat{y}|}$$

$$L_{dice} = 1 - \\text{Dice}$$

## U-Net 变体

| 变体 | 改进 |
|------|------|
| Attention U-Net | 门控注意力过滤无关区域 |
| U-Net++ | 密集跳跃连接，深度监督 |
| nnU-Net | 自动配置（预处理、网络、后处理） |
| 3D U-Net | 用于 CT/MRI 等三维数据 |

## PyTorch 实现要点

\`\`\`python
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

# 上采样：转置卷积或 nn.Upsample + Conv
# 跳跃连接：torch.cat([decoder_feat, encoder_feat], dim=1)
\`\`\`
`;export{n as default};
