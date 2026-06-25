const n=`# MobileNet / EfficientNet / ConvNeXt

## MobileNet 系列

### MobileNet v1 — 深度可分离卷积

将标准卷积分解为 Depthwise + Pointwise：

$$\\text{参数量比} = \\frac{D_K^2 \\cdot M + M \\cdot N}{D_K^2 \\cdot M \\cdot N} = \\frac{1}{N} + \\frac{1}{D_K^2}$$

当 $N=512, D_K=3$，约减少 8-9 倍。

两个超参数控制模型大小：
- **宽度乘数 $\\alpha$**：通道数缩放（0.25, 0.5, 0.75, 1.0）
- **分辨率乘数 $\\rho$**：输入分辨率缩放

### MobileNet v2 — 倒残差结构

\`\`\`python
# 倒残差块：Expand → Depthwise → Project
class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio=6):
        super().__init__()
        hidden_dim = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_ch, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
\`\`\`

关键：最后的 Pointwise 卷积后使用**线性激活**（ReLU 会破坏低维流形信息）。

### MobileNet v3

引入 SE 模块 (Squeeze-and-Excitation) 和 NAS 搜索，使用 h-swish 替代 swish。

---

## EfficientNet

### 复合缩放 (Compound Scaling)

通过 NAS 找到基线模型 EfficientNet-B0，然后统一缩放三个维度：

$$d = \\alpha^\\phi, \\quad w = \\beta^\\phi, \\quad r = \\gamma^\\phi$$

约束：$\\alpha \\cdot \\beta^2 \\cdot \\gamma^2 \\approx 2$

| 模型 | 参数量 | Top-1 Acc |
|------|--------|-----------|
| B0 | 5.3M | 77.1% |
| B4 | 19M | 82.9% |
| B7 | 66M | 84.3% |

---

## ConvNeXt — 卷积的现代化

借鉴 Transformer 设计理念"现代化"的标准卷积网络：

| 改动 | ResNet-50 → ConvNeXt |
|------|---------------------|
| 阶段比例 | [3,4,6,3] → [3,3,9,3] |
| Stem | 7×7 stride2 → 4×4 stride4 |
| 激活 | ReLU → GELU |
| 归一化 | BN → LN |
| 下采样 | stride-2 Conv → 单独的 2×2 Conv stride2 |

结果：在 ImageNet 上与 Swin Transformer 不相上下，同时保持了卷积的简单性。

## PyTorch 使用

\`\`\`python
import torchvision.models as models

mbn = models.mobilenet_v3_small(pretrained=True)
eff = models.efficientnet_b0(pretrained=True)
convnext = models.convnext_tiny(pretrained=True)
\`\`\`
`;export{n as default};
