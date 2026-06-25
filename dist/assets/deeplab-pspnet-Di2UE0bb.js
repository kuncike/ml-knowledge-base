const n=`# DeepLab / PSPNet

## DeepLab v1

### 核心创新：空洞卷积

用空洞卷积在不增加参数的情况下扩大感受野：

| dilation rate | 3×3 核的感受野 | 参数量 |
|:---|:---|:---|
| 1 | 3×3 | 9 |
| 2 | 5×5 | 9 |
| 4 | 9×9 | 9 |

### 条件随机场 (CRF)

后处理步骤：用 DenseCRF 精修分割边缘。

---

## DeepLab v2 — ASPP

### Atrous Spatial Pyramid Pooling

并行使用不同 dilation rate 的卷积，捕获多尺度上下文：

\`\`\`
输入特征 → 1×1 Conv → rate=6 Conv → rate=12 Conv → rate=18 Conv → Pool → Concat → 1×1 Conv
\`\`\`

多个尺度同时处理，适应不同大小的物体。

---

## DeepLab v3

- 改进 ASPP：加入全局平均池化和 Image Pooling
- 用级联空洞卷积模块替代 CRF（端到端）
- Multi-Grid 策略：不同层用不同 dilation rate

### 输出步幅 (Output Stride)

控制最终特征图分辨率的关键参数：

$$OS = \\frac{\\text{Input Resolution}}{\\text{Output Resolution}}$$

OS=16 是精度和速度的常用折中。

---

## PSPNet (Pyramid Scene Parsing Network)

### 核心创新：金字塔池化模块 (PPM)

将特征图池化到不同尺度（1×1, 2×2, 3×3, 6×6），然后上采样 + 拼接：

\`\`\`python
class PyramidPooling(nn.Module):
    def __init__(self, in_ch, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_ch, in_ch // 4, 1, bias=False),
                nn.BatchNorm2d(in_ch // 4),
                nn.ReLU(),
            ) for s in sizes
        ])

    def forward(self, x):
        h, w = x.shape[2:]
        pooled = [x]
        for pool in self.pools:
            p = pool(x)
            pooled.append(F.interpolate(p, (h, w), mode='bilinear'))
        return torch.cat(pooled, dim=1)
\`\`\`

### 辅助损失

在 ResNet 的 stage4 后添加辅助分类器，帮助中间层梯度传播。

## DeepLab vs PSPNet

| | DeepLab v3+ | PSPNet |
|------|------------|--------|
| 多尺度策略 | ASPP + Encoder-Decoder | PPM |
| 骨干 | Xception / ResNet | ResNet |
| 解码器 | 简单解码器 | 无（仅上采样） |
| 速度 | 较快 | 中等 |

## PyTorch 使用

\`\`\`python
import torchvision.models.segmentation as seg

deeplab = seg.deeplabv3_resnet50(pretrained=True)
# 或 deeplabv3_mobilenet_v3_large
\`\`\`
`;export{n as default};
