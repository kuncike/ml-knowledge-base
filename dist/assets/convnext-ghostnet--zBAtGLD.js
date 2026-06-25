const n=`# ConvNeXt

## 核心思想

2022 年，ConvNeXt 提出一个挑衅性问题：**如果纯粹用 ConvNet 技术、但按照 Swin Transformer 的设计哲学来重新设计 ResNet，会得到什么？** 答案是一个纯卷积模型，在 ImageNet 上匹配甚至超越 Swin Transformer，证明了卷积架构远未过时。

ConvNeXt V2 进一步引入**全局响应归一化（GRN）**替代 LayerNorm，解决了特征坍塌问题——深层特征图中不同通道的输出变得高度相关，GRN 通过通道间竞争恢复多样性。

---

## 数学定义与原理解析

### 从 ResNet 到 ConvNeXt 的现代化步骤

| 步骤 | ResNet-50 | → | ConvNeXt-T |
|------|-----------|---|------------|
| 训练策略 | 90 epochs | → | 300 epochs + 现代增强 |
| 阶段比 | (3,4,6,3) | → | (3,3,9,3) |
| Stem | 7×7, s=2 | → | 4×4, s=4 (patchify) |
| 卷积 | 3×3 分组 | → | 7×7 Depthwise |
| 瓶颈比 | 1:4:1 | → | 1:4:1 (inverted bottleneck) |
| 激活 | ReLU | → | GELU |
| 归一化 | BN | → | LayerNorm |
| 下采样 | 1×1, s=2 | → | 2×2, s=2 (separate) |

### ConvNeXt Block

$$
\\begin{aligned}
\\mathbf{x}' &= \\text{DepthwiseConv}_{7\\times7}(\\mathbf{x}) \\\\
\\mathbf{x}' &= \\text{LayerNorm}(\\mathbf{x}') \\\\
\\mathbf{x}' &= \\text{GELU}(\\mathbf{x}') \\\\
\\mathbf{x}' &= \\text{Conv}_{1\\times1, C \\to 4C}(\\mathbf{x}') \\\\
\\mathbf{x}' &= \\text{GELU}(\\mathbf{x}') \\\\
\\mathbf{x}' &= \\text{Conv}_{1\\times1, 4C \\to C}(\\mathbf{x}') \\\\
\\text{out} &= \\mathbf{x} + \\text{LayerScale}(\\mathbf{x}')
\\end{aligned}
$$

### GRN (ConvNeXt V2)

$$
G(\\mathbf{X}) = \\|\\mathbf{X}\\|_{\\text{channel}} = \\sqrt{\\sum_{c} X_c^2}
$$

$$
\\mathbf{X}' = \\frac{\\mathbf{X}}{\\mathbb{E}[G(\\mathbf{X})] + \\epsilon} \\cdot \\gamma + \\beta
$$

全局响应归一化：先计算全局特征范数，再在空间维度上归一化——强制通道间竞争。

---

## 可视化展示

### ResNet → ConvNeXt 现代化路径

\`\`\`mermaid
graph TD
    RN["ResNet-50<br/>BN · ReLU · 3×3 · 1:4:1"] -->
    S1["Stage Ratio: (3,4,6,3)→(3,3,9,3)"] -->
    S2["Stem: 7×7s2→4×4s4 (patchify)"] -->
    S3["ResNeXt-ify: 分组→Depthwise"] -->
    S4["Inverted Bottleneck: 1:4:1"] -->
    S5["Large Kernel: 3×3→7×7"] -->
    S6["Modern Activation: ReLU→GELU, BN→LN"] -->
    CN["ConvNeXt-T<br/>82.1% Top-1"]
\`\`\`

### ConvNeXt vs Swin 对比

\`\`\`echarts
return {
  title: { text: 'ConvNeXt vs Swin Transformer', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'value', name: 'FLOPs (G)' },
  yAxis: { type: 'value', name: 'ImageNet Top-1 (%)', min: 81, max: 88 },
  series: [
    { name: 'ConvNeXt', type: 'line', smooth: true,
      data: [[4.5,82.1], [8.7,83.8], [15.4,84.9], [34.4,85.8]],
      lineStyle: { color: '#16a085', width: 2.5 },
      symbolSize: 8 },
    { name: 'Swin Transformer', type: 'line', smooth: true,
      data: [[4.5,81.3], [8.7,83.3], [15.4,84.5], [34.5,85.2]],
      lineStyle: { color: '#2980b9', width: 2.5 },
      symbolSize: 8 }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — ConvNeXt Block

\`\`\`python
import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)  # depthwise
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)   # pointwise ↑
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)    # pointwise ↓

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] → [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]
        return shortcut + self.drop_path(x)


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 dims=[96, 192, 384, 768],
                 depths=[3, 3, 9, 3]):
        super().__init__()
        # Patchify stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], 4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6))

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)
            if i < 3:
                self.stages.append(nn.Sequential(
                    nn.LayerNorm(dims[i], eps=1e-6),
                    nn.Linear(dims[i], dims[i+1])))  # downsample

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            if isinstance(stage[0], ConvNeXtBlock):
                x = stage(x)
            else:
                x = stage(x.permute(0,2,3,1)).permute(0,3,1,2)
        return x.mean(dim=[2,3])  # global avg pool
\`\`\`
`;export{n as default};
