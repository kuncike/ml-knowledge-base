const n=`# SENet / ShuffleNet / GhostNet

## 核心思想

SENet (Squeeze-and-Excitation) 提出**通道注意力**——不是所有通道都一样重要，让网络自己学"哪个通道该被强调"。ShuffleNet 解决的是 $1 \\times 1$ 卷积在轻量模型中的瓶颈——通过通道混洗让分组卷积间信息流通。GhostNet 观测到 CNN 中间特征图存在大量"相似对"——廉价地用线性变换生成冗余特征，只为"必要特征"做真正的卷积。

三种设计的共同目标：**在极低 FLOPs 下保持精度**，用于移动端和嵌入式部署。

---

## 数学定义与原理解析

### SE Block (Squeeze-and-Excitation)

对特征图 $\\mathbf{X} \\in \\mathbb{R}^{C \\times H \\times W}$：

1. **Squeeze**：全局平均池化 → $\\mathbf{z} \\in \\mathbb{R}^C$，$z_c = \\frac{1}{HW}\\sum_{i,j} x_{c,i,j}$
2. **Excitation**：两层 MLP + Sigmoid → $\\mathbf{s} = \\sigma(\\mathbf{W}_2 \\cdot \\delta(\\mathbf{W}_1 \\cdot \\mathbf{z}))$
3. **Scale**：$\\tilde{\\mathbf{X}}_c = s_c \\cdot \\mathbf{X}_c$

中间降维比 $r=16$（如 $C=256$，中间 $16$ 维），参数量仅 $2C^2/r$。

### ShuffleNet V2 设计准则

1. 输入输出通道数相同时 MAC（Memory Access Cost）最小
2. 过多分组卷积增加 MAC
3. 网络碎片化（多分支）降低并行度
4. Element-wise 操作（ReLU/Add）不可忽略

**Channel Shuffle 操作**：将 $C$ 个通道 reshape 为 $(g, C/g)$ → transpose → flatten，等价于"分组间洗牌"。

### Ghost Module

普通卷积输出中，很多特征图两两之间存在简单的线性关系（"幽灵对"）。Ghost 模块：

1. 正常卷积产生 $\\frac{C_{out}}{2}$ 个"内在"特征图
2. 对每个内在特征图施加廉价操作 $\\Phi_i$（如 $3 \\times 3$ 深度卷积），各生成一个"幽灵"特征
3. 拼接 → $C_{out}$ 个特征图

计算量减少约 50%，因为一半的特征图由廉价操作生成。

---

## 可视化展示

### SE Block

\`\`\`mermaid
graph LR
    X["X (C×H×W)"] --> GAP["Global AvgPool"] --> Z["z (C×1×1)"]
    Z --> FC1["FC C→C/r"] --> RELU["ReLU"] --> FC2["FC C/r→C"] --> SIG["Sigmoid"]
    SIG --> S["s (C×1×1)"]
    S --> SCALE["Scale: X_c = X_c × s_c"]
    X --> SCALE --> XOUT["X̃ (C×H×W)"]
\`\`\`

### 轻量模型对比

\`\`\`echarts
return {
  title: { text: '轻量模型 ImageNet Top-1 vs FLOPs', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'value', name: 'FLOPs (M)', min: 0, max: 600 },
  yAxis: { type: 'value', name: 'Top-1 Accuracy (%)', min: 68, max: 78 },
  series: [
    { name: 'MobileNetV3', type: 'scatter', symbolSize: 14,
      data: [[219, 75.2]], itemStyle: { color: '#2980b9' } },
    { name: 'ShuffleNetV2', type: 'scatter', symbolSize: 14,
      data: [[299, 74.9], [149, 72.6]], itemStyle: { color: '#16a085' } },
    { name: 'GhostNet', type: 'scatter', symbolSize: 14,
      data: [[141, 73.9], [226, 75.0]], itemStyle: { color: '#d35400' } },
    { name: 'SENet', type: 'scatter', symbolSize: 14,
      data: [[390, 77.6]], itemStyle: { color: '#2c3e50' } }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — SE Block

\`\`\`python
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        B, C, _, _ = x.shape
        s = self.squeeze(x).view(B, C)
        s = self.excitation(s).view(B, C, 1, 1)
        return x * s
\`\`\`

### PyTorch — Channel Shuffle

\`\`\`python
def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    return x.view(B, groups, C // groups, H, W) \\
            .transpose(1, 2) \\
            .contiguous() \\
            .view(B, C, H, W)
\`\`\`

### PyTorch — ShuffleNetV2 Block

\`\`\`python
class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        mid = out_c // 2
        self.stride = stride
        # 分支1: 恒等映射 (或 depthwise conv if stride=2)
        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, mid, 1, bias=False),
                nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        # 分支2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_c if stride > 1 else mid, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.Conv2d(mid, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True))

    def forward(self, x):
        if self.stride > 1:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        return channel_shuffle(out, 2)
\`\`\`
`;export{n as default};
