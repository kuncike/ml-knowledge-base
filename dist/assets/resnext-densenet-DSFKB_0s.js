const n=`# ResNeXt / DenseNet

## 核心思想

ResNet 解决的是"深度增加→梯度消失"的问题，但宽度和基数的维度也被证明同样重要。ResNeXt 提出**基数（cardinality）**比深度和宽度更有效——通过分组卷积把一层的计算拆成多个并行分支。DenseNet 走得更极端：每一层都直接连接到前面所有层——不是加和（ResNet）而是拼接（concat），实现特征重用和梯度高速公路。Wide ResNet 则证明有时候"宽而浅"比"窄而深"更好。

---

## 数学定义与原理解析

### ResNeXt — 分组残差

将 ResNet 瓶颈层的中间 $3 \\times 3$ 卷积替换为 $C$ 个并行的分组卷积（$C$ = cardinality）：

每个组的变换 $\\mathcal{T}_i(\\mathbf{x})$（$1 \\times 1$ → $3 \\times 3$ → $1 \\times 1$ 瓶颈），最终输出为：

$$
\\mathbf{y} = \\mathbf{x} + \\sum_{i=1}^{C} \\mathcal{T}_i(\\mathbf{x})
$$

等效于分组卷积：中间 $3 \\times 3$ 卷积的 groups=32。ResNeXt-50（32×4d）的中间通道宽度 = $C \\times 4 = 128$。

### DenseNet — 密集连接

第 $l$ 层的输入是所有前面层的**拼接**（不是加和）：

$$
\\mathbf{x}_l = H_l([\\mathbf{x}_0, \\mathbf{x}_1, \\ldots, \\mathbf{x}_{l-1}])
$$

每层输出 $k$ 个特征图（**growth rate**）。第 $l$ 层的输入通道数为：
$$
C_{in}^{(l)} = C_0 + k \\times (l-1)
$$

即使 $k$ 很小（如 12），深层也会有很多输入通道——这是特征重用的代价。

### 参数量对比

| 网络 | 核心思路 | 参数效率 |
|------|---------|---------|
| ResNet-50 | 残差连接 | 25.6M |
| ResNeXt-50 (32×4d) | 分组卷积（基数=32） | 25.0M |
| DenseNet-121 | 密集连接（k=32） | 8.0M |
| Wide ResNet-50-2 | 加倍宽度 | 68.9M |

---

## 可视化展示

### ResNeXt vs ResNet Block

\`\`\`mermaid
graph TD
    subgraph ResNet["ResNet Bottleneck (1×1→3×3→1×1)"]
        R1["1×1, 64"] --> R2["3×3, 64"] --> R3["1×1, 256"]
    end
    subgraph ResNeXt["ResNeXt (C=32, grouped)"]
        X1["1×1, 128"] --> X2["3×3, 4, groups=32<br/>(32条独立路径)"]
        X1 --> X2A["..."]
        X2 --> X3["concat→1×1, 256"]
        X2A --> X3
    end
    R3 --> ADD1["+"] --> OUT1["out"]
    X3 --> ADD2["+"] --> OUT2["out"]
\`\`\`

### DenseNet 连接模式

\`\`\`mermaid
graph LR
    X0["x₀"] --> C1["concat"] --> H1["H₁"] --> X1["x₁"]
    X0 --> C2["concat"] --> H2["H₂"] --> X2["x₂"]
    X1 --> C2
    X0 --> C3["concat"] --> H3["H₃"] --> X3["x₃"]
    X1 --> C3
    X2 --> C3
\`\`\`

### 参数效率对比

\`\`\`echarts
return {
  title: { text: 'ImageNet Top-1 Accuracy vs Params', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'value', name: '参数量 (M)' },
  yAxis: { type: 'value', name: 'Top-1 Accuracy (%)', min: 75, max: 82 },
  series: [
    { name: 'ResNet', type: 'scatter', symbolSize: 14,
      data: [[25.6, 76.1], [44.5, 77.4], [60.2, 78.3]],
      itemStyle: { color: '#2980b9' } },
    { name: 'ResNeXt', type: 'scatter', symbolSize: 14,
      data: [[25.0, 77.8], [44.2, 78.8]],
      itemStyle: { color: '#16a085' } },
    { name: 'DenseNet', type: 'scatter', symbolSize: 14,
      data: [[8.0, 74.9], [20.0, 77.6], [28.4, 78.2]],
      itemStyle: { color: '#d35400' } }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — ResNeXt Bottleneck

\`\`\`python
import torch.nn as nn

class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, cardinality=32, stride=1):
        super().__init__()
        mid_planes = cardinality * planes // cardinality  # 每个组的宽度
        self.conv1 = nn.Conv2d(in_planes, mid_planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        # 分组卷积（cardinality 个组）
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, 3,
                               stride=stride, padding=1,
                               groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion))

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return nn.functional.relu(out)
\`\`\`

### PyTorch — DenseNet DenseBlock

\`\`\`python
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_c, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, 4 * growth_rate, 1, bias=False),   # bottleneck
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)
\`\`\`
`;export{n as default};
