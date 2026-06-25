const n=`# U-Net Variants (UNet++ / V-Net / Mask2Former)

## 核心思想

U-Net 的 U 形 encoder-decoder + skip connections 设计简洁优雅，但有几个局限：最优深度未知（不同任务需要不同的感受野），2D 分割无法处理 3D 数据，实例分割需要额外的检测头。UNet++ 用**密集跳跃连接**解决深度不确定性，V-Net 将 U-Net 扩展到 **3D 医学影像** + Dice Loss，Mask2Former 引入**掩码注意力**统一语义/实例/全景分割。

---

## 数学定义与原理解析

### UNet++ — 密集跳跃连接

在第 $j$ 层，节点 $\\mathbf{x}^{i,j}$ 的输入来自**所有浅层同分辨率节点的拼接**：

$$
\\mathbf{x}^{i,j} = 
\\begin{cases}
H(\\mathbf{x}^{i-1,j}) & \\text{if } j=0 \\text{ (纯下采样路径)} \\\\
H([\\mathbf{x}^{i,k}]_{k=0}^{j-1}, \\mathcal{U}(\\mathbf{x}^{i+1,j-1})) & \\text{if } j>0
\\end{cases}
$$

$\\mathcal{U}$ 是上采样，$[\\cdot]$ 是拼接。结果：每个深度都有一个分割输出，可用**深度监督**训练。

### V-Net — 3D 卷积 + Dice Loss

将 U-Net 所有 2D 操作替换为 3D 卷积。关键创新是 **Dice Loss**：

$$
\\text{Dice} = \\frac{2 \\sum_i p_i g_i}{\\sum_i p_i + \\sum_i g_i}
$$

Dice 系数直接优化重叠度，天然处理**前景/背景极度不平衡**（器官占体积不到 1%）。与交叉熵对比：CE 对类不平衡敏感，Dice 天然鲁棒。

### Mask2Former — 掩码注意力

Mask2Former 将分割统一为**掩码分类**问题。Transformer decoder 中的 cross-attention 被替换为 **masked attention**——每个 query 只关注它预测的掩码区域：

$$
\\text{MaskedAttn}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{softmax}\\left(\\frac{\\mathbf{QK}^T}{\\sqrt{d}} + M\\right)\\mathbf{V}
$$

其中 $M$ 在非掩码区域为 0，在掩码外区域为 $-\\infty$。这强制高层 query 只关注其目标区域。

---

## 可视化展示

### UNet++ 密集连接

\`\`\`mermaid
graph TD
    X00["X⁰⁰ (输入)"] --> X10["X¹⁰ ↓"]
    X10 --> X20["X²⁰ ↓"]
    X20 --> X30["X³⁰ ↓"]

    X00 -.->|"skip"| X01["X⁰¹ ↑"]
    X10 -.-> X01
    X10 --> X21["X²¹ ↓"]
    X20 -.-> X21

    X00 -.-> X02["X⁰² ↑"]
    X10 -.-> X02
    X01 --> X02
    X01 -.-> X11["X¹¹ ↑"]
    X10 -.-> X11
    X20 -.-> X11
    X21 --> X11

    X30 --> X31["X³¹ ↑"]
    X20 -.-> X31
    X21 -.-> X31
\`\`\`

### 分割模型对比

\`\`\`echarts
return {
  title: { text: '分割架构对比', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['语义分割', '实例分割', '全景分割', '3D分割'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '支持度' },
  legend: { data: ['U-Net', 'DeepLab', 'Mask R-CNN', 'Mask2Former'] },
  series: [
    { name: 'U-Net', type: 'bar', data: [1, 0, 0, 1], itemStyle: { color: '#2c3e50' } },
    { name: 'DeepLab', type: 'bar', data: [1, 0, 0, 0], itemStyle: { color: '#2980b9' } },
    { name: 'Mask R-CNN', type: 'bar', data: [0, 1, 0, 0], itemStyle: { color: '#95a5a6' } },
    { name: 'Mask2Former', type: 'bar', data: [1, 1, 1, 0], itemStyle: { color: '#16a085' } }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

Mask2Former 是第一个单架构统一三种分割任务的方法。

---

## 核心代码实现

### PyTorch — UNet++ Block

\`\`\`python
import torch
import torch.nn as nn

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, init_features=32):
        super().__init__()
        f = init_features
        self.enc0 = self._conv_block(in_channels, f)
        self.enc1 = self._conv_block(f, f*2)
        self.enc2 = self._conv_block(f*2, f*4)
        self.pool = nn.MaxPool2d(2)

        # 上采样路径: x[i,j] 来自 x[i,j-1] (同层) 和 x[i+1,j-1] (上一层)
        self.up1_0 = nn.ConvTranspose2d(f*2, f, 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)

        self.dec0_1 = self._conv_block(f*2, f)      # [x00, up(x10)]
        self.dec1_1 = self._conv_block(f*4, f*2)    # [x10, up(x20)]
        self.dec0_2 = self._conv_block(f*3, f)      # [x00, x01, up(x11)]

        self.outputs = nn.ModuleList([
            nn.Conv2d(f, num_classes, 1),   # x[0,0]
            nn.Conv2d(f, num_classes, 1),   # x[0,1]
            nn.Conv2d(f, num_classes, 1),   # x[0,2]
        ])

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

    def forward(self, x):
        x00 = self.enc0(x)
        x10 = self.enc1(self.pool(x00))
        x20 = self.enc2(self.pool(x10))

        x01 = self.dec0_1(torch.cat([x00, self.up1_0(x10)], dim=1))
        x11 = self.dec1_1(torch.cat([x10, self.up2_0(x20)], dim=1))
        x02 = self.dec0_2(torch.cat([x00, x01, self.up2_1(x11)], dim=1))

        # 深度监督: 三个不同深度的输出
        return [self.outputs[i](out) for i, out in enumerate([x00, x01, x02])]
\`\`\`

### Dice Loss

\`\`\`python
def dice_loss(pred, target, smooth=1e-5):
    """pred: [B, C, H, W] after sigmoid, target: [B, C, H, W]"""
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()
\`\`\`
`;export{n as default};
