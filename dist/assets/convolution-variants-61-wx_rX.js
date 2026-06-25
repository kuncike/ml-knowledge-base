const n=`# Convolution Variants (卷积变体)

## 核心思想

标准卷积用固定 kernel 在规则网格上滑动，适合"局部平移不变"的特征。但现实有多种特殊需求：扩大感受野但不想增加参数（空洞卷积）→ 用更大的间距采样；移动端部署需要极低计算量（深度可分离卷积）→ 先 per-channel 再 1×1；物体形变导致规则网格失效（可变形卷积）→ 让网络自己学采样位置的偏移。

---

## 数学定义与原理解析

### 空洞卷积 (Dilated Convolution)

在标准卷积的采样点之间引入空洞（dilation rate $r$）：

$$
y[p] = \\sum_{k} x[p + r \\cdot k] \\cdot w[k]
$$

有效感受野：$k' = k + (k-1)(r-1)$。不增加参数和计算量的前提下指数级扩大感受野——DeepLab 系列的核心。

### 深度可分离卷积 (Depthwise Separable)

分解为标准卷积的两步：

**Depthwise**：每个通道独立卷积（$C$ 个 $k \\times k$ 单通道卷积）
**Pointwise**：$1 \\times 1$ 卷积混合通道

计算量对比（$C_{in}=C_{out}=C$）：

$$
\\frac{\\text{FLOPs}_{ds}}{\\text{FLOPs}_{std}} = \\frac{C \\cdot k^2 \\cdot H \\cdot W + C^2 \\cdot H \\cdot W}{C^2 \\cdot k^2 \\cdot H \\cdot W} = \\frac{1}{C} + \\frac{1}{k^2} \\approx \\frac{1}{k^2}
$$

3×3 卷积大约节省 8-9 倍计算——MobileNet 的基石。

### 可变形卷积 (Deformable Convolution)

标准卷积在规则网格 $\\mathcal{R}$ 上采样。可变形卷积学习采样点的**偏移** $\\Delta p_n$：

$$
y[p_0] = \\sum_{p_n \\in \\mathcal{R}} w[p_n] \\cdot x[p_0 + p_n + \\Delta p_n]
$$

$\\Delta p_n$ 由另一个卷积层从输入中学习。偏移可以是小数，需要双线性插值采样。

### 转置卷积 (Transposed Convolution)

"逆卷积"的直觉：标准卷积的矩阵形式是 $\\mathbf{y} = \\mathbf{C} \\cdot \\mathbf{x}$（$\\mathbf{C}$ 是稀疏 Toeplitz 矩阵），转置卷积是 $\\mathbf{x}' = \\mathbf{C}^T \\cdot \\mathbf{y}$。用于上采样——GAN 的生成器和 U-Net 的解码器。

---

## 可视化展示

### 五种卷积对比

\`\`\`mermaid
graph TD
    STD["标准卷积<br/>k×k, 全通道"] -->|"分解"| DW["深度可分离<br/>Depthwise + 1×1"]
    STD -->|"增大 dilation"| DIL["空洞卷积<br/>扩大感受野, 不增参数"]
    STD -->|"学习采样偏移"| DEF["可变形卷积<br/>适应物体形变"]
    STD -->|"转置"| TRANS["转置卷积<br/>上采样"]
    STD -->|"通道分组"| GROUP["分组卷积<br/>ResNeXt"]
\`\`\`

### 空洞卷积的感受野增长

\`\`\`echarts
return {
  title: { text: '空洞卷积感受野 (k=3)', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['r=1(标准)', 'r=2', 'r=3', 'r=4', 'r=2^2层', 'r=2^3层'] },
  yAxis: { type: 'value', name: '感受野大小' },
  series: [{
    type: 'bar',
    data: [3, 5, 7, 9, 7, 15],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

空洞卷积在不增加参数的前提下指数级扩大感受野——DeepLab 用 $r=1,2,4,8,\\dots$ 的级联达到 224×224 的感受野。

---

## 核心代码实现

### PyTorch — 各类卷积

\`\`\`python
import torch
import torch.nn as nn

# 1. 空洞卷积 — 扩大感受野, 不增参数
dilated_conv = nn.Conv2d(64, 64, 3, dilation=2, padding=2)

# 2. 深度可分离卷积 = Depthwise + Pointwise
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, kernel=3):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, kernel,
                                    groups=in_c,
                                    padding=kernel//2, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# 3. 转置卷积 — 上采样
transposed_conv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

# 4. 分组卷积 — 通道间独立
grouped_conv = nn.Conv2d(64, 64, 3, groups=4, padding=1)

# 5. 可变形卷积 (PyTorch >= 1.9)
from torchvision.ops import DeformConv2d
deform_conv = DeformConv2d(64, 64, 3, padding=1)
offset = nn.Conv2d(64, 2 * 9, 3, padding=1)  # 每个采样点 (Δx, Δy)
# 使用: out = deform_conv(x, offset(x))
\`\`\`

### 计算量对比

\`\`\`python
def count_flops(in_c, out_c, k, h, w):
    std = in_c * out_c * k * k * h * w
    ds = in_c * k * k * h * w + in_c * out_c * h * w  # depthwise separable
    return std, ds, ds / std

# 3×3 Conv, 64→128, 56×56 feature map
std, ds, ratio = count_flops(64, 128, 3, 56, 56)
# 标准: 463M FLOPs, 深度可分离: 57M FLOPs, 约 12% 计算量
\`\`\`
`;export{n as default};
