const n=`# Pooling Layers (池化层)

## 核心思想

池化层的本质是**下采样 + 局部不变性**。它对特征图的每个局部窗口做聚合操作（取最大值或平均值），在降低空间维度、减小计算量的同时，赋予网络一定程度的平移不变性——物体稍微移动，池化后的特征依然稳定。

---

## 数学定义与原理解析

### Max Pooling（最大池化）

取窗口内最大值，保留最强响应：

$$
y_{i,j} = \\max_{p,q \\in [0, k-1]} x_{i \\cdot s + p,\\ j \\cdot s + q}
$$

- 保留纹理、边缘等强信号
- 反向传播时梯度**仅通过最大值位置**回传（稀疏梯度）
- $k=2, s=2$ 实现 $2\\times$ 下采样

### Average Pooling（平均池化）

取窗口内平均值，输出更平滑：

$$
y_{i,j} = \\frac{1}{k^2} \\sum_{p=0}^{k-1} \\sum_{q=0}^{k-1} x_{i \\cdot s + p,\\ j \\cdot s + q}
$$

- 保留背景和全局信息
- 反向传播时梯度**均匀分配**给窗口内每个位置

### Global Average Pooling (GAP)

整个空间维度取平均，得到一个标量：

$$
y_c = \\frac{1}{H \\times W} \\sum_{i=1}^{H} \\sum_{j=1}^{W} x_{i,j,c}
$$

**GAP 替代 Flatten + FC** 是 CNN 设计的重大转折：
- 参数量从 $HW \\times C_{in} \\times C_{out}$ 降为 **0**（无参数）
- 强制特征图与类别直接对应（可解释性强）
- 防止过拟合

### 反向传播对比

- **Max Pooling**：梯度只通过最大值位置回传——"赢者通吃"
- **Avg Pooling**：梯度均匀分配——"雨露均沾"

---

## 可视化展示

### 池化操作示意

| 3 | 1 | 2 | 5 |
|---|---|---|---|
| 4 | 8 | 1 | 3 |
| 2 | 6 | 7 | 2 |
| 1 | 3 | 4 | 9 |

$2 \\times 2$ Max Pooling, stride=2 → |8, 5| / |6, 9|

### Pooling vs Strided Conv 的效果对比

\`\`\`echarts
return {
  xAxis: { type: 'category', data: ['MaxPool 2×2', 'AvgPool 2×2', 'Strided Conv', 'GAP'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '相对得分' },
  legend: { data: ['可学习性', '平移不变性', '计算效率', '参数量(低=好)'] },
  series: [
    { name: '可学习性', type: 'bar', data: [0, 0, 1, 0], itemStyle: { color: '#2c3e50' } },
    { name: '平移不变性', type: 'bar', data: [0.9, 0.95, 0.6, 0.3], itemStyle: { color: '#16a085' } },
    { name: '计算效率', type: 'bar', data: [1, 1, 0.7, 1], itemStyle: { color: '#d35400' } },
    { name: '参数量(低=好)', type: 'bar', data: [1, 1, 0.3, 1], itemStyle: { color: '#2980b9' } }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch

\`\`\`python
import torch.nn as nn

# 最大池化 — 2× 下采样（最常用）
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# 平均池化
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

# 全局平均池化 — 将任意尺寸特征图压为 1×1
gap = nn.AdaptiveAvgPool2d((1, 1))

# 全局最大池化
gmp = nn.AdaptiveMaxPool2d((1, 1))
\`\`\`

### NumPy 手写

\`\`\`python
import numpy as np

def max_pool2d(x, kernel_size=2, stride=2):
    H, W = x.shape
    out_h, out_w = (H - kernel_size) // stride + 1, (W - kernel_size) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            ii, jj = i * stride, j * stride
            out[i, j] = np.max(x[ii:ii+kernel_size, jj:jj+kernel_size])
    return out

def global_avg_pool(x):
    """x shape: (C, H, W) → (C,)"""
    return np.mean(x, axis=(1, 2))
\`\`\`

---

## 池化 vs 步长卷积

| 对比维度 | Pooling | Strided Conv (s=2) |
|----------|---------|-------------------|
| 可学习参数 | 0 | $C_{in} \\times C_{out} \\times k^2$ |
| 平移不变性 | 强 | 弱 |
| 计算量 | 低 | 高 |
| 信息损失 | 不可逆 | 可学习（保留有用信息） |
| 现代趋势 | 逐渐被替代 | 成为主流 |

现代 CNN 架构（ResNet、EfficientNet）越来越倾向于用 **stride=2 的卷积**替代池化层来下采样。
`;export{n as default};
