const n=`# Video Understanding (视频理解)

## 核心思想

静态图像分类回答"这是什么"，视频理解回答"这在发生什么"——需要同时建模空间外观和时间动态。核心挑战：视频是 $T \\times H \\times W \\times 3$ 的 4D 张量，直接将 2D CNN 扩展为 3D 会带来巨大的计算开销。三种主流思路：**3D CNN**（直接处理时空立方体）、**Two-Stream**（RGB + 光流并行）、**时空分离**（先空间再时间，如 R(2+1)D）。

---

## 数学定义与原理解析

### 3D 卷积

将 2D 卷积扩展到时间维度，kernel 为 $T_k \\times H_k \\times W_k$：

$$
y_{t,i,j} = \\sum_{\\tau=0}^{T_k-1} \\sum_{h=0}^{H_k-1} \\sum_{w=0}^{W_k-1} x_{t+\\tau, i+h, j+w} \\cdot w_{\\tau, h, w}
$$

计算量约是 2D 卷积的 $T_k$ 倍。C3D 用 $3\\times3\\times3$ kernel，I3D 将预训练的 2D 权重膨胀（"2D kernel 沿时间复制"）初始化 3D 卷积。

### I3D (Inflated 3D ConvNets)

将 ImageNet 预训练的 2D ConvNet（如 ResNet）的每一层"膨胀"为 3D：

$$
W_{3D}^{(t,h,w)} = \\frac{1}{T} W_{2D}^{(h,w)} \\quad \\text{(权重沿时间均匀复制)}
$$

同时输入 RGB 帧和预计算的光流，两路分别处理后在最后合并。这利用了 ImageNet 预训练的巨大优势。

### R(2+1)D — 时空分解

将 $t \\times h \\times w$ 的 3D 卷积分解为 $(1 \\times h \\times w)$ + $(t \\times 1 \\times 1)$：

$$
\\text{3D Conv}(T_k, H_k, W_k) \\approx \\text{2D Conv}(1, H_k, W_k) + \\text{1D Conv}(T_k, 1, 1)
$$

额外的 ReLU 在两者之间，增加非线性。参数效率更高，训练更容易。

### SlowFast

两个并行的路径：
- **Slow pathway**：低帧率（如 4fps）、大网络 → 捕获空间语义
- **Fast pathway**：高帧率（如 32fps）、小网络、高时间分辨率 → 捕获运动

通过横向连接（lateral connections）将 Fast 的信息送入 Slow。

---

## 可视化展示

### 视频理解模型架构对比

\`\`\`mermaid
graph TD
    subgraph C3D["3D CNN (C3D)"]
        C3D_in["T帧 RGB"] --> C3D_conv["3D Conv 堆叠"] --> C3D_out["分类"]
    end
    subgraph TwoStream["Two-Stream"]
        RGB["RGB 帧"] --> RGB_CNN["空间 CNN"]
        FLOW["光流"] --> FLOW_CNN["时间 CNN"]
        RGB_CNN --> FUSE["融合"] --> OUT["分类"]
        FLOW_CNN --> FUSE
    end
    subgraph SlowFast["SlowFast"]
        SLOW["慢路径 (低帧率)"] --> LAT["横向连接"] --> SLOW_OUT["输出"]
        FAST["快路径 (高帧率)"] --> LAT
    end
\`\`\`

### 模型性能对比

\`\`\`echarts
return {
  title: { text: '视频理解模型 Kinetics-400 Top-1', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['C3D', 'Two-Stream', 'I3D', 'R(2+1)D', 'SlowFast', 'VideoMAE'] },
  yAxis: { type: 'value', min: 60, max: 85, name: 'Top-1 Accuracy (%)' },
  series: [{
    type: 'bar',
    data: [61.3, 68.4, 71.1, 74.3, 79.8, 82.5],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top', formatter: '{c}%' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — R(2+1)D Block

\`\`\`python
import torch
import torch.nn as nn

class R2Plus1DBlock(nn.Module):
    """将 3D 卷积分解为 (2+1)D"""
    def __init__(self, in_c, out_c, mid_c=None, temporal_kernel=3):
        super().__init__()
        mid_c = mid_c or out_c

        # Spatial conv: 1×k×k
        self.spatial = nn.Conv3d(in_c, mid_c, kernel_size=(1, 3, 3),
                                  padding=(0, 1, 1), bias=False)
        self.spatial_bn = nn.BatchNorm3d(mid_c)
        # Temporal conv: t×1×1
        self.temporal = nn.Conv3d(mid_c, out_c, kernel_size=(temporal_kernel, 1, 1),
                                   padding=(temporal_kernel//2, 0, 0), bias=False)
        self.temporal_bn = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Conv3d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        # x: [B, C, T, H, W]
        out = self.relu(self.spatial_bn(self.spatial(x)))
        out = self.relu(self.temporal_bn(self.temporal(out)))
        out += self.shortcut(x)
        return self.relu(out)


class SlowFastFusion(nn.Module):
    """SlowFast 横向连接"""
    def __init__(self, slow_dim, fast_dim, alpha=8):
        super().__init__()
        # Fast→Slow: 时间下采样, 通道变换
        self.fast_to_slow = nn.Sequential(
            nn.Conv3d(fast_dim // alpha, slow_dim // alpha, 1, bias=False),
            nn.BatchNorm3d(slow_dim // alpha))

    def forward(self, slow, fast):
        # slow: [B, C_slow, T_slow, H, W]
        # fast: [B, C_fast, T_fast, H, W]  (T_fast = α × T_slow)
        B, C_f, T_f, H, W = fast.shape
        T_s = slow.shape[2]
        alpha = T_f // T_s

        # 时间降采样: reshape→pool→reshape
        fast_reshaped = fast.view(B, C_f, T_s, alpha, H, W)
        fast_down = fast_reshaped.mean(dim=3)  # [B, C_f, T_s, H, W]
        fast_proj = self.fast_to_slow(fast_down)

        return torch.cat([slow, fast_proj], dim=1)
\`\`\`
`;export{n as default};
