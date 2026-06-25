const n=`# ConvLSTM (卷积 LSTM)

## 核心思想

标准 LSTM 用全连接处理空间维度——输入必须是向量。ConvLSTM 把全连接替换为卷积：输入、隐状态、门控之间全是卷积操作。这使得 LSTM 可以直接处理**时空序列**——每一帧是 2D 图像，时间维度由 LSTM 的循环连接建模。

典型应用：降水临近预报（雷达回波外推）、视频预测、交通流量预测。它同时保留了卷积的空间建模能力和 LSTM 的时间记忆能力。

---

## 数学定义与原理解析

### ConvLSTM 门控方程

将标准 LSTM 的门控方程中所有矩阵乘法变为卷积：

$$
\\mathbf{i}_t = \\sigma(\\mathbf{W}_{xi} * \\mathbf{X}_t + \\mathbf{W}_{hi} * \\mathbf{H}_{t-1} + \\mathbf{b}_i)
$$

$$
\\mathbf{f}_t = \\sigma(\\mathbf{W}_{xf} * \\mathbf{X}_t + \\mathbf{W}_{hf} * \\mathbf{H}_{t-1} + \\mathbf{b}_f)
$$

$$
\\mathbf{o}_t = \\sigma(\\mathbf{W}_{xo} * \\mathbf{X}_t + \\mathbf{W}_{ho} * \\mathbf{H}_{t-1} + \\mathbf{b}_o)
$$

$$
\\tilde{\\mathbf{C}}_t = \\tanh(\\mathbf{W}_{xc} * \\mathbf{X}_t + \\mathbf{W}_{hc} * \\mathbf{H}_{t-1} + \\mathbf{b}_c)
$$

$$
\\mathbf{C}_t = \\mathbf{f}_t \\odot \\mathbf{C}_{t-1} + \\mathbf{i}_t \\odot \\tilde{\\mathbf{C}}_t
$$

$$
\\mathbf{H}_t = \\mathbf{o}_t \\odot \\tanh(\\mathbf{C}_t)
$$

其中 $\\mathbf{X}_t, \\mathbf{H}_t, \\mathbf{C}_t \\in \\mathbb{R}^{C \\times H \\times W}$（3D 张量），$*$ 是卷积运算。

### 与标准 LSTM 的关键区别

| 组件 | 标准 LSTM | ConvLSTM |
|------|----------|----------|
| 输入形状 | $(B, d)$ | $(B, C, H, W)$ |
| 权重运算 | 矩阵乘 $\\mathbf{Wx}$ | 卷积 $\\mathbf{W} * \\mathbf{x}$ |
| 隐状态 | 向量 $\\mathbf{h}$ | 特征图 $\\mathbf{H}$ |
| 感受野 | N/A | 由 kernel size 决定 |

---

## 可视化展示

### ConvLSTM 单元结构

\`\`\`mermaid
graph TD
    X["Xₜ (当前帧)"]
    H["Hₜ₋₁ (上一隐状态)"]
    C["Cₜ₋₁ (上一细胞态)"]

    subgraph Gates["门控 (卷积 + Sigmoid/Tanh)"]
        I["iₜ = σ(Conv(Xₜ, Hₜ₋₁))"]
        F["fₜ = σ(Conv(Xₜ, Hₜ₋₁))"]
        O["oₜ = σ(Conv(Xₜ, Hₜ₋₁))"]
        G["gₜ = tanh(Conv(Xₜ, Hₜ₋₁))"]
    end

    X --> I & F & O & G
    H --> I & F & O & G

    I --> MUL1["⊙"]
    G --> MUL1
    C -->|"fₜ ⊙"| PLUS["+"] --> CNEW["Cₜ"]
    MUL1 --> PLUS
    PLUS --> TANH["tanh"] --> MUL2["⊙"]
    O --> MUL2 --> HNEW["Hₜ"]
\`\`\`

### ConvLSTM vs 3D CNN 对比

\`\`\`echarts
return {
  title: { text: '时空建模方法对比', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['参数量', '长时序', '局部空间', '训练难度'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '相对得分' },
  legend: { data: ['ConvLSTM', '3D CNN', 'ConvGRU'] },
  series: [
    { name: 'ConvLSTM', type: 'bar', data: [0.6, 0.85, 0.8, 0.55], itemStyle: { color: '#2c3e50' } },
    { name: '3D CNN', type: 'bar', data: [0.8, 0.4, 0.9, 0.75], itemStyle: { color: '#2980b9' } },
    { name: 'ConvGRU', type: 'bar', data: [0.5, 0.82, 0.78, 0.6], itemStyle: { color: '#16a085' } }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

---

## 核心代码实现

### PyTorch ConvLSTM Cell

\`\`\`python
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # 合并所有卷积为一个大卷积（效率优化）
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size, padding=padding)

    def forward(self, x, state):
        # x: [B, C_in, H, W]
        # state: (h, c) each [B, C_hidden, H, W]
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)  # [B, C_in+H, H, W]
        gates = self.conv(combined)  # [B, 4*H, H, W]
        i, f, o, g = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, (h, c)


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, num_layers=2):
        super().__init__()
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.cells.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        states = [None] * len(self.cells)
        outputs = []

        for t in range(T):
            h = x[:, t]
            for i, cell in enumerate(self.cells):
                h, states[i] = cell(h, states[i] or (
                    torch.zeros(B, cell.hidden_channels, H, W, device=x.device),
                    torch.zeros(B, cell.hidden_channels, H, W, device=x.device)
                ))
            outputs.append(h)

        return torch.stack(outputs, dim=1)  # [B, T, H, H, W]
\`\`\`
`;export{n as default};
