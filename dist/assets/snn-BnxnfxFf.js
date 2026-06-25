const n=`# SNN (脉冲神经网络)

## 核心思想

传统 ANN 用浮点激活值在连续时间上传递信息；SNN 用**脉冲（spike）**——0/1 离散事件在时间维度上传递信息。这更接近生物大脑的工作方式（神经元膜电位积累→达到阈值→发放脉冲→恢复）。核心优势：事件驱动的稀疏计算 → 极低功耗（神经形态芯片如 Loihi、TrueNorth）。核心挑战：脉冲是不可微的 → 训练难。

SNN 的信息编码在脉冲的**频率**（Rate Coding）或**精确时间**（Temporal Coding）中。

---

## 数学定义与原理解析

### LIF 神经元模型（最常用）

膜电位 $u(t)$ 的动态方程：

$$
\\tau_m \\frac{du}{dt} = -(u(t) - u_{rest}) + R \\cdot I(t)
$$

离散化形式（Euler 方法）：

$$
u[t] = \\underbrace{\\beta \\cdot u[t-1]}_{\\text{泄漏}} + \\underbrace{w \\cdot s_{in}[t]}_{\\text{输入脉冲}} - \\underbrace{\\theta \\cdot s_{out}[t-1]}_{\\text{复位}}
$$

其中 $\\beta = e^{-1/\\tau_m} < 1$ 是泄漏因子，$\\theta$ 是阈值。

**脉冲发放规则**：

$$
s_{out}[t] = \\begin{cases} 1 & \\text{if } u[t] \\geq \\theta \\\\ 0 & \\text{otherwise} \\end{cases}
$$

发放后：$u[t] \\leftarrow u[t] - \\theta$（硬复位）或 $u[t] \\leftarrow u[t] \\cdot (1 - s_{out}[t])$（软复位）。

### STDP (Spike-Timing-Dependent Plasticity)

无监督学习规则——突触权重变化取决于前后脉冲的时间差：

$$
\\Delta w = \\begin{cases}
A_+ \\cdot e^{-\\Delta t / \\tau_+} & \\text{if } \\Delta t > 0 \\text{ (前→后)} \\\\
-A_- \\cdot e^{\\Delta t / \\tau_-} & \\text{if } \\Delta t < 0 \\text{ (后→前)}
\\end{cases}
$$

$\\Delta t = t_{post} - t_{pre}$。"先因后果"增强连接，"先果后因"削弱连接。

### 替代梯度 (Surrogate Gradient) 训练

SNN 直接训练的核心技巧——前向用硬阈值（阶梯函数），反向用平滑近似：

$$
\\frac{\\partial s}{\\partial u} \\approx \\frac{1}{1 + \\alpha|u - \\theta|^2} \\quad \\text{或} \\quad \\max(0, 1 - |u - \\theta|)
$$

---

## 可视化展示

### LIF 神经元膜电位动态

\`\`\`echarts
return {
  title: { text: 'LIF 神经元膜电位变化', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'value', name: '时间步', min: 0, max: 50 },
  yAxis: { type: 'value', name: '膜电位 u(t)', min: -0.2, max: 1.2 },
  series: [{
    type: 'line', step: 'end',
    data: (function() {
      const d = [], theta = 1.0, beta = 0.9;
      let u = 0, input_spikes = [5,6,7, 20,21,22,23, 38,39,40,41,42];
      for (let t = 0; t <= 50; t++) {
        let I = input_spikes.includes(t) ? 0.3 : 0;
        u = beta * u + I;
        if (u >= theta) { u = 0; d.push([t, theta]); d.push([t, 0]); }
        else d.push([t, u]);
      }
      return d;
    })(),
    lineStyle: { color: '#2c3e50', width: 1.5 },
    markLine: { silent: true, data: [{ yAxis: 1.0, label: { formatter: '阈值 θ' }, lineStyle: { type: 'dashed', color: '#c0392b' } }] }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

### SNN 信息处理流程

\`\`\`mermaid
graph LR
    IN["输入脉冲序列"] --> ENC["编码"]
    ENC --> L1["LIF 层1"]
    L1 --> L2["LIF 层2"]
    L2 --> L3["LIF 层3"]
    L3 --> DEC["解码 (spike count)"]
    DEC --> OUT["输出"]

    subgraph "时间展开 (每一步稀疏激活)"
        T1["t=1"] --> T2["t=2"] --> T3["..."] --> TT["t=T"]
    end
\`\`\`

---

## 核心代码实现

### PyTorch — LIF 神经元

\`\`\`python
import torch
import torch.nn as nn

# 替代梯度：反正切型
class SurrogateArctan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=2.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_in = grad_out / (1 + alpha * x * x)  # arctan 导数近似
        return grad_in, None

class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, beta=0.9):
        super().__init__()
        self.threshold = threshold
        self.beta = beta

    def forward(self, x):
        # x: [B, T, N]
        B, T, N = x.shape
        u = torch.zeros(B, N, device=x.device)
        spikes = []
        for t in range(T):
            u = self.beta * u + x[:, t, :]
            s = SurrogateArctan.apply(u - self.threshold)
            u = u * (1 - s)  # 软复位
            spikes.append(s)
        return torch.stack(spikes, dim=1)  # [B, T, N]

# SNN 模型
class SimpleSNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, T=16):
        super().__init__()
        self.T = T
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.lif1 = LIFNeuron(threshold=1.0, beta=0.85)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.lif2 = LIFNeuron(threshold=1.0, beta=0.85)

    def forward(self, x):
        # x: [B, in_dim] → 展开为时间序列
        x = x.unsqueeze(1).repeat(1, self.T, 1)  # [B, T, in_dim]
        h = self.lif1(self.fc1(x))
        out = self.lif2(self.fc2(h))
        return out.sum(dim=1)  # rate coding: 总脉冲数作为输出
\`\`\`
`;export{n as default};
