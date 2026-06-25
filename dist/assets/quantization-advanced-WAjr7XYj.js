const n=`# Advanced Quantization (SmoothQuant / FP8 / NF4 / BitNet)

## 核心思想

传统量化方法对 LLM 不够友好——激活值中存在**极端 outlier**（某些通道数值比平均值大 100×），导致 INT8 量化直接崩溃。SmoothQuant 的洞察：**将激活 outlier 的量化难度通过数学等价变换"迁移"到权重上**（权重更容易量化）。FP8 是 NVIDIA H100 的硬件原生格式，比 INT8 有更大的动态范围。NF4 是 QLoRA 专门设计的 4-bit 正态分布量化。BitNet 更极端——用 1-bit 权重（±1）训练 LLM。

---

## 数学定义与原理解析

### SmoothQuant — 激活-权重迁移

原始线性层：$\\mathbf{Y} = \\mathbf{X} \\mathbf{W}$

激活 $\\mathbf{X}$ 的某些通道有巨大值 → 难以量化。SmoothQuant 引入对角缩放矩阵 $\\text{diag}(\\mathbf{s})$：

$$
\\mathbf{Y} = \\mathbf{X} \\mathbf{W} = \\mathbf{X} \\cdot \\text{diag}(\\mathbf{s})^{-1} \\cdot \\text{diag}(\\mathbf{s}) \\cdot \\mathbf{W} = \\hat{\\mathbf{X}} \\hat{\\mathbf{W}}
$$

数学上完全等价。但 $\\hat{\\mathbf{X}}$ 的 outlier 被 $\\mathbf{s}$ 压制（更好量化），$\\hat{\\mathbf{W}}$ 吸收了缩放因子（权重本身较小，可以承受）。

$\\mathbf{s}$ 的计算（per-channel smoothing factor）：

$$
s_j = \\max(|X_j|)^\\alpha \\cdot \\max(|W_j|)^{1-\\alpha}
$$

$\\alpha$ 是迁移强度（0.5 效果最好，均衡地迁移难度）。

### NF4 — 4-bit NormalFloat

QLoRA 发现标准 4-bit INT（均匀量化）对 LLM 权重效果不好——权重接近正态分布。NF4 假设 $\\mathbf{w} \\sim \\mathcal{N}(0, \\sigma^2)$，将量化区间按照正态分布的**等概率分位点**设计：

$$
Q_{NF4} = \\{q_i : \\Phi(q_i) = \\frac{i}{16}, i = 0, \\ldots, 15\\}
$$

其中 $\\Phi$ 是标准正态 CDF。这样在高概率密度区域有更细的分辨率，低密度区域更粗。

### BitNet — 1-bit 权重 + 8-bit 激活

BitNet 用 BitLinear 替代 nn.Linear：

$$
\\mathbf{y} = \\tilde{\\mathbf{W}} \\tilde{\\mathbf{x}} \\cdot \\frac{\\|\\mathbf{W}\\|_1}{n} \\cdot \\frac{\\|\\mathbf{x}\\|_\\infty}{\\gamma}
$$

其中 $\\tilde{\\mathbf{W}} = \\text{sign}(\\mathbf{W})$ 是 1-bit 权重（±1），$\\tilde{\\mathbf{x}}$ 被量化为 8-bit。缩放因子来自权重的 L1 范数和激活的 $\\infty$ 范数。

关键：**权重只存储符号**，内存占用为 FP16 的 1/16。

### FP8 (E4M3 / E5M2)

NVIDIA H100 原生支持的两种 FP8 格式：
- **E4M3**：4 位指数 + 3 位尾数 = 精度更高，用于前向传播
- **E5M2**：5 位指数 + 2 位尾数 = 动态范围更大，用于反向传播

FP8 相比 INT8 的优势：动态范围大（指数位更多），不需要逐通道/逐 token 标定。

---

## 可视化展示

### SmoothQuant 核心思想

\`\`\`mermaid
graph TD
    X["X (有 outlier 通道)<br/>量化困难"] --> S["÷ s (per-channel)"]
    S --> XHAT["X̂ = X/s (平滑)<br/>✅ 易量化"]
    W["W (权重较小)<br/>✅ 易量化"] --> SMUL["× s (per-channel)"]
    SMUL --> WHAT["Ŵ = s·W (增大)<br/>⚠ 可承受"]
    XHAT --> MUL["Ŵ · X̂"]
    WHAT --> MUL
    MUL --> Y["Y = XW (完全相同)"]
\`\`\`

### 量化格式对比

\`\`\`echarts
return {
  title: { text: '量化格式对比 (LLaMA-7B)', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['FP16', 'INT8', 'FP8', 'INT4+NF4', 'BitNet (1.58bit)'] },
  yAxis: { type: 'value', min: 0, max: 14, name: '显存占用 (GB)' },
  series: [{
    type: 'bar',
    data: [13, 7, 7, 3.5, 1.3],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top', formatter: '{c} GB' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

---

## 核心代码实现

### NF4 量化

\`\`\`python
import torch

def create_nf4_quantization_tensor(size):
    """生成 NF4 量化级别 (16 个值, 按正态分布分位点)"""
    from scipy.stats import norm

    # 16 个等概率分位点 (offset 方法避免边界)
    offsets = torch.tensor([-0.9986, -0.9674, -0.9085, -0.8416,
                             -0.7227, -0.5557, -0.3402, -0.1200,
                             0.1200, 0.3402, 0.5557, 0.7227,
                             0.8416, 0.9085, 0.9674, 0.9986])

    # 等概率分位点 → 概率 → 正态分位点
    probs = (offsets + 1) / 2  # → [0, 1]
    levels = norm.ppf(probs.numpy())
    return torch.tensor(levels)


def nf4_quantize(weight, quant_levels):
    """将浮点权重映射到最近的 NF4 级别"""
    # 归一化到 [-1, 1]
    scale = weight.abs().max()
    w_norm = weight / scale

    # 找最近的量化级别
    w_flat = w_norm.flatten()
    distances = torch.abs(w_flat.unsqueeze(1) - quant_levels.unsqueeze(0))
    indices = distances.argmin(dim=1)
    w_quantized = quant_levels[indices].view(weight.shape) * scale
    return w_quantized, scale, indices
\`\`\`

### SmoothQuant 激活尺度变换

\`\`\`python
def smooth_linear(linear, x, alpha=0.5):
    """SmoothQuant: 迁移激活 outlier 难度到权重"""
    W = linear.weight.data  # [out_c, in_c]
    x_max = x.abs().max(dim=0)[0]  # [in_c]
    w_max = W.abs().max(dim=0)[0]   # [in_c]

    # 平滑因子
    s = (x_max ** alpha) / (w_max ** (1 - alpha))
    s = torch.clamp(s, min=1e-5)
    s_inv = 1.0 / s

    # 应用变换
    x_smoothed = x * s_inv.unsqueeze(0)
    W_smoothed = W * s.unsqueeze(0)

    # 数学等价: x_smoothed @ W_smoothed.T = x @ W.T
    return x_smoothed, W_smoothed
\`\`\`

### BitLinear (1-bit 权重)

\`\`\`python
class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, weight: torch.Tensor):
        # x: [B, in_f]
        # weight: [out_f, in_f] — 存储为 FP, 使用时二值化
        w_binarized = weight.sign()  # ±1

        # AbsMax 量化激活到 8-bit
        gamma = x.abs().max(dim=-1, keepdim=True)[0]  # [B, 1]
        x_quantized = torch.clamp(x * 127 / (gamma + self.eps), -128, 127).round()

        # 缩放因子
        beta = weight.abs().mean(dim=1).unsqueeze(0)  # [1, out_f]

        # BitLinear 前向
        out = (x_quantized @ w_binarized.T) * beta * gamma / 127
        return out
\`\`\`
`;export{n as default};
