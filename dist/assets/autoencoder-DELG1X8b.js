const n=`# Autoencoder (自编码器)

## 核心思想

自编码器的直觉很简单：**把输入压缩到低维瓶颈，再重构出来**。如果模型能从压缩的表示中恢复原始输入，说明这个压缩表示抓住了数据的核心结构。AE 不是"记住了输入"，而是被迫学习了数据流形的参数化——瓶颈的维度限制阻止了简单的恒等映射。

变体：**稀疏 AE** 加 L1 惩罚让大部分神经元不激活（生物可解释性），**降噪 AE** 训练模型从损坏的输入中恢复干净数据（鲁棒性）。

---

## 数学定义与原理解析

### 标准 Autoencoder

编码器 $e_\\phi$ 和解码器 $d_\\theta$：

$$
\\mathbf{z} = e_\\phi(\\mathbf{x}), \\quad \\hat{\\mathbf{x}} = d_\\theta(\\mathbf{z})
$$

损失：$\\mathcal{L} = \\| \\mathbf{x} - \\hat{\\mathbf{x}} \\|^2$（MSE）或 BCE（二值数据）。

### 稀疏自编码器

在瓶颈层加稀疏性约束——通常用 KL 散度惩罚偏离目标激活率 $\\rho$：

$$
\\mathcal{L} = \\| \\mathbf{x} - \\hat{\\mathbf{x}} \\|^2 + \\beta \\sum_{j} \\text{KL}(\\rho \\| \\hat{\\rho}_j)
$$

其中 $\\hat{\\rho}_j = \\frac{1}{n}\\sum_i z_j^{(i)}$ 是神经元 $j$ 的平均激活，$\\text{KL}(\\rho \\| \\hat{\\rho}) = \\rho \\log\\frac{\\rho}{\\hat{\\rho}} + (1-\\rho)\\log\\frac{1-\\rho}{1-\\hat{\\rho}}$。

$\\rho$ 通常设得很小（如 0.05），强制每个神经元只在极少数样本上激活。

### 降噪自编码器 (DAE)

输入随机损坏（加噪声 / Dropout），输出要求还原干净数据：

$$
\\mathcal{L} = \\| \\mathbf{x} - d_\\theta(e_\\phi(\\tilde{\\mathbf{x}})) \\|^2
$$

其中 $\\tilde{\\mathbf{x}} \\sim q(\\tilde{\\mathbf{x}} | \\mathbf{x})$，如 $\\tilde{\\mathbf{x}} = \\mathbf{x} + \\mathcal{N}(0, \\sigma^2)$ 或随机置零。

---

## 可视化展示

### AE 架构流程

\`\`\`mermaid
graph LR
    X["输入 x (d维)"] --> ENC["编码器 e_ϕ"]
    ENC --> Z["瓶颈 z (k维, k<<d)"]
    Z --> DEC["解码器 d_θ"]
    DEC --> XHAT["重构 x̂ (d维)"]
    X --> LOSS["L = ‖x - x̂‖²"]
    XHAT --> LOSS
\`\`\`

### 原始 vs 重构对比（概念）

\`\`\`echarts
return {
  title: { text: 'MNIST AE 重构质量 vs 瓶颈维度', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['k=2', 'k=8', 'k=16', 'k=32', 'k=64', 'k=128'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '重构 MSE (↓ 越好)' },
  series: [{
    type: 'bar',
    data: [0.065, 0.028, 0.015, 0.010, 0.008, 0.007],
    itemStyle: { color: '#2c3e50' },
    markLine: { silent: true, data: [{ yAxis: 0.005, label: { formatter: '原始空间 784d' }, lineStyle: { type: 'dashed', color: '#95a5a6' } }] }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

瓶颈越小压缩越狠，但重构越差——这是"压缩-重构"的经典权衡。

---

## 核心代码实现

### PyTorch — 降噪自编码器

\`\`\`python
import torch
import torch.nn as nn

class DenoisingAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 64]):
        super().__init__()
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        for h in reversed(hidden_dims[:-1]):
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x, noise_std=0.2):
        # 注入噪声
        x_noisy = x + torch.randn_like(x) * noise_std
        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(self, x):
        x_hat, z = self.forward(x)
        return nn.functional.mse_loss(x_hat, x)
\`\`\`

### 稀疏自编码器

\`\`\`python
def kl_sparsity_loss(z, rho=0.05):
    # z: [B, k], 计算每个神经元的平均激活
    rho_hat = z.mean(dim=0)  # [k]
    rho = torch.full_like(rho_hat, rho)
    kl = rho * torch.log(rho / rho_hat) + \\
         (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    return kl.sum()
\`\`\`
`;export{n as default};
