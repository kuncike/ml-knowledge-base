const n=`# VAE (变分自编码器)

## 核心思想

标准自编码器的瓶颈 $z$ 是一个**确定的点**——给定输入 x，编码器产生一个固定的 z。VAE 的关键创新：**把瓶颈变成一个概率分布** $q(z|x)$。编码器输出 $\\mu$ 和 $\\sigma$（高斯参数），解码器从 $z \\sim \\mathcal{N}(\\mu, \\sigma^2)$ 采样重建。这使得 $z$ 空间变得连续和平滑——相近的 $z$ 对应相似的输出，$z$ 之间的插值也合理。

**重参数化技巧**是让梯度能流过采样的关键：$z = \\mu + \\sigma \\odot \\epsilon$，其中 $\\epsilon \\sim \\mathcal{N}(0,1)$。

---

## 数学定义与原理解析

### ELBO（证据下界）

VAE 最大化 ELBO，而非直接最大化 $\\log p(x)$：

$$
\\log p(x) \\geq \\underbrace{\\mathbb{E}_{q_\\phi(z|x)}[\\log p_\\theta(x|z)]}_{\\text{重构项}} - \\underbrace{D_{KL}(q_\\phi(z|x) \\| p(z))}_{\\text{KL 正则项}} = \\text{ELBO}
$$

直观解读：
- **重构项**：给定采样的 $z$，解码器还原 $x$ 的能力
- **KL 项**：让后验 $q(z|x)$ 不要偏离先验 $p(z) = \\mathcal{N}(0, I)$ 太远——这是正则化，保证 $z$ 空间的规整性

### 闭合形式 KL 散度

对于高斯先验和高斯后验：

$$
D_{KL}(\\mathcal{N}(\\mu, \\sigma^2) \\| \\mathcal{N}(0, 1))
= -\\frac{1}{2} \\sum_{j=1}^{d} (1 + \\log \\sigma_j^2 - \\mu_j^2 - \\sigma_j^2)
$$

### 重参数化技巧

$$
z = \\mu_\\phi(x) + \\sigma_\\phi(x) \\odot \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, I)
$$

梯度可以流过 $\\mu$ 和 $\\sigma$，因为随机性被隔离在 $\\epsilon$ 中。这就是 VAE 能端到端训练的根本原因。

### β-VAE

$$
\\mathcal{L}_{\\beta} = \\mathbb{E}_{q}[\\log p(x|z)] - \\beta \\cdot D_{KL}(q(z|x) \\| p(z))
$$

$\\beta > 1$ 强制更强的解耦（disentanglement），各维度对应独立的生成因子。

---

## 可视化展示

### VAE 概率图模型

\`\`\`mermaid
graph LR
    X["x (观测)"] --> ENC["编码器 q_ϕ(z|x)"]
    ENC --> MU["μ"]
    ENC --> SIG["σ"]
    EPS["ε ~ N(0,I)"] --> SAMP["z = μ + σ⊙ε"]
    MU --> SAMP
    SIG --> SAMP
    SAMP --> DEC["解码器 p_θ(x|z)"]
    DEC --> XHAT["x̂ (重构)"]
    SAMP -.->|KL 正则| PRIOR["p(z)=N(0,I)"]
\`\`\`

### KL 散度 vs 重构 Loss 的权衡

\`\`\`echarts
return {
  title: { text: 'β-VAE: β 对 KL 与重构的影响', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['β=0.1', 'β=0.5', 'β=1(VAE)', 'β=2', 'β=5', 'β=10'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '归一化得分' },
  legend: { data: ['KL 散度', '重构质量', '解耦性'] },
  series: [
    { name: 'KL 散度', type: 'bar', data: [0.05, 0.15, 0.3, 0.5, 0.75, 0.92], itemStyle: { color: '#2980b9' } },
    { name: '重构质量', type: 'bar', data: [0.95, 0.90, 0.82, 0.65, 0.40, 0.20], itemStyle: { color: '#16a085' } },
    { name: '解耦性', type: 'bar', data: [0.10, 0.25, 0.45, 0.65, 0.80, 0.90], itemStyle: { color: '#d35400' } }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

$\\beta$ 越大解耦越好但重构越差，这是"表示质量 vs 重构质量"的权衡。

---

## 核心代码实现

### PyTorch VAE

\`\`\`python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.enc_fc1(x))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.dec_fc1(z))
        return torch.sigmoid(self.dec_out(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x):
        x_hat, mu, logvar = self.forward(x)
        recon = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon + kl) / x.size(0)
\`\`\`
`;export{n as default};
