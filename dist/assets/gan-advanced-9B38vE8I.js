const n=`# GAN Advanced (WGAN-GP / CGAN / InfoGAN / StarGAN / BigGAN)

## 核心思想

原始 GAN 训练不稳定——模式坍塌、梯度消失、难以收敛。一系列改进从不同角度修补：WGAN 用 Wasserstein 距离替代 JS 散度（平滑梯度），WGAN-GP 用梯度惩罚强制 1-Lipschitz 约束，CGAN 让生成内容可控（条件输入），BigGAN 证明 GAN 也能大规模训练并产出高质量图像。共同思路：**更好的分布距离度量 + 更稳定的训练技巧**。

---

## 数学定义与原理解析

### WGAN — Wasserstein 距离

JS 散度的问题：当真假分布不重叠时，梯度为 0。WGAN 用 Earth-Mover 距离替代：

$$
W(P_r, P_g) = \\sup_{\\|f\\|_L \\leq 1} \\mathbb{E}_{\\mathbf{x} \\sim P_r}[f(\\mathbf{x})] - \\mathbb{E}_{\\tilde{\\mathbf{x}} \\sim P_g}[f(\\tilde{\\mathbf{x}})]
$$

$f$ 是 1-Lipschitz 函数（Critic，不再是"判别器"）。Critic 的输出是真实值而非概率。WGAN 损失：

$$
\\min_G \\max_{\\|D\\|_L \\leq 1} \\mathbb{E}_{x \\sim P_r}[D(x)] - \\mathbb{E}_{z \\sim P_z}[D(G(z))]
$$

### WGAN-GP — 梯度惩罚

WGAN 用权重裁剪实现 Lipschitz 约束，但会导致训练不稳定。WGAN-GP 用**梯度惩罚**替代：

$$
\\mathcal{L}_{GP} = \\lambda \\cdot \\mathbb{E}_{\\hat{\\mathbf{x}} \\sim P_{\\hat{x}}} \\left[(\\|\\nabla_{\\hat{\\mathbf{x}}} D(\\hat{\\mathbf{x}})\\|_2 - 1)^2\\right]
$$

其中 $\\hat{\\mathbf{x}} = \\epsilon \\mathbf{x} + (1-\\epsilon) G(\\mathbf{z})$ 是在真假样本连线上的采样，$\\lambda = 10$。

直觉：强制 Critic 在真-假连线上梯度范数接近 1，平滑地引导 Generator。

### InfoGAN — 解耦表示学习

在 GAN 的目标中加入互信息项，让一部分隐含编码 $c$ 控制特定的生成属性：

$$
\\min_G \\max_D V(D, G) - \\lambda \\cdot I(c; G(z, c))
$$

$I(c; G(z,c))$ 是互信息——衡量编码 $c$ 和生成图像之间的关联强度。用变分下界近似 $I$：

$$
I(c; G(z, c)) \\geq \\mathbb{E}_{c \\sim P(c), x \\sim G(z,c)} [\\log Q(c|x)] + H(c)
$$

$Q(c|x)$ 是一个辅助网络（与 D 共享大部分层），从生成图像中推断编码 $c$。

### Conditional GAN (CGAN)

将条件信息（如类别标签）同时输入 Generator 和 Discriminator：

$$
\\min_G \\max_D \\mathbb{E}_{x \\sim P_r}[\\log D(x|y)] + \\mathbb{E}_{z \\sim P_z}[\\log(1 - D(G(z|y)|y))]
$$

### BigGAN — 大规模 GAN 训练

关键工程技巧：
- **大 Batch**（2048）提升稳定性
- **正交正则化**：$R_\\beta(W) = \\beta \\|W^T W - I\\|_F^2$ 防止权重矩阵各向异性
- **截断技巧**（Truncation Trick）：采样 $z$ 时用阈值截断（$\\|z\\| \\leq r$ 则重采样）
- **共享嵌入**：分类条件嵌入与 G 的 BN 层参数共享

---

## 可视化展示

### GAN 演进树

\`\`\`mermaid
graph TD
    GAN["GAN (2014)<br/>JS Divergence"] --> DCGAN["DCGAN<br/>卷积 GAN"]
    GAN --> CGAN["CGAN<br/>条件生成"]
    GAN --> WGAN["WGAN<br/>Wasserstein 距离"]
    WGAN --> WGAN_GP["WGAN-GP<br/>梯度惩罚"]
    CGAN --> InfoGAN["InfoGAN<br/>解耦表示"]
    WGAN_GP --> BigGAN["BigGAN<br/>大规模训练"]
    DCGAN --> StyleGAN["StyleGAN<br/>风格控制"]
\`\`\`

### GAN 训练稳定性

\`\`\`echarts
return {
  title: { text: '不同 GAN 的 Inception Score (CIFAR-10)', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['DCGAN', 'WGAN', 'WGAN-GP', 'BigGAN', 'StyleGAN2'] },
  yAxis: { type: 'value', min: 5, max: 11, name: 'Inception Score' },
  series: [{
    type: 'bar',
    data: [6.4, 6.7, 7.9, 9.2, 9.9],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — WGAN-GP 梯度惩罚

\`\`\`python
import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device):
    """WGAN-GP 梯度惩罚"""
    B = real.size(0)
    epsilon = torch.rand(B, 1, 1, 1, device=device)
    # 在真-假之间插值
    interpolates = epsilon * real + (1 - epsilon) * fake
    interpolates.requires_grad_(True)

    critic_interpolates = critic(interpolates)

    # 计算对插值图像的梯度
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(B, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    return gp


def wgan_gp_loss(G, D, real, z, lambda_gp=10):
    fake = G(z)
    d_real = D(real)
    d_fake = D(fake.detach())

    # Critic 损失: 最大化真-假差异
    d_loss = d_fake.mean() - d_real.mean()
    gp = gradient_penalty(D, real, fake.detach(), z.device)
    d_loss_total = d_loss + lambda_gp * gp

    # Generator 损失
    g_loss = -D(fake).mean()

    return d_loss_total, g_loss
\`\`\`

### PyTorch — Conditional GAN

\`\`\`python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_channels=3):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, 512, 4, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, img_channels, 4, 2, 1), nn.Tanh())

    def forward(self, z, labels):
        # z: [B, latent_dim], labels: [B]
        label_emb = self.label_emb(labels)
        x = torch.cat([z, label_emb], dim=1)  # [B, 2*latent_dim]
        return self.main(x.unsqueeze(-1).unsqueeze(-1))
\`\`\`
`;export{n as default};
