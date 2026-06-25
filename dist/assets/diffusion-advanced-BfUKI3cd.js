const n=`# Diffusion Advanced (SGM / DiT / PixArt-α)

## 核心思想

标准扩散模型在 U-Net 中完成去噪，但 U-Net 是 CNN 架构——能否用更现代的设计？DiT (Diffusion Transformer) 大胆地将 U-Net 全部替换为 **ViT blocks + adaLN 条件注入**，验证了 Transformer 在扩散任务上同样优秀。PixArt-α 进一步用 **T5 文本编码器** 替代 CLIP，并引入训练效率优化（仅在压缩后的隐空间训练）。

核心发现：扩散模型的质量瓶颈**不在扩散过程，而在去噪网络的表达能力**。

---

## 数学定义与原理解析

### Score-Based Generative Models (SGM)

扩散模型可以统一理解为估计数据分布的**得分函数（score function）**：

$$
s_\\theta(\\mathbf{x}) = \\nabla_{\\mathbf{x}} \\log p(\\mathbf{x})
$$

去噪过程（逆向 SDE）：

$$
d\\mathbf{x} = [\\mathbf{f}(\\mathbf{x}, t) - g(t)^2 s_\\theta(\\mathbf{x}, t)] dt + g(t) d\\bar{\\mathbf{w}}
$$

DDPM 的噪声预测 $\\epsilon_\\theta(\\mathbf{x}_t, t)$ 和得分函数的关系：

$$
s_\\theta(\\mathbf{x}_t, t) \\approx -\\frac{\\epsilon_\\theta(\\mathbf{x}_t, t)}{\\sigma_t}
$$

### DiT — 用 Transformer 替代 U-Net

DiT 将 Latent Diffusion 中的 U-Net 替换为 ViT 架构：

1. **Patchify**：将 latent $\\mathbf{z} \\in \\mathbb{R}^{C \\times H \\times W}$ 转为 token 序列
2. **DiT Block**：标准 ViT block + **adaptive LayerNorm (adaLN)** 注入时间步和条件

adaLN 将时间步嵌入 $t_{emb}$ 和条件 $c$ 输入 MLP 生成 scale/shift 参数：

$$
\\text{adaLN}(x) = \\gamma(t, c) \\cdot \\text{LayerNorm}(x) + \\beta(t, c)
$$

### PixArt-α — T5 文本编码器

PixArt-α 用 **T5-XXL**（编码器）替代 SD 的 CLIP 作为文本条件，发现 T5 的语义理解深度显著优于 CLIP。同时优化训练策略：

1. 先在低分辨率（256²）用 ImageNet 预训练
2. 再用高质量 1024² 数据微调
3. **压缩比 8×**（SD 的 2 倍），进一步减少计算

---

## 可视化展示

### DiT 架构

\`\`\`mermaid
graph TD
    Z["Latent z (C×H×W)"] --> PATCH["Patchify → N tokens"]
    PATCH --> PE["+ Position Embedding"]
    T_EMB["Timestep t"] --> MLP["MLP"] --> ADALN["adaLN → γ, β"]
    COND["Condition c"] --> MLP
    PE --> DIT1["DiT Block 1"] --> DIT2["DiT Block 2"] --> DIT_N["... → DiT Block N"]
    ADALN --> DIT1 & DIT2 & DIT_N
    DIT_N --> UNPATCH["Unpatchify"] --> NOISE["噪声预测 ϵ̂"]
\`\`\`

### U-Net vs DiT

\`\`\`echarts
return {
  title: { text: '扩散模型 backbone 对比 (ImageNet 256×256)', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['U-Net (ADM)', 'U-Net (LDM)', 'DiT-L', 'DiT-XL', 'PixArt-α'] },
  yAxis: { type: 'value', min: 2, max: 6, name: 'FID (↓ 越低越好)' },
  series: [{
    type: 'bar',
    data: [3.94, 3.60, 3.07, 2.27, 2.53],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — DiT Block with adaLN

\`\`\`python
import torch
import torch.nn as nn

class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim))

        # adaLN: MLP 从时间+条件生成 6 组 scale/shift 参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x, c):
        # c: 时间 + 条件的联合嵌入 [B, dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \\
            self.adaLN_modulation(c).chunk(6, dim=1)

        # 自注意力 + adaLN
        x_norm = self.norm1(x)
        x_mod = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # FFN + adaLN
        x_norm = self.norm2(x)
        x_mod = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod)
        return x


class DiT(nn.Module):
    def __init__(self, in_channels=4, dim=512, depth=12, num_heads=8):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, 2, stride=2)
        self.time_embed = nn.Sequential(
            nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads) for _ in range(depth)])
        self.final = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, in_channels * 4))

    def forward(self, z, t, cond=None):
        # z: [B, C, H, W]
        x = self.patch_embed(z).flatten(2).transpose(1, 2)  # [B, N, D]
        c = self.time_embed(timestep_embedding(t, 256))
        if cond is not None:
            c = c + cond
        for block in self.blocks:
            x = block(x, c)
        B, N, D = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).view(B, D, H, W)
        return self.final(x)  # 预测的噪声或速度
\`\`\`
`;export{n as default};
