const n=`# DCGAN / WGAN

## DCGAN (Deep Convolutional GAN)

### 核心思想

将 CNN 架构引入 GAN，用卷积和转置卷积替代全连接层，大幅提升生成图像的质量和训练的稳定性。

### 架构设计准则

1. 用 **stride 卷积替代池化**（下采样）
2. 用 **转置卷积**上采样
3. 生成器用 **ReLU**（输出层 Tanh），判别器用 **LeakyReLU**
4. 全部使用 **Batch Normalization**（除了 G 输出层和 D 输入层）

### 生成器架构

\`\`\`
z (100) → FC → reshape[4×4×1024]
→ TransConv(512, 5×5, S=2) → BN + ReLU
→ TransConv(256, 5×5, S=2) → BN + ReLU
→ TransConv(128, 5×5, S=2) → BN + ReLU
→ TransConv(3, 5×5, S=2) → Tanh → [64×64×3]
\`\`\`

---

## WGAN (Wasserstein GAN)

### 动机：GAN 训练不稳定的根源

原始 GAN 的 JS 散度在两个分布不重叠时恒为 $\\log 2$（梯度消失）。WGAN 用 **Wasserstein 距离**（推土机距离）替代 JS 散度。

### Wasserstein 距离

$$W(P_r, P_g) = \\inf_{\\gamma \\in \\Pi(P_r, P_g)} \\mathbb{E}_{(x,y)\\sim \\gamma}[\\|x - y\\|]$$

直观理解：将分布 $P_g$ "搬运"成分布 $P_r$ 的最小成本。

### Kantorovich-Rubinstein 对偶

$$W(P_r, P_g) = \\sup_{\\|f\\|_L \\leq 1} \\mathbb{E}_{x\\sim P_r}[f(x)] - \\mathbb{E}_{\\tilde{x}\\sim P_g}[f(\\tilde{x})]$$

其中 $f$ 是 1-Lipschitz 函数，由**判别器**（称为 Critic）学习。

### WGAN 损失

**Critic**：

$$L_c = \\mathbb{E}_{\\tilde{x}\\sim P_g}[D(\\tilde{x})] - \\mathbb{E}_{x\\sim P_r}[D(x)]$$

**Generator**：

$$L_G = -\\mathbb{E}_{\\tilde{x}\\sim P_g}[D(\\tilde{x})]$$

### Lipchitz 约束实现

**WGAN** (原始)：权重裁剪到 $[-c, c]$

**WGAN-GP** (梯度惩罚)：

$$L_{GP} = \\lambda \\cdot \\mathbb{E}_{\\hat{x}} \\left[ (\\|\\nabla_{\\hat{x}} D(\\hat{x})\\|_2 - 1)^2 \\right]$$

在真实样本和生成样本的连线上采样 $\\hat{x}$，约束梯度范数接近 1。

## DCGAN vs WGAN(-GP)

| | DCGAN | WGAN-GP |
|------|-------|---------|
| 损失含义 | 真假概率 | Wasserstein 距离 |
| 训练稳定性 | 需仔细调参 | 稳定 |
| Mode Collapse | 常见 | 较少 |
| 收敛指示 | 不明确 | Loss 越小越好 |
`;export{n as default};
