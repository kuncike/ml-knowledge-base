const n=`# Stable Diffusion / Midjourney / DALL-E

## Latent Diffusion Model (Stable Diffusion)

### 核心思想

在高分辨率图像空间做扩散计算量巨大。Stable Diffusion 的核心创新是：**在 VAE 压缩的潜在空间中做扩散**，大幅降低计算量。

### 架构

\`\`\`
Text → CLIP Text Encoder → Cross-Attention Conditioning
                              ↓
Noise(z_T) → U-Net (Denoising) → z_0 → VAE Decoder → Image
\`\`\`

### 三个核心组件

#### 1. VAE (变分自编码器)

- **编码器**：将 512×512×3 图像压缩到 64×64×4 隐变量（~48× 压缩）
- **解码器**：从隐变量重建图像

#### 2. U-Net (去噪网络)

在隐空间中去噪，每个分辨率层通过 Cross-Attention 注入文本条件：

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d}}\\right)V$$

其中 $Q$ 来自 U-Net 特征，$K$ 和 $V$ 来自文本编码器。

#### 3. CLIP Text Encoder

将文本 prompt 编码为语义向量。

### 文本引导

#### Classifier-Free Guidance (CFG)

$$ \\hat{\\epsilon}_\\theta(x_t, c) = \\epsilon_\\theta(x_t, \\varnothing) + w \\cdot (\\epsilon_\\theta(x_t, c) - \\epsilon_\\theta(x_t, \\varnothing)) $$

- $w=1$：标准条件生成
- $w > 1$：更遵循文本提示（通常 $w=7.5$）
- $w$ 过大：图像过饱和、不自然

### SD 版本演进

| 版本 | 分辨率 | 关键改进 |
|------|--------|----------|
| SD 1.5 | 512×512 | 经典版本，社区生态最丰富 |
| SD 2.1 | 768×768 | 更强文本编码器 |
| SDXL | 1024×1024 | 更大的 U-Net + 双文本编码器 |
| SD 3 | 1024+ | MMDiT 架构，多模态扩散 Transformer |

---

## DALL-E 2/3

### DALL-E 2

- **CLIP + Diffusion**：先训练 CLIP 学习图文对齐，再用 Diffusion 从 CLIP 图像嵌入生成图像
- **unCLIP 解码器**：从 CLIP 嵌入反向生成图像

### DALL-E 3

- 用合成标注数据训练（图像 → 详细的文本描述 → 模型学习）
- 极其出色的指令遵循能力
- 集成 ChatGPT 进行 prompt 优化

---

## Midjourney

- 闭源商业产品
- 艺术质量和审美表现力领先
- 不断迭代（v1 → v6 质量持续飞跃）

## 对比

| 特性 | Stable Diffusion | DALL-E 3 | Midjourney |
|------|-----------------|----------|------------|
| 开源 | 是 | 否 | 否 |
| 本地运行 | 可 | 否 | 否 |
| 提示遵循 | 中 | 极高 | 高 |
| 艺术品质 | 中（取决于模型）| 高 | 极高 |
`;export{n as default};
