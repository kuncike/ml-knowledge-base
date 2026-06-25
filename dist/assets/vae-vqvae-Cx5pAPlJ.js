const n=`# VAE / VQ-VAE / 流模型

## VAE (变分自编码器)

### 核心思想

不是将输入映射到确定的隐向量，而是映射到隐空间的**概率分布**（均值和方差），实现连续平滑的隐空间。

### 网络结构

\`\`\`
Encoder: x → μ(x), σ(x) → Sample z ~ N(μ, σ²) → Decoder → x̂
\`\`\`

### 重参数化技巧

$$\\mathbf{z} = \\mu + \\sigma \\odot \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, \\mathbf{I})$$

将随机性从计算图中独立出来，使得梯度可以通过采样节点。

### ELBO 损失

$$\\mathcal{L} = \\mathbb{E}_{q(z|x)}[\\log p(x|z)] - D_{KL}(q(z|x) \\| p(z))$$

- 第一项：重建损失
- 第二项：KL 散度，推动后验逼近先验 $\\mathcal{N}(0, \\mathbf{I})$

### KL 散度闭式解

$$D_{KL} = -\\frac{1}{2} \\sum_{j=1}^{d} (1 + \\log \\sigma_j^2 - \\mu_j^2 - \\sigma_j^2)$$

### $\\beta$-VAE

$$\\mathcal{L} = \\mathbb{E}_{q(z|x)}[\\log p(x|z)] - \\beta D_{KL}(q(z|x) \\| p(z))$$

$\\beta > 1$ 鼓励解耦的（disentangled）表示学习。

---

## VQ-VAE (Vector Quantized VAE)

### 核心思想

将连续隐变量**离散化**：维护一个可学习的码本（Codebook），Encoder 输出被量化为最近的码本向量。

### 量化过程

$$\\mathbf{z}_q = \\mathbf{e}_k, \\quad k = \\arg\\min_j \\|\\mathbf{z}_e(\\mathbf{x}) - \\mathbf{e}_j\\|_2$$

### 损失函数

$$\\mathcal{L} = \\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|^2 + \\|\\text{sg}[\\mathbf{z}_e] - \\mathbf{e}\\|^2 + \\beta \\|\\mathbf{z}_e - \\text{sg}[\\mathbf{e}]\\|^2$$

- 重建损失
- 码本损失（更新码本向量）
- 承诺损失（更新 Encoder，约束不偏离码本太远）

sg[·] 是 stop-gradient 操作。

### VQ-VAE-2

层次化 VQ-VAE：先学习全局结构（顶层码），再学习局部细节（底层码）。配合 PixelCNN 在离散隐空间做自回归生成。

---

## 流模型 (Normalizing Flows)

### 核心思想

用一系列**可逆变换**将简单分布（如高斯）逐步变换为复杂的数据分布：

$$\\mathbf{z}_K = f_K \\circ \\cdots \\circ f_2 \\circ f_1(\\mathbf{z}_0), \\quad \\mathbf{z}_0 \\sim p_0$$

### 变量替换公式

$$\\log p(\\mathbf{x}) = \\log p_0(\\mathbf{z}_0) - \\sum_{i=1}^{K} \\log \\left| \\det \\frac{\\partial f_i}{\\partial \\mathbf{z}_{i-1}} \\right|$$

### 关键约束

每个 $f_i$ 必须可逆且雅可比行列式易于计算。

## 对比

| | VAE | VQ-VAE | Flow | GAN | Diffusion |
|------|-----|--------|------|-----|-----------|
| 似然 | 近似 | 近似 | 精确 | N/A | 近似 |
| 隐空间 | 连续 | 离散 | 连续 | 连续 | 连续 |
| 生成质量 | 中 | 中 | 中 | 高 | 最高 |
| 推理速度 | 快 | 快 | 快 | 快 | 慢 |
`;export{n as default};
