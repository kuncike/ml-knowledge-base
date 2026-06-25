const n=`# DDPM / DDIM (扩散模型)

## DDPM (Denoising Diffusion Probabilistic Models)

### 核心思想

扩散模型包含两个过程：
1. **前向过程**：逐步向数据添加高斯噪声，直到变成纯噪声
2. **反向过程**：学习从噪声中逐步去噪，恢复原始数据

### 前向过程 (Diffusion)

$$q(x_t \\mid x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1 - \\beta_t} x_{t-1}, \\beta_t \\mathbf{I})$$

$\\beta_t$ 是噪声调度（noise schedule），通常从 $\\beta_1=10^{-4}$ 线性增加到 $\\beta_T=0.02$。

**重参数化**：任意时间步可以一步采样：

$$x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon$$

其中 $\\alpha_t = 1 - \\beta_t$，$\\bar{\\alpha}_t = \\prod_{s=1}^{t} \\alpha_s$。

### 反向过程 (Denoising)

$$p_\\theta(x_{t-1} \\mid x_t) = \\mathcal{N}(x_{t-1}; \\mu_\\theta(x_t, t), \\Sigma_\\theta(x_t, t))$$

### 训练目标

简化后等价于预测添加的噪声：

$$L_{simple} = \\mathbb{E}_{t, x_0, \\epsilon} \\left[ \\|\\epsilon - \\epsilon_\\theta(x_t, t)\\|^2 \\right]$$

其中 $x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon$。

### U-Net 预测器

$\\epsilon_\\theta$ 是 U-Net，输入 $x_t$ 和时间步 $t$（通过位置编码注入），输出预测的噪声。

---

## DDIM (Denoising Diffusion Implicit Models)

### 动机

DDPM 的反向过程是随机的且非常慢（$T=1000$ 步）。

### 核心创新：非马尔可夫前向过程

DDIM 重新定义了前向过程，使其不再是马尔可夫的，从而反向过程可以**确定性**且**大幅加速**。

### 确定性采样

$$x_{t-1} = \\sqrt{\\bar{\\alpha}_{t-1}} \\cdot \\hat{x}_0 + \\sqrt{1 - \\bar{\\alpha}_{t-1}} \\cdot \\epsilon_\\theta(x_t, t)$$

$$\\hat{x}_0 = \\frac{x_t - \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon_\\theta(x_t, t)}{\\sqrt{\\bar{\\alpha}_t}}$$

- **$\\eta = 0$**：DDIM（确定性）→ 极少步数（~50 步）即可生成好图像
- **$\\eta = 1$**：退化为 DDPM 的随机采样

## DDPM vs DDIM

| | DDPM | DDIM |
|------|------|------|
| 采样步数 | 1000 | 50-100 |
| 采样方式 | 随机 | 确定/随机 |
| 隐空间插值 | 不可控 | 可控 |
| 关键优势 | 质量高 | 速度快 |

DDIM 的确定性采样使得**对隐变量 $x_T$ 做插值可以产生语义平滑的过渡**，这对 Stable Diffusion 等应用至关重要。
`;export{n as default};
