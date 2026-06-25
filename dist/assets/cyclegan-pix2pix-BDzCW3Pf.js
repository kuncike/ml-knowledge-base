const n=`# CycleGAN / Pix2Pix

## Pix2Pix — 有监督的图像翻译

### 核心思想

用**成对数据**训练图像翻译（如：语义图 → 真实照片）。使用条件 GAN + L1 Loss。

### 损失函数

$$G^* = \\arg\\min_G \\max_D \\mathcal{L}_{cGAN}(G, D) + \\lambda \\mathcal{L}_{L1}(G)$$

#### 条件 GAN Loss

$$\\mathcal{L}_{cGAN}(G, D) = \\mathbb{E}_{x,y}[\\log D(x,y)] + \\mathbb{E}_{x,z}[\\log(1 - D(x, G(x,z)))]$$

#### L1 Loss

$$\\mathcal{L}_{L1}(G) = \\mathbb{E}_{x,y,z}[\\|y - G(x, z)\\|_1]$$

- L1 比 L2 产生更清晰的图像（对模糊惩罚更大）
- $\\lambda$ 通常取 100

### U-Net Generator

编码器-解码器 + 跳跃连接（保留高频细节）。

### PatchGAN Discriminator

不判断整图真假，而是判断 $N \\times N$ 的 patch 真假（局部纹理 vs 全局结构）。

---

## CycleGAN — 无监督的图像翻译

### 核心思想

不需要成对数据！只需两类图片（如 马 和 斑马）的集合，通过**循环一致性**学习映射。

### 核心创新：循环一致性损失

$$G: X \\to Y, \\quad F: Y \\to X$$

$$\\mathcal{L}_{cyc}(G, F) = \\mathbb{E}_{x\\sim X}[\\|F(G(x)) - x\\|_1] + \\mathbb{E}_{y\\sim Y}[\\|G(F(y)) - y\\|_1]$$

翻译过去再翻译回来，应得到原始图像。

### 总损失

$$\\mathcal{L} = \\mathcal{L}_{GAN}(G, D_Y, X, Y) + \\mathcal{L}_{GAN}(F, D_X, Y, X) + \\lambda \\mathcal{L}_{cyc}(G, F)$$

两个方向的 GAN 损失 + 循环一致性损失。

### 身份保持损失（可选）

$$\\mathcal{L}_{identity}(G, F) = \\mathbb{E}_{y\\sim Y}[\\|G(y) - y\\|_1] + \\mathbb{E}_{x\\sim X}[\\|F(x) - x\\|_1]$$

把 Y 域图片输入 G，应该保持不变（因为已经是 Y 域）。

## Pix2Pix vs CycleGAN

| | Pix2Pix | CycleGAN |
|------|---------|----------|
| 训练数据 | 成对 | 不成对 |
| 输入约束 | 强配对 | 弱（两个集合） |
| 应用场景 | Labels→照片，边缘→图 | 风格迁移，季节转换 |
| 训练难度 | 较低 | 中等 |
`;export{n as default};
