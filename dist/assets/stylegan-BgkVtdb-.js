const n=`# StyleGAN (v1-v3)

## StyleGAN v1 — 基于风格的生成器

### 核心创新

借鉴**风格迁移**的思想，将隐向量通过映射网络转换为**风格向量**，在不同分辨率层级注入到生成过程中。

### 架构

\`\`\`
z → Mapping Network (8×FC) → w (中间隐空间)
                                 ↓
      w → Affine Transform → (γ, β) — 控制 AdaIN
                                      ↓
常量输入 (4×4×512) → Synthesis Network (逐步上采样)
\`\`\`

### AdaIN (自适应实例归一化)

$$\\text{AdaIN}(x_i, y) = y_{s,i} \\cdot \\frac{x_i - \\mu(x_i)}{\\sigma(x_i)} + y_{b,i}$$

- $y_s$（风格缩放）和 $y_b$（风格偏置）从风格向量 $w$ 计算
- 改变统计量即改变图像的"风格"

### 噪声注入

每层加入高斯噪声 → 控制细节变化（如头发纹理、皮肤毛孔）

---

## StyleGAN v2

### 关键修复：水滴伪影

问题：AdaIN 导致生成图像出现水滴状伪影。

**解决**：用**权重解调 (Weight Demodulation)** 替代 AdaIN：

$$w'_{ijk} = s_i \\cdot w_{ijk}, \\quad w''_{ijk} = \\frac{w'_{ijk}}{\\sqrt{\\sum_{i,k} w_{ijk}'^2 + \\epsilon}}$$

### 其他改进

- 路径长度正则化：使隐空间的线性插值产生平滑的视觉变化
- 无渐进式增长：直接从最终分辨率训练

---

## StyleGAN v3 — 等变性

### 核心问题

图像平移时，纹理细节不随平移变化（纹理粘连现象）。

### 解决方案

强制生成器满足**平移等变性**：如果特征图发生平移，输出图像也应平移同样的量 → 自然的动画效果。

---

## 关键概念

| 概念 | 说明 |
|------|------|
| $W$ 空间 | 映射网络输出的中间隐空间 |
| $W+$ 空间 | 每层不同风格向量的扩展空间 |
| 样式混合 | 不同层用不同的 $w$ 向量 |
| 截断技巧 | $\\bar{w} + \\psi \\cdot (w - \\bar{w})$，控制多样性 |

## 使用

\`\`\`python
# StyleGAN3 生成
import dnnlib, legacy
import torch

with dnnlib.util.open_url('stylegan3-t.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema']
z = torch.randn([1, G.z_dim])
img = G(z, c=None)  # [1, 3, 1024, 1024]
\`\`\`
`;export{n as default};
