const n=`# ViT / Swin Transformer

## Vision Transformer (ViT)

### 核心思想

将图像切成固定大小的 patch，像 NLP 中的词一样输入 Transformer。

### 架构

\`\`\`
Image → Split into patches → Linear Projection → + Position Embedding
→ [CLS] Token + Patch Tokens → Transformer Encoder × L
→ [CLS] Token → MLP Head → Class
\`\`\`

### 关键公式

图像 $\\mathbf{x} \\in \\mathbb{R}^{H \\times W \\times C}$，patch 大小 $P \\times P$，生成 $N = HW/P^2$ 个 patch：

$$\\mathbf{z}_0 = [\\mathbf{x}_{class}; \\mathbf{x}_p^1 \\mathbf{E}; \\ldots; \\mathbf{x}_p^N \\mathbf{E}] + \\mathbf{E}_{pos}$$

然后经过 $L$ 层 Transformer：

$$\\mathbf{z}'_\\ell = \\text{MSA}(\\text{LN}(\\mathbf{z}_{\\ell-1})) + \\mathbf{z}_{\\ell-1}$$
$$\\mathbf{z}_\\ell = \\text{MLP}(\\text{LN}(\\mathbf{z}'_\\ell)) + \\mathbf{z}'_\\ell$$

### 数据需求

ViT 缺少 CNN 的归纳偏置（局部性、平移等变性），因此需要**海量数据**（JFT-300M）才能发挥优势。在小数据上不如 CNN。

---

## Swin Transformer

### 核心创新：分层结构 + 移动窗口

1. **分层特征图**：像 CNN 一样逐步下采样（4× → 8× → 16× → 32×）
2. **窗口内自注意力**：每个窗口内独立计算，降低复杂度
3. **移动窗口 (Shifted Window)**：相邻窗口间信息交互

### 移动窗口机制

\`\`\`
第 l 层：标准窗口划分   ┌──┬──┐
                        ├──┼──┤
                        └──┴──┘

第 l+1 层：窗口移动       ┌──┬──┐
  (shift = window_size/2) ├──┼──┤
                          └──┴──┘
\`\`\`

### 复杂度对比

| 方法 | 复杂度 |
|------|--------|
| 全局自注意力 | $O(N^2 d)$ |
| Swin (窗口) | $O(M^2 \\cdot N \\cdot d)$ = $O(NMd)$ |

其中 $M$ 是窗口大小（通常 7），$N$ 是 patch 数。当 $N$ 很大时优势明显。

### Swin 变体

| 模型 | 参数量 | 计算量 | ImageNet Acc |
|------|--------|--------|-------------|
| Swin-T | 29M | 4.5G | 81.3% |
| Swin-S | 50M | 8.7G | 83.0% |
| Swin-B | 88M | 15.4G | 83.5% |
| Swin-L | 197M | 34.5G | 83.8% |

### Self-Supervised 扩展

Swin Transformer 结合 SimMIM (Masked Image Modeling)，利用自监督预训练取得更好的迁移学习效果。
`;export{n as default};
