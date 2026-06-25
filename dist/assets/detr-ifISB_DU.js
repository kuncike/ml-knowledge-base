const n=`# DETR (DEtection TRansformer)

## 核心思想

DETR 将目标检测完全建模为一个**集合预测 (Set Prediction)** 问题，使用 Transformer 的编码器-解码器架构直接输出 $N$ 个物体的类别和边界框，无需 Anchor、NMS 等手工组件。

## 架构

\`\`\`
Image → CNN Backbone → Feature Map → + Position Encoding
→ Transformer Encoder → Transformer Decoder
→ FFN (每个 object query) → (class, bbox) × N
\`\`\`

### Object Queries

$N$ 个可学习的向量（通常 $N=100$），每个 query 负责"询问"图像中是否存在某一类物体。经过 Decoder 后每个 query 输出一个预测。

## 二分图匹配 (Bipartite Matching)

### 核心问题

预测的 $N$ 个框与真实的 $M$ 个框如何配对？（$N > M$，且预测无特定顺序）

### 匈牙利算法

使用 Hungarian 算法找到最优二分图匹配，最小化总成本：

$$\\hat{\\sigma} = \\arg\\min_{\\sigma} \\sum_{i=1}^{N} \\mathcal{L}_{match}(y_i, \\hat{y}_{\\sigma(i)})$$

$$\\mathcal{L}_{match} = -\\mathbb{1}_{\\{c_i \\neq \\varnothing\\}} \\cdot \\hat{p}_{\\sigma(i)}(c_i) + \\mathbb{1}_{\\{c_i \\neq \\varnothing\\}} \\cdot \\mathcal{L}_{box}(b_i, \\hat{b}_{\\sigma(i)})$$

### 损失函数

找到最优匹配后，计算损失：

$$L = \\sum_{i=1}^{N} \\left[ -\\log \\hat{p}_{\\hat{\\sigma}(i)}(c_i) + \\mathbb{1}_{\\{c_i \\neq \\varnothing\\}} \\cdot \\mathcal{L}_{box}(b_i, \\hat{b}_{\\hat{\\sigma}(i)}) \\right]$$

边界框损失使用 L1 + GIoU 的组合。

## DETR 的局限性

- 训练收敛慢（需要 ~500 epoch vs Faster R-CNN 的 ~12 epoch）
- 小物体检测效果差（Transformer 的全局注意力对小分辨率特征不友好）

## Deformable DETR

用**可变形注意力 (Deformable Attention)** 替代全局注意力：

- 每个 query 只关注特征图中少数关键位置
- 收敛速度大幅提升（~50 epoch）
- 小物体检测改善

## 对比

| | DETR | Faster R-CNN | YOLO |
|------|------|-------------|------|
| NMS | 不需要 | 需要 | 需要 (v8 前) |
| Anchor | 不需要 | 需要 | v8 后不需要 |
| 架构 | Transformer | CNN | CNN |
| 收敛 | 慢 | 快 | 快 |
| 优雅度 | 极高 | 中等 | 中等 |
`;export{n as default};
