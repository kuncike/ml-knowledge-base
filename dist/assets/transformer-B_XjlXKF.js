const n=`# Transformer

## 核心思想

2017 年 "Attention Is All You Need" 论文提出的 Transformer 架构，完全基于注意力机制，摒弃了 RNN 的递归结构，开启了 NLP 和整个深度学习的新纪元。

## 架构总览

\`\`\`
Encoder:                          Decoder:
  Input → Embedding + PE           Output (shifted) → Embedding + PE
    ↓                                ↓
  Multi-Head Self-Attention       Masked Multi-Head Self-Attention
    ↓ Add & Norm                    ↓ Add & Norm
  Feed-Forward Network            Cross-Attention (with Encoder output)
    ↓ Add & Norm                    ↓ Add & Norm
  (× N)                           Feed-Forward Network
                                     ↓ Add & Norm
                                   (× N)
                                     ↓
                                   Linear + Softmax
\`\`\`

## 编码器 (Encoder)

### 自注意力子层

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

在 Encoder 中 $Q = K = V$（来自前一层输出）。

### 前馈网络子层 (FFN)

$$\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2$$

（原始论文用 ReLU，现代用 GELU）

### 残差连接 + 层归一化

每个子层后：

$$\\text{Output} = \\text{LayerNorm}(x + \\text{Sublayer}(x))$$

**Post-Norm**（原始）：先加子层再归一化
**Pre-Norm**（现代）：先归一化再加子层 → 训练更稳定（GPT/LLaMA 选择）

## 解码器 (Decoder)

### 因果掩膜 (Causal Masking)

确保位置 $i$ 只能关注 $< i$ 的位置（自回归生成）：

\`\`\`python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
\`\`\`

### 交叉注意力

解码器从编码器输出获取输入序列信息。Q 来自解码器，K、V 来自编码器。

## 位置编码

$$\\text{PE}(pos, 2i) = \\sin(pos / 10000^{2i/d_{model}})$$
$$\\text{PE}(pos, 2i+1) = \\cos(pos / 10000^{2i/d_{model}})$$

## 超参数

| 参数 | Base | Big |
|------|------|-----|
| $d_{model}$ | 512 | 1024 |
| $h$ (头数) | 8 | 16 |
| $d_{ff}$ | 2048 | 4096 |
| $N$ (层数) | 6 | 6 |
| Dropout | 0.1 | 0.3 (BIG)|

## 为什么 Transformer 替代了 RNN？

| | RNN/LSTM | Transformer |
|------|----------|-------------|
| 并行度 | 串行（$O(n)$） | 并行（$O(1)$） |
| 长距离依赖 | 间接（需逐步传递） | 直接（任意两位置 O(1)） |
| 训练速度 | 慢 | 快 |

Transformer 是现代 AI（BERT, GPT, ViT, DALL-E, Sora）的统一架构基础。
`;export{n as default};
