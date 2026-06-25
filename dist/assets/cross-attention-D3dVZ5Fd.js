const n=`# 交叉注意力 (Cross-Attention)

## 核心思想

与自注意力（Q, K, V 来自同一序列）不同，交叉注意力的 **Q 来自一个序列，K 和 V 来自另一个序列**。它使一个序列可以"查询"另一个序列中的信息。

## 数学定义

$$\\text{CrossAttention}(\\mathbf{Q}_x, \\mathbf{K}_y, \\mathbf{V}_y) = \\text{softmax}\\left(\\frac{\\mathbf{Q}_x \\mathbf{K}_y^T}{\\sqrt{d_k}}\\right) \\mathbf{V}_y$$

其中：
- $\\mathbf{Q}_x$ 来自序列 $x$（查询方）
- $\\mathbf{K}_y, \\mathbf{V}_y$ 来自序列 $y$（被查询方）

## 典型应用

### 1. Transformer 解码器中的 Encoder-Decoder Attention

解码器的每一层在自注意力后，用解码器的输出作为 Q，编码器的输出作为 K 和 V。这使得解码时能"看"整个输入序列。

### 2. 多模态模型 (如 Stable Diffusion)

\`\`\`python
# 文本特征作为条件，指导图像生成
Q = image_features       # 图像查询"需要什么文本信息"
K = text_features        # 文本"提供什么语义"
V = text_features        # 文本的实际贡献
output = CrossAttention(Q, K, V)
\`\`\`

### 3. CLIP / 图文检索

图像和文本互为查询方和键值方进行交互。

## PyTorch 实现

\`\`\`python
import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, context):
        """
        x: 查询方 [B, N_x, D]
        context: 被查询方 [B, N_c, D]
        """
        B, N_x, D = x.shape
        _, N_c, _ = context.shape

        Q = self.W_q(x).view(B, N_x, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(context).view(B, N_c, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(context).view(B, N_c, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N_x, D)
        return self.W_o(out)
\`\`\`

## Self-Attention vs Cross-Attention

| | Self-Attention | Cross-Attention |
|------|---------------|-----------------|
| Q, K, V 来源 | 同一序列 | Q 和 K,V 不同源 |
| 功能 | 序列内部建模 | 序列间信息融合 |
| 典型位置 | Encoder 和 Decoder 首层 | Decoder 中间层 |
| 注意力矩阵形状 | $n \\times n$ | $n_x \\times n_y$ |
`;export{n as default};
