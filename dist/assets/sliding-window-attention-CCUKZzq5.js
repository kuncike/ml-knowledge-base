const n=`# 滑动窗口注意力 (Sliding Window Attention)

## 核心思想

不是让每个 token 关注所有 token，而是限制每个 token 只关注其周围的 **固定大小窗口**。这是处理长序列的稀疏注意力方法之一，由 Longformer 和 Mistral 等模型采用。

## 数学定义

对位置 $i$，其注意力范围：

$$\\text{Attention}(i) = \\{j: \\max(0, i-w) \\leq j \\leq \\min(n-1, i+w)\\}$$

其中 $w$ 是窗口大小。

复杂度从 $O(n^2)$ 降到 $O(n \\cdot w)$，当 $w \\ll n$ 时，近乎线性。

## 模型使用

### Mistral / Mixtral

使用 $w = 4096$ 的滑动窗口注意力，使得可以在长序列上高效推理。配合 KV Cache，内存使用显著降低。

### Longformer

结合三种注意力模式：
1. **滑动窗口**：局部上下文
2. **全局注意力**：特定位置（如 [CLS]）关注全序列
3. **扩张滑动窗口**：空洞式的稀疏窗口，扩大感受野

## 金字塔形的感受野

堆叠多层滑动窗口注意力后，顶层 token 的感受野可以覆盖整条序列（类似于 CNN 的逐层扩大感受野）。

第 $l$ 层的感受野：

$$RF_l = 2lw$$

例如 $w=512, L=24$ 层，顶层感受野 $= 24 \\times 2 \\times 512 = 24576$。

## PyTorch 实现

\`\`\`python
import torch

def sliding_window_attention(Q, K, V, window_size):
    """
    Q, K, V: [batch, heads, seq_len, d_k]
    """
    B, H, N, D = Q.shape
    scores = Q @ K.transpose(-2, -1)  # [B, H, N, N]

    # 构建滑动窗口 mask
    mask = torch.ones(N, N, device=Q.device)
    for i in range(N):
        left = max(0, i - window_size)
        right = min(N, i + window_size + 1)
        mask[i, left:right] = 0
    mask = mask.bool()

    scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores / (D ** 0.5), dim=-1)
    return attn @ V
\`\`\`

## 与其他稀疏注意力对比

| 方法 | 复杂度 | 特点 |
|------|--------|------|
| 标准 | $O(n^2)$ | 全局感受野 |
| 滑动窗口 | $O(nw)$ | 局部 + 层叠全局感受野 |
| 空洞滑动窗口 | $O(nw)$ | 更大感受野 |
| 全局+局部 | $O(n(g+w))$ | 指定位置全局，其他局部 |

## 优缺点

- **优点**：线性复杂度，易于实现，适合流式处理
- **缺点**：需要多层堆叠才能获得全局感受野，某些长距离依赖可能丢失
`;export{n as default};
