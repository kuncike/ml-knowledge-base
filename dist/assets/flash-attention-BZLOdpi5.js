const n=`# FlashAttention

## 核心思想

FlashAttention 是一种 **IO-Aware**（感知输入输出）的精确注意力算法。它通过分块（Tiling）和重计算（Recomputation）技术，大幅减少 GPU HBM (High Bandwidth Memory) 与 SRAM 之间的数据读写，在不损失精度的情况下实现 2-4 倍的加速和 10-20 倍的内存节省。

## 问题背景

标准注意力实现的瓶颈：

$$\\mathbf{S} = \\mathbf{Q}\\mathbf{K}^T \\in \\mathbb{R}^{n \\times n}$$

$$\\mathbf{P} = \\text{softmax}(\\mathbf{S})$$

$$\\mathbf{O} = \\mathbf{P}\\mathbf{V}$$

$n \\times n$ 的注意力矩阵需要从 HBM 读写，当 $n$ 很大时（如 64K），这成为内存带宽瓶颈，而非计算瓶颈。

## 核心技巧

### 1. 分块计算 (Tiling)

将 Q, K, V 切分成小块，在 SRAM 中完成局部计算，避免将完整的 $n \\times n$ 矩阵写入 HBM。

### 2. Online Softmax

通过维护 running max 和 running sum，逐步更新 Softmax：

\`\`\`python
# 伪代码：Online Softmax
m_i = max(m_{i-1}, row_max_i)
sum_i = sum_{i-1} * exp(m_{i-1} - m_i) + row_sum_i * exp(row_max_i - m_i)
\`\`\`

### 3. 重计算 (Recomputation)

反向传播时不保存中间注意力矩阵，而是重新计算。用计算换内存。

## FlashAttention-2 改进

- 减少非矩阵乘法运算
- 更好地并行化（沿序列长度维度）
- 优化 warp 级别的调度
- 达到理论峰值的约 70%

## FlashAttention-3

针对 H100 GPU 的异步特性（WGMMA 指令、TMA 加速器）进一步优化。

## 使用

\`\`\`python
# PyTorch 2.0+ 内置支持
import torch.nn.functional as F

# 自动使用 FlashAttention (如果可用)
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# 或手动指定
from flash_attn import flash_attn_func
out = flash_attn_func(Q, K, V, causal=True)
\`\`\`

## 复杂度对比

| 方法 | 时间复杂度 | 空间复杂度 | 精确 |
|------|-----------|-----------|------|
| 标准 Attention | $O(n^2 d)$ | $O(n^2)$ | 是 |
| FlashAttention | $O(n^2 d)$ | $O(n)$ | 是 |
| Sparse Attention | $O(n \\sqrt{n} d)$ | $O(n \\sqrt{n})$ | 近似 |

FlashAttention 是目前长序列 LLM 推理和训练的标配。
`;export{n as default};
