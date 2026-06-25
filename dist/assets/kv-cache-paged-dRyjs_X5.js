const n=`# KV Cache / PagedAttention

## KV Cache

### 为什么需要 KV Cache？

自回归生成时，每个 token 都需要计算所有之前 token 的 Key 和 Value。如果没有 Cache，每个新 token 都会重新计算整个序列的 K、V → 计算量 $O(n^2)$。

### KV Cache 原理

将已计算的 K、V 缓存在显存中，每步只计算新 token 的 K、V：

\`\`\`python
# 第 t 步（有 Cache）
Q_t = W_Q @ x_t  # 只计算新 token 的 Q
K_t = W_K @ x_t  # 只计算新 token 的 K
V_t = W_V @ x_t  # 只计算新 token 的 V

# 拼接 Cache
K = concat(K_cache, K_t)  # [t, d_k]
V = concat(V_cache, V_t)  # [t, d_v]

# 只对 Q_t 和 K 计算注意力
attn = softmax(Q_t @ K^T / sqrt(d_k)) @ V  # O(t) 而非 O(t^2)
\`\`\`

### 内存分析

KV Cache 的内存占用：

$$\\text{Memory} = 2 \\times n_{layers} \\times n_{heads} \\times d_{head} \\times \\text{seq\\_len} \\times \\text{dtype\\_size} \\times \\text{batch\\_size}$$

例如 LLaMA-7B (32 层，32 头，$d_{head}=128$，FP16)：
- 序列长度 4096：~2GB
- 序列长度 32K：~16GB

KV Cache 是大 batch / 长序列推理的主要内存瓶颈！

### GQA (Grouped-Query Attention) 的作用

多个 Query 头共享同一组 K、V 头 → KV Cache 大小成倍减少。

---

## PagedAttention (vLLM)

### 核心问题

KV Cache 像操作系统中的内存一样面临**碎片化**问题：
- 预分配固定大小 → 浪费（多数请求较短）
- 动态分配 → 碎片化

### 解决方案

将 KV Cache 划分为固定大小的 **Page**（灵感来自操作系统的虚拟内存分页）：

\`\`\`
Physical KV Blocks:  [Block 0] [Block 1] [Block 2] [Block 3]
                          ↖         ↗
Request 1:    Block 0 → Block 1
Request 2:    Block 2 → Block 3
\`\`\`

- **Block Table**：逻辑 token 位置 → 物理 block 映射
- **Copy-on-Write**：Beam Search 中共享 prompt 的 KV Cache

### 优势

- 内存利用率接近最优（<4% 浪费 vs 传统 ~40%）
- 支持更大 batch size → 更高吞吐

## vLLM 吞吐量优势

| | 传统 | vLLM (PagedAttention) |
|------|------|----------------------|
| KV Cache 浪费 | ~40% | <4% |
| Batch Size | 受限 | 可根据内存动态扩展 |
| 吞吐量 | 基准 | 2-4× |
`;export{n as default};
