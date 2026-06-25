const n=`# Sentence-BERT / ColBERT

## Sentence-BERT (SBERT)

### 核心问题

BERT 用于语义相似度搜索时需要将两个句子拼接送入模型 → 寻找最相似的句子对需要 $O(n^2)$ 次推理（不可行）。

### 解决方案

SBERT 使用**孪生网络**架构，独立编码每个句子为固定大小的向量，然后用余弦相似度比较：

\`\`\`python
# 两个句子独立编码
emb_A = pool(bert(A))  # [768]
emb_B = pool(bert(B))  # [768]
similarity = cosine(emb_A, emb_B)
\`\`\`

### 训练目标

#### 分类目标函数

$$o = \\text{softmax}(W_t \\cdot |u-v|; u*v)$$

将两个句子的向量之差和逐元素乘积拼接后分类。

#### 回归目标函数

$$\\text{MSE}(\\text{cosine}(u, v), \\text{label})$$

#### 三元组目标函数

$$L = \\max(0, \\|s_a - s_p\\| - \\|s_a - s_n\\| + \\epsilon)$$

### 池化策略

| 策略 | 描述 |
|------|------|
| CLS | 使用 [CLS] token 的输出 |
| MEAN | 所有 token 输出的平均值（**推荐**） |
| MAX | 所有 token 输出的按维度最大值 |

---

## ColBERT (Contextualized Late Interaction)

### 核心思想

在效率（双塔独立编码）和效果（交叉注意力交互）之间找到一个中间点。

### Late Interaction

**独立编码，延迟交互**：

- Query 编码：$Q = [q_1, q_2, \\ldots, q_m]$（$m$ 个 token 向量）
- Doc 编码：$D = [d_1, d_2, \\ldots, d_n]$（$n$ 个 token 向量）

相似度计算（MaxSim）：

$$S(q, d) = \\sum_{i=1}^{m} \\max_{j=1..n} q_i^T d_j$$

每个 query token 找到最相似的 doc token，求和。

### 索引

文档的所有 token 向量都存入向量索引（而非仅一个 [CLS] 向量）→ 索引更大但交互更精细。

## SBERT vs ColBERT

| | SBERT | ColBERT |
|------|-------|---------|
| 向量数/文档 | 1 | N (每个 token) |
| 索引大小 | 小 | 大（数十倍） |
| 交互粒度 | 粗（句子级） | 细（token 级） |
| 速度 | 快 | 较快 |
| 效果 | 好 | 更好 |

## 使用

\`\`\`python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
similarities = model.similarity(emb_A, emb_B)
\`\`\`
`;export{n as default};
