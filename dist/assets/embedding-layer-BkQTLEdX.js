const n=`# 嵌入层 (Embedding Layer)

## 核心思想

嵌入层将**离散的、高维的类别特征**（如词、用户 ID、物品 ID）映射到**低维稠密向量**空间，使得语义相似的对象的向量也彼此接近。

## 数学定义

嵌入层本质上是一个**可学习的查找表 (Lookup Table)**：

$$\\mathbf{E} \\in \\mathbb{R}^{V \\times d}$$

给定索引 $i$，输出第 $i$ 行：$\\mathbf{e}_i = \\mathbf{E}[i, :]$。

其中：
- $V$：词汇表大小
- $d$：嵌入维度

## 为什么需要 Embedding

One-hot 编码的问题：
- 维度 = 词汇量（数万到数十万）
- 向量之间两两正交（无法表达语义相似度）

Embedding 的优势：
- 维度可控（50~1024）
- 语义相近的词向量距离也近（分布假说）

## 词嵌入的经典理论

$$\\text{king} - \\text{man} + \\text{woman} \\approx \\text{queen}$$

嵌入空间中的线性操作可以编码语义关系。

## PyTorch 实现

\`\`\`python
import torch.nn as nn

# 词嵌入
embed = nn.Embedding(num_embeddings=10000, embedding_dim=256, padding_idx=0)

# 输入：词的索引 [batch, seq_len]
indices = torch.LongTensor([[1, 5, 3], [2, 7, 0]])  # [2, 3]
out = embed(indices)  # [2, 3, 256]
\`\`\`

## 位置嵌入 (Positional Embedding)

Transformer 需要位置信息（因为自注意力不具备顺序感知）：

**正弦位置编码**（原始 Transformer）：

$$PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d}}\\right)$$
$$PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d}}\\right)$$

**可学习位置嵌入**：直接将位置 $pos$ 视为索引查表（BERT, GPT 使用）。

## Embedding 维度选择经验

| 场景 | 推荐维度 |
|------|----------|
| 小词汇（< 10K） | 64–128 |
| 中词汇（10K–100K） | 256–512 |
| 大词汇/LLM | 768–4096+ |

嵌入层参数量 = $V \\times d$，是很多 NLP 模型参数的大头。
`;export{n as default};
