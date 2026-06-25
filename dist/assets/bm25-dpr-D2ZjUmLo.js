const n=`# BM25 / DPR

## BM25 (稀疏检索 / Sparse Retrieval)

经典的概率检索模型，自 1990 年代以来一直是搜索引擎的标配。

$$\\text{BM25}(q, d) = \\sum_{t \\in q} \\text{IDF}(t) \\cdot \\frac{f_{t,d}(k_1+1)}{f_{t,d} + k_1(1-b+b \\cdot \\frac{|d|}{avgdl})}$$

- $k_1$（通常 1.5）：词频饱和参数
- $b$（通常 0.75）：文档长度归一化

### 优势

- 极快（倒排索引 + 精确词匹配）
- 可解释（可以精确知道为什么某文档被召回）
- 无需训练数据

### 劣势

- 词汇不匹配问题（"汽车" 检索不到 "轿车"）
- 无法利用语义信息

---

## DPR (Dense Passage Retrieval)

### 核心思想

用**两个独立的 BERT** 分别编码问题和文档，用点积计算相似度：

$$\\text{sim}(q, d) = E_Q(q)^T \\cdot E_D(d)$$

### 训练目标：对比学习

$$\\text{loss} = -\\log \\frac{e^{\\text{sim}(q, d^+)}}{e^{\\text{sim}(q, d^+)} + \\sum_{d^-} e^{\\text{sim}(q, d^-)}}$$

- $d^+$：正样本（相关文档）
- $d^-$：负样本（不相关文档）

### 负样本策略

- **In-batch negatives**：同 batch 内其他问题的正文档作为负样本
- **Hard negatives**：BM25 高分但与问题不相关的文档（关键！）

### 训练数据

通常从 QA 数据集（如 Natural Questions, TriviaQA）构建 (q, d+, d-) 三元组。

## BM25 vs DPR

| | BM25 | DPR |
|------|------|-----|
| 检索方式 | 精确词匹配 | 稠密向量相似度 |
| 表示 | 稀疏 (词汇量维度) | 稠密 (768 维) |
| 语义理解 | 无 | 有 |
| 速度 | 极快 | 需向量索引 (FAISS) |
| 冷启动 | 无 | 需要训练数据 |

## 混合检索

目前 SOTA：结合 BM25 和稠密检索：

- **第一阶段**：BM25 召回 + 向量召回 → 合并
- **第二阶段**：交叉编码器 (Cross-Encoder) 精排

## 使用

\`\`\`python
# BM25
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(tokenized_query)

# Dense Retrieval (简单示例)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/e5-base-v2')
q_emb = model.encode(query)
d_embs = model.encode(docs)  # 预先索引
scores = q_emb @ d_embs.T
\`\`\`
`;export{n as default};
