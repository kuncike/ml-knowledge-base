const n=`# BGE / BCEmbedding / E5 嵌入模型

## 嵌入模型概览

这些模型是 RAG (检索增强生成) 系统的核心组件，将文本转化为稠密向量用于语义搜索。

## BGE (BAAI General Embedding)

### 核心特点

智源研究院推出的开源嵌入模型系列。

### 训练策略

1. **预训练**：RetroMAE（自编码预训练）
2. **微调**：对比学习 + 指令微调

### BGE 变体

| 模型 | 维度 | 参数量 | MTEB 平均 |
|------|------|--------|-----------|
| BGE-Small-EN | 384 | 33M | 62.5 |
| BGE-Base-EN | 768 | 109M | 63.5 |
| BGE-Large-EN | 1024 | 335M | 64.2 |
| BGE-M3 | 1024 | 568M | 多语言+多粒度 |

### BGE-M3 亮点

- **Multi-Lingual**：支持 100+ 语言
- **Multi-Granularity**：同时输出 Dense + Sparse (Lexical) 向量
- **Multi-Functionality**：Dense Retrieval + Sparse Retrieval + ColBERT

### 使用

\`\`\`python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3')

# Dense embedding
dense = model.encode("什么是机器学习？")['dense_vecs']  # [1024]

# Sparse embedding (lexical weights)
sparse = model.encode("什么是机器学习？")['lexical_weights']
# [{'机': 0.32, '器': 0.28, '学': 0.25, '习': 0.24, ...}]
\`\`\`

---

## E5 (EmbEddings from bidirEctional Encoder rEpresentations)

### 核心特点

微软推出的嵌入模型。关键创新在于**统一的文本前缀格式**：

\`\`\`
Query:  "query: 什么是机器学习？"
Doc:    "passage: 机器学习是人工智能的一个分支..."
\`\`\`

训练和推理时使用对应前缀，使模型区分查询和文档的语义空间。

### E5 变体

| 模型 | 维度 | 特点 |
|------|------|------|
| e5-small-v2 | 384 | 轻量 |
| e5-base-v2 | 768 | 通用 |
| e5-large-v2 | 1024 | 高性能 |
| multilingual-e5-large | 1024 | 多语言 |

---

## BCE (BAAI Client Embedding)

与 BGE 同源，针对**企业级 RAG** 优化，特别强调：
- **双编码器 + 交叉编码器重排序** 的联合优化
- 对中文的特别优化

---

## 模型选择建议

| 场景 | 推荐 |
|------|------|
| 中文为主 | BGE-M3 / BCE |
| 英文为主 | E5 / BGE |
| 多语言 | BGE-M3 / multilingual-e5 |
| 轻量部署 | BGE-Small / e5-small |
| 需要 Lexical | BGE-M3 (唯一支持) |

## 评估基准：MTEB

Massive Text Embedding Benchmark，包含分类、聚类、配对、重排序、检索、STS、摘要等子任务。
`;export{n as default};
