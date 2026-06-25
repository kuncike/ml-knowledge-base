const n=`# RAG / GraphRAG (检索增强生成)

## RAG (Retrieval-Augmented Generation)

### 核心思想

在 LLM 生成回答之前，先从外部知识库中**检索相关文档**，将检索到的内容作为上下文拼入 prompt，让 LLM 基于检索结果生成回答。

### 系统架构

\`\`\`
Query → Embedding Model → Query Vector
                                  ↓
                    Vector Database (FAISS/Milvus/Weaviate)
                                  ↓
                    Top-K Relevant Chunks
                                  ↓
           Prompt = Query + Retrieved Context
                                  ↓
                             LLM → Answer
\`\`\`

### 索引阶段 (Offline)

1. **文档解析**：提取文本（PDF/HTML/Markdown）
2. **文本切分 (Chunking)**：将文档切分为合适大小的片段
3. **向量化**：用 Embedding 模型将每个 chunk 转为向量
4. **存储**：存入向量数据库

### 检索 + 生成阶段 (Online)

1. Query → Embedding → 向量搜索 → Top-K chunks
2. 拼接 Prompt：\`基于以下知识回答问题：\\n{chunks}\\n\\n问题：{query}\`
3. LLM 生成答案

## 关键设计决策

### 分块策略 (Chunking)

| 策略 | 描述 |
|------|------|
| 固定大小 | 按 token 数切（如 512 tokens, overlap 50） |
| 语义分块 | 按自然段落/句子边界 |
| 递归分块 | 尝试不同分隔符（段落 → 句子 → 词） |
| 父子分块 | 检索小块，给 LLM 大的上下文窗口 |

### 检索策略

- **Top-K 直接检索**：返回相似度最高的 K 个
- **MMR (最大边际相关性)**：平衡相关性和多样性
- **多级检索**：粗排（向量召回） → 精排（Cross-Encoder）

---

## GraphRAG

### 核心思想

RAG 只能检索碎片化的文本块，无法理解实体之间的**全局关系和层次结构**。GraphRAG 构建知识图谱来增强理解。

### 构建流程

1. **实体提取**：LLM 从文档中提取实体和关系
2. **社区发现**：用 Leiden 算法在图中发现主题社区
3. **社区总结**：LLM 为每个社区生成摘要
4. **查询时**：根据问题匹配社区/实体 → 获取结构化+非结构化上下文

### GraphRAG vs 标准 RAG

| | RAG | GraphRAG |
|------|-----|----------|
| 检索粒度 | 文本块 | 实体 + 文本块 |
| 全局理解 | 弱 | 强 |
| 构建成本 | 低 | 高 |
| 适合查询 | 事实性 | 总结/跨文档推理 |

## 评估指标

- **检索质量**：Recall@K, MRR, NDCG
- **生成质量**：Faithfulness（是否忠于检索内容）、Answer Relevance
- **端到端**：与 Ground Truth 的语义相似度

## 简单实现

\`\`\`python
import chromadb
client = chromadb.Client()
collection = client.create_collection("knowledge")

# 索引
for i, chunk in enumerate(chunks):
    collection.add(ids=str(i), embeddings=embed(chunk), documents=chunk)

# 检索
results = collection.query(query_embeddings=embed(query), n_results=5)
context = "\\n".join(results["documents"][0])
prompt = f"基于以下知识回答问题：\\n{context}\\n\\n问题：{query}"
answer = llm(prompt)
\`\`\`
`;export{n as default};
