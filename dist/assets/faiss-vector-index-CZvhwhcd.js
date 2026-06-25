const n=`# FAISS (向量索引与近似最近邻搜索)

## 核心思想

暴力搜索（遍历所有向量计算相似度）在百万级以上不可行。FAISS 用**索引结构**来组织向量空间，使得查询时只检查一小部分候选。从最简单的 IVF（聚类 + 同簇搜索）到最复杂的 HNSW（多层图遍历），本质上都是在"搜索精度"和"搜索速度"之间做权衡。Product Quantization (PQ) 进一步压缩向量本身，把内存占用压到原来的 4-16 倍以下。

---

## 数学定义与原理解析

### IVF (Inverted File)

1. 用 K-Means 将所有向量聚类为 $K$ 个簇
2. 建倒排索引：每个簇存其成员向量 ID
3. 查询时：计算 query 与 $K$ 个聚类中心的距离，只搜最近的 $n_{probe}$ 个簇

搜索复杂度：$O(n_{probe} \\times N/K \\times d)$，vs 暴力 $O(Nd)$。

### HNSW (Hierarchical Navigable Small World)

构建多层图结构，每层是一个近似 Delaunay 图：
- **上层**：长距离连接（高速公路），快速导航到目标区域
- **底层**：短距离连接（小路），精确搜索最近邻

查询时从顶层开始贪婪搜索，逐层下降。搜索复杂度 $O(\\log N)$。

### Product Quantization (PQ)

将 $d$ 维向量均分为 $M$ 段，每段 $d/M$ 维：

$$
\\mathbf{x} \\mapsto (q_1(x_{1:d/M}), q_2(x_{d/M:2d/M}), \\ldots, q_M(x_{(M-1)d/M:d}))
$$

每段独立用 K-Means（256 个中心 → 1 byte），整个向量压缩为 $M$ bytes。$d=1024$ 时：4096 bytes (float32) → 64 bytes (64×1 byte)，压缩比 64×。

搜索时用**非对称距离计算**（Asymmetric Distance Computation）：query 向量不量化，只有数据库向量量化，精度损失更小。

### IVFPQ — 工业标配

结合 IVF（过滤候选）+ PQ（压缩向量）：

1. IVF 粗量化找到最近的 $n_{probe}$ 个簇
2. 在这部分子集上用 PQ code 做精确距离计算
3. 返回 Top-K

---

## 可视化展示

### FAISS 索引类型

\`\`\`mermaid
graph TD
    FLAT["IndexFlatIP<br/>暴力内积搜索<br/>✅ 100% 精确<br/>❌ O(N)"] -->
    IVF["IndexIVF<br/>聚类+倒排<br/>✅ O(K+n_probe)<br/>❌ 精度损失"] -->
    IVFPQ["IndexIVFPQ<br/>IVF + PQ 压缩<br/>✅ 内存友好<br/>❌ 更多精度损失"] -->
    HNSW["IndexHNSW<br/>多层图搜索<br/>✅ O(log N)<br/>❌ 建索引慢"]
\`\`\`

### 召回率 vs 速度

\`\`\`echarts
return {
  title: { text: 'FAISS 索引召回率 vs QPS (1M vectors, 128d)', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'value', name: 'QPS (越高越快)' },
  yAxis: { type: 'value', name: 'Recall@10', min: 0.5, max: 1.0 },
  series: [
    { type: 'scatter', symbolSize: 14,
      data: [[100, 1.0], [5000, 0.98], [20000, 0.92], [80000, 0.85], [300000, 0.72]],
      label: { show: true,
        formatter: (p) => ['Flat','IVF','IVFPQ','HNSW','PQ+HNSW'][p.dataIndex],
        position: 'bottom' }
    }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 60 }
}
\`\`\`

---

## 核心代码实现

### FAISS 索引构建与搜索

\`\`\`python
import faiss
import numpy as np

# 1. IndexFlatIP — 内积搜索 (精确)
flat_index = faiss.IndexFlatIP(128)   # 128 维向量
flat_index.add(vectors)               # vectors: [N, 128]
D, I = flat_index.search(query, k=10) # D: distances, I: indices

# 2. IVFPQ — 工业级索引
d = 128
nlist = 1024          # 聚类中心数
m = 8                 # PQ 分段数
nbits = 8             # 每段聚类中心数 = 256 (1 byte)

quantizer = faiss.IndexFlatIP(d)  # 粗量化器
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
index.train(train_vectors)        # 必须先训练!
index.add(vectors)
index.nprobe = 32                 # 搜索时检查的簇数
D, I = index.search(query, k=10)

# 3. HNSW — 图搜索
index_hnsw = faiss.IndexHNSWFlat(d, 32)  # 32 connections per node
index_hnsw.hnsw.efConstruction = 200      # 构建时的搜索宽度
index_hnsw.hnsw.efSearch = 64             # 查询时的搜索宽度
index_hnsw.add(vectors)
D, I = index_hnsw.search(query, k=10)

# 4. GPU 加速
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
D, I = gpu_index.search(query, k=10)
\`\`\`

### 基于 FAISS 的检索 Pipeline

\`\`\`python
class VectorRetriever:
    def __init__(self, embed_dim, nlist=1024):
        self.dim = embed_dim
        quantizer = faiss.IndexFlatIP(embed_dim)
        self.index = faiss.IndexIVFPQ(quantizer, embed_dim, nlist, 8, 8)
        self.index.nprobe = 32
        self.id_map = []

    def build(self, embeddings, doc_ids):
        """embeddings: [N, D], doc_ids: [N]"""
        self.index.train(embeddings)
        self.index.add(embeddings)
        self.id_map = list(doc_ids)

    def search(self, query_emb, top_k=10):
        """query_emb: [D]"""
        D, I = self.index.search(query_emb.reshape(1, -1), top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.id_map):
                results.append({'doc_id': self.id_map[idx], 'score': float(dist)})
        return results
\`\`\`
`;export{n as default};
