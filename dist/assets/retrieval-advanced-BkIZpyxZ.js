const n=`# Dense Retrieval Advanced (Contriever / ANCE / PLAID)

## 核心思想

DPR 用 BM25 产生的负样本训练密集检索器，但 BM25 的负样本不够"难"——模型很快就能区分，这限制了检索精度的进一步提升。三个方向突破：Contriever 完全无监督训练（连 BM25 都不用）、ANCE 在训练中动态生成最难的负样本、PLAID 在 ColBERT 基础上大幅加速推理。核心命题：**更好的负样本 → 更好的检索器**。

---

## 数学定义与原理解析

### Contriever — 无监督对比检索

完全不需要人工标注或 BM25，只用无监督数据通过对比学习训练。关键技巧是 **MoCo 风格的动量编码器** + **随机裁剪正样本对**：

学习目标（InfoNCE）：

$$
\\mathcal{L} = -\\log \\frac{\\exp(\\text{sim}(q, k_+) / \\tau)}{\\sum_j \\exp(\\text{sim}(q, k_j) / \\tau)}
$$

正样本 $k_+$ 来自同一文档的**不同片段**（随机裁剪），负样本来自 batch 内其他文档或动量队列。这与 SimCLR 一样，但用于文本检索场景。

### ANCE — 难负样本挖掘

DPR 的问题：用 BM25 或 in-batch negatives 训练，这些负样本对模型来说太简单了。ANCE 的核心思想是**异步难负样本挖掘**：

1. 用当前模型对所有候选文档编码
2. 对每个 query，检索 Top-K —— 其中排在高位但不是正样本的就是"难的负样本"
3. 用这些难负样本重新训练模型
4. 重复 1-3

\`\`\`
每轮训练: 编码所有文档 → 检索 → 选难负样本 → 更新模型
\`\`\`

$$
\\mathcal{L}_{ANCE} = -\\log \\frac{\\exp(q \\cdot d_+ / \\tau)}{\\exp(q \\cdot d_+ / \\tau) + \\sum_{d_- \\in H(q)} \\exp(q \\cdot d_- / \\tau)}
$$

$H(q)$ 是从当前 Top-K 检索结果中选出的难负样本集合。

### PLAID — 高效 ColBERT 检索

ColBERT 的 Late Interaction（每个 token 单独编码然后做 MaxSim）虽然效果好但推理慢。PLAID 通过两阶段候选生成大幅加速：

1. **Candidate Generation**：用 centroid 近似（所有 token embedding 的聚类中心），快速过滤掉 99.9% 的文档
2. **Late Interaction**：只对剩下的候选文档做精确 MaxSim 计算

速度提升 100× 以上，同时保持几乎相同的精度。

---

## 可视化展示

### 难负样本的影响

\`\`\`echarts
return {
  title: { text: '负样本类型对检索性能的影响', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['随机负样本', 'BM25 负样本', 'In-Batch 负样本', '难负样本 (ANCE)'] },
  yAxis: { type: 'value', min: 60, max: 90, name: 'MRR@10' },
  series: [{
    type: 'bar',
    data: [65, 72, 78, 87],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — Contriever 训练循环

\`\`\`python
import torch
import torch.nn.functional as F

def contriever_loss(q_emb, k_emb, temperature=0.05, queue=None):
    """
    q_emb: [B, D] — query 编码
    k_emb: [B, D] — key 编码 (正样本, 来自同一文档的另一段)
    queue: [Q, D] — 动量队列 (来自其他文档的历史 key)
    """
    B = q_emb.shape[0]

    # 正样本相似度
    pos_sim = torch.sum(q_emb * k_emb, dim=-1).unsqueeze(1)  # [B, 1]

    # 负样本相似度: batch 内其他 key
    neg_sim = q_emb @ k_emb.T  # [B, B]
    diag_mask = torch.eye(B, device=q_emb.device).bool()
    neg_sim = neg_sim.masked_fill(diag_mask, float('-inf'))

    # 如果有队列，拼接
    if queue is not None:
        queue_sim = q_emb @ queue.T  # [B, Q]
        logits = torch.cat([pos_sim, neg_sim, queue_sim], dim=1)
    else:
        logits = torch.cat([pos_sim, neg_sim], dim=1)

    labels = torch.zeros(B, dtype=torch.long, device=q_emb.device)
    return F.cross_entropy(logits / temperature, labels)


# 动量编码器更新
@torch.no_grad()
def momentum_update(student, teacher, m=0.999):
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data = m * t_param.data + (1 - m) * s_param.data
\`\`\`

### PyTorch — MaxSim (ColBERT Late Interaction)

\`\`\`python
def colbert_score(q_embs, d_embs, doc_mask=None):
    """
    q_embs: [B, T_q, D] — query 的 token 级编码
    d_embs: [B, T_d, D] — doc 的 token 级编码
    """
    # [B, T_q, T_d] — 每个 query token 与每个 doc token 的相似度
    sim = q_embs @ d_embs.transpose(-1, -2)
    if doc_mask is not None:
        sim = sim.masked_fill(doc_mask.unsqueeze(1) == 0, float('-inf'))
    # MaxSim: 每个 query token 取其最相似的 doc token
    max_per_query = sim.max(dim=-1)[0]  # [B, T_q]
    return max_per_query.sum(dim=-1)    # [B]
\`\`\`
`;export{n as default};
