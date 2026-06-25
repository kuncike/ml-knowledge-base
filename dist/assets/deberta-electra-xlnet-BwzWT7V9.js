const n=`# DeBERTa / ELECTRA / XLNet

## 核心思想

BERT 的 MLM 简单有效，但三个改进方向分别解决了不同痛点：DeBERTa 将**位置编码解耦**（内容和位置应该独立建模），ELECTRA 将 MLM 改写为**替换词检测**（更高效——对所有 token 计算损失而非仅 15%），XLNet 用**排列语言模型**（permutation LM）实现真正的双向上下文，而非 BERT 的 [MASK] 噪声。

共同目标：**比 BERT 更高效地学习更好的上下文表示**。

---

## 数学定义与原理解析

### DeBERTa — 解耦注意力

标准注意力将内容和位置混合投影。DeBERTa 将两者分离开：

$$
A_{ij} = \\underbrace{\\mathbf{H}_i \\mathbf{H}_j^T}_{\\text{内容-内容}} + \\underbrace{\\mathbf{H}_i (\\mathbf{P}_{j-i})^T}_{\\text{内容-位置}} + \\underbrace{\\mathbf{P}_{j-i} (\\mathbf{H}_j)^T}_{\\text{位置-内容}} + \\dots
$$

位置编码 $\\mathbf{P}_{j-i}$ 是相对位置嵌入。额外引入**增强掩码解码器**（EMD）——在 MLM 预训练时用 absolute position 帮助解码 masked token。

### ELECTRA — 替换词检测

用两个网络：**Generator**（小型 MLM，BERT 的 1/4 大小）和 **Discriminator**（主模型）。

1. Generator 对 15% token 做 MLM → 生成替代词
2. Discriminator 对**所有 token** 做二分类：每个 token 是原始的还是 Generator 替换的

损失函数：

$$
\\mathcal{L} = \\mathcal{L}_{MLM}(G) + \\lambda \\cdot \\mathcal{L}_{RTD}(D)
$$

关键效率提升：MLM 只对 15% token 监督，RTD 对所有 token 监督。同等训练下 ELECTRA 比 BERT 好得多。

### XLNet — 排列语言模型

对长度为 $T$ 的序列，随机采样一个排列 $\\mathbf{z}$，按排列顺序预测：

$$
\\max_\\theta \\mathbb{E}_{\\mathbf{z} \\sim \\mathcal{Z}_T} \\left[ \\sum_{t=1}^{T} \\log p_\\theta(x_{z_t} | \\mathbf{x}_{z_{<t}}) \\right]
$$

**Two-Stream Self-Attention**：Content Stream（能看自己和上文）和 Query Stream（不能看自己的内容）——这是实现排列 LM 同时保持自回归特性的关键。

---

## 可视化展示

### ELECTRA 训练流程

\`\`\`mermaid
graph TD
    X["原始输入: The cat sat"] --> MASK["随机 Mask 15%<br/>The [MASK] sat"]
    MASK --> GEN["Generator (小MLM)<br/>预测: The dog sat"]
    GEN --> DISK["替换后: The dog sat"]
    DISK --> DISC["Discriminator (主模型)<br/>对每个token判断<br/>Original/Replaced"]
    ORIG["原始: The cat sat"] --> DISC
    DISC --> LOSS["Fake: ✓<br/>Real: ✓ Fake: ✗ Real: ✓"]
\`\`\`

### BERT 家族对比

\`\`\`echarts
return {
  title: { text: 'BERT 家族 GLUE 基准对比 (Base)', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['BERT', 'RoBERTa', 'XLNet', 'ELECTRA', 'DeBERTa'] },
  yAxis: { type: 'value', min: 78, max: 88, name: 'GLUE Avg Score' },
  series: [{
    type: 'bar',
    data: [79.6, 82.1, 83.1, 84.0, 86.8],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

DeBERTa 凭借解耦注意力 + EMD，在 BERT 家族中取得了最高的 GLUE 分数。

---

## 核心代码实现

### PyTorch — ELECTRA Discriminator Head

\`\`\`python
import torch
import torch.nn as nn

class ELECTRADiscriminator(nn.Module):
    def __init__(self, bert_encoder):
        super().__init__()
        self.encoder = bert_encoder  # 共享 BERT encoder 结构
        self.disc_head = nn.Linear(768, 1)  # 二分类头 (原始/替换)

    def forward(self, input_ids, attention_mask):
        # input_ids: generator 替换后的序列
        hidden = self.encoder(input_ids, attention_mask).last_hidden_state
        # [B, T, 768]
        logits = self.disc_head(hidden).squeeze(-1)  # [B, T]
        return logits  # 每个 token 的替换概率

    def loss_fn(self, logits, labels, attention_mask):
        """labels: 1=原始, 0=被替换"""
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction='none')
        return (loss * attention_mask).sum() / attention_mask.sum()
\`\`\`

### PyTorch — DeBERTa 解耦注意力

\`\`\`python
class DisentangledSelfAttention(nn.Module):
    def __init__(self, dim, max_rel_pos=512):
        super().__init__()
        self.dim = dim
        self.max_rel_pos = max_rel_pos
        # 相对位置嵌入
        self.rel_embeddings = nn.Embedding(2 * max_rel_pos + 1, dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, hidden_states, attention_mask=None):
        B, N, D = hidden_states.shape
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        # 内容-内容注意力
        content_score = torch.matmul(Q, K.transpose(-2, -1))

        # 内容-位置注意力 (简化: 仅计算相对位置偏置)
        rel_pos = self._get_rel_pos(N, hidden_states.device)
        rel_pos_emb = self.rel_embeddings(rel_pos)  # [N, N, D]
        pos_score = torch.einsum('bnd,bnmd->bnm', Q, rel_pos_emb.unsqueeze(0))

        attn = content_score + pos_score
        attn = attn / (D ** 0.5)

        if attention_mask is not None:
            attn = attn + attention_mask

        attn_weights = torch.softmax(attn, dim=-1)
        return self.out_proj(torch.matmul(attn_weights, V))

    def _get_rel_pos(self, length, device):
        """生成相对位置索引"""
        range_vec = torch.arange(length, device=device)
        rel_pos = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        rel_pos = torch.clamp(rel_pos, -self.max_rel_pos, self.max_rel_pos)
        return rel_pos + self.max_rel_pos
\`\`\`
`;export{n as default};
