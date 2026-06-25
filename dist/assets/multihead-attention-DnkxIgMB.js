const n=`# Multi-Head Attention (多头注意力)

## 核心思想

单头注意力只能学一种"关系模式"——比如总是关注主语-谓语关系。多头注意力将 Q、K、V 切成 $h$ 份，每份（一个"头"）在**不同的表示子空间**中独立做注意力，最后拼接。这相当于让模型同时从 $h$ 个不同"视角"理解序列：一个头看语法，一个头看共指，一个头看长距离依赖。

---

## 数学定义与原理解析

### 完整公式

$$
\\text{MultiHead}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_h) \\mathbf{W}^O
$$

$$
\\text{head}_i = \\text{Attention}(\\mathbf{Q} \\mathbf{W}_i^Q, \\mathbf{K} \\mathbf{W}_i^K, \\mathbf{V} \\mathbf{W}_i^V)
$$

其中投影矩阵维度：

$$
\\mathbf{W}_i^Q, \\mathbf{W}_i^K \\in \\mathbb{R}^{d_{model} \\times d_k}, \\quad \\mathbf{W}_i^V \\in \\mathbb{R}^{d_{model} \\times d_v}, \\quad \\mathbf{W}^O \\in \\mathbb{R}^{h d_v \\times d_{model}}
$$

通常设 $d_k = d_v = d_{model} / h$，使得**多头总计算量与单头几乎相同**（仅多了一个 $W^O$ 投影）。

### 为什么需要多头？

**单头的问题**：平均化效应。如果将 Q、K、V 的所有信息压缩到一个注意力分布，不同关系被混在一起。

**多头的优势**：

- Head 1 可能关注**句法关系**（如"定语-中心词"）
- Head 2 可能关注**共指关系**（如"Trump → he"）
- Head 3 可能关注**位置邻近性**（相邻词）
- Head 4 可能关注**全局主题词**

### 多头合并的张量操作

\`\`\`python
# 输入: [B, N, d_model]
# 投影后: [B, N, h, d_k]
# transpose: [B, h, N, d_k]  → 每头独立做 attention
# 合并: [B, h, N, d_k] → [B, N, h·d_k] → W_o → [B, N, d_model]
\`\`\`

---

## 可视化展示

### Multi-Head Attention 架构

\`\`\`mermaid
graph TD
    X["输入 X (n×d_model)"] --> QL["Linear → Q"]
    X --> KL["Linear → K"]
    X --> VL["Linear → V"]

    QL --> S1["Split → h 份"]
    KL --> S2["Split → h 份"]
    VL --> S3["Split → h 份"]

    subgraph "h 个头并行计算"
        H1["Head 1: Attention(Q₁,K₁,V₁)"]
        H2["Head 2: Attention(Q₂,K₂,V₂)"]
        H3["..."]
        Hh["Head h: Attention(Qₕ,Kₕ,Vₕ)"]
    end

    S1 --> H1
    S1 --> H2
    S1 --> H3
    S1 --> Hh
    S2 --> H1
    S2 --> H2
    S2 --> H3
    S2 --> Hh
    S3 --> H1
    S3 --> H2
    S3 --> H3
    S3 --> Hh

    H1 --> CAT["Concat"]
    H2 --> CAT
    H3 --> CAT
    Hh --> CAT

    CAT --> WO["Linear · W^O"]
    WO --> OUT["输出 (n×d_model)"]
\`\`\`

### MHA / MQA / GQA 对比

\`\`\`echarts
return {
  xAxis: { type: 'category', data: ['MHA (标准)', 'GQA (LLaMA 2)', 'MQA (PaLM)'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '相对得分' },
  legend: { data: ['推理速度', '显存占用(低=好)', '模型质量'] },
  series: [
    { name: '推理速度', type: 'bar', data: [0.5, 0.78, 0.95], itemStyle: { color: '#2c3e50' } },
    { name: '显存占用(低=好)', type: 'bar', data: [0.5, 0.75, 0.95], itemStyle: { color: '#16a085' } },
    { name: '模型质量', type: 'bar', data: [0.95, 0.92, 0.82], itemStyle: { color: '#d35400' } }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 60 }
}
\`\`\`

- **MHA**（Multi-Head）：每头独立 K, V → 质量最高，显存大
- **MQA**（Multi-Query）：所有头共享 K, V → 推理最快，KV Cache 最小
- **GQA**（Grouped-Query）：分组共享 K, V → 折中方案（LLaMA 2/3 使用）

---

## 核心代码实现

### PyTorch 完整实现

\`\`\`python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, D = x.shape
        # 投影 + 拆分为多头: [B, N, D] → [B, h, N, d_k]
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V  # [B, h, N, d_k]

        # 合并多头: [B, h, N, d_k] → [B, N, D]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)
\`\`\`

### 主流模型配置

| 模型 | $d_{model}$ | $h$ | $d_k$ | 注意力类型 |
|------|-------------|-----|-------|-----------|
| Transformer (base) | 512 | 8 | 64 | MHA |
| BERT-base | 768 | 12 | 64 | MHA |
| GPT-3 (175B) | 12288 | 96 | 128 | MHA |
| LLaMA-7B | 4096 | 32 | 128 | MHA |
| LLaMA 2-70B | 8192 | 64 | 128 | **GQA** (8 groups) |
| PaLM | 18432 | 48 | 384 | **MQA** |
`;export{n as default};
