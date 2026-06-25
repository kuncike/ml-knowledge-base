const n=`# MASS / Pegasus (序列到序列预训练)

## 核心思想

BERT 是 encoder-only（理解），GPT 是 decoder-only（生成）。但是摘要、翻译等任务需要 **encoder-decoder**——先理解输入，再生成输出。MASS 和 Pegasus 专门为 Seq2Seq 架构设计了预训练任务：MASS 将输入的一部分 masked 掉，让 decoder 从 encoder 的压缩表示中恢复被 mask 的部分；Pegasus 更进一步——设计了一个巧妙的 gap sentence generation 任务，让模型从整篇文章中生成摘要式的"关键句"。

两者都证明了：**为下游任务设计特定的预训练目标，远比通用的 LM/MLM 效果更好**。

---

## 数学定义与原理解析

### MASS — 联合 mask encoder 和 decoder

给定句子 $x_{1:n}$，随机 mask 一个连续片段 $x_{u:v}$（长度 $k = v - u + 1$）：

- **Encoder 输入**：$x_{\\setminus u:v}$（被 mask 的片段替换为 [M]，其余保留）
- **Decoder 输入**：前面 $k-1$ 个位置是 [M]，第 $k$ 个位置预测第一个被 mask 的词，依次自回归预测
- **Decoder 输出**：$x_{u:v}$（被 mask 的片段）

损失：
$$
\\mathcal{L} = -\\frac{1}{k} \\sum_{t=u}^{v} \\log P(x_t | x_{\\setminus u:v}, x_{u:t-1})
$$

关键设计：$k$ 设为句子长度的约 50%——恰好介于 BERT 的 15% 和 GPT 的 100% 之间，让 encoder 学会压缩理解、decoder 学会条件生成。

### Pegasus — Gap Sentence Generation (GSG)

将多句子文档中"最重要的句子"mask 掉，让模型生成这些 gap sentences。

**重要性选择**：用 ROUGE-1 F1 衡量每个句子与文档其余部分的相似度，选最相似的（作为该文档的"摘要式中心句"）。

$$
\\text{Importance}(s_i) = \\text{ROUGE-1}(s_i, D \\setminus s_i)
$$

Mask 文档中 top-$m$ (通常 $m=1$) 个重要句子，输入剩余部分到 encoder，decoder 自回归生成被 mask 的句子。

### 与 BERT/GPT 的对比

| 方法 | 架构 | 预训练目标 | $k$ (mask比例) |
|------|------|-----------|----------------|
| BERT | Encoder | MLM | ~15% |
| GPT | Decoder | LM | 0% (自回归) |
| MASS | Enc-Dec | Seq-to-Seq MLM | ~50% |
| Pegasus | Enc-Dec | Gap Sentence Gen | 1-2 个整句 |

---

## 可视化展示

### MASS 预训练流程

\`\`\`mermaid
graph TD
    SRC["原文: A B [M] [M] [M] F G H"] --> ENC["Encoder"]
    ENC --> HIDDEN["(压缩的语义表示)"]
    HIDDEN --> DEC["Decoder"]
    TGT1["[M]"] --> DEC
    DEC --> P1["预测 C"]
    TGT2["[M]"] --> DEC
    DEC --> P2["预测 D"]
    TGT3["[M]"] --> DEC
    DEC --> P3["预测 E"]
\`\`\`

### 预训练对比

\`\`\`echarts
return {
  title: { text: 'Seq2Seq 预训练方法摘要性能 (CNN/DailyMail)', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['Random Init', 'BERT2BERT', 'MASS', 'Pegasus', 'BART'] },
  yAxis: { type: 'value', min: 30, max: 45, name: 'ROUGE-L' },
  series: [{
    type: 'bar',
    data: [31.2, 36.8, 39.6, 44.2, 44.2],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — MASS 风格的前向传播

\`\`\`python
import torch
import torch.nn as nn
import random

class MASSModel(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 8), num_layers=6)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, 8), num_layers=6)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids):
        # src: masked encoder 输入
        # tgt: 前缀 [M] + 被 mask 的片段
        src_emb = self.embed(src_ids)          # [B, T_src, D]
        tgt_emb = self.embed(tgt_ids)          # [B, T_tgt, D]
        memory = self.encoder(src_emb)
        out = self.decoder(tgt_emb, memory)
        return self.out(out)


def create_mass_masks(tokens, mask_ratio=0.5):
    """生成 MASS 的 encoder 输入和 decoder 目标"""
    n = len(tokens)
    k = max(1, int(n * mask_ratio))
    start = random.randint(0, n - k)

    # Encoder 输入: mask 掉片段
    src = tokens.copy()
    src[start:start + k] = ['[MASK]'] * k

    # Decoder 输入: [MASK] * (k-1) + 片段[:-1]
    tgt_input = ['[MASK]'] * (k - 1) + tokens[start:start + k - 1]

    # Decoder 目标: 被 mask 的片段
    tgt_output = tokens[start:start + k]

    return src, tgt_input, tgt_output
\`\`\`

### Pegasus GSG 核心逻辑

\`\`\`python
from rouge_score import rouge_scorer

def select_gap_sentences(document, num_gaps=1):
    """选择最重要的句子作为 gap"""
    sentences = document.split('. ')
    scorer = rouge_scorer.RougeScorer(['rouge1'])
    scores = []

    for i, sent in enumerate(sentences):
        rest = '. '.join(sentences[:i] + sentences[i+1:])
        rouge = scorer.score(sent, rest)
        scores.append((i, rouge['rouge1'].fmeasure))

    # 选 ROUGE-1 F1 最高的作为 pseudo-summary / gap
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores[:num_gaps]]
\`\`\`
`;export{n as default};
