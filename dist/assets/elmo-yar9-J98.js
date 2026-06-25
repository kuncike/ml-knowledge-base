const n=`# ELMo (深度上下文词向量)

## 核心思想

2018 年之前，每个词只有一个固定的 embedding（Word2Vec/GloVe），"bank"在"river bank"和"bank account"中向量完全一样。ELMo 开创性地提出**深度上下文词向量**：每个词的表示是其所在句子的函数，而非固定的查找表。核心架构是两层双向 LSTM 在大规模语料上做**语言模型预训练**（前向预测下一个词 + 后向预测上一个词），下游任务使用时取各层隐状态的**加权组合**。

这直接启发了 BERT 和 GPT 的双向/单向预训练范式——ELMo 是"预训练"时代的开端。

---

## 数学定义与原理解析

### 双向语言模型

前向语言模型：$p(t_1, \\ldots, t_N) = \\prod_{k=1}^{N} p(t_k | t_1, \\ldots, t_{k-1})$

后向语言模型：$p(t_1, \\ldots, t_N) = \\prod_{k=1}^{N} p(t_k | t_{k+1}, \\ldots, t_N)$

联合目标（负对数似然最小化）：

$$
\\mathcal{L} = -\\sum_{k=1}^{N} \\left( \\log p(t_k | t_{<k}; \\Theta) + \\log p(t_k | t_{>k}; \\Theta) \\right)
$$

### 词表示 = 各层加权和

对于 token $t_k$，ELMo 输出 $L+1$ 个表示（$L$ 层 BiLM + 输入嵌入）：

$$
\\mathbf{R}_k = \\{\\mathbf{x}_k^{LM}, \\overrightarrow{\\mathbf{h}}_{k,1}, \\overleftarrow{\\mathbf{h}}_{k,1}, \\ldots, \\overrightarrow{\\mathbf{h}}_{k,L}, \\overleftarrow{\\mathbf{h}}_{k,L}\\}
$$

$$
\\text{ELMo}_k = \\gamma \\sum_{j=0}^{L} s_j \\cdot \\mathbf{h}_{k,j}
$$

- $s_j$：softmax 归一化的层权重（每层的重要性由任务学习）
- $\\gamma$：全局缩放因子（任务相关）

### 字符级 CNN 输入

ELMo 不使用固定词表——用字符 CNN 从字符序列生成词表示，天然处理 OOV 词：

$$
\\text{CharCNN}(c_1, \\ldots, c_m) = \\text{MaxPool}(\\text{CNN}(\\text{CharEmbed}(c_1, \\ldots, c_m)))
$$

---

## 可视化展示

### ELMo 架构

\`\`\`mermaid
graph TD
    CHARS["字符序列 (CharCNN)"] --> EMB["词嵌入 (L=0)"]
    EMB --> LSTM1F["LSTM Layer 1 →"]
    EMB --> LSTM1B["← LSTM Layer 1"]
    LSTM1F --> LSTM2F["LSTM Layer 2 →"]
    LSTM1B --> LSTM2B["← LSTM Layer 2"]
    EMB --> WEIGHT["加权求和<br/>s₀·h₀ + s₁·h₁ + s₂·h₂"]
    LSTM1F --> WEIGHT
    LSTM1B --> WEIGHT
    LSTM2F --> WEIGHT
    LSTM2B --> WEIGHT
    WEIGHT --> SCALE["× γ"] --> ELMO["ELMo 向量"]
\`\`\`

### 不同任务学到的层权重

\`\`\`echarts
return {
  title: { text: 'ELMo 各层在下游任务中的权重', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['Layer 0 (词法)', 'Layer 1 (句法)', 'Layer 2 (语义)'] },
  yAxis: { type: 'value', min: 0, max: 1, name: 'softmax 权重' },
  legend: { data: ['SQuAD (QA)', 'SNLI (推理)', 'POS Tagging'] },
  series: [
    { name: 'SQuAD (QA)', type: 'bar', data: [0.1, 0.3, 0.6], itemStyle: { color: '#2c3e50' } },
    { name: 'SNLI (推理)', type: 'bar', data: [0.05, 0.2, 0.75], itemStyle: { color: '#2980b9' } },
    { name: 'POS Tagging', type: 'bar', data: [0.25, 0.65, 0.1], itemStyle: { color: '#16a085' } }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

浅层更适合词法任务（词性标注），深层更适合语义理解（QA、推理）。

---

## 核心代码实现

### PyTorch — ELMo 风格的 BiLM

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=4096, n_layers=2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        # 前后向 LSTM
        self.forward_lstm = nn.LSTM(embed_dim, hidden_dim,
                                     num_layers=n_layers, batch_first=True)
        self.backward_lstm = nn.LSTM(embed_dim, hidden_dim,
                                      num_layers=n_layers, batch_first=True)
        # 输出映射
        self.fwd_proj = nn.Linear(hidden_dim, vocab_size)
        self.bwd_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.token_embed(x)  # [B, T, D]

        # 前向 LSTM: 预测下一个 token
        fwd_out, fwd_h = self.forward_lstm(emb[:, :-1])
        fwd_logits = self.fwd_proj(fwd_out)  # [B, T-1, V]
        # 后向 LSTM: 翻转序列, 预测上一个 token
        rev_emb = torch.flip(emb[:, 1:], [1])
        bwd_out, bwd_h = self.backward_lstm(rev_emb)
        bwd_logits = self.bwd_proj(bwd_out)

        return fwd_logits, bwd_logits, (fwd_h, bwd_h)

    def get_elmo_representations(self, x):
        """提取 ELMo 风格的多层表示"""
        emb = self.token_embed(x)
        layers = [emb]

        fwd_out, _ = self.forward_lstm(emb)
        bwd_out, _ = self.backward_lstm(torch.flip(emb, [1]))
        bwd_out = torch.flip(bwd_out, [1])

        layers.append(torch.cat([fwd_out, bwd_out], dim=-1))
        return layers  # [layer0_emb, layer1_bilm_outputs]
\`\`\`
`;export{n as default};
