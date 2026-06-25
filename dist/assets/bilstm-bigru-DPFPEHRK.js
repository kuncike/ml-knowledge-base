const n=`# BiLSTM / BiGRU (双向循环网络)

## 核心思想

单向 LSTM 只能看到过去——"我 _ 吃饭"无法确定中间是"在"还是"没"。双向设计让一个 LSTM 从左到右扫描（过去→未来），另一个从右到左扫描（未来→过去），拼接后的隐状态同时包含两侧上下文。这对 NLP 序列标注（NER、POS、分词）至关重要，因为每个 token 的标签依赖于前后文。

BiLSTM-CRF 曾是命名实体识别的 SOTA 架构——BiLSTM 抽取特征，CRF 建模标签之间的转移约束。

---

## 数学定义与原理解析

### 双向结构

对于输入序列 $\\mathbf{x}_{1:T}$：

前向 LSTM：
$$
\\overrightarrow{\\mathbf{h}}_t = \\text{LSTM}(\\mathbf{x}_t, \\overrightarrow{\\mathbf{h}}_{t-1})
$$

后向 LSTM：
$$
\\overleftarrow{\\mathbf{h}}_t = \\text{LSTM}(\\mathbf{x}_t, \\overleftarrow{\\mathbf{h}}_{t+1})
$$

拼接输出：
$$
\\mathbf{h}_t = [\\overrightarrow{\\mathbf{h}}_t ; \\overleftarrow{\\mathbf{h}}_t] \\in \\mathbb{R}^{2d_h}
$$

### BiGRU

GRU 只有重置门和更新门（比 LSTM 少一个输出门），双向结构完全类似：

$$
\\overrightarrow{\\mathbf{h}}_t = \\text{GRU}(\\mathbf{x}_t, \\overrightarrow{\\mathbf{h}}_{t-1}), \\quad
\\overleftarrow{\\mathbf{h}}_t = \\text{GRU}(\\mathbf{x}_t, \\overleftarrow{\\mathbf{h}}_{t+1})
$$

### 与 Transformer 的对比

| 特性 | BiLSTM | Transformer |
|------|--------|-------------|
| 上下文方向 | 分开处理再拼接 | 天然双向（同时看全部）|
| 时间复杂度 | $O(n)$ | $O(n^2)$ |
| 长距离依赖 | 仍受梯度路径限制 | $O(1)$ 路径长度 |
| 序列长度适应性 | 任意长度 | 受 $n^2$ 限制 |

---

## 可视化展示

### BiLSTM 结构

\`\`\`mermaid
graph LR
    subgraph Forward["前向 LSTM"]
        F1["h⃗₁"] --> F2["h⃗₂"] --> F3["h⃗₃"] --> FDOT["..."]
    end
    subgraph Backward["后向 LSTM"]
        B1["h⃖₁"] --> B2["h⃖₂"] --> B3["h⃖₃"] --> BDOT["..."]
    end
    X1["x₁"] --> F1 & B1
    X2["x₂"] --> F2 & B2
    X3["x₃"] --> F3 & B3

    F1 --> C1["[h⃗₁; h⃖₁]"]
    B1 --> C1
    F2 --> C2["[h⃗₂; h⃖₂]"]
    B2 --> C2
    F3 --> C3["[h⃗₃; h⃖₃]"]
    B3 --> C3
\`\`\`

每个位置的输出同时包含左侧（前向）和右侧（后向）的全部序列信息。

### BiLSTM vs BiGRU 参数对比

\`\`\`echarts
return {
  title: { text: 'LSTM vs GRU 参数量对比 (hidden=256)', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['单向LSTM', 'BiLSTM', '单向GRU', 'BiGRU'] },
  yAxis: { type: 'value', name: '参数量 (M)' },
  series: [{
    type: 'bar',
    data: [3.1, 6.2, 2.3, 4.6],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — BiLSTM

\`\`\`python
import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=0.3 if num_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None):
        # x: [B, T]
        emb = self.embedding(x)  # [B, T, E]
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (hn, cn) = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(emb)  # [B, T, 2H]
        return self.classifier(out)  # [B, T, C]

    def get_last_output(self, x):
        """获取序列最后一个有效位置的隐状态 (用于分类)"""
        emb = self.embedding(x)
        out, (hn, cn) = self.lstm(emb)
        # hn: [2*layers, B, H] → 取最后一层前向+后向
        fwd = hn[-2, :, :]  # 前向最后一步
        bwd = hn[-1, :, :]  # 后向最后一步
        return torch.cat([fwd, bwd], dim=-1)  # [B, 2H]
\`\`\`

### PyTorch — BiGRU

\`\`\`python
class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim,
                          num_layers=num_layers,
                          bidirectional=True,
                          batch_first=True,
                          dropout=0.3 if num_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        out, hn = self.gru(emb)
        return self.classifier(out)
\`\`\`
`;export{n as default};
