const n=`# BERT / RoBERTa / ALBERT / DistilBERT

## BERT (Bidirectional Encoder Representations from Transformers)

### 核心思想

BERT 使用 Transformer 的**编码器**部分，通过**掩码语言模型 (MLM)** 和**下一句预测 (NSP)** 进行双向预训练。

### 预训练任务

#### 1. 掩码语言模型 (Masked Language Model, MLM)

随机掩盖 15% 的 token：

- 80% 替换为 [MASK]
- 10% 替换为随机 token
- 10% 保持不变

模型从双向上下文中预测被掩盖的 token。

#### 2. 下一句预测 (Next Sentence Prediction, NSP)

给定两个句子 A 和 B，判断 B 是否是 A 的下一句。50% 是正样本，50% 随机句子。

### 输入表示

\`\`\`
[CLS] Token1 Token2 ... [SEP] TokenA ... [SEP]
  ↑                            ↑
  句子级表示                  分隔符
\`\`\`

输入嵌入 = Token Embedding + Segment Embedding + Position Embedding 之和。

### BERT 变体

| 模型 | 层数 | 隐藏维度 | 头数 | 参数量 |
|------|------|----------|------|--------|
| BERT-Tiny | 2 | 128 | 2 | 4.4M |
| BERT-Mini | 4 | 256 | 4 | 11M |
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

---

## RoBERTa — 更鲁棒的 BERT

### 关键改进

1. **去掉 NSP**：NSP 其实没有帮助（甚至有害）
2. **动态掩码**：每个 epoch 使用不同的掩码
3. **更大的 batch size**：8K（vs BERT 的 256）
4. **更多的数据**：160GB（vs BERT 的 16GB）
5. **Byte-level BPE**：更大词汇量（50K → 50K+）

### 效果

在不改架构的前提下，仅通过更好的训练策略，显著超越 BERT。

---

## ALBERT — 轻量级 BERT

### 参数共享

所有层共享参数（跨层参数共享）：

- 参数量大幅减少（ALBERT-large 只有 BERT-large 的 ~1/18）
- 计算量不变（每层仍需要前向传播）

### 分解嵌入参数化

$$\\mathbf{E} = \\mathbf{W}_{V \\times E} \\quad \\rightarrow \\quad \\mathbf{E} = \\mathbf{W}_{V \\times d} \\cdot \\mathbf{W}_{d \\times H}$$

将词汇嵌入矩阵分解为两个小矩阵，减少参数。

### Sentence Order Prediction (SOP)

替代 NSP：判断两句是否被交换。比 NSP 更难（不能仅靠主题判断）。

---

## DistilBERT

通过**知识蒸馏**将 BERT-Base 压缩为 6 层（参数减半，速度提升 60%，精度保留 95%）。

## 使用

\`\`\`python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
# outputs.last_hidden_state: [B, L, 768]
# outputs.pooler_output: [B, 768] (CLS token)
\`\`\`
`;export{n as default};
