const n=`# T5 / BART / ELECTRA

## T5 (Text-to-Text Transfer Transformer)

### 核心思想：万物皆文本到文本

将所有 NLP 任务统一为 **Text-to-Text 格式**：

\`\`\`
翻译:    "translate English to German: Hello" → "Hallo"
摘要:    "summarize: ..." → "摘要文本"
分类:    "mnli premise: ... hypothesis: ..." → "entailment"
问答:    "question: ... context: ..." → "答案"
\`\`\`

### 预训练目标：Span Corruption

随机掩盖连续的 token span（而非单 token），用一个特殊 token 替换：

\`\`\`
Input:  "Thank you [X] me to your party [Y] week"
Output: "[X] for inviting [Y] last [Z]"  (Z 表示结束)
\`\`\`

平均 span 长度 = 3，掩盖 15% token。

### 架构

标准 Transformer Encoder-Decoder：

| 配置 | 层数 | $d_{model}$ | 头数 | 参数量 |
|------|------|-------------|------|--------|
| Small | 6+6 | 512 | 8 | 60M |
| Base | 12+12 | 768 | 12 | 220M |
| Large | 24+24 | 1024 | 16 | 770M |
| 3B | 24+24 | 1024 | 32 | 2.8B |
| 11B | 24+24 | 1024 | 128 | 11B |

---

## BART

### 核心思想

BART 是结合 BERT（编码器）和 GPT（解码器）的序列到序列模型。关键在于**破坏输入的多种噪声方式**：

1. **Token Masking**：像 BERT 一样随机替换 token
2. **Token Deletion**：随机删除 token
3. **Text Infilling**：用单个 [MASK] 替换连续 span
4. **Sentence Permutation**：打乱句子顺序
5. **Document Rotation**：随机旋转文档

模型学习从损坏文本中重建原始文本。

---

## ELECTRA

### 核心思想：效率优先

不是预测被掩盖的 token（MLM），而是训练一个**判别器**判断每个 token 是否被替换：

1. 用一个小的 MLM Generator 替换部分 token
2. 用大的 Discriminator 判断每个 token 是否被替换
3. 对所有 token 做二分类（而非仅 15%）

所有 token 都参与损失计算，效率远高于 MLM。

### 损失函数

$$L = L_{MLM}(G) + \\lambda L_{Disc}(D)$$

Generator 的损失不反向传播到 Discriminator。

参数量与 BERT 相同的情况下，ELECTRA 效果更好（或同等效果更快）。

## 对比总结

| 模型 | 架构 | 预训练目标 | 特点 |
|------|------|-----------|------|
| BERT | Encoder | MLM + NSP | 理解任务 |
| GPT | Decoder | LM | 生成任务 |
| T5 | Enc-Dec | Span Corruption | 统一文本任务 |
| BART | Enc-Dec | 多种噪声重建 | 生成+理解 |
| ELECTRA | Encoder | RTD | 训练效率高 |
`;export{n as default};
