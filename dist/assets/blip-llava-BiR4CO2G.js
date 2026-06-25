const n=`# BLIP / LLaVA / MiniGPT-4

## BLIP (Bootstrapping Language-Image Pre-training)

### 核心创新

结合**理解**和**生成**能力的视觉语言模型，并能通过自举（Bootstrapping）提升数据质量。

### 多任务架构

\`\`\`
图像 → ViT → [CLS] Token ─┬→ ITM Head (Image-Text Matching)
              Text Tokens ─┼→ LM Head (Language Modeling)
                          └→ ITC Head (Image-Text Contrastive)
\`\`\`

三种损失：

1. **ITC**：对比损失（同 CLIP）
2. **ITM**：二分类，判断图文是否匹配
3. **LM**：自回归生成，给定图像生成文本描述

### Captioning + Filtering (CapFilt)

1. **Captioner**：为网上图片生成高质量描述
2. **Filter**：过滤掉图文不匹配的噪声样本
3. 用清洗后的数据训练更优的模型

---

## BLIP-2 — Q-Former 桥接

### 核心创新：Q-Former

用轻量的 Querying Transformer 桥接**冻结的**图像编码器和**冻结的**大语言模型。

\`\`\`
图像 → 冻结的 ViT → 图像特征
                        ↓
      Learnable Queries → Q-Former (Cross-Attention)
                        ↓
      输出 → 线性投影 → 冻结的 LLM → 文本
\`\`\`

只训练 Q-Former 和投影层，两个大型模型都冻结！

---

## LLaVA

### 核心思想

将视觉信息作为 LLM 的"外语"输入。用简单的线性投影将 CLIP 图像特征映射到 LLM 的词嵌入空间。

### 训练数据

用 GPT-4（或类似强模型）为 COCO 图像生成多轮对话：

\`\`\`
Human: 图中有什么？
GPT-4 (看图): 图中有一个男人在遛狗，背景是公园...
\`\`\`

### 两阶段训练

1. **视觉-语言对齐**：冻结 LLM 和 ViT，只训练投影层
2. **指令微调**：解冻 LLM，端到端微调

---

## MiniGPT-4

### 核心思想

与 LLaVA 类似，但使用 **Q-Former + 线性投影** 桥接 ViT 和 LLM（Vicuna）。

MiniGPT-4 证明：只需训练一个投影层 + 少量微调，就能赋予 LLM 强大的视觉理解能力。

## 对比

| | BLIP-2 | LLaVA | MiniGPT-4 |
|------|--------|-------|-----------|
| 视觉-语言桥接 | Q-Former | 线性投影 | Q-Former + 投影 |
| LLM | OPT/FlanT5 | LLaMA/Vicuna | Vicuna |
| 训练效率 | 高（冻结核心模型）| 中 | 高 |
| 对话能力 | 弱 | 强 | 强 |
`;export{n as default};
