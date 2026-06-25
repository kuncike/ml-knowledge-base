const n=`# Segment Anything Model (SAM)

## 核心思想

SAM 是一个**可提示的分割基础模型**。给定一张图像和一个提示（点、框、掩膜或文本），SAM 可以为图像中的任何物体生成分割掩膜，而无需针对特定类别训练。

## 架构：三个组件

\`\`\`
Image ─→ Image Encoder (ViT) ─→ Image Embedding
                                        │
Prompt ─→ Prompt Encoder ───────────────┤
  (点/框/掩膜/文本)                      │
                                         ↓
                              Mask Decoder ─→ Masks + Scores + IoU
\`\`\`

### 1. Image Encoder

ViT-H/16（基于 MAE 预训练），一次编码可供后续多次提示复用。

### 2. Prompt Encoder

将不同类型提示编码为统一格式：
- **稀疏提示**（点、框、文本）：位置编码 + 可学习嵌入
- **稠密提示**（掩膜）：卷积编码 + 逐元素加和

### 3. Mask Decoder

轻量的 Transformer Decoder + 动态 MLP，输出多个候选掩膜和 IoU 分数。

## 歧义处理

一个提示可能对应多个有效掩膜（如衣服的领口 vs 整件衣服）。SAM 输出**3 个候选掩膜**（整体、部分、子部分），让下游任务选择。

## 数据引擎

SAM 的成功离不开其数据引擎的"三步循环"：

1. **辅助人工标注**：标注员使用 SAM 辅助标注
2. **半自动标注**：SAM 自动标注，人工确认
3. **全自动标注**：密集网格点提示，自动生成掩膜

最终数据集：**SA-1B** — 1100 万张图像，10 亿个掩膜。

## 零样本泛化

SAM 在未见过的数据分布上表现出色，是真正的"基础模型"：

- 不需要微调即可泛化到新视觉领域
- 支持多种提示模式的组合

## PyTorch 使用

\`\`\`python
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image)
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
\`\`\`

## 意义

SAM 对计算机视觉的影响类似于 GPT 对 NLP 的影响——证明了**大规模预训练 + 提示范式**在视觉领域同样可行。
`;export{n as default};
