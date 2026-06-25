const n=`# CLIP / ALIGN

## CLIP (Contrastive Language-Image Pre-training)

### 核心思想

CLIP 通过**对比学习**将图像和文本映射到同一向量空间，使得匹配的图文对距离近，不匹配的距离远。

### 训练方法

\`\`\`
图像 → Image Encoder (ViT/ResNet)  → I₁, I₂, ..., I_N  (N 个图像向量)
文本 → Text Encoder (Transformer)  → T₁, T₂, ..., T_N  (N 个文本向量)

相似度矩阵 S[i,j] = I_i^T · T_j / τ
目标：最大化对角线 (配对的) 相似度
\`\`\`

### 对比损失 (InfoNCE)

$$\\mathcal{L} = -\\frac{1}{2N} \\left( \\sum_{i} \\log \\frac{e^{I_i^T T_i / \\tau}}{\\sum_j e^{I_i^T T_j / \\tau}} + \\sum_{i} \\log \\frac{e^{T_i^T I_i / \\tau}}{\\sum_j e^{T_i^T I_j / \\tau}} \\right)$$

两个方向（图像→文本 和 文本→图像）的交叉熵之和。

### 数据规模

400M 图文对，从互联网收集。数据量是关键——CLIP 证明了**大规模嘈杂数据 + 简单对比目标**可以学习强大的视觉表示。

### 零样本分类

\`\`\`
1. 将所有类别名放入 prompt 模板： "A photo of {class}"
2. 用 Text Encoder 编码 → T₁, ..., T_K
3. 用 Image Encoder 编码测试图像 → I
4. 预测 = argmax(I^T · T_k)
\`\`\`

### 使用

\`\`\`python
import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(text=["a cat", "a dog"], images=image, return_tensors="pt")
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # 图文相似度
\`\`\`

---

## ALIGN

### 核心改进：噪声数据中的鲁棒学习

CLIP 需要相对干净的数据（CLIP 经过了数据清洗），ALIGN 证明**即使不清理数据**（用 1.8B 高噪声图文对），只要数据量足够大，也能学到优秀的表示。

### ALIGN vs CLIP

| | CLIP | ALIGN |
|------|------|-------|
| 数据量 | 400M | 1.8B |
| 数据清洗 | 是 | 否 |
| 核心发现 | 对比学习有效 | 数据量 > 数据质量 |
| 图像 Encoder | ViT/ResNet | EfficientNet |
`;export{n as default};
