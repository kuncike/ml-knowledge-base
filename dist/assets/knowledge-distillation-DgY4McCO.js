const n=`# 知识蒸馏 (Knowledge Distillation)

## 核心思想

用一个大型的、性能好的**教师模型 (Teacher)** 指导一个小型的**学生模型 (Student)** 学习。学生不仅学习真实标签（硬标签），还学习教师输出的概率分布（软标签）。

## 蒸馏损失

$$L = (1 - \\alpha) \\cdot L_{CE}(y, \\sigma(z_s)) + \\alpha \\cdot T^2 \\cdot L_{KL}(\\sigma(z_t/T), \\sigma(z_s/T))$$

- 第一项：学生与真实标签的交叉熵
- 第二项：学生与教师软标签的 KL 散度
- $T$：温度（软化概率分布）

### 温度的作用

$$\\text{softmax}(z / T)_i = \\frac{e^{z_i/T}}{\\sum_j e^{z_j/T}}$$

- $T=1$：标准 Softmax
- $T > 1$：软化分布，露出"暗知识"（非目标类的概率包含大量信息）

例如分类 "汽车" 时，教师可能认为它有 $10^{-3}$ 概率是"卡车"（而背景是 $10^{-8}$），这个相对关系编码了类别之间的语义相似性。

## 蒸馏类型

### Logit 蒸馏 (Response-based)

仅匹配教师和学生最终输出的概率分布（上面公式）。最简单。

### 特征蒸馏 (Feature-based)

匹配中间层的特征图：

$$L_{feat} = \\|F_t - \\phi(F_s)\\|^2$$

其中 $\\phi$ 是一个适配层（学生对教师的特征维度可能不同）。

### 关系蒸馏 (Relation-based)

匹配样本间的关系（而非单个样本的表示）：

$$L_{rel} = \\| \\psi(F_t^i, F_t^j) - \\psi(F_s^i, F_s^j) \\|^2$$

$\\psi$ 可以是距离、角度或其他关系度量。

## 特殊案例

### DistilBERT

- 6 层学生（BERT-base 的一半）
- 损失：MLM + KL + Cosine Embedding Loss
- 保留 95% 的效果，速度提升 60%

### TinyBERT

- 两阶段蒸馏：通用蒸馏（预训练阶段） + 任务蒸馏（微调阶段）
- 同时蒸馏 Embedding、Attention 矩阵、Hidden States 和 Logits

### ViT 蒸馏

DeiT (Data-efficient Image Transformers)：用 CNN 教师（RegNet）蒸馏训练 ViT，大幅减少 ViT 的数据需求。

## 为什么蒸馏有效？

1. **软标签提供更多信息**：告诉学生哪些错误更"合理"
2. **正则化效果**：教师输出比 one-hot 标签更平滑
3. **迁移暗知识**：教师从大数据中学到的类别间关系
`;export{n as default};
