const n=`# OpenPose / HRNet

## OpenPose

### 核心思想

OpenPose 是第一个实时多人 2D 姿态估计系统，使用 **Part Affinity Fields (PAFs)** 解决"哪个关键点属于哪个人"的关联问题。

### 双分支架构

输入图像经过 VGG-19 骨干，然后分为两个分支迭代预测：

1. **关键点热力图 (Confidence Maps)**：$J$ 个通道，每个通道对应一个关键点类型
2. **部位亲和场 (Part Affinity Fields, PAFs)**：$2C$ 个通道（每肢体的 x, y 方向向量）

### 贪婪关联算法

1. 在热力图上找关键点候选
2. 用 PAF 计算两个关键点之间连线的"亲和度"
3. 用二分图匹配构建完整骨架

### 部位亲和场

对于肢体 $c$ 上的点 $p$：

$$\\mathbf{L}_{c,k}^*(p) = \\begin{cases} \\mathbf{v} & \\text{if } p \\text{ on limb } c \\\\ 0 & \\text{otherwise} \\end{cases}$$

其中 $\\mathbf{v}$ 是肢体方向上的单位向量。

---

## HRNet (High-Resolution Network)

### 核心思想

传统姿势估计网络（如 Stacked Hourglass）走"高→低→高"分辨率路线，HRNet 则**始终保持高分辨率表示**，并通过多分辨率并行交互逐步融合信息。

### 多分辨率并行

\`\`\`
Stage 1: [1×]
Stage 2: [1×] ←→ [1/2×]
Stage 3: [1×] ←→ [1/2×] ←→ [1/4×]
Stage 4: [1×] ←→ [1/2×] ←→ [1/4×] ←→ [1/8×]
\`\`\`

各分辨率之间通过**多尺度融合模块**交换信息（上采样+下采样）。

### 优势

- 高分辨率特征保留精确的空间位置信息
- 低分辨率特征提供语义理解
- 融合过程兼顾两者

## 对比

| | OpenPose | HRNet |
|------|----------|-------|
| 方法 | 自底向上 | 自顶向下/自底向上 |
| 关键创新 | PAF | 高分辨率的保持 |
| 速度 | 快 | 较快 |
| 精度 | 中等 | 高 (COCO SOTA) |
`;export{n as default};
