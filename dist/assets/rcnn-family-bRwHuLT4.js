const n=`# R-CNN / Fast R-CNN / Faster R-CNN

## R-CNN (2014)

### 流程

1. **Selective Search** 生成约 2000 个候选区域
2. 每个区域 warp 到固定大小，送入 CNN 提取特征
3. SVM 分类 + 边界框回归

### 问题

- 每个候选区域独立跑 CNN → ~2000 次前向传播
- 训练多阶段、不端到端
- 推理极慢（~47s/图）

---

## Fast R-CNN (2015)

### 核心改进

1. **共享卷积**：整张图只过一遍 CNN 得到特征图
2. **RoI Pooling**：将任意大小的候选区域映射为固定大小的特征
3. **多任务损失**：分类 + BBox 回归联合训练

### RoI Pooling

将候选区域划分成 $H \\times W$ 网格，对每个格子做 Max Pooling。

### 损失函数

$$L = L_{cls}(p, u) + \\lambda [u \\geq 1] \\cdot L_{loc}(t^u, v)$$

- $L_{cls}$：交叉熵
- $L_{loc}$：Smooth L1 Loss
- $[u \\geq 1]$：仅对非背景类计算回归损失

速度：~3s/图（vs R-CNN 的 47s）。

---

## Faster R-CNN (2015)

### 核心创新：RPN (Region Proposal Network)

不再依赖外部的 Selective Search，用神经网络自己生成候选区域！

### RPN 结构

\`\`\`
特征图 → 3×3 Conv → 两个分支:
  ├─ 1×1 Conv → 2K 个分数 (K 个锚框各两个：前景/背景)
  └─ 1×1 Conv → 4K 个坐标偏移
\`\`\`

### Anchor 机制

每个位置预设 $K$ 个锚框（3 种尺度 × 3 种长宽比 = 9 个锚框），RPN 学习预测偏移。

**Anchor 大小**：$128^2, 256^2, 512^2$
**长宽比**：1:1, 1:2, 2:1

### 端到端训练

整个网络（RPN + Fast R-CNN）可以端到端联合训练。速度：~0.2s/图（实时！）

## R-CNN 家族技术演进

| 方法 | 候选区域 | 特征提取 | 分类器 | 速度 |
|------|----------|----------|--------|------|
| R-CNN | SS | 每区域独立 | SVM | 极慢 |
| Fast R-CNN | SS | 共享卷积 | Softmax | 慢 |
| Faster R-CNN | RPN | 共享卷积 | Softmax | 快 |

## PyTorch 使用

\`\`\`python
import torchvision.models.detection as detection

model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
predictions = model(images)
\`\`\`
`;export{n as default};
