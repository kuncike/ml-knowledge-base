const n=`# YOLO (You Only Look Once)

## 核心思想

YOLO 将目标检测重新定义为**回归问题**：一次前向传播直接预测边界框坐标和类别概率，实现真正的"一眼看全图"。

## YOLOv1 — 网格预测

将图像划分为 $S \\times S$ 网格（通常 $S=7$），每个网格预测 $B$ 个边界框及其置信度 + $C$ 个类别概率。

### 损失函数

$$L = \\lambda_{coord} \\sum_{i} \\mathbb{1}_{ij}^{obj} \\left[(x_i - \\hat{x}_i)^2 + (y_i - \\hat{y}_i)^2\\right]$$

$$+ \\lambda_{coord} \\sum_{i} \\mathbb{1}_{ij}^{obj} \\left[(\\sqrt{w_i} - \\sqrt{\\hat{w}_i})^2 + (\\sqrt{h_i} - \\sqrt{\\hat{h}_i})^2\\right]$$

$$+ \\sum_{i} \\mathbb{1}_{ij}^{obj} (C_i - \\hat{C}_i)^2$$

$$+ \\lambda_{noobj} \\sum_{i} \\mathbb{1}_{ij}^{noobj} (C_i - \\hat{C}_i)^2$$

$$+ \\sum_{i} \\mathbb{1}_{i}^{obj} \\sum_{c} (p_i(c) - \\hat{p}_i(c))^2$$

> 对 $w, h$ 取平方根是为了让大框和小框的偏移惩罚相近。

## YOLO 版本演进

| 版本 | 年份 | 关键改进 |
|------|------|----------|
| v1 | 2016 | 网格预测，端到端检测 |
| v2 | 2017 | Anchor 机制，BatchNorm，多尺度训练 |
| v3 | 2018 | FPN 多尺度检测，Darknet-53 骨干 |
| v4 | 2020 | CSPNet, Mish 激活, Mosaic 数据增强 |
| v5 | 2020 | 工程化（非官方），AutoAnchor |
| v7 | 2022 | E-ELAN, 模型重参数化 |
| v8 | 2023 | Anchor-Free, 解耦头, C2f 模块 |
| v11 | 2024 | 持续优化，支持多任务 |

## YOLOv8 架构要点

### Anchor-Free 检测头

不再预设 Anchor，直接预测：
- 每个位置是否为物体中心
- 从该中心到四条边的距离 (t, l, b, r)

### 解耦头 (Decoupled Head)

分类和回归分支分开（不像 v5 共享卷积）：

\`\`\`
特征 → Conv → 分类分支 → 类别概率
      → Conv → 回归分支 → bbox 偏移
\`\`\`

## YOLO vs Faster R-CNN

| | YOLO | Faster R-CNN |
|------|------|-------------|
| 检测方式 | 单阶段 | 两阶段 |
| 速度 | 极快 | 较快 |
| 小物体 | 较弱（早期版本） | 较好 |
| 精度 | 略低（早期版本） | 较高 |

## Python 使用 (Ultralytics)

\`\`\`python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # nano, s, m, l, x
results = model('image.jpg')
results[0].show()
\`\`\`
`;export{n as default};
