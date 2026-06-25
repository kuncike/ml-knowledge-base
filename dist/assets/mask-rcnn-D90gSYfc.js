const n=`# Mask R-CNN

## 核心思想

在 Faster R-CNN 的基础上增加一个**并行的分割分支**，为每个 RoI 预测一个二值掩膜（Mask），实现**实例分割**（区分同一类别中的不同个体）。

## 架构

\`\`\`
图像 → CNN + FPN → RPN → RoIAlign → 三个并行分支:
  ├─ 分类: [K+1] 类概率
  ├─ 回归: 4K 个 bbox 偏移
  └─ 分割: K × m × m 的掩膜
\`\`\`

## RoIAlign — 核心创新

### RoI Pooling 的问题

两次量化导致 misalignment：
1. RoI 坐标取整（浮点 → 整数）
2. 子区域划分取整

### RoIAlign 解决方案

使用**双线性插值**在连续坐标上采样：

\`\`\`python
def roi_align(feature_map, roi, output_size):
    # 在规则采样点上做双线性插值
    for y, x in sampling_points:
        value = bilinear_interpolate(feature_map, y, x)
\`\`\`

这使得掩膜精度大幅提升（从 10% 到 50% mAP mask）。

## 多任务损失

$$L = L_{cls} + L_{box} + L_{mask}$$

### 掩膜分支的特殊处理

- 输出 $K$ 个掩膜（每个类一个），但只使用 Ground Truth 类的掩膜计算损失
- 每个掩膜用 Sigmoid + Binary Cross-Entropy（而非 Softmax）
- **类别解耦**：分类和分割独立，避免类间竞争

## 关键设计细节

1. **FPN + Mask R-CNN**：多尺度特征显著提升分割质量
2. **分割分支是 FC 还是 FCN**：FCN（小型全卷积网络）效果更好
3. **掩膜分辨率**：$28 \\times 28$（v1），可调整

## PyTorch 使用

\`\`\`python
import torchvision.models.detection as detection

model = detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
predictions = model(images)
# predictions contain: boxes, labels, scores, masks
\`\`\`

## 技术演进

| 方法 | 任务 | 创新 |
|------|------|------|
| R-CNN | 检测 | CNN 特征 |
| Fast R-CNN | 检测 | RoI Pooling |
| Faster R-CNN | 检测 | RPN |
| Mask R-CNN | 实例分割 | RoIAlign + Mask Head |
| Cascade Mask R-CNN | 实例分割 | 级联 IOU 阈值 |
| HTC | 实例分割 | 混合任务级联 |
`;export{n as default};
