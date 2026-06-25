const n=`# SSD / RetinaNet / CenterNet

## SSD (Single Shot MultiBox Detector)

### 核心思想

在多个不同分辨率的特征图上同时做检测，高分辨率图检测小物体，低分辨率图检测大物体。

### 默认框 (Default Boxes)

每个特征图位置预设不同尺度和长宽比的默认框：

$$s_k = s_{min} + \\frac{s_{max} - s_{min}}{m - 1}(k - 1), \\quad k \\in [1, m]$$

### 损失函数

$$L = \\frac{1}{N}(L_{conf} + \\alpha L_{loc})$$

- $L_{conf}$：交叉熵（正负样本 1:3 难负样本挖掘）
- $L_{loc}$：Smooth L1

---

## RetinaNet

### 核心创新：Focal Loss

解决单阶段检测器的核心问题——**极端的前景-背景类别不平衡**。

$$\\text{FL}(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)$$

- $\\gamma = 2, \\alpha = 0.25$ 是最佳参数
- 简单负样本（$p_t$ 接近 1）的损失被大幅缩减
- 困难样本保持较高损失

### 架构

\`\`\`
ResNet + FPN → 两个子网络:
  ├─ 分类子网络 (4 × Conv + 输出 K×A 个类别)
  └─ 回归子网络 (4 × Conv + 输出 4×A 个偏移)
\`\`\`

两个子网络**不共享参数**（与 SSD 不同）。

---

## CenterNet

### 核心思想

将目标表示为**边界框的中心点**，用热力图预测中心位置，然后回归宽高。是一种 Anchor-Free 方法。

### 三个分支

1. **Heatmap Head**：预测每个位置是中心点的概率
2. **Offset Head**：补偿下采样带来的量化误差
3. **Size Head**：预测 $(w, h)$

### 中心点热力图的制作

以真实框的中心为高斯核中心生成目标热力图：

$$Y_{xyc} = \\exp\\left(-\\frac{(x - \\tilde{p}_x)^2 + (y - \\tilde{p}_y)^2}{2\\sigma_p^2}\\right)$$

其中 $\\sigma_p$ 与物体大小相关。

### 损失函数

$$L = L_k + \\lambda_{size} L_{size} + \\lambda_{off} L_{off}$$

- $L_k$：Focal Loss（关键点热力图）
- $L_{size}$：L1 Loss（宽高回归）
- $L_{off}$：L1 Loss（偏移补偿）

## 对比总结

| 方法 | 检测范式 | 特征 | 核心创新 |
|------|----------|------|----------|
| SSD | 密集 Anchor | 多尺度特征图 | 单次前向 |
| RetinaNet | 密集 Anchor | FPN + Focal Loss | 解决类别不平衡 |
| CenterNet | Anchor-Free | 中心点 + 宽高 | 极简设计 |
`;export{n as default};
