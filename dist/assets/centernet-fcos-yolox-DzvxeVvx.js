const n=`# CenterNet / FCOS / YOLOX (Anchor-Free 检测)

## 核心思想

传统检测器依赖 anchor boxes——需要预设大量不同尺度和比例的锚框，正负样本严重不平衡，超参数多。Anchor-Free 检测器的核心主张：**去掉 anchor，直接预测关键点或边界**。三种主流思路：CenterNet 将目标建模为热力图上的中心点（keypoint detection），FCOS 对每个位置回归到四条边的距离，YOLOX 将 YOLO 改造为 anchor-free + 解耦头 + SimOTA 匹配。

共同的数学背景：都使用 **Focal Loss 或变体** 来处理正负样本极度不平衡（一张图数十万个候选位置中只有几十个正样本）。

---

## 数学定义与原理解析

### CenterNet (Objects as Points)

将目标检测转化为中心点热力图回归：

$$\\hat{Y} \\in [0, 1]^{\\frac{H}{R} \\times \\frac{W}{R} \\times C}$$

其中 $\\hat{Y}_{xyc} = 1$ 表示 $(x,y)$ 处有类别 $c$ 的目标中心。热力图用高斯核在 GT 点周围扩散：

$$
Y_{xyc} = \\exp\\left(-\\frac{(x - \\tilde{p}_x)^2 + (y - \\tilde{p}_y)^2}{2\\sigma_p^2}\\right)
$$

同时回归宽高 + 中心点偏移。Focal Loss 用于热力图训练。

### FCOS (Fully Convolutional One-Stage)

每个位置 $(x, y)$ 预测到边界框四条边的距离 $(l, t, r, b)$：

$$
l = x - x_0, \\; t = y - y_0, \\; r = x_1 - x, \\; b = y_1 - y
$$

核心创新是 **Centerness** 分支——衡量该位置有多接近目标中心：

$$
\\text{centerness} = \\sqrt{\\frac{\\min(l, r)}{\\max(l, r)} \\times \\frac{\\min(t, b)}{\\max(t, b)}}
$$

远离中心的位置产生的低质量预测被 centerness 抑制（NMS 时乘上 centerness 分数）。

### YOLOX

关键改进：
- **Decoupled Head**：分类和回归分家（YOLO 以前共享一个 head）
- **Anchor-Free**：每格预测 1 个框（而非 3 个 anchor × 3 种比例 = 9 个）
- **SimOTA**：动态 Top-k 标签分配，根据 IoU 自动决定每个 GT 匹配哪些位置

---

## 可视化展示

### Anchor-Based vs Anchor-Free 范式对比

\`\`\`mermaid
graph TD
    subgraph Anchor["Anchor-Based (Faster R-CNN / YOLOv3)"]
        A1["预设 k 个 anchor<br/>(3 scale × 3 ratio = 9)"]
        A2["每个位置预测<br/>9 组 (Δx, Δy, Δw, Δh, obj, cls)"]
        A3["正样本: IoU > 0.7<br/>负样本: IoU < 0.3"]
    end
    subgraph Free["Anchor-Free (FCOS / CenterNet)"]
        B1["无预设 anchor"]
        B2["每个位置预测<br/>1 组 (l, t, r, b, cls) 或热力图"]
        B3["正样本: 落在 GT box 内<br/>用 centerness 过滤"]
    end
    Anchor -->|"简化"| Free
\`\`\`

### FCOS 核心流程

\`\`\`mermaid
graph TD
    FPN["FPN 多尺度特征<br/>P3~P7"] --> CLS["分类分支<br/>(C 类 + centerness)"]
    FPN --> REG["回归分支<br/>(l, t, r, b)"]
    CLS --> MUL["centerness × cls_score"]
    REG --> NMS["NMS"]
    MUL --> NMS --> OUT["检测结果"]
\`\`\`

### 检测范式对比

\`\`\`echarts
return {
  title: { text: 'Anchor-Based vs Anchor-Free', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['Anchor 数量', '正负样本比', '超参数', '小目标'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '相对得分 (越低越好/越高越好)' },
  legend: { data: ['Faster R-CNN', 'RetinaNet', 'FCOS', 'YOLOX'] },
  series: [
    { name: 'Faster R-CNN', type: 'bar', data: [0.2, 0.1, 0.3, 0.7], itemStyle: { color: '#2980b9' } },
    { name: 'RetinaNet', type: 'bar', data: [0.3, 0.2, 0.4, 0.7], itemStyle: { color: '#95a5a6' } },
    { name: 'FCOS', type: 'bar', data: [0.9, 0.7, 0.8, 0.8], itemStyle: { color: '#16a085' } },
    { name: 'YOLOX', type: 'bar', data: [0.9, 0.7, 0.9, 0.75], itemStyle: { color: '#d35400' } }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — FCOS Head

\`\`\`python
import torch
import torch.nn as nn

class FCOSHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 分类分支 (含 centerness)
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels), nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels), nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels), nn.ReLU())
        self.cls_out = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)

        # 回归分支
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels), nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels), nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels), nn.ReLU())
        self.reg_out = nn.Conv2d(in_channels, 4, 3, padding=1)  # l, t, r, b

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.cls_out.bias, -4.595)  # focal loss prior

    def forward(self, features):
        cls_feat = self.cls_conv(features)
        cls_score = self.cls_out(cls_feat)
        centerness = self.centerness(cls_feat)

        reg_feat = self.reg_conv(features)
        bbox_pred = torch.exp(self.reg_out(reg_feat))  # 确保正值

        return cls_score, centerness, bbox_pred

    def compute_centerness_target(self, l, t, r, b):
        """计算 centerness 标签: 越居中越接近 1"""
        left_right = torch.min(l, r) / torch.max(l, r)
        top_bottom = torch.min(t, b) / torch.max(t, b)
        return torch.sqrt(left_right * top_bottom)
\`\`\`
`;export{n as default};
