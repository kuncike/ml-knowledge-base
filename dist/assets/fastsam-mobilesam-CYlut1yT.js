const n=`# FastSAM / MobileSAM (轻量 SAM)

## 核心思想

SAM 效果好但太大——ViT-H 编码器 632M 参数，单张推理数秒。FastSAM 和 MobileSAM 从不同方向解决效率问题：FastSAM 完全抛弃 Transformer，用 YOLOv8 做实例分割（将 SAM 的"提示→掩码"流程转换为传统的"检测+分割"），MobileSAM 保留 SAM 架构但将 ViT-H 蒸馏为轻量网络。

结论：SAM 的知识可以被蒸馏——一个比 ViT-H 小 60 倍的编码器也能产出可比的结果。

---

## 数学定义与原理解析

### FastSAM — 两阶段流水线

**阶段 1**：YOLOv8-seg 对所有实例生成分割掩码（无提示的一阶段检测+分割）
**阶段 2**：根据用户提示（点/框）从所有提议中选择匹配的掩码

提示匹配策略：
- **Point prompt**：选择包含该点的所有掩码，按面积排序取前三
- **Box prompt**：选择与框 IoU 最大的掩码
- **Text prompt**：CLIP 嵌入 + 掩码区域 → 最高相似度

### MobileSAM — 知识蒸馏

将 SAM 的 ViT-H 编码器蒸馏为 TinyViT（仅 6M 参数）：

$$
\\mathcal{L}_{distill} = \\text{MSE}(\\mathbf{f}_{student}, \\mathbf{f}_{teacher})
$$

关键在于**解耦的蒸馏策略**——编码器和解码器独立蒸馏：
1. 训练 student 编码器近似 teacher 的特征输出
2. 冻结编码器 + 用原始数据训练 decoder

结果：MobileSAM 在单 GPU 上 1 天内训完（vs SAM 需要 68 个 V100 训 3 天）。

### SAM 2 (SAM 2.0 提示)

2024 年的 SAM 2 扩展到了视频——"segment anything in video"。核心是**记忆机制**：
- Memory Bank 存储历史帧的 prompt 和掩码预测
- 当前帧通过 Memory Attention 访问 Bank
- 支持交互式修正：在任意帧点击 → 向前向后传播

---

## 可视化展示

### FastSAM vs SAM 流程对比

\`\`\`mermaid
graph TD
    subgraph SAM["原始 SAM"]
        S1["ViT-H Encoder"] --> S2["Image Embedding"]
        S2 --> S3["Prompt Encoder<br/>(point/box/text)"]
        S3 --> S4["Mask Decoder<br/>(cross-attn)"]
        S4 --> S5["Mask Output"]
    end
    subgraph FastSAM["FastSAM"]
        F1["YOLOv8-seg"] --> F2["所有实例掩码"]
        F2 --> F3["Prompt Matcher<br/>(点筛选/框IoU)"]
        F3 --> F4["最终 Mask"]
    end
\`\`\`

### SAM 系列参数量对比

\`\`\`echarts
return {
  title: { text: 'SAM 系列参数量 vs 推理速度', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'value', name: '参数量 (M)' },
  yAxis: { type: 'value', name: '推理时间 (ms)', min: 0, max: 2000 },
  series: [
    { type: 'scatter', symbolSize: 20,
      data: [[632, 1800], [312, 900], [11, 50], [6, 35]],
      label: { show: true, formatter: (p) => ['SAM-H','SAM-B','MobileSAM','TinyViT'][p.dataIndex], position:'right' }
    }
  ],
  grid: { left: 60, right: 60, top: 50, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### FastSAM 推理

\`\`\`python
from ultralytics import FastSAM

# 自动下载模型 + 推理
model = FastSAM('FastSAM-x.pt')

# 推理一切 (prompt='everything')
results = model('image.jpg', device='mps')
masks = results[0].masks.data  # 所有实例掩码

# 提示模式
results = model('image.jpg', points=[[100, 200]], point_label=[1])
\`\`\`

### MobileSAM 蒸馏思路

\`\`\`python
import torch
import torch.nn as nn

class DistillSAMEncoder(nn.Module):
    """学生编码器 + 蒸馏损失"""
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher  # 冻结的 ViT-H
        self.student = student  # TinyViT

    def forward(self, x):
        with torch.no_grad():
            t_feat = self.teacher(x)  # [B, D, H, W]
        s_feat = self.student(x)      # [B, D, H, W]
        return s_feat, t_feat

    def distill_loss(self, x):
        s_feat, t_feat = self.forward(x)
        return nn.functional.mse_loss(s_feat, t_feat)
\`\`\`

### SAM 2 记忆注意力的概念

\`\`\`python
class MemoryAttention(nn.Module):
    """SAM 2: 当前帧通过 cross-attn 访问历史记忆"""
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, curr_frame, memory_bank):
        # curr_frame: [B, N, D]  当前帧的 token
        # memory_bank: [B, M, D]  存储的 (帧+prompt+掩码) 嵌入
        Q = self.q_proj(curr_frame)
        K = self.k_proj(memory_bank)
        V = self.v_proj(memory_bank)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / (Q.shape[-1] ** 0.5), dim=-1)
        return curr_frame + attn @ V
\`\`\`
`;export{n as default};
