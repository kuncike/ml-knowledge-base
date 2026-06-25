const n=`# ViT Variants (DeiT / PVT / CaiT)

## 核心思想

ViT 的痛点：需要 JFT-300M 级别的数据才能训好，小数据集上不如 ResNet。DeiT 用**知识蒸馏**解决了这个问题——让教师 CNN 指导 ViT 训练，在 ImageNet-1K 上也能达到好效果。PVT (Pyramid Vision Transformer) 引入 CNN 风格的**金字塔结构**（多尺度特征图），让 ViT 可用于密集预测任务（检测/分割）。CaiT 则发现在深层用 Class Attention 替代 Self-Attention 能提升效果——前面的层负责提取特征，最后几层才让 CLS token 做决策。

---

## 数学定义与原理解析

### DeiT 蒸馏

除了 class token，DeiT 还引入一个 **distillation token**——它也参与注意力计算，但最终由教师模型的预测来监督：

$$
\\mathcal{L}_{total} = \\mathcal{L}_{CE}(y_{cls}, y) + \\mathcal{L}_{CE}(y_{distill}, y_{teacher})
$$

其中 $y_{cls}$ 是 class token 的输出，$y_{distill}$ 是 distillation token 的输出。

### PVT — 空间缩减注意力 (SRA)

标准 MSA 的 $O(n^2)$ 在高分辨率特征图上不可行。PVT 引入 SRA——将 K 和 V 下采样 $R$ 倍后再做注意力：

$$
\\text{SRA}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{Concat}(\\text{head}_1, \\dots) \\mathbf{W}^O
$$

$$
\\text{head}_i = \\text{Attention}(\\mathbf{Q}\\mathbf{W}_i^Q, \\text{SR}(\\mathbf{K})\\mathbf{W}_i^K, \\text{SR}(\\mathbf{V})\\mathbf{W}_i^V)
$$

SR 是用 stride=R 的平均池化（或卷积）。注意力矩阵从 $N \\times N$ 降为 $N \\times (N/R^2)$。

### CaiT 的 Class Attention

CaiT 分两阶段：
- **Self-Attention 阶段**（前几层）：标准 ViT 的自注意力，CLS token 和别人一起更新
- **Class Attention 阶段**（后 2 层）：**冻结 patch tokens**，只用 CLS token 作为 Query 去 attend patches

\`\`\`python
# Class Attention: Q 只来自 CLS token, K/V 来自 patches + CLS
Q = W_q(cls_token)          # [1, D]
K = W_k(torch.cat([cls, patches]))  # [N+1, D]
V = W_v(torch.cat([cls, patches]))  # [N+1, D]
cls_new = Attention(Q, K, V)  # 只更新 CLS
\`\`\`

---

## 可视化展示

### DeiT 架构

\`\`\`mermaid
graph TD
    IMG["图像 Patches"] --> ENC["Patch Embedding"]
    ENC --> PE["+ Position Embedding"]
    PE --> CONCAT["Concat [CLS token, Distill token, Patches]"]
    CONCAT --> TRANS["Transformer × L"]
    TRANS --> CLS_OUT["CLS → Class Prediction"]
    TRANS --> DIST_OUT["Distill → 教师软标签匹配"]
\`\`\`

### 金字塔 ViT vs 标准 ViT

\`\`\`echarts
return {
  title: { text: 'ViT 架构家族对比', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['ViT', 'DeiT', 'PVT', 'CaiT', 'Swin'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '相对能力' },
  legend: { data: ['小数据性能', '密集预测', '可扩展性'] },
  series: [
    { name: '小数据性能', type: 'bar', data: [0.2, 0.75, 0.3, 0.4, 0.8], itemStyle: { color: '#2c3e50' } },
    { name: '密集预测', type: 'bar', data: [0.1, 0.1, 0.9, 0.1, 0.9], itemStyle: { color: '#16a085' } },
    { name: '可扩展性', type: 'bar', data: [0.8, 0.7, 0.7, 0.75, 0.9], itemStyle: { color: '#2980b9' } }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — DeiT 蒸馏前向传播

\`\`\`python
import torch
import torch.nn as nn

class DistilledTransformer(nn.Module):
    def __init__(self, embed_dim=384, num_classes=1000, depth=12, num_heads=6):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.distill_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 197, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=4*embed_dim)
            for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_distill = nn.Linear(embed_dim, num_classes)

    def forward(self, patches):
        # patches: [B, N, D], N = 196 (14×14 patches)
        B = patches.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        distill_token = self.distill_token.expand(B, -1, -1)
        x = torch.cat([cls_token, distill_token, patches], dim=1)  # [B, 198, D]
        x = x + self.pos_embed[:, :x.shape[1]]

        for blk in self.blocks:
            x = blk(x)

        cls_out = self.head(x[:, 0])        # CLS 分类
        distill_out = self.head_distill(x[:, 1])  # Distill 分类
        return cls_out, distill_out
\`\`\`

### PyTorch — Class Attention (CaiT 风格)

\`\`\`python
class ClassAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, N+1, D], 第0位是 CLS token
        cls_token = x[:, :1]
        patches = x[:, 1:]

        Q = self.q(cls_token)  # [B, 1, D]
        K = self.k(x)          # [B, N+1, D]
        V = self.v(x)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        cls_new = attn @ V
        cls_new = self.proj(cls_new)

        return torch.cat([cls_new, patches], dim=1)  # patches 冻结不动
\`\`\`
`;export{n as default};
