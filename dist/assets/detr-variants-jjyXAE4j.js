const n=`# DETR Variants (Deformable DETR / DINO)

## 核心思想

DETR 用 Transformer 彻底简化了检测流水线——没有 anchor、没有 NMS、没有 RPN，直接端到端预测集合。但原始 DETR 有两个致命弱点：**收敛慢**（500 epochs vs Faster R-CNN 的 12 epochs）和**小目标差**。Deformable DETR 用可变形注意力替代全局注意力，将复杂度从 $O(H^2W^2)$ 降到 $O(HW \\cdot K)$，收敛速度飙升。DINO 进一步引入去噪训练和对比匹配，成为 DETR 系列的集大成者。

---

## 数学定义与原理解析

### Deformable DETR — 可变形注意力

标准自注意力关注所有 $N = HW$ 个位置。可变形注意力只采样 $K$ 个学习到的参考点（$K \\ll N$）：

$$
\\text{DeformAttn}(\\mathbf{z}, \\mathbf{p}, \\mathbf{x}) = \\sum_m \\mathbf{W}_m \\left[ \\sum_k A_{mqk} \\cdot \\mathbf{W}_m' \\mathbf{x}(\\mathbf{p} + \\Delta \\mathbf{p}_{mqk}) \\right]
$$

- $\\mathbf{p}$：参考点位置（query 坐标映射到特征图）
- $\\Delta\\mathbf{p}_{mqk}$：学习到的采样偏移
- $A_{mqk}$：注意力权重

计算复杂度：$O(2N_q C^2 + N_q K C)$，与 $HW$ 线性增长。

### DINO — 对比去噪训练 (CDN)

关键创新：
1. **Contrastive DeNoising**：在 GT 框上加噪声生成"负查询"，正查询是 learnable queries，负查询被训练为"不对应任何真实目标"
2. **Mixed Query Selection**：从 encoder 输出中选择 top-K 特征初始化 decoder queries（非全零初始化）
3. **Look Forward Twice**：用本层输出更新下一层的梯度和参数

### DINO 的匹配机制

$$\\hat{\\sigma} = \\arg\\min_\\sigma \\sum_i \\mathcal{L}_{match}(y_i, \\hat{y}_{\\sigma(i)})$$

与 DETR 相同（Hungarian 匹配），但加入了 CDN 去噪的辅助损失。

---

## 可视化展示

### DETR 系列演进

\`\`\`mermaid
graph TD
    DETR["DETR (2020)<br/>Transformer + Bipartite Match<br/>❌ 收敛慢, 500 epoch"] -->
    DefDETR["Deformable DETR (2021)<br/>可变形注意力 / 多尺度<br/>✅ 收敛 50× 加速"] -->
    DN["DN-DETR (2022)<br/>去噪训练<br/>✅ 稳定性提升"] -->
    DINO["DINO (2023)<br/>对比去噪 + 混合查询<br/>✅ SOTA 性能"]
\`\`\`

### 注意力复杂度对比

\`\`\`echarts
return {
  title: { text: 'DETR vs Deformable DETR 收敛速度', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'value', name: 'Training Epochs' },
  yAxis: { type: 'value', name: 'COCO AP', min: 25, max: 50 },
  series: [
    { name: 'DETR', type: 'line', smooth: true,
      data: [[12,31],[25,37],[50,40],[100,42],[300,44],[500,45]],
      lineStyle: { color: '#95a5a6', width: 2 } },
    { name: 'Deformable DETR', type: 'line', smooth: true,
      data: [[12,44],[25,46],[50,47],[100,47.5]],
      lineStyle: { color: '#16a085', width: 2.5 } },
    { name: 'DINO', type: 'line', smooth: true,
      data: [[12,49.5],[25,50.8],[36,51.2]],
      lineStyle: { color: '#d35400', width: 2.5 } }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — Deformable Attention 简化版

\`\`\`python
import torch
import torch.nn as nn

class DeformableAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_points=4):
        super().__init__()
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, feat_map):
        """
        query: [B, Nq, C] — decoder queries
        reference_points: [B, Nq, 2] — 归一化坐标 [0,1]
        feat_map: [B, C, H, W] — encoder 输出
        """
        B, Nq, _ = query.shape
        _, C, H, W = feat_map.shape

        # 学习采样偏移和注意力权重
        offsets = self.sampling_offsets(query)      # [B, Nq, H*P*2]
        offsets = offsets.view(B, Nq, self.n_heads, self.n_points, 2)
        attn_w = self.attention_weights(query)       # [B, Nq, H*P]
        attn_w = torch.softmax(attn_w.view(B, Nq, self.n_heads, self.n_points), dim=-1)

        # 采样位置 = 参考点 + 偏移
        sample_locs = reference_points.unsqueeze(2).unsqueeze(3) + offsets
        sample_locs = sample_locs * 2 - 1  # 转到 [-1, 1] 给 grid_sample

        # 使用 grid_sample 进行稀疏采样
        # feat_map: [B, C, H, W] → [B*H, C/H, Nq, P]
        V = self.value_proj(feat_map.flatten(2).transpose(1, 2))
        V = V.view(B, Nq, self.n_heads, -1, self.head_dim)

        output = torch.zeros(B, Nq, C, device=query.device)
        # 简化: 实际需要对每个 head 和 point 做 grid_sample
        # 完整实现参考 mmdetection / deformable_attention 官方仓库

        return self.output_proj(output)
\`\`\`
`;export{n as default};
