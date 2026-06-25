const n=`# 模型剪枝 (Pruning)

## 核心思想

剪枝通过移除神经网络中不重要的权重或结构来减小模型体积和计算量，以最小的精度损失换取显著的加速和压缩。

## 剪枝的分类

### 非结构化剪枝

逐权重地将接近 0 的权重置为 0：

\`\`\`python
# 按绝对值大小剪枝
mask = (|W| > threshold)
W_pruned = W * mask
\`\`\`

- 可以达到很高的稀疏率（90%+）
- 但需要专门的稀疏矩阵硬件/库来加速（通用 GPU 加速效果有限）

### 结构化剪枝

移除整个神经元、通道或层：

\`\`\`python
# 通道剪枝：删除 L1 范数小的通道
channel_importance = W.norm(dim=[1,2,3])  # 对每个输出通道
keep_idx = topk(channel_importance, k=keep_channels)
W_pruned = W[keep_idx]
\`\`\`

- 稀疏率不如非结构化，但可以直接在标准硬件上加速
- 通道剪枝是实际部署中最常用的方法

---

## 剪枝策略

### 基于幅度的剪枝 (Magnitude-based)

最简单有效的方法：权重绝对值越小越不重要。

$$ \\text{importance} = |w_i| $$

### 迭代剪枝 vs 一次性剪枝

| 策略 | 方法 | 效果 |
|------|------|------|
| One-shot | 一次剪完 + 微调 | 较快 |
| Iterative | 剪一点 → 微调 → 重复 | 更好 |

### 基于梯度的剪枝

$$ \\text{importance} = |w_i \\cdot \\frac{\\partial L}{\\partial w_i}| $$

同时考虑权重大小和其对损失的影响。

### Lottery Ticket Hypothesis (彩票假设)

> 一个随机初始化的稠密网络包含一个子网络（"中奖彩票"），当独立训练时，可以达到与原网络相当的测试精度。

这个假设推动了从"剪枝后微调"到"直接训练稀疏子网络"的范式转变。

---

## 剪枝时机

| 时机 | 方法 | 说明 |
|------|------|------|
| 训练后 | 剪枝 → 微调 | 最常用 |
| 训练中 | 边训练边剪枝 | 逐渐增加稀疏度 |
| 训练前 | SNIP / GraSP | 初始化时就选好子网络 |

## 评估指标

- **稀疏率**：零值权重占比
- **压缩比**：原始大小 / 压缩后大小
- **加速比**：原始推理时间 / 压缩后推理时间
- **精度损失**：$\\text{Acc}_{original} - \\text{Acc}_{pruned}$

## PyTorch 剪枝

\`\`\`python
import torch.nn.utils.prune as prune

module = nn.Linear(512, 256)
# L1 非结构化剪枝 30%
prune.l1_unstructured(module, name='weight', amount=0.3)
# 结构化剪枝：移除整个神经元
prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=0)
\`\`\`
`;export{n as default};
