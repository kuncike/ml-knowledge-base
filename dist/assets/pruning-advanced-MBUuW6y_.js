const n=`# Advanced Pruning (彩票假设 / 结构化剪枝 / N:M 稀疏)

## 核心思想

标准剪枝按权重大小逐元素剪——精度不错但硬件不友好（稀疏模式不规则）。**彩票假设**给出了一个震撼的解释：大网络中存在一个"中奖子网络"，如果找到它并重新训练，可以恢复全部精度。**结构化剪枝**直接剪整个 channel/filter，不改变矩阵稠密性，硬件友好。**N:M 稀疏**（如 2:4）介于结构化和非结构化之间——每组 N 个连续权重只保留 M 个最大的，Ampere GPU 原生支持。

---

## 数学定义与原理解析

### 彩票假设 (Lottery Ticket Hypothesis)

> 一个随机初始化的密集网络包含一个子网络，该子网络在**相同的初始化 + 独立训练**下，可以用更少的参数达到原网络的测试精度。

**步骤**：

1. 随机初始化网络 $f(x; \\theta_0)$
2. 训练 $t$ 步 → $\\theta_t$
3. 剪枝 $p\\%$ 的最小 $|\\theta_t|$ → 生成 mask $m$（剪掉 → $m_j=0$）
4. **关键**：将剩余权重**回退到初始值** $\\theta_0 \\odot m$，重新训练

LTH 的回退步骤（rewinding）保证了"中奖票"被发现——子网络在初始状态就具备学习能力，而不只是训练的副产品。

### 结构化剪枝

剪掉整个通道而非单个权重：

**Filter Pruning**：剪掉第 $l$ 层卷积的一个 filter = 同时剪掉该层的输出通道和第 $l+1$ 层的对应输入通道。前后层的参数同时减少。

**Channel Pruning**：先评估通道重要性（如 L1 范数、BN 的 $\\gamma$ 值），剪掉最不重要的，微调恢复精度。

### N:M 稀疏 (2:4 Sparsity)

NVIDIA Ampere 的 Sparse Tensor Core 要求：每组 4 个连续值中恰好有 2 个非零。实现：

1. 将权重按 4 个一组分割
2. 每组保留最大的 2 个值，其余置零
3. 用 **permutation** 技巧找到最优的分组排列，最小化精度损失

FLOPs 减半，但**必须使用特定排列**（不是任意 50% 稀疏度都能加速）。

### 移动剪枝 (Movement Pruning)

标准 magnitude pruning 只看权重的大小。Movement pruning 将重要性定义为"权重在训练中的变化方向"：

$$
\\text{score}(w) = |w \\cdot \\nabla_w \\mathcal{L}|
$$

如果一个权重很小但在快速增长，它可能是重要的。这比纯 magnitude 更准确——特别是在微调场景下。

---

## 可视化展示

### 三种剪枝模式

\`\`\`mermaid
graph TD
    subgraph Unstruct["非结构化 (Fine-grained)"]
        U_MATRIX["不规则稀疏矩阵<br/>❌ 无硬件加速<br/>✅ 最佳精度"]
    end
    subgraph Struct["结构化 (Coarse-grained)"]
        S_MATRIX["整行/整列去除<br/>✅ 标准矩阵乘法<br/>❌ 精度损失大"]
    end
    subgraph NM["N:M 稀疏 (2:4)"]
        N_MATRIX["每组4保留2<br/>✅ Ampere Sparse Tensor Core<br/>✅ 精度接近非结构化"]
    end
\`\`\`

### 剪枝率 vs 精度

\`\`\`echarts
return {
  title: { text: 'ResNet-50 不同剪枝策略对比 (ImageNet)', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'value', name: '剪枝率 (%)' },
  yAxis: { type: 'value', name: 'Top-1 Accuracy Drop (%)', min: 0, max: 10 },
  legend: { data: ['Magnitude (非结构)', 'Channel Pruning', 'N:M (2:4)'] },
  series: [
    { name: 'Magnitude (非结构)', type: 'line', smooth: true,
      data: [[30,0.5],[50,1.0],[70,2.2],[90,5.0]],
      lineStyle: { color: '#16a085', width: 2.5 } },
    { name: 'Channel Pruning', type: 'line', smooth: true,
      data: [[30,1.5],[50,3.5],[70,7.0]],
      lineStyle: { color: '#c0392b', width: 2.5 } },
    { name: 'N:M (2:4)', type: 'line', smooth: true,
      data: [[50,0.8],[50,0.8]],
      itemStyle: { color: '#2980b9' }, lineStyle: { color: '#2980b9', width: 2.5, type: 'dashed' },
      markPoint: { data: [{ coord: [50, 0.8], label: { formatter: '固定50%' } }] } }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — 全局 Magnitude 剪枝

\`\`\`python
import torch
import torch.nn.utils.prune as prune

# 非结构化全局剪枝 (不推荐, 仅示例)
parameters_to_prune = [(model.conv1, 'weight'), (model.fc, 'weight')]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5)

# 结构化剪枝: 剪掉整个通道 (基于 L2 范数)
def channel_prune(conv_layer, prune_ratio=0.3):
    weight = conv_layer.weight.data  # [out_c, in_c, k, k]
    l2_norm = weight.norm(p=2, dim=(1, 2, 3))  # [out_c]
    n_keep = int(weight.shape[0] * (1 - prune_ratio))
    keep_idx = torch.topk(l2_norm, n_keep).indices
    return keep_idx
\`\`\`

### Lottery Ticket 寻找

\`\`\`python
def find_lottery_ticket(model, train_fn, pruner, sparsity=0.8, rewinding_steps=500):
    """寻找中奖票"""
    init_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 1. 训练 → 剪枝 → 迭代
    for iteration in range(10):
        train_fn(model, steps=rewinding_steps)
        mask = pruner.prune(model, sparsity)
        # 回退到初始状态, 但保留 mask
        model.load_state_dict({k: init_state[k] * mask[k] for k in init_state})

    return model, mask
\`\`\`

### N:M (2:4) 稀疏

\`\`\`python
def apply_nm_sparsity(weight, n=2, m=4):
    """每组 m 个连续权重中保留最大的 n 个"""
    out_features = weight.shape[0]
    weight_abs = weight.abs()
    for i in range(0, weight.shape[1], m):
        if i + m > weight.shape[1]:
            break
        block = weight_abs[:, i:i+m]
        # 每行找到第 n 大的值作为阈值
        threshold = torch.topk(block, n, dim=1).values[:, -1:]
        mask = block >= threshold
        weight[:, i:i+m] *= mask
    return weight

# 使用 PyTorch 2.0+ 原生 N:M 稀疏
# torch.sparse.semi_structured_tensor.to_sparse_semi_structured(weight)
\`\`\`
`;export{n as default};
