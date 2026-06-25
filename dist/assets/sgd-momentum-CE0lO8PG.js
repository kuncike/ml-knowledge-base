const n=`# SGD / Momentum / Nesterov

## SGD (随机梯度下降)

$$\\mathbf{w}_{t+1} = \\mathbf{w}_t - \\eta \\cdot \\nabla L(\\mathbf{w}_t)$$

- 每次迭代只用一个 mini-batch 估计梯度（而非全量数据）
- 梯度噪声有助于逃离局部最优（鞍点）
- 收敛速度受学习率 $\\eta$ 限制

## Momentum (动量)

### 物理直觉

将优化过程看作小球在损失面上滚动，动量积累之前梯度方向的速度，减少震荡：

$$\\mathbf{v}_{t+1} = \\beta \\mathbf{v}_t + \\nabla L(\\mathbf{w}_t)$$

$$\\mathbf{w}_{t+1} = \\mathbf{w}_t - \\eta \\cdot \\mathbf{v}_{t+1}$$

- $\\beta$ 通常取 0.9
- $\\mathbf{v}$ 是历史梯度的指数移动平均
- 加速在一致梯度方向上的移动，抑制震荡

## Nesterov Accelerated Gradient (NAG)

### "前瞻"一步

先按累积动量走一步，再在那一点计算梯度：

$$\\mathbf{v}_{t+1} = \\beta \\mathbf{v}_t + \\nabla L(\\mathbf{w}_t - \\eta \\beta \\mathbf{v}_t)$$

$$\\mathbf{w}_{t+1} = \\mathbf{w}_t - \\eta \\cdot \\mathbf{v}_{t+1}$$

相比标准 Momentum，Nesterov 有"前瞻矫正"能力，在凸优化中有更快的理论收敛率 $O(1/t^2)$（vs 标准动量 $O(1/t)$）。

## 学习率调度

### 阶梯衰减 (Step Decay)

每 $k$ 个 epoch 将学习率乘以 $\\gamma$（如每 30 epoch × 0.1）。

### 余弦退火 (Cosine Annealing)

$$\\eta_t = \\eta_{min} + \\frac{1}{2}(\\eta_{max} - \\eta_{min}) \\left(1 + \\cos\\left(\\frac{t}{T_{max}} \\pi\\right)\\right)$$

平滑地将学习率从最大值衰减到最小值，现代训练（ViT, GPT 等）的标配。

### 预热 (Warmup)

训练开始时线性增加学习率，避免初期不稳定：

$$\\eta_t = \\eta_{max} \\cdot \\min\\left(1, \\frac{t}{T_{warmup}}\\right)$$

### 周期重启 (Cosine Annealing with Warm Restarts)

训练过程中周期性将学习率重置为最高值，帮助跳出局部最优。

## PyTorch 实现

\`\`\`python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                      weight_decay=5e-4, nesterov=True)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# 或
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
\`\`\`
`;export{n as default};
