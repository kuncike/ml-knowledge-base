const n=`# Adam / AdamW

## Adam (Adaptive Moment Estimation)

### 核心思想

Adam 结合了 Momentum（一阶矩）和 RMSprop（二阶矩），为每个参数自适应地调整学习率。

### 算法

**一阶矩**（梯度均值）：

$$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$$

**二阶矩**（梯度未中心化方差）：

$$v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$$

**偏差校正**（初始时 $m_0=v_0=0$ 导致估计偏低）：

$$\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$$

**参数更新**：

$$\\theta_t = \\theta_{t-1} - \\eta \\cdot \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$$

### 默认超参数

| 参数 | 值 | 含义 |
|------|-----|------|
| $\\eta$ | 1e-3 | 学习率 |
| $\\beta_1$ | 0.9 | 一阶矩衰减 |
| $\\beta_2$ | 0.999 | 二阶矩衰减 |
| $\\epsilon$ | 1e-8 | 数值稳定 |

---

## AdamW

### Adam 的 Weight Decay 问题

Adam 原始论文中将 L2 正则化（weight decay）加入梯度中，但这在自适应学习率下效果等同于"学习率相关"的正则化，不是真正的 weight decay。

### 解耦 Weight Decay

AdamW 将 weight decay 从梯度更新中**解耦**出来：

$$\\theta_t = \\theta_{t-1} - \\eta \\cdot \\left( \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda \\theta_{t-1} \\right)$$

这样 weight decay 的强度不再受自适应学习率的影响。这是现代 Transformer 训练的标配。

## 其他自适应优化器对比

| 优化器 | 核心思路 | 适用场景 |
|--------|----------|----------|
| AdaGrad | 累积平方梯度，学习率递减 | 稀疏特征 |
| RMSprop | 滑动平均替代累积 | RNN |
| Adam | 一阶 + 二阶矩 | 通用首选 |
| AdamW | Adam + 解耦 WD | Transformer |
| LAMB | Layer-wise 自适应 | 大 batch 训练 |

## PyTorch 使用

\`\`\`python
import torch.optim as optim

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
\`\`\`

## Adam 的局限性

- 泛化性能有时不如带有良好调参的 SGD + Momentum
- 对 $\\beta_2$ 敏感（太大导致学习率被"固定在"早期梯度）
- 某些任务中，自适应学习率的"遗忘"特性可能导致收敛到较差的解
`;export{n as default};
