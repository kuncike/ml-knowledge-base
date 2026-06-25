const n=`# RNN / LSTM / GRU / BiLSTM

## 循环神经网络 (RNN)

### 核心思想

RNN 通过**隐藏状态** $h_t$ 在时间步之间传递信息，天然适合处理序列数据。

$$h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

$$y_t = W_{hy} h_t + b_y$$

### 梯度消失/爆炸

BPTT (Backpropagation Through Time) 中对隐藏状态的梯度包含连乘：

$$\\frac{\\partial L}{\\partial h_0} \\sim \\prod_{t=1}^{T} W_{hh}^T \\cdot \\text{diag}(\\tanh')$$

- $\\lambda_{\\max} < 1$：梯度指数衰减（梯度消失）
- $\\lambda_{\\max} > 1$：梯度指数增长（梯度爆炸）

梯度爆炸用**梯度裁剪**解决；梯度消失催生了 LSTM 和 GRU。

---

## LSTM (长短期记忆网络)

### 三个门控制信息流

**遗忘门**：$f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)$

**输入门**：$i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)$

**候选记忆**：$\\tilde{C}_t = \\tanh(W_C \\cdot [h_{t-1}, x_t] + b_C)$

**更新记忆**：$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$

**输出门**：$o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)$

**隐藏状态**：$h_t = o_t \\odot \\tanh(C_t)$

$C_t$ 的设计使得梯度可以在时间维上**直接传递**（无乘门），解决了梯度消失问题。

---

## GRU (门控循环单元)

### LSTM 的简化版 — 两个门

**更新门**：$z_t = \\sigma(W_z \\cdot [h_{t-1}, x_t])$

**重置门**：$r_t = \\sigma(W_r \\cdot [h_{t-1}, x_t])$

**候选隐藏状态**：$\\tilde{h}_t = \\tanh(W \\cdot [r_t \\odot h_{t-1}, x_t])$

**新隐藏状态**：$h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t$

比 LSTM 参数少约 25%，在很多任务上效果相当。

---

## BiLSTM (双向 LSTM)

同时从前向和后向处理序列，将两个方向的隐藏状态拼接：

$$h_t^{\\text{bi}} = [\\overrightarrow{h_t}; \\overleftarrow{h_t}]$$

这在 NLP 任务中非常有效，因为一个词的上下文同时包含前文和后文。

## PyTorch 实现

\`\`\`python
import torch.nn as nn

lstm = nn.LSTM(input_size=128, hidden_size=256,
               num_layers=2, batch_first=True, bidirectional=True)

gru = nn.GRU(input_size=128, hidden_size=256,
             num_layers=2, batch_first=True)
\`\`\`

## 对比总结

| | RNN | LSTM | GRU |
|------|-----|------|-----|
| 门数量 | 0 | 3 | 2 |
| 长期依赖 | 差 | 好 | 好 |
| 参数量 | 少 | 多 | 中等 |
| 训练速度 | 快 | 慢 | 较快 |
`;export{n as default};
