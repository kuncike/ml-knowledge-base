const n=`# Dropout

## 核心思想

训练时以概率 $p$ 随机"杀死"神经元，迫使每个神经元学会独立提取有用特征，而非依赖特定同伴——这本质上是**训练 $2^n$ 个子网络的隐式集成**，以极低的计算代价实现强大的正则化效果。

---

## 数学定义与原理解析

### 标准 Dropout

训练时：

$$
h' = h \\odot \\mathbf{m}, \\quad m_i \\sim \\text{Bernoulli}(1-p)
$$

其中 $\\mathbf{m}$ 是随机二值 mask。测试时所有神经元激活，但需乘 $(1-p)$ 以保持期望一致：

$$
\\mathbb{E}[h'_{\\text{test}}] = (1-p) \\cdot h = \\mathbb{E}[h'_{\\text{train}}]
$$

### Inverted Dropout（PyTorch 默认方式）

训练时缩放，测试时原样输出——更优雅：

$$
h'_{\\text{train}} = \\frac{1}{1-p} \\cdot h \\odot \\mathbf{m}, \\quad h'_{\\text{test}} = h
$$

这样测试阶段无需任何额外操作（"inverted" 指将缩放从测试阶段移到了训练阶段）。

### 为什么有效？三个视角

1. **集成学习视角**：每次前向传播随机丢弃不同神经元 = 训练不同的子网络。$n$ 个神经元对应 $2^n$ 个可能的子网络，测试时近似为它们的集成
2. **共适应（Co-adaptation）抑制**：神经元不能依赖特定"搭档"，必须各自独立学习鲁棒特征
3. **噪声注入**：Dropout 等效于在激活上乘以乘性伯努利噪声，类似数据增强在特征空间的作用

---

## 可视化展示

### Dropout 在前向传播中的随机丢弃示意

\`\`\`mermaid
graph TD
    subgraph "前一层 (4 个神经元)"
        N1[●]; N2[●]; N3[●]; N4[●]
    end
    subgraph "Dropout Mask (p=0.5)"
        M1[✓]; M2[✗]; M3[✓]; M4[✗]
    end
    subgraph "后一层 (保留 2 个)"
        O1[●]; O2[○灰色]; O3[●]; O4[○灰色]
    end
    N1 --> M1 --> O1
    N2 --> M2 -.-> O2
    N3 --> M3 --> O3
    N4 --> M4 -.-> O4
\`\`\`

### Dropout 概率与子网络数量关系

\`\`\`echarts
return {
  xAxis: { type: 'value', min: 0, max: 1, name: 'Dropout 概率 p' },
  yAxis: { type: 'value', name: '有效子网络比例' },
  series: [{
    type: 'line',
    smooth: true,
    data: (function() {
      const d = [];
      for (let p = 0; p <= 1; p += 0.01) d.push([p, Math.pow(1 - p, 10)]);
      return d;
    })(),
    lineStyle: { color: '#c0392b', width: 2 },
    areaStyle: { color: 'rgba(192, 57, 43, 0.1)' }
  }],
  tooltip: { trigger: 'axis', formatter: 'p = {b}<br/>子网络比例: {c}' },
  grid: { left: 60, right: 20, top: 20, bottom: 50 }
}
\`\`\`

$n=10$ 个神经元的层中，不同 $p$ 值下活跃子网络的比例。$p$ 越大，集成的子网络越多，但每个子网络容量越小。

---

## 核心代码实现

### PyTorch

\`\`\`python
import torch.nn as nn

# 全连接层后的标准 Dropout
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),   # 全连接层常用 p=0.5
)

# CNN 用较小的 Dropout
nn.Dropout2d(p=0.1)      # 丢弃整个通道（空间 Dropout）

# Transformer 中的 Dropout（p=0.1 远小于 CNN）
nn.Dropout(p=0.1)
\`\`\`

### DropPath (Stochastic Depth)

ViT、Swin Transformer 中使用——丢弃整个残差块而非单个神经元：

\`\`\`python
import torch

def drop_path(x, drop_prob, training):
    if not training or drop_prob == 0.:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x / keep_prob * random_tensor
\`\`\`

---

## Dropout 变体对比

| 变体 | 粒度 | 使用场景 |
|------|------|----------|
| Standard Dropout | 单个神经元 | 全连接层 |
| Spatial Dropout (Dropout2d) | 整个通道 | CNN |
| DropPath (Stochastic Depth) | 整个残差块 | ViT, Swin |
| DropConnect | 权重（而非激活） | 理论研究 |
| Variational Dropout | RNN 时间步共享 mask | RNN/LSTM |

## 注意事项

- **必须切换模式**：\`model.train()\` 和 \`model.eval()\` 的行为不同
- **BN + Dropout 的方差偏移**：两者同时使用可能导致训练和测试的统计量不一致（"variance shift"），一般避免在相邻层同时使用
`;export{n as default};
