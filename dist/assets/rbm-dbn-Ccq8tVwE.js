const n=`# RBM / DBN (受限玻尔兹曼机 / 深度信念网络)

## 核心思想

RBM 是能量模型最简单的形式——可见层与隐藏层之间双向连接，层内无连接（"受限"的含义）。训练通过对比散度（Contrastive Divergence）近似最大似然，核心直觉是**"让模型对真实数据的期望与模型自身生成的期望对齐"**。

DBN 将多个 RBM 堆叠起来：第一个 RBM 的隐藏层作为第二个 RBM 的可见层，层层贪婪预训练。这是深度学习复兴的关键里程碑（Hinton, 2006）——它解决了深层网络难以训练的问题，为后来的监督学习铺平了道路。

---

## 数学定义与原理解析

### RBM 能量函数

$$
E(\\mathbf{v}, \\mathbf{h}) = -\\mathbf{v}^T \\mathbf{W} \\mathbf{h} - \\mathbf{b}^T \\mathbf{v} - \\mathbf{c}^T \\mathbf{h}
$$

联合概率：$P(\\mathbf{v}, \\mathbf{h}) = \\frac{1}{Z} e^{-E(\\mathbf{v}, \\mathbf{h})}$，其中 $Z = \\sum_{\\mathbf{v},\\mathbf{h}} e^{-E(\\mathbf{v},\\mathbf{h})}$ 是配分函数。

### 条件概率（推导自能量）

由于层内无连接，条件概率独立：

$$
P(h_j = 1 | \\mathbf{v}) = \\sigma(\\mathbf{W}_{:j}^T \\mathbf{v} + c_j)
$$

$$
P(v_i = 1 | \\mathbf{h}) = \\sigma(\\mathbf{W}_{i:} \\mathbf{h} + b_i)
$$

### 对比散度 (CD-k)

直接计算 $Z$ 不可行，CD-k 用 Gibbs 采样近似：

1. 从数据 $\\mathbf{v}^{(0)}$ 出发
2. 交替更新：$\\mathbf{h}^{(t)} \\sim P(\\mathbf{h}|\\mathbf{v}^{(t)})$，$\\mathbf{v}^{(t+1)} \\sim P(\\mathbf{v}|\\mathbf{h}^{(t)})$
3. 重复 k 步（通常 k=1 就够）

权重更新（CD-1）：

$$
\\Delta \\mathbf{W} = \\mathbf{v}^{(0)} \\mathbf{h}^{(0)T} - \\mathbf{v}^{(1)} \\mathbf{h}^{(1)T}
$$

$\\mathbf{v}^{(0)}\\mathbf{h}^{(0)T}$ 是"数据驱动"的期望，$\\mathbf{v}^{(1)}\\mathbf{h}^{(1)T}$ 是"模型生成"的期望。

### DBN 贪婪预训练

1. 训练 RBM1：$(\\mathbf{x})$ → $\\mathbf{h}^{(1)}$
2. 冻结 RBM1，用 $\\mathbf{h}^{(1)}$ 训练 RBM2：$\\mathbf{h}^{(1)}$ → $\\mathbf{h}^{(2)}$
3. 重复至所需深度
4. 最后加分类层，用 BP 微调全局

---

## 可视化展示

### RBM → DBN 堆叠

\`\`\`mermaid
graph TD
    subgraph RBM1["RBM 1"]
        V0["可见层 v (输入)"] <-->|"W₁"| H1["隐藏层 h⁽¹⁾"]
    end
    subgraph RBM2["RBM 2"]
        H1B["h⁽¹⁾ (作可见层)"] <-->|"W₂"| H2["隐藏层 h⁽²⁾"]
    end
    subgraph RBM3["RBM 3"]
        H2B["h⁽²⁾ (作可见层)"] <-->|"W₃"| H3["隐藏层 h⁽³⁾"]
    end
    V0 --> H1 --> H2 --> H3
    H3 --> OUT["分类器"]
\`\`\`

### CD-1 采样过程

\`\`\`mermaid
graph LR
    V0["v⁽⁰⁾ (数据)"] -->|"P(h|v)"| H0["h⁽⁰⁾"]
    H0 -->|"P(v|h)"| V1["v⁽¹⁾ (重构)"]
    V1 -->|"P(h|v)"| H1["h⁽¹⁾"]
\`\`\`

---

## 核心代码实现

### NumPy — RBM with CD-1

\`\`\`python
import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden, lr=0.01):
        self.lr = lr
        self.W = np.random.randn(n_visible, n_hidden) * 0.01
        self.b = np.zeros(n_visible)
        self.c = np.zeros(n_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sample_hidden(self, v):
        return self.sigmoid(v @ self.W + self.c)

    def sample_visible(self, h):
        return self.sigmoid(h @ self.W.T + self.b)

    def cd1(self, v0):
        """对比散度 CD-1"""
        # 正向：数据 → 隐藏
        p_h0 = self.sample_hidden(v0)
        h0 = (np.random.rand(*p_h0.shape) < p_h0).astype(float)

        # 反向：隐藏 → 重构
        p_v1 = self.sample_visible(h0)
        p_h1 = self.sample_hidden(p_v1)

        # 梯度 = 数据期望 - 模型期望
        dW = v0.T @ p_h0 - p_v1.T @ p_h1
        db = v0.sum(axis=0) - p_v1.sum(axis=0)
        dc = p_h0.sum(axis=0) - p_h1.sum(axis=0)

        self.W += self.lr * dW / v0.shape[0]
        self.b += self.lr * db / v0.shape[0]
        self.c += self.lr * dc / v0.shape[0]

        return np.mean((v0 - p_v1) ** 2)

    def reconstruct(self, v):
        return self.sample_visible(self.sample_hidden(v))
\`\`\`

### DBN 贪婪预训练

\`\`\`python
def pretrain_dbn(rbm_list, X, epochs=10):
    """逐层预训练 DBN"""
    data = X.copy()
    for i, rbm in enumerate(rbm_list):
        print(f'Training RBM {i+1}: {rbm.W.shape}')
        for _ in range(epochs):
            rbm.cd1(data)
        # 下一层的输入 = 当前隐藏层的激活
        data = rbm.sample_hidden(data)
    return rbm_list
\`\`\`
`;export{n as default};
