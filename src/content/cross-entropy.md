# Cross-Entropy Loss (交叉熵损失)

## 知识地图

```mermaid
graph LR
    InfoTheory[信息论] --> Entropy[熵 H(p)<br/>衡量分布的不确定性]
    Entropy --> KL[KL 散度<br/>衡量两个分布的差异]
    KL --> CE[交叉熵 H(p,q)<br/>KL = CE - H(p)]
    CE --> BCE[二分类交叉熵<br/>BCE / Sigmoid]
    CE --> CCE[多分类交叉熵<br/>Categorical CE / Softmax]
    CE --> LabelSmooth[标签平滑<br/>防止过拟合]
    CCE --> SoftmaxGrad[Softmax + CE 优雅梯度<br/>∂L/∂z = ŷ - y]
    BCE --> Imbalance[类别不平衡<br/>加权交叉熵 / Focal Loss]
    CCE --> NLP[NLP 语言模型<br/>下一个 token 预测]
    CCE --> CV[图像分类<br/>ImageNet 标配]

    style CE fill:#2980b9,stroke:#1a5276,color:#fff
    style SoftmaxGrad fill:#16a085,stroke:#0e6655,color:#fff
```

## 前置知识

- **信息论基础**：理解熵（不确定性度量）、KL 散度（分布差异度量）的基本概念
- **Softmax 函数**：知道 $\text{Softmax}(z_i) = e^{z_i} / \sum_j e^{z_j}$ 将 logits 转为概率分布
- **Sigmoid 函数**：知道 $\sigma(z) = 1/(1+e^{-z})$ 用于二分类概率估计
- **最大似然估计 (MLE)**：理解交叉熵最小化等价于分类问题下的最大似然估计
- **梯度下降**：理解优化器利用损失函数的梯度来更新模型参数

## 为什么会出现 (Why)

在分类问题中，最早人们尝试直接用 **MSE** 作为损失函数——预测概率和真实 one-hot 标签之间算平方误差。但实践证明 MSE 在分类任务上收敛极慢、精度也差。

根本原因有两层：

**第一层（信息论视角）：** 分类问题本质上是在比较两个概率分布——真实分布 $p$（one-hot）和预测分布 $q$（模型输出）。比较两个概率分布最自然的方式是 KL 散度。而交叉熵 = KL 散度 + 常数（$H(p)$ 对 one-hot 为 0），所以最小化交叉熵等价于最小化 KL 散度，等价于让预测分布逼近真实分布。

**第二层（梯度视角）：** Softmax + Cross-Entropy 组合产生的梯度形式极其简洁——$\frac{\partial L}{\partial z} = \hat{y} - y$（预测值减真实值）。而 Softmax + MSE 的梯度会多出一个 $\text{Softmax}'(z)$ 因子，在预测概率趋近 0 或 1 时梯度消失，导致"学不动"。

交叉熵的这两个优势（信息论正确性 + 梯度优雅性）使其成为分类任务的唯一标配，从未被挑战。

## 解决什么问题 (Problem)

| 问题 | 交叉熵如何解决 |
|------|---------------|
| **分类任务用什么 Loss** | 交叉熵是信息论上比较概率分布的唯一正确方式 |
| **MSE 在分类上梯度消失** | Softmax+CE 的梯度 = $\hat{y} - y$，永不消失 |
| **模型过度自信** | 标签平滑（Label Smoothing）防止预测概率极端化 |
| **类别不平衡** | 加权交叉熵给少数类更大的惩罚权重 |

## 核心思想 (Core Idea)

**交叉熵衡量"如果你用分布 $q$ 来编码来自分布 $p$ 的信息，平均需要多少比特"。在机器学习中，$p$ 是真实标签分布（one-hot），$q$ 是模型预测分布——最小化交叉熵等价于让预测分布尽可能接近真实分布。**

---

## 数学模型/公式

### 信息论定义

$$
H(p, q) = -\sum_{k} p_k \log q_k
$$

> **通俗解释：** 假设你有一个"编码本"（预测分布 $q$），你想用这个编码本来传输来自真实分布 $p$ 的消息。交叉熵就是使用错误编码本时，平均每条消息需要多少比特。$p_k \log q_k$ 的意思是：真实类别 $k$ 出现了（概率 $p_k$），但你用 $-\log q_k$ 个比特去编码它。如果 $q_k$ 很小（你觉得它不太可能出现），编码长度就很长——这就是"惩罚"。

其中 $p$ 是真实分布，$q$ 是预测分布。与 KL 散度的关系：

$$
\text{KL}(p \parallel q) = H(p, q) - H(p)
$$

> **通俗解释：** KL 散度 = 用错误编码本 $q$ 的代价 - 用正确编码本 $p$ 的代价，衡量的是"因为用了错误的编码本而多花的比特数"。对于 one-hot 标签，$H(p) = 0$（确定性事件的熵为 0），所以 KL = CE——两者完全等价。

对于 one-hot 标签 $H(p) = 0$，因此 $\text{KL} = \text{CE}$——最小化交叉熵等价于最小化 KL 散度。

### 二分类交叉熵 (Binary Cross-Entropy)

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]
$$

> **通俗解释：** 公式有两项，但每次只有一项生效。当 $y=1$ 时，第一项 $-\log(\hat{y})$ 生效——如果模型说概率是 0.99，惩罚很小（$-\log 0.99 \approx 0.01$）；如果模型说概率是 0.01，惩罚巨大（$-\log 0.01 \approx 4.6$）。当 $y=0$ 时第二项 $-\log(1-\hat{y})$ 生效，逻辑对称。

其中 $\hat{y}_i = \sigma(z_i)$ 是 Sigmoid 输出。当 $y=1$ 时，$L = -\log(\hat{y})$（惩罚低预测概率）；当 $y=0$ 时，$L = -\log(1-\hat{y})$（惩罚高预测概率）。

### 多分类交叉熵 (Categorical Cross-Entropy)

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})
$$

> **通俗解释：** 对于每个样本，在所有 $K$ 个类别中，只有真实类别 $k^*$ 的 $y_{ik^*} = 1$（one-hot），其他都是 0。所以这个双重求和实际上只会留下一个项：$-\log(\hat{y}_{i, k^*})$——只惩罚真实类别的预测概率。如果真实类别的预测概率高 → loss 小；低 → loss 大。

对于 one-hot 标签，简化为只惩罚正确类别的预测概率：

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \log(\hat{y}_{i, k_i^*})
$$

> **通俗解释：** 这是交叉熵最简洁的形式。假设 3 分类问题中真实类别是第 2 类，模型输出概率 $[0.1, 0.8, 0.1]$，loss = $-\log(0.8) \approx 0.22$。如果输出 $[0.1, 0.2, 0.7]$，loss = $-\log(0.2) \approx 1.61$——模型越"看不起"真实类别，惩罚越重。

### Softmax + Cross-Entropy 优雅梯度

这是 Softmax + CE 组合被广泛使用的深层原因：

$$
\frac{\partial L}{\partial z_{ik}} = \hat{y}_{ik} - y_{ik}
$$

> **通俗解释：** 梯度 = 预测概率 - 真实概率（one-hot）。这个形式极其优美——对于真实类别，如果预测概率是 0.2，梯度就是 0.2 - 1 = -0.8（方向是增大该类的 logit）；对于错误类别，如果预测概率是 0.3，梯度就是 0.3 - 0 = 0.3（方向是减小该类的 logit）。而且这个梯度永远不会消失——即使预测概率到了 0.99，梯度仍有 -0.01 继续推动优化。这与 MSE + Sigmoid 形成鲜明对比（MSE+ Sigmoid 在饱和区梯度趋近于 0）。

梯度 = 预测值 - 真实值——形式与线性回归的 MSE 梯度一致，极其简洁。

### 标签平滑 (Label Smoothing)

防止模型对训练标签过于自信：

$$
y_{ik}^{LS} = (1 - \epsilon) y_{ik} + \frac{\epsilon}{K}
$$

> **通俗解释：** 原始 one-hot 标签是 $[0, 1, 0]$（100% 确定是第 2 类）。经过 $\epsilon = 0.1$ 的标签平滑后变成 $[0.033, 0.9, 0.033]$——不再是 100% 肯定，而是"大概率是第 2 类，但也有小概率是别的"。这告诉模型：别太自信，留点余地。这能有效防止模型 overfitting 到训练标签的噪声。

等价于在交叉熵基础上增加模型输出与均匀分布的 KL 散度惩罚，缓解过拟合，提升泛化能力。

---

## 可视化展示

### BCE Loss 曲线：预测概率 vs 损失值

```echarts
return {
  xAxis: { type: 'value', min: 0.001, max: 0.999, name: '预测概率 p̂' },
  yAxis: { type: 'value', min: 0, max: 8, name: 'Loss' },
  legend: { top: 28,  data: ['y=1: -log(p̂)', 'y=0: -log(1-p̂)'] },
  series: [
    {
      name: 'y=1: -log(p̂)', type: 'line', smooth: true,
      lineStyle: { color: '#2980b9', width: 2 },
      data: (function() { const d = []; for (let i = 0.001; i <= 0.999; i += 0.001) d.push([i, -Math.log(i)]); return d; })()
    },
    {
      name: 'y=0: -log(1-p̂)', type: 'line', smooth: true,
      lineStyle: { color: '#c0392b', width: 2 },
      data: (function() { const d = []; for (let i = 0.001; i <= 0.999; i += 0.001) d.push([i, -Math.log(1 - i)]); return d; })()
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 60 }
}
```

当 $\hat{y} \to 0$ 而 $y=1$ 时，loss $\to \infty$——交叉熵对错误预测的惩罚非常严厉，这是它相对 MSE 收敛更快的核心原因。

### 标签平滑效果对比

```echarts
return {
  xAxis: { type: 'category', data: ['无标签平滑', 'ε=0.05', 'ε=0.1', 'ε=0.2'] },
  yAxis: { type: 'value', min: 0, max: 1, name: 'Top-1 Accuracy' },
  legend: { top: 28,  data: ['训练集', '验证集', '校准误差'] },
  series: [
    { name: '训练集', type: 'bar', data: [0.95, 0.93, 0.91, 0.88], itemStyle: { color: '#2c3e50' } },
    { name: '验证集', type: 'bar', data: [0.83, 0.85, 0.86, 0.84], itemStyle: { color: '#16a085' } },
    { name: '校准误差', type: 'bar', data: [0.12, 0.06, 0.04, 0.05], itemStyle: { color: '#d35400' } }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 60 }
}
```

### 交叉熵 vs MSE 在分类任务中的梯度对比

```mermaid
graph TD
    subgraph "Softmax + Cross-Entropy"
        A1[Logit z] --> A2[Softmax: ŷ]
        A2 --> A3[CE: -Σ y log ŷ]
        A3 --> A4["梯度 = ŷ - y<br/>永不消失，形式简洁"]
    end
    subgraph "Softmax + MSE"
        B1[Logit z] --> B2[Softmax: ŷ]
        B2 --> B3[MSE: (ŷ - y)²]
        B3 --> B4["梯度 = (ŷ-y)·ŷ(1-ŷ)<br/>饱和区梯度消失"]
    end

    style A4 fill:#16a085,stroke:#0e6655,color:#fff
    style B4 fill:#e74c3c,stroke:#c0392b,color:#fff
```

---

## 最小可运行代码

### PyTorch

```python
import torch
import torch.nn as nn

# 多分类 — 内置 LogSoftmax + NLLLoss（数值稳定）
ce = nn.CrossEntropyLoss(label_smoothing=0.1)

# 二分类 — 内置 Sigmoid
bce = nn.BCEWithLogitsLoss()

# 多标签分类 — 每个类别独立做二分类
mlbce = nn.BCEWithLogitsLoss()

# 不要这么做（数值不稳定）：
# loss = nn.NLLLoss()(torch.log(nn.Softmax(dim=1)(logits)), targets)
```

### NumPy 手写

```python
import numpy as np

def cross_entropy(y_true, y_pred, eps=1e-12):
    """y_true: one-hot, y_pred: probabilities"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def binary_cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

### 训练示例：Softmax + Cross-Entropy vs Softmax + MSE

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 简单三分类数据
X = torch.randn(300, 5)
y = torch.randint(0, 3, (300,))

def train_with_loss(loss_fn, name, epochs=200):
    model = nn.Linear(5, 3)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        if loss_fn == 'ce':
            loss = nn.CrossEntropyLoss()(logits, y)
        else:
            loss = nn.MSELoss()(probs, nn.functional.one_hot(y, 3).float())
        loss.backward()
        optimizer.step()
    acc = (probs.argmax(dim=1) == y).float().mean().item()
    print(f"{name}: Accuracy={acc:.3f}")

train_with_loss('ce', "Cross-Entropy")   # 预期精度高
train_with_loss('mse', "MSE")            # 预期精度低
```

---

## 工业界应用

| 应用场景 | 使用的 Loss | 为什么 | 优点 | 缺点 |
|----------|------------|--------|------|------|
| **图像分类 (ImageNet)** | Categorical Cross-Entropy + Label Smoothing | 多分类标配，Label Smoothing 提升泛化 | 收敛快，精度高 | 对噪声标签敏感 |
| **BERT / GPT 预训练** | Cross-Entropy (MLM / Next Token) | 语言建模本质是分类（词表大小 ~30K-50K） | 训练稳定 | 词表大时计算量高 |
| **二分类 (垃圾邮件 / 欺诈)** | BCEWithLogitsLoss | 输出概率可解释 | 梯度简洁 | 正负样本不平衡需加权 |
| **目标检测分类分支** | Cross-Entropy | 前景/背景 + 类别分类 | 与回归分支独立优化 | 正负样本极度不平衡需 Focal Loss |
| **多标签分类** | BCEWithLogitsLoss | 每个标签独立二分类 | 支持重叠标签 | 不考虑标签间关联 |
| **知识蒸馏** | KL 散度 (等价于 CE) | 让学生分布逼近教师分布 | 软标签含更多信息 | 需要预训练教师模型 |

---

## 优缺点对比

| 维度 | 优点 | 缺点 |
|------|------|------|
| **梯度行为** | Softmax + CE 梯度 = $\hat{y} - y$，简洁优雅，永不消失 | 对错误预测惩罚极大（loss → ∞），可能产生梯度爆炸 |
| **理论正确性** | 信息论上比较概率分布的最优选择 | 假设特征独立，不建模标签间关系 |
| **收敛速度** | 比 MSE 快得多（预测概率低时梯度极大） | 收敛太快可能预成熟收敛（premature convergence） |
| **数值稳定性** | PyTorch 的 `CrossEntropyLoss` 内置 LogSoftmax，避免 exp/log 数值问题 | 手动实现（Softmax → Log → NLL）极易出现 NaN |
| **可解释性** | 输出直接是概率，loss 单位是 nat/bit | 对模型校准要求高（需要 Label Smoothing 辅助） |

---

## 对比表格

### 交叉熵 vs MSE 在分类任务中

| 维度 | Cross-Entropy | MSE |
|------|---------------|-----|
| **信息论解释** | 概率分布间的差异（KL 散度） | 点估计间的欧氏距离 |
| **Softmax 配合梯度** | $\hat{y} - y$（极其简洁） | $(\hat{y}-y) \cdot \hat{y}(1-\hat{y})$（含饱和因子） |
| **饱和区 (ŷ→1)** | 梯度 → -0.0001（小但不为零） | 梯度 → 0（消失） |
| **错误预测惩罚** | $-\log(0.001) \approx 6.9$（极重） | $(1 - 0.001)^2 \approx 1$（温和） |
| **收敛速度** | 快 | 慢 |
| **最终精度** | 高 | 低 |
| **概率解释** | 自然的概率比较 | 不自然 |
| **业界使用** | 分类任务的唯一选择 | 基本不用 |

### 各种分类 Loss 对比

| Loss 类型 | PyTorch | 适用场景 | 关键特性 |
|-----------|---------|----------|----------|
| **CrossEntropyLoss** | `nn.CrossEntropyLoss()` | 标准多分类 | 内置 LogSoftmax + NLLLoss |
| **BCEWithLogitsLoss** | `nn.BCEWithLogitsLoss()` | 二分类 / 多标签 | 内置 Sigmoid，数值稳定 |
| **NLLLoss** | `nn.NLLLoss()` | 需手动提供 log-probability | 一般不直接用 |
| **KLDivLoss** | `nn.KLDivLoss()` | 知识蒸馏 | 输入需为 log-probability |
| **Focal Loss** | 自定义 | 极度类别不平衡 | 降低易分类样本的权重 |
| **Label Smoothing** | `CrossEntropyLoss(label_smoothing=0.1)` | 需要更好泛化和校准 | 防止过度自信 |

---

## 学完后建议继续学习

- [MSE / MAE / Huber Loss](mse-mae-huber.md) — 理解回归损失的梯度行为，对比交叉熵在分类上的优势
- [SGD / Momentum / Nesterov](sgd-momentum.md) — 交叉熵的梯度直接驱动 SGD 的优化过程
- [Adam 与 AdamW 优化器详解](adam-adamw.md) — 理解自适应优化器如何处理交叉熵在训练初期的极端梯度
- [L1 / L2 正则化](l1-l2-regularization.md) — 正则化与 Label Smoothing 都是防止过拟合的手段

---

## 高频面试题

### Q1: 为什么分类任务用交叉熵而不用 MSE？请从梯度角度详细说明。

**标准回答：** 以二分类为例，使用 Sigmoid + Cross-Entropy：

$$\frac{\partial L}{\partial z} = \sigma(z) - y = \hat{y} - y$$

梯度只依赖于预测值与真实值的差，与 Sigmoid 的导数无关。当 $\hat{y} = 0.99, y=1$ 时，梯度 = -0.01，虽然小但不为零，优化仍然持续。

而 Sigmoid + MSE：

$$\frac{\partial L}{\partial z} = 2(\hat{y} - y) \cdot \hat{y}(1 - \hat{y})$$

当 $\hat{y} \to 0$ 或 $\hat{y} \to 1$ 时，$\hat{y}(1-\hat{y}) \to 0$，梯度消失。模型在预测正确（饱和区）时几乎停止学习，无法进一步优化。

另外，MSE 的损失面是非凸的（通过 Sigmoid 后），存在大量局部最优；而 Cross-Entropy + Sigmoid/Softmax 的损失面是凸的（对于最后一层），优化容易找到全局最优。

### Q2: Softmax + CrossEntropy 为什么梯度形式是 $\hat{y} - y$？推导过程。

**标准回答：** 设 logits 为 $z_k$，Softmax 输出为 $\hat{y}_k = e^{z_k} / \sum_j e^{z_j}$，CE loss 为 $L = -\sum_k y_k \log \hat{y}_k$。

对于 Softmax 的导数（分两种情况）：

- $k = i$（对角线）：$\frac{\partial \hat{y}_k}{\partial z_k} = \hat{y}_k(1 - \hat{y}_k)$
- $k \neq i$（非对角线）：$\frac{\partial \hat{y}_i}{\partial z_k} = -\hat{y}_i \hat{y}_k$

代入链式法则：

$$\frac{\partial L}{\partial z_k} = -\sum_i y_i \frac{1}{\hat{y}_i} \frac{\partial \hat{y}_i}{\partial z_k}$$

展开后化简：利用 $\sum_i y_i = 1$（one-hot 的概率和为 1），最终得到 $\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k$。

这个优雅的形式是巧合吗？不是——这是对数似然损失配合指数族分布的一般性质（广义线性模型的 canonical link function）。

### Q3: 交叉熵和 KL 散度有什么关系？为什么最小化交叉熵等价于最小化 KL 散度？

**标准回答：** KL 散度定义：$KL(p \parallel q) = \sum_k p_k \log(p_k / q_k) = \sum_k p_k \log p_k - \sum_k p_k \log q_k = -H(p) + H(p, q)$。

所以 $H(p, q) = KL(p \parallel q) + H(p)$。

在分类任务中，$p$ 是真实标签分布（one-hot），$H(p) = 0$（因为只有一个类别的概率为 1，$1 \cdot \log 1 = 0$）。因此 $H(p, q) = KL(p \parallel q)$，两者完全等价。

即使使用标签平滑，$H(p)$ 虽然不再是 0，但它是一个常数（不依赖模型参数 $\theta$）。所以最小化 $H(p, q)$ 依然等价于最小化 $KL(p \parallel q)$——常数项不影响优化。

### Q4: 标签平滑（Label Smoothing）的原理和效果是什么？

**标准回答：** 标签平滑将 one-hot 标签 $[0, 1, 0]$ 替换为 $[0.033, 0.9, 0.033]$（$\epsilon=0.1$ 时）。数学上等价于在原始交叉熵基础上增加模型输出与均匀分布的 KL 散度惩罚。

三个核心效果：
1. **防止过度自信**：没有标签平滑时，模型会尝试让正确类别的 logit 趋近无穷大（Softmax 输出 → 1），这会导致权重范数不断增大，过拟合训练集中的噪声和错误标注。
2. **提升泛化**：模型不把 100% 概率给任何一个类，这天然起到正则化作用，尤其在小数据集上提升显著。
3. **改善模型校准**：无标签平滑时，模型输出的概率往往过于极端（置信度 0.99 但实际正确率只有 0.85）。标签平滑后，预测概率更接近实际正确率（置信度 0.9 对应实际正确率约 0.9）。

Inception-v3 论文（Szegedy et al., 2016）首次大规模使用，ImageNet 上 Top-1 提升约 0.2-0.3%。

### Q5: 对于极度类别不平衡的数据，交叉熵有什么改进方案？

**标准回答：** 三种主流方案：

**1. 加权交叉熵 (Weighted Cross-Entropy)：**
$$L = -\frac{1}{n} \sum_i w_{y_i} \log(\hat{y}_{i, y_i})$$
少数类的 $w$ 设大一些（如与类频倒数成正比）。简单直接，但只是放大了少数类的梯度，没改变梯度结构。

**2. Focal Loss (Lin et al., 2017)：**
$$L = -\frac{1}{n} \sum_i (1 - \hat{y}_{i, y_i})^\gamma \log(\hat{y}_{i, y_i})$$
当 $\gamma > 0$ 时，对于已经预测正确的样本（$\hat{y}$ 接近 1），$(1-\hat{y})^\gamma \to 0$，loss 被大幅降低。这自动让模型聚焦于难分类的样本，而难分类的样本往往是少数类。$\gamma=2$ 是常用值。

**3. OHEM (Online Hard Example Mining)：**
在每批数据中只保留 loss 最大的 k% 样本做反向传播，相当于硬性地聚焦难样本。配合 Focal Loss 的"软"聚焦，两者可以叠加使用。

这三种方案中，Focal Loss 在目标检测领域（RetinaNet）取得了最大成功，解决了前景-背景极端不平衡（~1:1000）的问题。
