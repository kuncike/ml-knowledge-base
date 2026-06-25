const a=`# L1 / L2 正则化

## 核心思想

正则化通过在损失函数中增加惩罚项来限制模型复杂度，防止过拟合。

### L2 正则化 (Weight Decay / Ridge)

$$L_{total} = L_{data} + \\frac{\\lambda}{2} \\|\\mathbf{w}\\|^2$$

梯度更新：

$$\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta \\frac{\\partial L_{data}}{\\partial \\mathbf{w}} - \\eta \\lambda \\mathbf{w} = (1 - \\eta \\lambda) \\mathbf{w} - \\eta \\frac{\\partial L_{data}}{\\partial \\mathbf{w}}$$

每次更新权重会**衰减**一定比例（weight decay 名称的由来）。

### L1 正则化 (Lasso)

$$L_{total} = L_{data} + \\lambda \\|\\mathbf{w}\\|_1$$

产生**稀疏解**（很多权重为 0），自动特征选择。

## 几何直觉

在二维权空间中：

- L1 约束区域是菱形（有尖角）→ 最优解常落在坐标轴上 → 稀疏
- L2 约束区域是圆形 → 最优解通常不在坐标轴上 → 非稀疏

## PyTorch 中的 Weight Decay

\`\`\`python
import torch.optim as optim

# AdamW — 解耦 weight decay（推荐）
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# SGD + weight decay
optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-4)
\`\`\`

### Adam vs AdamW

Adam 原始实现将 weight decay 混入了自适应学习率中，效果不如显式解耦。AdamW 将 weight decay 与梯度更新解耦：

$$\\mathbf{w} \\leftarrow (1 - \\eta \\lambda) \\mathbf{w} - \\eta \\cdot \\text{AdamUpdate}$$

## 经验法则

| 模型类型 | Weight Decay 参考值 |
|----------|---------------------|
| 一般分类 | 1e-4 ~ 5e-4 |
| Transformer (GPT/LLaMA) | 0.1 (较大!) |
| ViT | 0.05 ~ 0.1 |
| ResNet | 1e-4 ~ 5e-4 |

## 早停 (Early Stopping)

早停可以看作正则化的一种形式。其 L2 正则化等价性：在二次近似下，早停在梯度下降路径上的某个点停止 ≈ 选定一个特定 $\\lambda$ 的 L2 正则化。
`;export{a as default};
