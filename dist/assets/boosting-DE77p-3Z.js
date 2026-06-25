const n=`# Boosting — AdaBoost / GBDT / XGBoost / LightGBM / CatBoost

## 核心思想

Boosting 的哲学是**"知错就改"**：串行训练弱学习器，每个新学习器专注于修正前一个犯的错。与 Bagging（并行、降方差）相反，Boosting 的核心目标是降低**偏差**——将一系列弱学习器（如浅层决策树）逐步组合成强学习器。

---

## 数学定义与原理解析

### AdaBoost — 加权纠错

核心：错误样本获得更高权重，下一轮必须"重点关照"。

**算法流程**：

1. 初始化样本权重 $w_i = 1/n$
2. 对 $t = 1, \\ldots, T$：
   - 在加权数据上训练弱分类器 $h_t$
   - 计算加权错误率 $\\epsilon_t = \\sum_i w_i \\cdot \\mathbb{I}(y_i \\neq h_t(x_i))$
   - 计算分类器权重 $\\alpha_t = \\frac{1}{2} \\ln\\frac{1-\\epsilon_t}{\\epsilon_t}$
   - 更新样本权重 $w_i \\leftarrow w_i \\cdot \\exp(-\\alpha_t y_i h_t(x_i))$，然后归一化
3. 最终分类器：

$$
H(x) = \\text{sign}\\left(\\sum_{t=1}^{T} \\alpha_t h_t(x)\\right)
$$

**直觉**：$\\alpha_t$ 是分类器的"话语权"——错误率越低，权重越大；被错误分类的样本权重指数增长。

\`\`\`mermaid
graph LR
    D1["数据集 D<br/>权重均匀"] --> M1["弱学习器 h₁<br/>错误率 ε₁"]
    M1 -->|"更新权重<br/>错分样本 ↑ 正确样本 ↓"| D2["加权数据集"]
    D2 --> M2["弱学习器 h₂<br/>错误率 ε₂"]
    M2 -->|"更新权重"| D3["加权数据集"]
    D3 --> M3["弱学习器 h₃<br/>错误率 ε₃"]
    M1 -->|"α₁ = ½ln(1-ε₁)/ε₁"| H["最终分类器<br/>H(x) = sign(Σαₜhₜ)"]
    M2 -->|"α₂"| H
    M3 -->|"α₃"| H
\`\`\`

### GBDT — 梯度提升决策树

GBDT 用**梯度下降**视角统一了 Boosting：每轮新树拟合的是**负梯度方向**（伪残差）。

$$
r_{ti} = -\\left[ \\frac{\\partial L(y_i, F(x_i))}{\\partial F(x_i)} \\right]_{F=F_{t-1}}
$$

更新：

$$
F_t(x) = F_{t-1}(x) + \\eta \\cdot h_t(x)
$$

其中 $\\eta$（学习率，通常 0.01-0.1）是最关键的调参旋钮：$\\eta$ 越小需要越多树，但泛化更好。

### XGBoost — 工程化极致

相比 GBDT 的核心改进：

1. **二阶泰勒展开**：

$$
\\text{Obj}^{(t)} \\approx \\sum_i \\left[ g_i h_t(x_i) + \\frac{1}{2} h_i h_t^2(x_i) \\right] + \\Omega(h_t)
$$

其中 $g_i = \\partial_{\\hat{y}} L(y_i, \\hat{y}^{(t-1)})$，$h_i = \\partial^2_{\\hat{y}} L(y_i, \\hat{y}^{(t-1)})$。使用 Hessian 信息 → 收敛更快，分裂更准。

2. **正则化目标函数**：

$$
\\Omega(f) = \\gamma T + \\frac{1}{2}\\lambda \\|\\mathbf{w}\\|^2
$$

直接惩罚树的复杂度（叶子数 $T$ + 叶权重模长），有效防止过拟合。

### LightGBM — 速度优先

- **GOSS**（单边梯度采样）：保留所有大梯度样本，随机采样小梯度样本 → 大幅减少计算量
- **EFB**（互斥特征捆绑）：将互斥的稀疏特征合并，降低特征维度
- **Leaf-wise 生长**：每次分裂增益最大的叶子 → 更快收敛，但需控制 \`max_depth\` 防过拟合

### CatBoost — 类别特征专家

- **Ordered Boosting**：对每个样本用"排除该样本的历史数据"做预测 → 解决训练-测试分布偏移
- **Ordered Target Encoding**：对类别特征做无偏编码（避免传统 Target Encoding 的过拟合）

---

## 可视化展示

### Boosting 的核心：偏差逐步降低

\`\`\`echarts
return {
  xAxis: { type: 'value', min: 0, max: 100, name: '迭代轮数 T' },
  yAxis: { type: 'value', min: 0, max: 1, name: '错误率' },
  legend: { data: ['训练误差', '测试误差'] },
  series: [
    {
      name: '训练误差', type: 'line', smooth: true,
      lineStyle: { color: '#2c3e50', width: 2 },
      data: (function() { const d = []; for (let t = 0; t <= 100; t++) d.push([t, 0.4 * Math.exp(-t/20) + 0.02]); return d; })()
    },
    {
      name: '测试误差', type: 'line', smooth: true,
      lineStyle: { color: '#c0392b', width: 2 },
      data: (function() { const d = []; for (let t = 0; t <= 100; t++) d.push([t, 0.40*Math.exp(-t/15) + 0.05 + 0.0001*t]); return d; })()
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

训练误差不断下降（Boosting 可以逼近零训练误差），但 $T$ 过大测试误差可能上升（过拟合）。

---

## 框架对比总结

| 框架 | 核心改进 | 最佳场景 | 特点 |
|------|----------|----------|------|
| XGBoost | 二阶导数 + 正则化 | 通用，中小规模 | 精度最高，特征工程友好 |
| LightGBM | GOSS + EFB + Leaf-wise | 大规模高维数据 | 训练速度最快，内存占用小 |
| CatBoost | 类别特征编码 + Ordered Boosting | 大量类别特征 | 开箱即用，调参最少 |

## 核心代码实现

### XGBoost

\`\`\`python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,           # 行采样
    colsample_bytree=0.8,    # 列采样
    reg_lambda=1.0,          # L2 正则
    early_stopping_rounds=10
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
\`\`\`

### LightGBM

\`\`\`python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=100,
    num_leaves=31,           # 控制树复杂度（代替 max_depth）
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
\`\`\`
`;export{n as default};
