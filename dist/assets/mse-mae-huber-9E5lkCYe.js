const n=`# MSE / MAE / Huber Loss

## 核心思想

三条损失函数的核心区别在于**如何惩罚预测误差**。MSE 平方放大误差（对大误差零容忍），MAE 平等对待所有误差（对异常值友好），Huber 在两者之间平滑过渡——误差小用二次，误差大用线性。

---

## 数学定义与原理解析

### MSE (Mean Squared Error / L2 Loss)

$$
\\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2
$$

**梯度**：$\\frac{\\partial \\text{MSE}}{\\partial \\hat{y}_i} = \\frac{2}{n} (\\hat{y}_i - y_i)$

- 梯度与误差大小**成正比** → 大误差产生强梯度信号，收敛快
- 对大误差（异常值）极度敏感 → 模型会被异常值"牵着走"
- 最优预测是**条件均值** $E[Y|X]$

### MAE (Mean Absolute Error / L1 Loss)

$$
\\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|
$$

**梯度**：$\\frac{\\partial \\text{MAE}}{\\partial \\hat{y}_i} = \\frac{1}{n} \\cdot \\text{sign}(\\hat{y}_i - y_i)$

- 梯度恒为 $\\pm 1/n$（除完美预测外）→ 训练后期振荡，难以精细收敛
- 对异常值**鲁棒** → 所有误差一视同仁
- 最优预测是**条件中位数** $\\text{median}(Y|X)$
- 在 $\\hat{y} = y$ 处**不可导**

### Huber Loss (Smooth L1)

$$
\\text{Huber}(y, \\hat{y}) = \\begin{cases} \\frac{1}{2}(y - \\hat{y})^2 & |y - \\hat{y}| \\leq \\delta \\\\ \\delta(|y - \\hat{y}| - \\frac{1}{2}\\delta) & |y - \\hat{y}| > \\delta \\end{cases}
$$

- $|e| \\leq \\delta$：二次（平滑、梯度含大小信息）
- $|e| > \\delta$：线性（对异常值鲁棒）
- $\\delta$ 控制切换阈值，常用 $\\delta = 1.0$

---

## 可视化展示

### 三条损失函数曲线对比

\`\`\`echarts
return {
  xAxis: { type: 'value', min: -4, max: 4, name: '预测误差 e = ŷ - y' },
  yAxis: { type: 'value', min: 0, max: 8, name: 'Loss' },
  legend: { data: ['MSE (e²)', 'MAE (|e|)', 'Huber (δ=1.0)'] },
  series: [
    {
      name: 'MSE (e²)', type: 'line', smooth: true,
      lineStyle: { color: '#c0392b', width: 2 },
      data: (function() { const d = []; for (let i = -4; i <= 4; i += 0.02) d.push([i, i * i]); return d; })()
    },
    {
      name: 'MAE (|e|)', type: 'line', smooth: false,
      lineStyle: { color: '#2c3e50', width: 2 },
      data: (function() { const d = []; for (let i = -4; i <= 4; i += 0.02) d.push([i, Math.abs(i)]); return d; })()
    },
    {
      name: 'Huber (δ=1.0)', type: 'line', smooth: true,
      lineStyle: { color: '#16a085', width: 2.5 },
      data: (function() {
        const d = [];
        const delta = 1.0;
        for (let i = -4; i <= 4; i += 0.02) {
          const abs = Math.abs(i);
          d.push([i, abs <= delta ? 0.5 * i * i : delta * (abs - 0.5 * delta)]);
        }
        return d;
      })()
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

观察 MSE 曲线在 $|e| > 2$ 后急剧上升——这正是它对异常值敏感的根源。

### 异常值对最优解的影响

\`\`\`echarts
return {
  title: { text: '不同损失函数的最优解位置' },
  xAxis: { type: 'value', name: 'x' },
  yAxis: { type: 'value', name: 'Loss' },
  legend: { data: ['数据点', 'MSE最优(均值)', 'MAE最优(中位数)'] },
  series: [
    { name: '数据点', type: 'scatter', data: [[1,0],[1.2,0],[1.1,0],[0.9,0],[1.3,0],[10,0]], symbolSize: 8, itemStyle: { color: '#2c3e50' } },
    { name: 'MSE最优(均值)', type: 'line', markLine: { data: [{ xAxis: 2.58, label: { formatter: '均值=2.58' } }], silent: true, lineStyle: { color: '#c0392b' } } },
    { name: 'MAE最优(中位数)', type: 'line', markLine: { data: [{ xAxis: 1.15, label: { formatter: '中位数=1.15' } }], silent: true, lineStyle: { color: '#16a085' } } },
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

当数据中存在一个异常值（$x=10$），MSE 最优解被"拉向"异常值，而 MAE 最优解保持在中位数附近。

---

## 核心代码实现

### PyTorch

\`\`\`python
import torch.nn as nn

mse = nn.MSELoss()
mae = nn.L1Loss()
huber = nn.SmoothL1Loss(beta=1.0)  # beta 即 δ
\`\`\`

### NumPy 手写

\`\`\`python
import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def huber(y_true, y_pred, delta=1.0):
    e = np.abs(y_true - y_pred)
    return np.mean(np.where(e <= delta, 0.5 * e**2, delta * (e - 0.5 * delta)))
\`\`\`

---

## 选择指南

| 损失函数 | 对异常值 | 梯度行为 | 最优解 | 典型场景 |
|----------|----------|----------|--------|----------|
| MSE | 敏感 | 梯度 ∝ 误差大小 | 条件均值 | 常规回归，数据干净 |
| MAE | 鲁棒 | 梯度 = ±1 恒定 | 条件中位数 | 有异常值的回归 |
| Huber | 鲁棒（可调） | 平滑过渡 | 介于均值/中位数之间 | 目标检测 BBox 回归 |
`;export{n as default};
