const n=`# Decision Tree (决策树)

## 核心思想

决策树通过**递归地选择最优特征进行二分（或多分）**，将数据不断划分为纯度更高的子集。如果把分类过程想象成 20 Questions 游戏，决策树就是学到的"最佳提问顺序"——每次都在当前数据上问一个最"一针见血"的问题。

---

## 数学定义与原理解析

### 三种经典算法

| 算法 | 年份 | 分裂准则 | 树类型 | 特征支持 |
|------|------|----------|--------|----------|
| ID3 | 1986 | 信息增益 | 多叉 | 仅离散 |
| C4.5 | 1993 | 信息增益比 | 多叉 | 离散+连续 |
| CART | 1984 | Gini (分类) / MSE (回归) | **二叉** | 离散+连续 |

现代主流是 CART（二叉、高效、可用 Gini 和 MSE）。

### 熵与信息增益 (ID3)

**熵**衡量数据的不确定性：

$$
H(D) = -\\sum_{k=1}^{K} p_k \\log_2 p_k
$$

- 纯数据集（所有样本同类）：$H=0$
- 极度混乱（各类均匀）：$H = \\log_2 K$（最大）

**信息增益** = 划分前后的熵差：

$$
\\text{Gain}(D, A) = H(D) - \\sum_{v} \\frac{|D_v|}{|D|} H(D_v)
$$

**缺陷**：倾向选择取值多的特征（如 ID 列）→ 极易过拟合。

### 信息增益比 (C4.5)

用特征自身的熵做归一化，惩罚取值多的特征：

$$
\\text{GainRatio}(D, A) = \\frac{\\text{Gain}(D, A)}{H_A(D)}
$$

其中 $H_A(D) = -\\sum_v \\frac{|D_v|}{|D|} \\log_2 \\frac{|D_v|}{|D|}$ 是特征 $A$ 的固有熵。

### Gini 系数 (CART)

$$
\\text{Gini}(D) = 1 - \\sum_{k=1}^{K} p_k^2
$$

- Gini 越小 → 数据越纯
- 相比熵，Gini 计算更快（无对数运算）
- 两者在实际中的分裂效果高度相似

### 剪枝

防止过拟合的核心手段：
- **预剪枝**：\`max_depth\`、\`min_samples_split\`、\`min_samples_leaf\`
- **后剪枝**：先生成完整树，再从底向上剪去增益小的分支

---

## 可视化展示

### 决策树结构示意（以是否打网球为例）

\`\`\`mermaid
graph TD
    N1["天气 outlook"] -->|"sunny"| N2["湿度 humidity"]
    N1 -->|"overcast"| P["✓ 打网球 (4/0)"]
    N1 -->|"rain"| N3["风力 windy"]
    N2 -->|"高 ≤70"| N4["✗ 不打 (0/3)"]
    N2 -->|"正常 >70"| N5["✓ 打网球 (2/0)"]
    N3 -->|"强 TRUE"| N6["✗ 不打 (0/2)"]
    N3 -->|"弱 FALSE"| N7["✓ 打网球 (3/0)"]

    style P fill:#16a085,stroke:#fff,color:#fff
    style N5 fill:#16a085,stroke:#fff,color:#fff
    style N7 fill:#16a085,stroke:#fff,color:#fff
    style N4 fill:#c0392b,stroke:#fff,color:#fff
    style N6 fill:#c0392b,stroke:#fff,color:#fff
\`\`\`

### Gini 系数 vs 熵

\`\`\`echarts
return {
  xAxis: { type: 'value', min: 0, max: 1, name: '正类概率 p' },
  yAxis: { type: 'value', min: 0, max: 1.2, name: '不纯度' },
  legend: { data: ['Gini = 1-p²-(1-p)²', 'Entropy = -p·log₂(p)-(1-p)·log₂(1-p)', '分类误差 = 1-max(p,1-p)'] },
  series: [
    {
      name: 'Gini = 1-p²-(1-p)²', type: 'line', smooth: true,
      lineStyle: { color: '#2c3e50', width: 2 },
      data: (function() { const d = []; for (let p = 0; p <= 1; p += 0.01) d.push([p, 1 - p*p - (1-p)*(1-p)]); return d; })()
    },
    {
      name: 'Entropy', type: 'line', smooth: true,
      lineStyle: { color: '#d35400', width: 2 },
      data: (function() { const d = []; for (let p = 0.001; p <= 0.999; p += 0.01) d.push([p, -p*Math.log2(p) - (1-p)*Math.log2(1-p)]); return d; })()
    },
    {
      name: '分类误差', type: 'line', smooth: false,
      lineStyle: { color: '#95a5a6', width: 1.5, type: 'dashed' },
      data: (function() { const d = []; for (let p = 0; p <= 1; p += 0.01) d.push([p, 1 - Math.max(p, 1-p)]); return d; })()
    }
  ],
  tooltip: { trigger: 'axis' },
  grid: { left: 60, right: 20, top: 40, bottom: 50 }
}
\`\`\`

Gini 和熵的形状几乎一致——Gini 是熵的一阶泰勒近似，因此分裂效果相近，但 Gini 计算更快。

---

## 核心代码实现

### Scikit-learn

\`\`\`python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# 分类树
clf = DecisionTreeClassifier(
    criterion='gini',        # 或 'entropy'
    max_depth=5,             # 预剪枝
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
clf.fit(X_train, y_train)

# 回归树
reg = DecisionTreeRegressor(criterion='squared_error', max_depth=5)
reg.fit(X_train, y_train)
\`\`\`

### NumPy 手写 CART（核心分裂逻辑）

\`\`\`python
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = 0, None, None
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                left, right = y[left_mask], y[~left_mask]
                if len(left) < self.min_samples or len(right) < self.min_samples:
                    continue
                gini = (len(left) * self._gini(left) + len(right) * self._gini(right)) / len(y)
                gain = self._gini(y) - gini
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh
        return best_feat, best_thresh

    def fit(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            self.label = Counter(y).most_common(1)[0][0]
            return
        feat, thresh = self._best_split(X, y)
        if feat is None:
            self.label = Counter(y).most_common(1)[0][0]
            return
        self.feat, self.thresh = feat, thresh
        left_mask = X[:, feat] <= thresh
        self.left = DecisionTree(self.max_depth, self.min_samples)
        self.right = DecisionTree(self.max_depth, self.min_samples)
        self.left.fit(X[left_mask], y[left_mask], depth + 1)
        self.right.fit(X[~left_mask], y[~left_mask], depth + 1)

    def predict_one(self, x):
        if hasattr(self, 'label'): return self.label
        return self.left.predict_one(x) if x[self.feat] <= self.thresh else self.right.predict_one(x)
\`\`\`

## 优缺点

| 优点 | 缺点 |
|------|------|
| 可解释性强（可视化即规则） | 容易过拟合（需要剪枝） |
| 无需特征缩放 | 对数据变化敏感（高方差） |
| 自动处理非线性关系 | 偏向类别多的特征（需用 GainRatio 修正） |
| 可处理混合类型数据 | 难以捕捉 XOR 等线性不可分关系 |
`;export{n as default};
