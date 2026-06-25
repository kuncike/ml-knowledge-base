const n=`# 随机森林 (Random Forest)

## 核心思想

随机森林是 Bagging (Bootstrap Aggregating) 的代表算法。通过对训练数据进行**自助采样 (Bootstrap)** 训练多个决策树，再对树进行**特征随机抽样**，最终投票或平均得到结果。

## Bagging 算法流程

1. 从 $n$ 个样本中有放回地抽取 $n$ 个样本（Bootstrap）
2. 用该样本集训练一个基学习器
3. 重复 $T$ 次
4. 分类：多数投票；回归：取平均

每个样本被抽中的概率：$1 - \\left(1 - \\frac{1}{n}\\right)^n \\approx 1 - \\frac{1}{e} \\approx 63.2\\%$

约 36.8% 的样本未被抽中，称为**袋外数据 (OOB)**，可用于无偏估计泛化误差。

## 随机森林的特殊性

在 Bagging 基础上增加了**特征随机性**：

- 每个节点分裂时，只从随机选取的 $m$ 个特征中选择最优分裂
- 分类：$m = \\lfloor\\sqrt{d}\\rfloor$
- 回归：$m = \\lfloor d/3\\rfloor$

这使得树之间更加独立，降低了方差。

## Python 实现

\`\`\`python
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

class RandomForest:
    def __init__(self, n_trees=100, max_features='sqrt', max_depth=None):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.trees = []
        self.feature_subsets = []

    def fit(self, X, y):
        n, d = X.shape
        m = int(np.sqrt(d)) if self.max_features == 'sqrt' else d
        for _ in range(self.n_trees):
            idx = np.random.choice(n, n, replace=True)  # Bootstrap
            feats = np.random.choice(d, m, replace=False)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X[idx][:, feats], y[idx])
            self.trees.append(tree)
            self.feature_subsets.append(feats)

    def predict(self, X):
        preds = []
        for tree, feats in zip(self.trees, self.feature_subsets):
            preds.append(tree.predict(X[:, feats]))
        preds = np.array(preds).T
        return np.array([Counter(row).most_common(1)[0][0] for row in preds])
\`\`\`

## 泛化误差分析

随机森林的泛化误差上界：

$$\\text{PE}^* \\leq \\frac{\\bar{\\rho}(1 - s^2)}{s^2}$$

其中 $\\bar{\\rho}$ 是树之间的平均相关性，$s$ 是单棵树的强度。**降低树间相关性**是降误差的关键！

## OOB 误差

无需交叉验证即可估计泛化性能：

$$\\text{OOB Error} = \\frac{1}{n} \\sum_{i=1}^{n} \\ell(y_i, \\hat{y}_i^{\\text{OOB}})$$

## 优缺点

- **优点**：精度高，不易过拟合，可处理高维数据，OOB 提供免费验证
- **缺点**：模型较大，训练和推理较慢，可解释性不如单棵决策树
`;export{n as default};
