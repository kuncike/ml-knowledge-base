const n=`# 朴素贝叶斯 (Naive Bayes)

## 核心思想

朴素贝叶斯基于贝叶斯定理，并做了一个**强假设**：所有特征在给定类别的条件下**相互独立**。这个假设虽然"朴素"，但在实践中效果出奇地好，尤其是在文本分类中。

## 贝叶斯定理

$$P(y \\mid \\mathbf{x}) = \\frac{P(\\mathbf{x} \\mid y) \\cdot P(y)}{P(\\mathbf{x})}$$

朴素假设：$P(\\mathbf{x} \\mid y) = \\prod_{j=1}^{d} P(x_j \\mid y)$

决策规则：

$$\\hat{y} = \\arg\\max_y P(y) \\prod_{j=1}^{d} P(x_j \\mid y)$$

## 三种常见变体

### 1. 高斯朴素贝叶斯

适用于**连续特征**，假设每个特征在各类别下服从正态分布：

$$P(x_j \\mid y) = \\frac{1}{\\sqrt{2\\pi\\sigma_{y,j}^2}} \\exp\\left(-\\frac{(x_j - \\mu_{y,j})^2}{2\\sigma_{y,j}^2}\\right)$$

### 2. 多项式朴素贝叶斯

适用于**离散计数特征**（如词频）：

$$P(x_j \\mid y) = \\frac{N_{y,j} + \\alpha}{N_y + \\alpha \\cdot d}$$

其中 $\\alpha$ 是拉普拉斯平滑参数。

### 3. 伯努利朴素贝叶斯

适用于**二元特征**（如词是否出现）：

$$P(x_j \\mid y) = P(j \\mid y)^{x_j} \\cdot (1 - P(j \\mid y))^{(1-x_j)}$$

## Python 实现 (高斯朴素贝叶斯)

\`\`\`python
import numpy as np
from scipy.stats import norm

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.params = {}
        for c in self.classes:
            X_c = X[y == c]
            self.params[c] = {
                'mean': X_c.mean(axis=0),
                'var': X_c.var(axis=0) + 1e-9,  # 防除零
                'prior': len(X_c) / len(X),
            }

    def predict(self, X):
        log_probs = []
        for c in self.classes:
            p = self.params[c]
            # 对数空间计算避免下溢
            log_p = np.log(p['prior'])
            log_p += norm.logpdf(X, p['mean'], np.sqrt(p['var'])).sum(axis=1)
            log_probs.append(log_p)
        return self.classes[np.argmax(log_probs, axis=0)]
\`\`\`

## 拉普拉斯平滑

防止零概率问题：当某个特征值在训练集中未出现时，概率为 0 会导致整个乘积为 0：

$$P(x_j \\mid y) = \\frac{\\text{count}(x_j, y) + \\alpha}{\\text{count}(y) + \\alpha \\cdot |V|}$$

## 优缺点

- **优点**：训练和预测极快，少量数据也有效，可解释性强，天然支持增量学习
- **缺点**：特征独立假设在现实中通常不成立，对输入数据的分布假设敏感
`;export{n as default};
