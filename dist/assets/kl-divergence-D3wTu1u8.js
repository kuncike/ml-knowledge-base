const n=`# KL 散度 (Kullback-Leibler Divergence)

## 核心定义

KL 散度衡量一个概率分布 $Q$ 与另一个参考分布 $P$ 之间的差异：

$$D_{KL}(P \\| Q) = \\sum_x P(x) \\log \\frac{P(x)}{Q(x)} = \\mathbb{E}_{x \\sim P} \\left[ \\log \\frac{P(x)}{Q(x)} \\right]$$

## 关键性质

### 非对称性

$$D_{KL}(P \\| Q) \\neq D_{KL}(Q \\| P)$$

- **前向 KL** $D_{KL}(P \\| Q)$：$P$ 大但 $Q$ 小 → 惩罚重（mode-covering）
- **反向 KL** $D_{KL}(Q \\| P)$：$Q$ 大但 $P$ 小 → 惩罚重（mode-seeking）

### 非负性

$$D_{KL}(P \\| Q) \\geq 0$$

当且仅当 $P = Q$ 时取等号（Gibbs 不等式）。

### 与交叉熵的关系

$$D_{KL}(P \\| Q) = H(P, Q) - H(P)$$

当 $P$ 固定时（如真实标签），最小化 $D_{KL}(P \\| Q)$ 等价于最小化交叉熵 $H(P, Q)$。

## 应用场景

| 场景 | 描述 |
|------|------|
| 分类损失 | KL → CE → 负对数似然 |
| 知识蒸馏 | 学生输出匹配教师输出 |
| VAE | 后验 $q(z|x)$ 逼近先验 $p(z)$ |
| 强化学习 | PPO 中限制策略更新幅度 |
| t-SNE | 高维和低维相似度的匹配 |

## VAE 中的应用

$$\\text{ELBO} = \\mathbb{E}_{q(z|x)}[\\log p(x|z)] - D_{KL}(q(z|x) \\| p(z))$$

其中 $p(z) \\sim \\mathcal{N}(0, \\mathbf{I})$。

对两个高斯分布，KL 散度有闭式解：

$$D_{KL}(\\mathcal{N}(\\mu, \\sigma^2) \\| \\mathcal{N}(0, 1)) = -\\frac{1}{2} \\sum_j (1 + \\log \\sigma_j^2 - \\mu_j^2 - \\sigma_j^2)$$

## 与 Jensen-Shannon 散度的对比

JS 散度是 KL 的对称化版本：

$$\\text{JSD}(P \\| Q) = \\frac{1}{2} D_{KL}(P \\| M) + \\frac{1}{2} D_{KL}(Q \\| M)$$

其中 $M = \\frac{P + Q}{2}$。JS 散度是 GAN 中原始判别器的理论基础。

## Python 实现

\`\`\`python
import torch
import torch.nn.functional as F

# 方式 1：使用 PyTorch 内置（注意输入是 log 概率）
kl_div = F.kl_div(
    F.log_softmax(student_logits, dim=-1),
    F.softmax(teacher_logits, dim=-1),
    reduction='batchmean'
)

# 方式 2：手动计算
def kl_divergence(p_logits, q_logits, T=1.0):
    p = F.softmax(p_logits / T, dim=-1)
    q = F.softmax(q_logits / T, dim=-1)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()
\`\`\`
`;export{n as default};
