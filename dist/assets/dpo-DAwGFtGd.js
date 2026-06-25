const n=`# DPO / PPO / KTO

## DPO (Direct Preference Optimization)

### 为什么需要 DPO？

RLHF 需要 4 个模型（policy, reference, reward model, value model），极其消耗资源且训练不稳定。DPO 直接用偏好数据优化策略，不需要显式训练 RM。

### 核心推导

从 RLHF 的最优策略解析解出发：

$$\\pi^*(y|x) = \\frac{1}{Z(x)} \\pi_{ref}(y|x) \\exp\\left(\\frac{1}{\\beta} r(x, y)\\right)$$

解出奖励：

$$r(x, y) = \\beta \\log \\frac{\\pi^*(y|x)}{\\pi_{ref}(y|x)} + \\beta \\log Z(x)$$

代入 Bradley-Terry 偏好模型，$Z(x)$ 项抵消掉：

### DPO 损失

$$L_{DPO} = -\\mathbb{E}_{(x, y_w, y_l) \\sim D} \\left[ \\log \\sigma\\left( \\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{ref}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{ref}(y_l|x)} \\right) \\right]$$

**直觉**：$\\pi_\\theta$ 对 $y_w$（优）提高概率，对 $y_l$（差）降低概率，$\\pi_{ref}$ 作为锚点防止偏离太远。

### DPO vs RLHF

| | RLHF | DPO |
|------|------|-----|
| 模型数 | 4 | 2 (policy + ref) |
| 训练稳定性 | 不稳定 | 稳定 |
| 显式 RM | 需要 | 不需要 |
| 在线采样 | 需要 | 不需要 |
| 效果 | SOTA | 接近或达到 SOTA |

---

## PPO (Proximal Policy Optimization)

### 核心公式

$$L^{CLIP}(\\theta) = \\hat{\\mathbb{E}}_t \\left[ \\min(r_t(\\theta)\\hat{A}_t, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon)\\hat{A}_t) \\right]$$

其中 $r_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{old}(a_t|s_t)}$。

两种情况下裁剪生效：
- 优势为正 → 限制 $r_t$ 不能 > $1+\\epsilon$（不要太激进）
- 优势为负 → 限制 $r_t$ 不能 < $1-\\epsilon$（不要太保守）

### RLHF 中的 PPO

$$\\text{reward} = r_{RM}(x, y) - \\beta \\cdot \\text{KL}(\\pi_\\theta \\| \\pi_{ref})$$

---

## KTO (Kahneman-Tversky Optimization)

### 核心思想

基于前景理论的洞察：**人类对损失的敏感度高于收益**。

KTO 不需要成对的偏好数据（A > B），只需要"满意/不满意"的二元信号：

$$L_{KTO} = \\mathbb{E}_{(x,y) \\sim D} \\left[ w(y) \\cdot \\left(1 - h(r_\\theta(x, y))\\right) \\right]$$

对想要的输出：鼓励 KL 散度大；对不想要的输出：惩罚 KL 散度大。

## 什么时候用什么方法？

| 场景 | 推荐方法 |
|------|----------|
| 有成对偏好数据 | DPO |
| 有绝对打分 | KTO |
| 可以在线采样 | Online DPO / PPO |
| 资源受限 | DPO |
| 追求极致性能 | Iterative DPO / PPO |
`;export{n as default};
