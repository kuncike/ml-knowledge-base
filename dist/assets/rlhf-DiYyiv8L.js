const n=`# RLHF (基于人类反馈的强化学习)

## 核心思想

RLHF 通过人类偏好信号来训练语言模型，使其输出更符合人类期望（有帮助、真实、无害）。

## 三步流程

### Step 1: 监督微调 (SFT)

用高质量人工标注的 (指令, 回答) 对微调模型。

### Step 2: 训练奖励模型 (Reward Model, RM)

对同一个 prompt，模型生成多个回答，人类标注员对回答进行**排序**（标注排序比标注绝对分数更可靠）。

训练目标（Bradley-Terry 模型）：

$$L(\\theta) = -\\mathbb{E}_{(x, y_w, y_l) \\sim D} \\left[ \\log \\sigma(r_\\theta(x, y_w) - r_\\theta(x, y_l)) \\right]$$

其中 $y_w$ 是人类偏好的回答，$y_l$ 是较差的回答。

### Step 3: 用 PPO 优化策略

将语言模型视为 RL 的 agent：

- **状态** $s_t$：当前生成的文本序列 $y_{<t}$
- **动作** $a_t$：生成下一个 token $y_t$
- **策略** $\\pi_\\theta$：语言模型（待优化）
- **奖励**：RM 打分 + KL 惩罚（防止偏离初始策略太远）

目标函数：

$$R_{total} = r_\\theta(x, y) - \\beta \\cdot D_{KL}(\\pi_\\theta(y|x) \\| \\pi_{ref}(y|x))$$

## KL 惩罚的重要性

没有 KL 惩罚，策略很快会滥用奖励模型的漏洞（Reward Hacking），生成高奖励但无意义的文本。

$$\\text{KL penalty} = \\log \\pi_\\theta(y_t|x, y_{<t}) - \\log \\pi_{ref}(y_t|x, y_{<t})$$

## PPO 算法核心

$$\\max_\\theta \\mathbb{E} \\left[ \\frac{\\pi_\\theta(a|s)}{\\pi_{old}(a|s)} \\cdot A(s,a) - \\beta \\cdot \\text{KL} \\right]$$

裁剪目标防止策略更新过大：

$$L^{CLIP} = \\mathbb{E} \\left[ \\min\\left(r(\\theta)A, \\text{clip}(r(\\theta), 1-\\epsilon, 1+\\epsilon)A\\right) \\right]$$

其中 $r(\\theta) = \\pi_\\theta/\\pi_{old}$，$\\epsilon$ 通常取 0.2。

## 数据收集流程

1. 从当前策略采样多个回答
2. RM 打分
3. 计算优势函数（GAE）
4. 用 PPO 更新策略
5. 重复

## RLHF 的挑战

- **训练不稳定**：同时维护 4 个模型（policy, ref, RM, value）
- **奖励黑客**：模型找到 RM 漏洞而非真正改善质量
- **标注成本高**：需要大量高质量人类偏好标注
- **偏好漂移**：不同文化/人群偏好不同

## RLHF 的简化方案

- **DPO**：直接用偏好数据优化策略，省去 RM
- **KTO**：使用非对比的信号（逐条打分而非排序）
- **ORPO**：在 SFT 过程中同时优化偏好
`;export{n as default};
