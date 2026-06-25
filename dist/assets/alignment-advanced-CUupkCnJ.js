const n=`# Alignment Advanced (KTO / ORPO / SimPO)

## 核心思想

RLHF/DPO 都需要**成对偏好数据**（同一个 prompt 的 chosen vs rejected response），标注成本高。KTO 革新性地提出：不需要成对数据——只要知道单个回答是"好"还是"坏"即可（基于 Kahneman 的前景理论）。ORPO 更进一步：**在对齐的同时保留 SFT 能力**——在 SFT 目标中加入 odds ratio 惩罚，一步完成"微调+对齐"。SimPO 则证明：用生成概率作为奖励信号、无需参考模型，也能达到 DPO 的效果。

核心趋势：**减少对齐所需的数据量和计算开销**。

---

## 数学定义与原理解析

### KTO — 前景理论损失

受经济学 Kahneman-Tversky 前景理论启发，KTO 分别处理"好"和"坏"的响应：

对于 **desired** 响应（标签 $y=1$）：

$$
\\mathcal{L}_{KTO}^+ = -\\lambda_D \\cdot \\sigma\\left( \\beta \\left( \\log \\frac{\\pi_\\theta(y|x)}{\\pi_{ref}(y|x)} - \\mathbb{E}_{y' \\sim \\pi_{ref}} \\left[ \\log \\frac{\\pi_\\theta(y'|x)}{\\pi_{ref}(y'|x)} \\right] \\right) \\right)
$$

对于 **undesired** 响应（标签 $y=0$）：

$$
\\mathcal{L}_{KTO}^- = -\\lambda_U \\cdot \\sigma\\left( -\\beta \\left( \\log \\frac{\\pi_\\theta(y|x)}{\\pi_{ref}(y|x)} - \\mathbb{E}_{y' \\sim \\pi_{ref}}[\\ldots] \\right) \\right)
$$

关键：两种响应**不需要来自同一个 prompt**。$\\lambda_D$ 和 $\\lambda_U$ 分别控制好/坏样本的权重（前景理论的 loss aversion 不对称性）。

### ORPO — 单阶段对齐

将 SFT 的 CE 损失和偏好对齐的 OR 损失合并：

$$
\\mathcal{L}_{ORPO} = \\mathcal{L}_{SFT} + \\lambda \\cdot \\mathcal{L}_{OR}
$$

其中 OR（Odds Ratio）项为：

$$
\\mathcal{L}_{OR} = -\\log \\sigma\\left( \\log \\frac{\\text{odds}_\\theta(y_w|x)}{\\text{odds}_\\theta(y_l|x)} \\right)
$$

$$
\\text{odds}_\\theta(y|x) = \\frac{\\pi_\\theta(y|x)}{1 - \\pi_\\theta(y|x)}
$$

### SimPO — 无需参考模型

直接用**生成概率**作为隐式奖励，无需参考模型：

$$
\\mathcal{L}_{SimPO} = -\\log \\sigma\\left( \\frac{\\beta}{|y_w|} \\sum_i \\log \\pi_\\theta(y_w^{(i)}|x, y_w^{(<i)}) - \\frac{\\beta}{|y_l|} \\sum_i \\log \\pi_\\theta(y_l^{(i)}|x, y_l^{(<i)}) - \\gamma \\right)
$$

$\\gamma$ 是 target reward margin。核心简化：奖励直接来自**长度归一化的平均对数概率**。

---

## 可视化展示

### 对齐方法对比

\`\`\`echarts
return {
  title: { text: '对齐方法需求对比', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['需成对数据', '需参考模型', '需奖励模型', '单阶段'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '是否需要 (0=否, 1=是)' },
  legend: { data: ['RLHF', 'DPO', 'KTO', 'ORPO', 'SimPO'] },
  series: [
    { name: 'RLHF', type: 'bar', data: [1, 1, 1, 0], itemStyle: { color: '#c0392b' } },
    { name: 'DPO', type: 'bar', data: [1, 1, 0, 0], itemStyle: { color: '#2980b9' } },
    { name: 'KTO', type: 'bar', data: [0, 1, 0, 0], itemStyle: { color: '#16a085' } },
    { name: 'ORPO', type: 'bar', data: [1, 0, 0, 1], itemStyle: { color: '#d35400' } },
    { name: 'SimPO', type: 'bar', data: [1, 0, 0, 0], itemStyle: { color: '#2c3e50' } }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — KTO Loss

\`\`\`python
import torch
import torch.nn.functional as F

def kto_loss(policy_logps, ref_logps, labels, beta=0.1,
              lambda_desired=1.0, lambda_undesired=1.0):
    """
    policy_logps: [B] — log π_θ(y|x) / |y|
    ref_logps:    [B] — log π_ref(y|x) / |y|
    labels:       [B] — 1=desired, 0=undesired
    """
    kl = policy_logps - ref_logps  # [B]

    # 参考分布下的平均 KL
    kl_ref_mean = kl.mean()

    # 分开处理 desired 和 undesired
    loss = 0
    for group, lam, sign in [
        (labels == 1, lambda_desired, 1.0),
        (labels == 0, lambda_undesired, -1.0)
    ]:
        if group.any():
            adjusted = kl[group] - kl_ref_mean
            loss += lam * (
                -F.logsigmoid(sign * beta * adjusted)
            ).mean()
    return loss
\`\`\`

### PyTorch — ORPO Loss

\`\`\`python
def orpo_loss(policy_logps, policy_logprobs_chosen,
              policy_logprobs_rejected, beta=0.1, lambda_or=1.0):
    """
    policy_logps: [B] — cross-entropy SFT 损失
    policy_logprobs_chosen:   [B] — log π_θ(y_win|x)
    policy_logprobs_rejected: [B] — log π_θ(y_lose|x)
    """
    # SFT 损失
    sft_loss = -policy_logps.mean()

    # Odds ratio 损失
    odds_chosen = policy_logprobs_chosen - torch.log1p(
        -torch.exp(policy_logprobs_chosen).clamp(max=0.999))
    odds_rejected = policy_logprobs_rejected - torch.log1p(
        -torch.exp(policy_logprobs_rejected).clamp(max=0.999))
    log_odds_ratio = odds_chosen - odds_rejected
    or_loss = -F.logsigmoid(beta * log_odds_ratio).mean()

    return sft_loss + lambda_or * or_loss
\`\`\`

### PyTorch — SimPO Loss

\`\`\`python
def simpo_loss(policy_model, batch, beta=2.0, gamma=0.5):
    """batch: {prompt, chosen, rejected}"""
    # 计算 chosen 和 rejected 的生成概率（长度归一化）
    chosen_logp = policy_model.log_prob(batch['prompt'], batch['chosen'])
    rejected_logp = policy_model.log_prob(batch['prompt'], batch['rejected'])

    chosen_len = batch['chosen'].ne(tokenizer.pad_token_id).sum(dim=-1)
    rejected_len = batch['rejected'].ne(tokenizer.pad_token_id).sum(dim=-1)

    chosen_avg = chosen_logp.sum(dim=-1) / chosen_len  # [B]
    rejected_avg = rejected_logp.sum(dim=-1) / rejected_len  # [B]

    # 隐式奖励差异
    loss = -F.logsigmoid(beta * (chosen_avg - rejected_avg) - gamma)
    return loss.mean()
\`\`\`
`;export{n as default};
