const n=`# GPT 系列

## GPT-1 (2018)

### 核心思想

在大规模无标注文本上做**生成式预训练**，然后针对下游任务做**判别式微调**。

### 架构

12 层 Transformer Decoder，$d_{model}=768$，12 头，117M 参数。

### 两阶段训练

1. **无监督预训练**：标准语言模型（预测下一个 token）
2. **有监督微调**：将下游任务输入格式化为特殊 token 序列

---

## GPT-2 (2019)

### 核心理念：Zero-shot

**语言模型是无监督多任务学习器**。不需要微调，只需提供自然语言提示。

### 规模升级

| 版本 | 参数量 | 层数 |
|------|--------|------|
| GPT-2 Small | 124M | 12 |
| GPT-2 Medium | 355M | 24 |
| GPT-2 Large | 774M | 36 |
| GPT-2 XL | 1.5B | 48 |

---

## GPT-3 (2020)

### 核心理念：In-Context Learning

175B 参数。不更新模型参数，仅通过提示中的几个示例（Few-Shot）学习任务。

### 涌现能力 (Emergent Abilities)

当模型规模达到一定阈值时才出现的能力：
- 算术推理
- 代码生成
- 翻译
- 多步推理

## GPT-3.5 / InstructGPT

### RLHF 三步流程

1. **SFT**（监督微调）：用高质量人类回答微调
2. **RM**（奖励模型训练）：对多个回答排序，训练奖励模型
3. **PPO**（强化学习）：用 PPO 优化策略，使奖励模型打分最高

## GPT-4 (2023)

### 关键能力

- **多模态**：支持图像输入
- **长上下文**：32K → 128K tokens
- **指令遵循**：大幅提升
- **安全性**：对抗越狱的鲁棒性增强

### MoE 架构

GPT-4 使用 8 × 220B 的 Mixture-of-Experts 架构，每次推理仅激活部分专家。

## 自回归生成

GPT 的核心是**自回归生成**：

$$P(x) = \\prod_{t=1}^{T} P(x_t \\mid x_{<t})$$

解码策略：
- **Greedy**：选概率最大的 token
- **Temperature**：调整概率分布熵
- **Top-k**：从 top-k 候选采样
- **Top-p (Nucleus)**：从累积概率 ≥ p 的候选中采样

## 关键公式：注意力中的 Causal Mask

\`\`\`python
# 自回归遮罩：token i 只能看到 0..i
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
# [[False, True,  True],
#  [False, False, True],
#  [False, False, False]]
\`\`\`
`;export{n as default};
