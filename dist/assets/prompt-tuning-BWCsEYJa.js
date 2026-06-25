const n=`# Prompt Tuning / Prefix Tuning / P-Tuning

这些方法统称**参数高效微调 (PEFT)** 中的"提示类"方法，不修改模型权重，只学习额外的提示参数。

## Prompt Tuning

### 核心思想

在输入前添加**可学习的软提示 (Soft Prompt)**，训练时只更新这些 token 的嵌入，模型参数全部冻结。

\`\`\`
[P1] [P2] ... [Pk] [原始输入 tokens]
 ↑ 可学习的虚拟 tokens
\`\`\`

### 两个变体

- **Prompt Tuning**：只学习嵌入层的虚拟 token
- **Prompt Tuning v2**：每层都插入可学习向量（类似 Prefix Tuning）

### 适用场景

- **小模型**（< 1B）：Prompt Tuning 效果不佳
- **大模型**（> 10B）：Prompt Tuning 可接近全参数微调的效果

---

## Prefix Tuning

### 核心思想

在每一层 Transformer 的 Key 和 Value 前面拼接可学习向量：

\`\`\`
第 l 层的 KV：
  K_l' = [P^K_1, ..., P^K_m, original_K]
  V_l' = [P^V_1, ..., P^V_m, original_V]
\`\`\`

### 重参数化

直接用梯度优化前缀向量不稳定。使用 MLP 重参数化：

$$\\mathbf{P} = \\text{MLP}_\\theta(\\mathbf{P}')$$

训练完只保存 $\\mathbf{P}$（丢弃 MLP）。

---

## P-Tuning v1 / v2

### P-Tuning v1

针对自然语言理解任务，在输入嵌入中插入可学习的 prompt tokens，并增加 LSTM 或 MLP 编码器对 prompt 进行编码。

### P-Tuning v2

将所有层的输入中加入可学习的连续提示向量（不再限于嵌入层）。

---

## 对比

| 方法 | 插入位置 | 参数量 | 适用 |
|------|----------|--------|------|
| Prompt Tuning | 输入嵌入 | 极低 | 大模型 |
| Prefix Tuning | 每层的 K,V | 中等 | 生成任务 |
| P-Tuning v1 | 输入嵌入 + LSTM | 中 | NLU 任务 |
| P-Tuning v2 | 每层输入 | 中 | NLU + NLG |
| LoRA | 权重更新 | 中 | 通用 |

## 与 LoRA 的选择

| 场景 | 推荐 |
|------|------|
| 追求最佳效果 | LoRA / QLoRA |
| 极致低参数 | Prompt Tuning |
| 多任务快速切换 | Prompt Tuning（只切换 prompt 向量） |
| 生成任务 | Prefix Tuning / LoRA |
| 分类等理解任务 | P-Tuning / LoRA |
`;export{n as default};
