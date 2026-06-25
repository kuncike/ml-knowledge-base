const n=`# Qwen / DeepSeek / ChatGLM (国产大模型)

## Qwen 系列 (通义千问)

阿里巴巴推出的大模型系列。

### Qwen2

开源，提供 0.5B / 1.5B / 7B / 72B 多个规模。

### 技术特点

- **GQA**：分组查询注意力
- **SwiGLU**：激活函数
- **QKV Bias**：注意力中的偏置项
- **Tie Embedding**：输入输出嵌入共享权重
- **多语言**：支持 27+ 语言

### Qwen-VL

多模态版本，支持图文理解。

---

## DeepSeek 系列

深度求索公司推出的开源大模型。

### DeepSeek-V2 / V3

#### 核心创新：MLA (Multi-head Latent Attention)

将 KV Cache 压缩到低维潜在空间：

$$\\mathbf{C}^{KV} = \\mathbf{h}_t \\cdot \\mathbf{W}^{DKV}$$

注意力计算中，K 和 V 从压缩表示 $\\mathbf{C}^{KV}$ 重构：

$$\\mathbf{K} = \\mathbf{C}^{KV} \\cdot \\mathbf{W}^{UK}, \\quad \\mathbf{V} = \\mathbf{C}^{KV} \\cdot \\mathbf{W}^{UV}$$

→ KV Cache 大小大幅降低，长序列推理更高效。

#### DeepSeek-R1

专注推理能力。核心创新：
- **纯强化学习推理**：无需 SFT，直接用 RL 提升推理
- **GRPO (Group Relative Policy Optimization)**：组内相对优化

---

## ChatGLM 系列

智谱 AI 的对话模型。

### GLM 架构

#### 双向注意力前缀

GLM 使用特殊的注意力掩码：

- 前缀部分（上文）：双向注意力
- 生成部分：自回归注意力

结合了 BERT（双向理解）和 GPT（自回归生成）的优势。

### ChatGLM 版本演进

| 版本 | 参数量 | 特点 |
|------|--------|------|
| ChatGLM-6B | 6B | 中英双语 |
| ChatGLM2-6B | 6B | 更长上下文 (32K) |
| ChatGLM3-6B | 6B | 工具调用、代码解释 |
| GLM-4 | 130B+ | 多模态、128K 上下文 |

---

## 对比

| 模型 | 参数量 | 独特创新 | 开源 |
|------|--------|----------|------|
| Qwen2 | 0.5B–72B | 多语言 SOTA | 全部 |
| DeepSeek-V3 | 671B (MoE) | MLA + 极低成本训练 | 全部 |
| ChatGLM | 6B–130B+ | 双向+自回归掩码 | 部分 |

### DeepSeek 的颠覆性

DeepSeek-V3 使用 ~$5M 的训练成本达到 GPT-4o 级别表现（传统需要 $100M+），通过 MLA、FP8 训练、MoE 架构等创新大幅降低成本。
`;export{n as default};
