const n=`# LLaMA / Mistral / Mixtral (MoE)

## LLaMA 系列

### LLaMA 1 (2023)

Meta 推出的开源大模型，证明了**高质量小数据 > 低质量大数据**。

| 版本 | 参数量 | $d_{model}$ | 头数 | 层数 |
|------|--------|-------------|------|------|
| 7B | 6.7B | 4096 | 32 | 32 |
| 13B | 13.0B | 5120 | 40 | 40 |
| 33B | 32.5B | 6656 | 52 | 60 |
| 65B | 65.2B | 8192 | 64 | 80 |

### 架构改进

1. **Pre-Norm**：用 RMSNorm 替代 LayerNorm
2. **SwiGLU**：替代 ReLU 激活
3. **RoPE**：旋转位置编码

### LLaMA 2

#### 关键改进

- **GQA (Grouped-Query Attention)**：减少 KV Cache
- **更长上下文**：4096 tokens
- **RLHF 对齐**：开源 SFT + RM + PPO 流程
- **Ghost Attention**：多轮对话的指令遵循

### LLaMA 3 / 3.1

- 支持多模态
- 128K 上下文窗口
- 显著提升推理和代码能力

---

## Mistral 7B

### 核心创新

1. **滑动窗口注意力** (Window=4096)
2. **分组查询注意力 (GQA)**
3. **滚动缓冲区缓存 (Rolling Buffer Cache)**

7B 参数达到超越 LLaMA 2 13B 的效果。

---

## Mixtral 8×7B (MoE)

### Mixture of Experts (MoE)

$$\\mathbf{y} = \\sum_{i=1}^{E} g_i(\\mathbf{x}) \\cdot \\text{Expert}_i(\\mathbf{x})$$

- 8 个专家，每次选择 Top-2
- 总参数量：46.7B（但推理时仅激活 12.9B）
- 门控网络：$\\text{softmax}(\\text{TopK}(\\mathbf{x} \\cdot \\mathbf{W}_g, k=2))$

### MoE 的优势

| | Dense | MoE |
|------|-------|-----|
| 总参数 | 大 | 更大 |
| 每 token 激活参数 | 全部 | 部分 |
| 推理速度 | 固定 | 更快（激活更少） |
| 训练难度 | 简单 | 需负载均衡 |

### 负载均衡损失

防止门控网络总是选同样的几个专家：

$$L_{aux} = \\alpha \\cdot E \\cdot \\sum_{i=1}^{E} f_i \\cdot P_i$$

其中 $f_i$ 是分配给专家 $i$ 的 token 比例，$P_i$ 是路由概率均值。
`;export{n as default};
