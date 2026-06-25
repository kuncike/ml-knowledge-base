const n=`# LoRA / QLoRA / AdaLoRA

## LoRA (Low-Rank Adaptation)

### 核心思想

预训练权重是满秩的，但微调时的**权重更新是低秩的**。LoRA 用两个小矩阵的乘积来近似权重更新：

$$\\mathbf{W}' = \\mathbf{W} + \\Delta\\mathbf{W} = \\mathbf{W} + \\mathbf{B}\\mathbf{A}$$

其中 $\\mathbf{W} \\in \\mathbb{R}^{d \\times k}$，$\\mathbf{B} \\in \\mathbb{R}^{d \\times r}$，$\\mathbf{A} \\in \\mathbb{R}^{r \\times k}$，$r \\ll \\min(d, k)$。

### 参数量对比

对 $d=4096, k=4096, r=16$ 的投影矩阵：

- 原始：$4096 \\times 4096 = 16.8M$
- LoRA：$4096 \\times 16 \\times 2 = 131K$ → 约 128× 减少！

### 缩放因子

$$\\mathbf{h} = \\mathbf{W}_0 \\mathbf{x} + \\frac{\\alpha}{r} \\cdot \\mathbf{B}\\mathbf{A}\\mathbf{x}$$

$\\alpha$ 控制 LoRA 更新的强度，通常 $\\alpha = 16$ 或 $32$。

### 在哪里加 LoRA？

Transformer 中的**所有线性层**都可以加：

- **Attention**：$W_Q, W_K, W_V, W_O$
- **FFN**：$W_1, W_2, W_3$（SwiGLU 有三个线性层）

一般至少加在 Attention 的 $Q$ 和 $V$ 上。QLoRA 论文建议加在**所有线性层**以获得最佳效果。

### 合并与切换

\`\`\`python
# 推理时合并（零额外延迟）
merged_W = W + (alpha / r) * B @ A

# 切换任务只需切换 LoRA 权重
model.load_lora("task_A_lora.pt")  # 基础模型不动
\`\`\`

---

## QLoRA

### 核心创新

1. **NF4 (4-bit NormalFloat)**：信息论最优的正态分布 4-bit 量化格式
2. **双重量化**：对量化常数再量化
3. **分页优化器**：用 CPU 内存处理 OOM 的梯度检查点

### NF4 vs INT4

| | INT4 | NF4 |
|------|------|-----|
| 分布假设 | 均匀 | 正态 |
| 量化间隔 | 均匀 | 不等距（密度匹配） |
| 适合场景 | 均匀分布 | 正态分布（神经网络权重） |

### 效果

在 4-bit 量化的 65B 模型上用 QLoRA 微调，可以达到 16-bit 全参数微调的性能，只需 48GB GPU（单卡）。

---

## AdaLoRA

### 核心思想

不是所有层都一样重要。AdaLoRA 根据**参数重要性**动态分配不同的秩：

$$\\min_{\\mathcal{P}} \\sum_{i} \\|\\mathbf{P}_i \\odot (\\mathbf{W}_i - \\mathbf{W}_i^0)\\|_F^2$$

$$\\text{s.t.} \\quad \\sum_i \\text{rank}(\\mathbf{P}_i) \\leq \\text{budget}$$

重要的层用更高的秩，不重要的层用更低的秩，在相同参数预算下获得更好性能。

## 使用 (PEFT)

\`\`\`python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: 8.4M || all params: 7B || trainable%: 0.12%
\`\`\`
`;export{n as default};
