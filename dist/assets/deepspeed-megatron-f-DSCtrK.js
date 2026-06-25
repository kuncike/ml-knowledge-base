const n=`# DeepSpeed / Megatron-LM

## 大规模训练的挑战

训练 GPT-4 级别的大模型需要分布式训练：

- 内存：175B 参数 × 2-4 bytes = 350-700 GB（单 GPU 只有 80GB）
- 计算：需要数万亿 token 的训练 → 数千 GPU 并行

---

## DeepSpeed (Microsoft)

### ZeRO (Zero Redundancy Optimizer)

#### 优化器状态冗余问题

标准数据并行 (DP) 中，每个 GPU 存储**完整的**模型参数、梯度和优化器状态。这导致巨大的冗余。

#### ZeRO 的三个阶段

| 阶段 | 分片内容 | 内存节省 |
|------|----------|----------|
| ZeRO-1 | 优化器状态 (Adam 的 m, v) | 4× |
| ZeRO-2 | + 梯度 | 8× |
| ZeRO-3 | + 模型参数 | 与 GPU 数量线性 |

#### ZeRO-Offload

将优化器状态和计算卸载到 CPU 内存，进一步增大可训练的模型规模。

#### ZeRO-Infinity

将模型状态卸载到 NVMe SSD，实现**万亿参数**级别的训练。

### DeepSpeed 使用

\`\`\`python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json",
)
# 训练循环
for data in dataloader:
    loss = model_engine(data)
    model_engine.backward(loss)
    model_engine.step()
\`\`\`

---

## Megatron-LM (NVIDIA)

### 张量并行 (Tensor Parallelism)

将单个 Transformer 层的权重矩阵沿列或行切分到多个 GPU：

\`\`\`
Attention 中的 QKV 投影:
  W × X → [W₁; W₂] × X → [W₁X; W₂X] (沿列切分)

FFN 的第一个线性层:
  W × X → [W₁; W₂] × X → [W₁X; W₂X] (沿列切分)
\`\`\`

### 流水线并行 (Pipeline Parallelism)

将不同层放在不同 GPU 上，用微批次 (Micro-batch) 流水线执行：

\`\`\`
GPU 0: Layers 0-7
GPU 1: Layers 8-15
GPU 2: Layers 16-23
GPU 3: Layers 24-31
\`\`\`

### 序列并行 (Sequence Parallelism)

将长序列沿序列维度切分，减少 LayerNorm 和 Dropout 中的激活内存。

### 3D 并行

结合三种并行策略：

$$\\text{总 GPU 数} = TP \\times PP \\times DP$$

- TP（张量并行）：层内权重的切分（通常 8 GPU）
- PP（流水线并行）：层间模型的分布
- DP（数据并行）：数据切分（ZeRO 优化）

## DeepSpeed vs Megatron-LM

| | DeepSpeed | Megatron-LM |
|------|-----------|-------------|
| 核心贡献 | ZeRO 优化器 | 模型并行策略 |
| 适用 | 通用分布式训练 | 超大规模 Transformer |
| 内存优化 | ZeRO-1/2/3 + Offload | TP + SP + Activation Recomputation |
| 易用性 | 配置驱动，易上手 | 需要修改模型代码 |

### 实际使用

两者常结合使用（如 DeepSpeed + Megatron），各自发挥优势：Megatron 提供模型并行，DeepSpeed 提供 ZeRO 数据并行优化。
`;export{n as default};
