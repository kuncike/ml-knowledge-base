const n=`# 模型量化 (Quantization)

## 核心思想

将浮点权重和激活值从 FP32/FP16 降低到更低精度（INT8/INT4/FP8），以减少模型体积和推理延迟。

## 统一量化公式

$$q = \\text{round}\\left(\\frac{r}{s}\\right) + z$$

$$r = s \\cdot (q - z)$$

- $r$：浮点值
- $q$：量化后的整数值
- $s$：缩放因子 (scale)
- $z$：零点 (zero point，用于非对称量化)

### 对称量化 vs 非对称量化

| | 对称 | 非对称 |
|------|------|--------|
| $z$ | 0 | $z \\neq 0$ |
| 表示范围 | $[-127, 127]$ | $[-128, 127]$ |
| 实现复杂度 | 简单 | 略复杂 |
| 适用 | 对称分布数据 | 非对称分布（如 ReLU 输出） |

---

## 量化方法

### PTQ (Post-Training Quantization)

直接对训练好的模型做量化，不需要重新训练（或只需少量校准数据）：

\`\`\`python
import torch.quantization as quant

model_fp32.eval()
model_fp32.qconfig = quant.get_default_qconfig('fbgemm')
model_int8 = quant.quantize_dynamic(model_fp32)
\`\`\`

### QAT (Quantization-Aware Training)

在训练过程中模拟量化误差，让模型适应量化后的精度损失：

\`\`\`python
model.train()
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
model = quant.prepare_qat(model)
# ... 训练 ...
model = quant.convert(model.eval())
\`\`\`

### 动态量化 vs 静态量化

| | 动态量化 | 静态量化 |
|------|----------|----------|
| 量化时机 | 推理时动态计算 scale/z | 推理前预先计算 scale/z |
| 需要校准数据 | 否 | 是 |
| 速度提升 | 中等 | 更大 |
| 常用场景 | LSTM, Transformer (权重 INT8, 激活 FP16) | CNN |

---

## 先进量化方法

### GPTQ

逐列量化权重，每次量化一列后用剩余列的更新来补偿量化误差：

$$w_{:,j}^q = \\text{quant}(w_{:,j}), \\quad \\delta = \\frac{w_{:,j} - w_{:,j}^q}{[H^{-1}]_{jj}}$$

$$w_{:,j+1:} \\leftarrow w_{:,j+1:} - \\delta \\cdot H_{j,j+1:}^{-1}$$

### AWQ (Activation-aware Weight Quantization)

观察：不是所有权重对结果同样重要。AWQ 根据激活值的幅度缩放重要通道的权重：

$$\\mathbf{s} = \\mathbf{s}_{\\mathbf{x}}^\\alpha, \\quad \\alpha \\in [0, 1]$$

对重要通道使用更大的 scale（保留更多精度），对不重要的通道使用更小的 scale。

## 精度对比

| 格式 | 位数 | 精度损失 | 推理加速 |
|------|------|----------|----------|
| FP32 | 32 | 0 (基准) | 1× |
| FP16 | 16 | 极小 | 2-4× |
| INT8 | 8 | 小 | 4-8× |
| INT4 (GPTQ) | 4 | 中等 | 8-16× |
| NF4 (QLoRA) | 4 | 中 | 8-16× |
`;export{n as default};
