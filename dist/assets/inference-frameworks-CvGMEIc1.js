const n=`# TensorRT / ONNX / OpenVINO / vLLM / llama.cpp

## TensorRT (NVIDIA)

### 核心原理

对训练好的模型做**图优化**和**精度降级**，生成针对特定 GPU 架构高度优化的推理引擎。

### 优化技术

- **层融合**：Conv + BN + ReLU → 单个 kernel
- **张量内存优化**：减少显存分配/释放
- **内核自动调优**：为每层选择最优 CUDA kernel
- **INT8/FP8 量化**：结合校准数据的精度优化

### 使用流程

\`\`\`
PyTorch → ONNX → TensorRT Engine → 推理
         或
PyTorch → torch2trc → TensorRT Engine
\`\`\`

---

## ONNX Runtime

### 核心思想

ONNX 是模型交换格式，ONNX Runtime 是跨平台推理引擎，支持多种后端（CPU、CUDA、TensorRT、OpenVINO）。

\`\`\`python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(["output_name"], {"input_name": input_data})
\`\`\`

---

## OpenVINO (Intel)

### 定位

Intel 的推理优化工具，针对 Intel CPU/GPU/VPU 深度优化。

### 关键优化

- **Post-training Optimization Tool (POT)**：INT8 量化
- **Neural Network Compression Framework (NNCF)**：QAT + 剪枝
- 针对 Xeon CPU 的向量化指令优化

---

## vLLM

### 核心创新

- **PagedAttention**：解决 KV Cache 内存碎片
- **Continuous Batching**：动态合并请求到 batch（不等整个 batch 完成）
- **高吞吐**：比 HuggingFace Transformers 提升 10-20×

\`\`\`python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b")
outputs = llm.generate(prompts, SamplingParams(temperature=0.8))
\`\`\`

---

## llama.cpp

### 核心思想

纯 C/C++ 实现的 LLaMA 推理，优化 CPU 推理速度。

### 关键优化

- **INT4/INT8 量化**：GGUF 格式
- **mmap 加载**：模型快速加载，支持部分加载
- **Apple Silicon 优化**：Accelerate + Metal
- **内存效率**：可在 RAM 有限的设备上运行

\`\`\`bash
./llama-cli -m model.gguf -p "Hello, my name is"
\`\`\`

## 对比总结

| 框架 | 适用硬件 | 量化支持 | 吞吐优化 | 易用性 |
|------|----------|----------|----------|--------|
| TensorRT | NVIDIA GPU | INT8/FP8/FP16 | 极致 | 中 |
| ONNX Runtime | 全平台 | INT8/FP16 | 中 | 高 |
| OpenVINO | Intel CPU/GPU | INT8/FP16 | 高 | 中 |
| vLLM | NVIDIA/AMD GPU | AWQ/GPTQ | 极高 | 高 |
| llama.cpp | CPU/Apple Silicon | INT4/INT8 | 中 | 极高 |

## 选择建议

- **GPU 部署（最大吞吐）**：TensorRT 或 vLLM
- **CPU 部署**：ONNX Runtime 或 OpenVINO
- **个人设备/边缘**：llama.cpp
- **LLM 在线服务**：vLLM 或 TGI
`;export{n as default};
