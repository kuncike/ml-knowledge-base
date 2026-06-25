const n=`# LLM Serving Frameworks (vLLM / SGLang / TGI / LMDeploy)

## 核心思想

裸跑 HuggingFace Transformers 做 LLM 推理，GPU 利用率通常不到 30%。专业推理框架解决三大痛点：**KV Cache 内存管理**（vLLM 的 PagedAttention）、**请求调度**（continuous batching）、**前缀缓存和结构化输出**（SGLang 的 RadixAttention）。vLLM 将 OS 的虚拟内存/分页思想引入 KV Cache，SGLang 将常见前缀（如 few-shot prompt）只存一次，LMDeploy 用 C++ 重写了关键路径以最小化 Python 开销。

---

## 数学定义与原理解析

### PagedAttention (vLLM)

将 KV Cache 切分为固定大小的 **block**（类似 OS 的 page），非连续存储：

$$
\\text{KV Cache} = \\{\\text{Block}_1, \\text{Block}_2, \\ldots, \\text{Block}_k\\}
$$

每个 block 存 $B$ 个 token 的 K 和 V。注意力计算时"拼接"需要的 block：

$$
A_{ij} = \\frac{\\exp(Q_i K_j^T / \\sqrt{d})}{\\sum_{j'} \\exp(Q_i K_{j'}^T / \\sqrt{d})}
$$

但与 FlashAttention 的 tile 计算完美兼容——每个 block 内部独立计算部分注意力。

**核心优势**：
- 按需分配（不像 contiguous allocation 必须预留最大长度）
- 零碎片化（blocks 可以来自不同物理位置）
- 内存共享（beam search 或 parallel sampling 时共享 prompt 的 KV Cache blocks）

### RadixAttention (SGLang)

LRU 缓存整棵前缀树——相同的 prompt 前缀只计算一次 KV Cache：

\`\`\`
Prefix Tree Cache:
"Translate to French: "  ← 根节点 (prompt 前缀)
├── "The cat sits"        ← 子节点 (用户 query1)
│   └── generated tokens...
└── "A dog runs"          ← 子节点 (用户 query2)
    └── generated tokens...
\`\`\`

新请求到达时，在 Trie 中查找最长匹配前缀 → 直接复用其 KV Cache → 只计算新 token。

### 框架对比

| 框架 | 核心优化 | 语言 | 特色 |
|------|---------|------|------|
| vLLM | PagedAttention | Python/C++ | 生态最完善 |
| SGLang | RadixAttention + Structured Outputs | Python | 多轮对话最优 |
| TGI (HF) | Watermarking + FlashAttn | Rust/Python | HuggingFace 官方 |
| LMDeploy | TurboMind (C++ runtime) | C++/Python | 推理延迟最低 |
| llama.cpp | 纯 CPU + 量化 | C++ | 个人设备 |

---

## 可视化展示

### PagedAttention 内存管理

\`\`\`mermaid
graph TD
    subgraph Traditional["传统 (Contiguous)"]
        T1["Req1: ████████░░░░ (预留 max_len)"]
        T2["Req2: ██░░░░░░░░░░ (预留 max_len)"]
        T3["大量内部碎片"]
    end
    subgraph Paged["PagedAttention (vLLM)"]
        P1["Block 1 ██"] --> P2["Block 3 ██"] --> P3["Block 7 ███"]
        P4["Block 2 █"] --> P5["Block 5 ██"]
        P6["按需分配, 无碎片"]
    end
\`\`\`

### 推理框架性能

\`\`\`echarts
return {
  title: { text: '推理框架吞吐量对比 (LLaMA2-7B, A100)', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['HF Transformers', 'TGI', 'vLLM', 'SGLang', 'LMDeploy'] },
  yAxis: { type: 'value', min: 0, max: 5000, name: 'Throughput (tokens/s)' },
  series: [{
    type: 'bar',
    data: [250, 1800, 3200, 3800, 4200],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

---

## 核心代码实现

### vLLM 异步推理

\`\`\`python
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# 1. 离线批处理
llm = LLM(model="meta-llama/Llama-2-7b-hf",
          max_model_len=4096,
          gpu_memory_utilization=0.9)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

prompts = ["Explain quantum computing:", "Write a Python function to:"]
outputs = llm.generate(prompts, sampling_params)
for o in outputs:
    print(o.outputs[0].text)

# 2. 在线异步服务
async def serve():
    engine_args = AsyncEngineArgs(model="meta-llama/Llama-2-7b-hf")
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    results_generator = engine.generate("What is AI?", sampling_params, request_id="req-1")
    async for request_output in results_generator:
        if request_output.finished:
            print(request_output.outputs[0].text)
\`\`\`

### SGLang 前缀缓存

\`\`\`python
import sglang as sgl

@sgl.function
def translate(s, text, target_lang):
    s += "You are a translator. Translate to " + target_lang + ":\\n\\n"
    s += text + "\\n\\nTranslation:"
    s += sgl.gen("translation", max_tokens=256)

# 多个请求共享 "You are a translator..." 前缀 → 自动缓存 KV
# SGLang 的 RadixAttention 自动管理前缀缓存, 无需手动操作
\`\`\`

### LMDeploy TurboMind Python API

\`\`\`python
from lmdeploy import pipeline, TurbomindEngineConfig

# C++ 后端的低延迟推理
backend_config = TurbomindEngineConfig(
    model_name="internlm2",
    tp=1,                     # tensor parallel (单卡)
    session_len=4096,
    max_batch_size=64)
pipe = pipeline("internlm/internlm2-chat-7b", backend_config=backend_config)

response = pipe(["你好, 请介绍一下深度学习"],
                gen_config=dict(top_p=0.8, temperature=0.7, max_new_tokens=512))
print(response[0].text)
\`\`\`
`;export{n as default};
