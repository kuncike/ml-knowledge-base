const n=`# Gemma / Phi / Falcon (小型大语言模型)

## 核心思想

不是所有场景都需要 GPT-4 的 1.7T 参数。三款代表性小型模型展示了三条路径：Gemma（Google）用知识蒸馏从 Gemini 压低到 2B/7B，Phi（Microsoft）用**教科书质量数据**让 1.3B 模型在代码和推理上匹敌更大模型，Falcon（TII）证明了**精心清洗的大规模数据**+ MQA 也能在 7B 级别达到顶尖效果。

共同启示：**数据质量 > 模型大小**。高质量数据的蒸馏和筛选可以让小模型完成很多以前被认为需要大模型的任务。

---

## 数学定义与原理解析

### Gemma — 从 Gemini 蒸馏

Gemma 继承 Gemini 的架构选择，但通过知识蒸馏大幅压缩：

- **GeGLU 激活**：GELU 的门控变体（与 LLaMA 的 SwiGLU 类似）
- **Multi-Query Attention**：所有头共享一组 K, V，大幅减小 KV Cache
- **RMSNorm**：移除了 LayerNorm 的均值中心化操作
- **RoPE**：旋转位置编码

蒸馏损失：
$$
\\mathcal{L} = \\alpha \\cdot \\mathcal{L}_{CE}(y, y_{true}) + (1-\\alpha) \\cdot \\mathcal{L}_{KL}(p_{student} \\| p_{teacher})
$$

### Phi — 数据中心方法

Phi 的核心发现：**教科书的因果链**比互联网文本更适合模型学习推理。数据策略：

1. 用 GPT-4 生成"教科书"——带推理步骤的高质量文本
2. 筛选：保留有明确因果逻辑的内容
3. 合成 + 精选 ≈ 1.4T tokens 训练 Phi-3

Phi-3.8B 在 HumanEval 上匹敌 LLaMA 3 70B。

### Falcon — 大规模数据 + MQA

Falcon 技术栈：
- **RefinedWeb**：从 CommonCrawl 中经过严格过滤（语言识别、困惑度筛选、去重）产生的 5T token 数据集
- **FlashAttention + MQA**：减少训练显存占用
- **Parallel Attention**：在 attention 和 FFN 间用并行路径
- **BPE tokenizer**（而非 SentencePiece）

---

## 可视化展示

### 小型模型设计哲学

\`\`\`mermaid
graph TD
    subgraph Gemma["Gemma (Google)"]
        G1["大模型蒸馏"] --> G2["GeGLU + MQA + RMSNorm"] --> G3["2B/7B 参数"]
    end
    subgraph Phi["Phi (Microsoft)"]
        P1["教科书数据合成"] --> P2["因果链筛选"] --> P3["1.3B/3.8B 参数"]
    end
    subgraph Falcon["Falcon (TII)"]
        F1["RefinedWeb 5T tokens"] --> F2["MQA + Parallel Attn"] --> F3["7B/40B 参数"]
    end
\`\`\`

### 小型模型性能

\`\`\`echarts
return {
  title: { text: '小型 LLM HumanEval 对比', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['LLaMA3-8B', 'Gemma-7B', 'Phi-3.8B', 'Falcon-7B', 'Mistral-7B'] },
  yAxis: { type: 'value', min: 0, max: 70, name: 'HumanEval Pass@1 (%)' },
  series: [{
    type: 'bar',
    data: [62.2, 54.3, 68.1, 34.7, 52.6],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top', formatter: '{c}%' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### HuggingFace 加载示例

\`\`\`python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Gemma
gemma = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b", torch_dtype="auto", device_map="auto")
# Phi-3
phi = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True, torch_dtype="auto", device_map="auto")
# Falcon
falcon = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b-instruct",
    trust_remote_code=True, torch_dtype="auto", device_map="auto")

def generate(model, tokenizer, prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens,
                              temperature=0.7, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
\`\`\`

### Multi-Query Attention 示意

\`\`\`python
import torch
import torch.nn as nn

class MultiQueryAttention(nn.Module):
    """MQA: 多个 Q head 共享一组 K, V"""
    def __init__(self, dim=512, n_query_heads=8, n_kv_heads=1):
        super().__init__()
        self.n_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_query_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, n_kv_heads * self.head_dim)  # 小 K
        self.W_v = nn.Linear(dim, n_kv_heads * self.head_dim)  # 小 V
        self.W_o = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # K, V 广播到所有 Q heads
        K = K.expand(-1, self.n_heads, -1, -1)
        V = V.expand(-1, self.n_heads, -1, -1)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)
\`\`\`
`;export{n as default};
