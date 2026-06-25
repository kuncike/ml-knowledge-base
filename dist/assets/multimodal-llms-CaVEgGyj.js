const n=`# Multimodal LLMs (Qwen-VL / CogVLM / InternVL)

## 核心思想

多模态大模型 = LLM + 视觉编码器 + 对齐层。关键是**怎么把视觉信息喂给 LLM**——三种主流设计：LLaVA 用简单的 MLP 投影（便宜有效），CogVLM 给每个 Transformer 层加一个独立的**视觉专家**（效果最好但参数多），Qwen-VL 用 Q-former 风格的 resampler 压缩视觉 token（平衡效果与效率）。

---

## 数学定义与原理解析

### LLaVA — MLP 投影

最简洁的设计：用一/两层 MLP 将 ViT 的特征投影到 LLM 的嵌入空间：

$$
\\mathbf{H}_{vision} = \\text{ViT}(I) \\in \\mathbb{R}^{N \\times d_v}, \\quad \\mathbf{H}_{proj} = \\text{MLP}(\\mathbf{H}_{vision}) \\in \\mathbb{R}^{N \\times d_{llm}}
$$

投影后的视觉 token 和文本 token 拼接，作为 LLM 的输入。LLaVA-1.5 只需 13B 参数就能匹敌许多更大的模型。

### CogVLM — 视觉专家

在每个 Transformer 层内添加一个视觉专家（Visual Expert）模块：

$$
\\mathbf{x}_{text}^{(l)} = \\text{FFN}(\\text{SelfAttn}(\\mathbf{x}^{(l-1)}))
$$

$$
\\mathbf{x}_{vision}^{(l)} = \\text{VisualExpert}(\\text{SelfAttn}([\\mathbf{x}_{text}; \\mathbf{x}_{vision}]^{(l-1)}))
$$

Visual Expert 是一个独立的 QKV 投影 + FFN，参数约为 LLM 层参数量的一半。视觉 token 和文本 token 在注意力层交互，但在 FFN 层分开处理。

### Qwen-VL — Resampler 压缩

ViT 输出的视觉 token 数量可能很大（$14 \\times 14 = 196$ 或更多）。Qwen-VL 用 **resampler**（类似于 BLIP-2 的 Q-Former）将可变数量的视觉 token 压缩为固定长度的"可学习查询向量"：

$$
\\mathbf{H}_{compressed} = \\text{Resampler}(\\mathbf{H}_{vision}) \\in \\mathbb{R}^{K \\times d_{llm}}
$$

$K$ 通常为 64-256，比原始 ViT 输出小很多。使用 cross-attention 机制，$K$ 个 learnable queries 去 attend 视觉 token。

---

## 可视化展示

### 多模态 LLM 架构对比

\`\`\`mermaid
graph TD
    subgraph LLaVA["LLaVA"]
        L_ViT["ViT-L"] --> L_MLP["MLP 投影"] --> L_LLM["LLaMA"]
    end
    subgraph CogVLM["CogVLM"]
        C_ViT["ViT-G"] --> C_Adapter["MLP Adapter"]
        C_Adapter --> C_LLM["LLaMA + 每层的 Visual Expert"]
    end
    subgraph QwenVL["Qwen-VL"]
        Q_ViT["ViT-G"] --> Q_Resampler["Resampler (cross-attn)"]
        Q_Resampler --> Q_LLM["Qwen"]
    end
\`\`\`

### 多模态 LLM 性能

\`\`\`echarts
return {
  title: { text: '多模态 LLM MMBench 得分', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['LLaVA-1.5 7B', 'LLaVA-1.5 13B', 'CogVLM 17B', 'Qwen-VL-Chat', 'InternVL 26B'] },
  yAxis: { type: 'value', min: 60, max: 85, name: 'MMBench Score' },
  series: [{
    type: 'bar',
    data: [65.4, 68.9, 77.6, 72.5, 82.3],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — LLaVA 风格多模态前向传播

\`\`\`python
import torch
import torch.nn as nn

class LLaVAForward(nn.Module):
    def __init__(self, vision_encoder, llm, projector):
        super().__init__()
        self.vision_encoder = vision_encoder  # ViT
        self.llm = llm                          # LLaMA
        self.projector = projector              # MLP: d_v → d_llm

    def forward(self, images, input_ids, attention_mask):
        # 1. 视觉编码
        with torch.no_grad():
            vision_feat = self.vision_encoder(images)  # [B, N_v, d_v]
        vision_emb = self.projector(vision_feat)        # [B, N_v, d_llm]

        # 2. 文本嵌入 (含 <image> token 占位)
        text_emb = self.llm.model.embed_tokens(input_ids)  # [B, N_t, d_llm]

        # 3. 将视觉嵌入插入 <image> token 位置
        image_positions = (input_ids == IMAGE_TOKEN_ID)
        # ... (实际实现中需要 broadcast 到对应位置)

        # 4. 拼接: [vision_tokens, text_tokens]
        inputs_embeds = torch.cat([vision_emb, text_emb], dim=1)
        outputs = self.llm(inputs_embeds=inputs_embeds)
        return outputs.logits
\`\`\`

### CogVLM 视觉专家

\`\`\`python
class VisualExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        # 视觉专用的 QKV（只处理视觉 token）
        self.qkv_vision = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        # 视觉专用的 FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size))

    def forward(self, text_hidden, vision_hidden):
        # vision_hidden: 经过 joint self-attention 后的视觉 token
        # 用视觉专用 QKV 投影
        qkv_v = self.qkv_vision(vision_hidden)
        # ... attention 操作
        # 视觉专用 FFN
        return vision_hidden + self.ffn(vision_hidden)
\`\`\`
`;export{n as default};
