const n=`# T2I Adapters (IP-Adapter / GLIGEN / InstantID)

## 核心思想

ControlNet 让图像生成可以被**空间条件**（边缘图、深度图）控制。但更实用的场景是：用户有一张参考图想要"保持身份/风格"生成新图。IP-Adapter 用**图像作为 prompt**（而非文本），通过解耦的交叉注意力注入图像特征。GLIGEN 用**门控自注意力**实现可变数量的布局条件。InstantID 更进一步——用一张人脸照实现**零样本身份保持**的图像生成。

共同原理：**在不破坏预训练扩散模型的前提下，注入额外的控制信号**。

---

## 数学定义与原理解析

### IP-Adapter — 解耦交叉注意力

传统交叉注意力将文本特征 $c_{text}$ 和视觉 token 交互。IP-Adapter **新增一条独立的交叉注意力路径**处理图像特征 $c_{image}$：

目标函数：

$$
\\epsilon_\\theta(\\mathbf{z}_t, c_{text}, c_{image})
$$

IP-Adapter 的注意力层：

$$
\\mathbf{Z}_{new} = \\text{Softmax}\\left(\\frac{\\mathbf{Q} \\mathbf{K}_{text}^T}{\\sqrt{d}}\\right) \\mathbf{V}_{text} + \\lambda \\cdot \\text{Softmax}\\left(\\frac{\\mathbf{Q} \\mathbf{K}_{image}^T}{\\sqrt{d}}\\right) \\mathbf{V}_{image}
$$

关键设计：**可训练的只有 image cross-attention 的投影层**（$\\mathbf{W}_K^{image}, \\mathbf{W}_V^{image}$），UNet 其他参数全部冻结。$\\lambda$ 控制图像条件强度。

### GLIGEN — 门控自注意力

在 UNet 的门控自注意力层中插入**grounding tokens**（表示空间位置条件）：

$$
\\mathbf{A}_{gated} = \\text{Softmax}\\left(\\frac{\\mathbf{Q} \\mathbf{K}^T}{\\sqrt{d}} + \\gamma \\cdot \\mathbf{G}\\right) \\mathbf{V}
$$

其中 $\\mathbf{G}$ 是 grounding attention mask——指定哪些 token 应该 attend 到 grounding tokens。$\\gamma$ 是可学习的门控参数（初始化为 0，训练中逐渐生效）。

Grounding token 的表示：$\\mathbf{h}_g = \\text{FFN}([\\text{Fourier}(x,y,w,h), \\mathbf{e}_{text}])$

### InstantID — 身份保持

用专用的 **ID Encoder**（基于人脸识别模型）提取人脸身份嵌入，与 CLIP 文本嵌入一起指导扩散模型：

$$
c_{combined} = \\text{concat}(c_{id}, c_{text})
$$

同时使用轻量级 ControlNet 适配器保持面部关键点的一致性。

---

## 可视化展示

### IP-Adapter 架构

\`\`\`mermaid
graph TD
    IMG["参考图像"] --> IMG_ENC["CLIP Image Encoder"] --> IMG_FEAT["c_image"]
    TEXT["文本 Prompt"] --> TXT_ENC["CLIP Text Encoder"] --> TXT_FEAT["c_text"]

    subgraph UNet["Frozen UNet (SD)"]
        SA["Self-Attention"]
        CA_TXT["Cross-Attn (text)"]
        CA_IMG["New Cross-Attn (image, 可训练)"]
    end

    Z_T["z_t (噪声 latent)"] --> SA
    SA --> CA_TXT
    CA_TXT --> CA_IMG
    TXT_FEAT --> CA_TXT
    IMG_FEAT --> CA_IMG
    CA_IMG --> ADD["+"] --> OUT["ϵ̂"]
\`\`\`

### 适配器对比

\`\`\`echarts
return {
  title: { text: 'T2I 适配器对比', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['Prompt 控制', '空间控制', '身份保持', '可训练参数'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '能力得分' },
  legend: { data: ['ControlNet', 'GLIGEN', 'IP-Adapter', 'InstantID'] },
  series: [
    { name: 'ControlNet', type: 'bar', data: [0.3, 1, 0, 0.5], itemStyle: { color: '#2c3e50' } },
    { name: 'GLIGEN', type: 'bar', data: [0.5, 1, 0, 0.6], itemStyle: { color: '#2980b9' } },
    { name: 'IP-Adapter', type: 'bar', data: [0.3, 0, 1, 0.1], itemStyle: { color: '#d35400' } },
    { name: 'InstantID', type: 'bar', data: [0.3, 0.5, 1, 0.3], itemStyle: { color: '#16a085' } }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — IP-Adapter 解耦交叉注意力

\`\`\`python
import torch
import torch.nn as nn

class IPAdapterCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads=8):
        super().__init__()
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)
        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads

    def forward(self, x, image_context, scale=1.0):
        # x: [B, N, D] — UNet 的 latent token
        # image_context: [B, M, D'] — CLIP 图像特征
        B = x.shape[0]
        Q = self.to_q(x).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.to_k(image_context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.to_v(image_context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = Q @ K.transpose(-2, -1) * (self.head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.head_dim)
        return self.to_out(out) * scale
\`\`\`

### GLIGEN Grounding Token

\`\`\`python
class GLIGENGrounding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, attn_map, grounding_tokens, grounding_mask):
        # grounding_tokens 插入 latent 序列中
        # grounding_mask: [B, N, N_g] — 哪些 latent token attend 到 grounding token
        gate = torch.tanh(self.gamma)
        return attn_map + gate * grounding_mask
\`\`\`
`;export{n as default};
