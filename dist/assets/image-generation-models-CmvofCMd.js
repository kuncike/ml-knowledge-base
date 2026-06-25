const n=`# Image Generation Models (Midjourney / DALL-E / Imagen)

## 核心思想

Stable Diffusion 是开源扩散模型的标杆，但闭源模型在图像质量和文本遵循度上仍有显著领先。三个闭源模型的共同特征：**更强的文本编码器 + 美学数据筛选 + 多阶段级联**。Midjourney 擅长美学品质（通过偏好数据强化学习微调），DALL-E 3 强在文本遵循度（LLM 重写 prompt），Imagen 证明 T5 编码器远优于 CLIP（但成本更高）。

---

## 数学定义与原理解析

### Imagen — 级联扩散 + T5 编码

两级级联：

1. **Base model**：$64 \\times 64$ 低分辨率生成，以**冻结的 T5-XXL** 文本编码为条件
2. **Super-Resolution model**：$64 \\to 256 \\to 1024$，两个级联的 SR 扩散模型

关键发现：CLIP 文本编码器在文本-图像对齐任务上训练，但**语义理解和推理能力远不如 T5**（T5 是纯文本模型，无图像侧监督）。Imagen 证明了"无条件绑定图像侧的纯文本编码器"可以更好地指导图像生成。

### DALL-E 3 — Prompt 重写

DALL-E 3 的核心创新不在于扩散模型本身，而在于训练了一个**caption rewriter**——将用户的简短 prompt 展开为详细的视觉描述。这解决了扩散模型的一个根本问题：用户写的 prompt 通常很短，而模型训练时的 caption 很长。

\`\`\`
User: "a cat sitting on a chair"
Rewriter: "A photograph of an orange tabby cat with green eyes, sitting gracefully 
_on a wooden kitchen chair with spindle back, soft afternoon light streaming through 
_a window to the left, creating warm highlights on the cat's fur..."
\`\`\`

### Midjourney — 美学微调

Midjourney 的技术细节未公开，但普遍认为其卓越的美学品质来自：
- **偏好优化**：用大量人类美学偏好数据微调（类似 RLHF）
- **多阶段变分自编码器**：比 SD 的 VAE 更高分辨率的 latent space
- **数据策展**：精心挑选的高美学训练数据（ArtStation、专业摄影等）

---

## 可视化展示

### 文本编码器选择的影响

\`\`\`mermaid
graph TD
    PROMPT["用户 Prompt<br/>'a red fox in a forest'"] --> ENC
    ENC["文本编码器选择"] --> CLIP["CLIP (SD)<br/>✅ 图文对齐<br/>❌ 语义推理弱"]
    ENC --> T5["T5-XXL (Imagen)<br/>❌ 无图文对齐<br/>✅ 强大语义理解"]
    ENC --> REWRITE["LLM Rewrite (DALL-E 3)<br/>✅ 自动展开细节<br/>✅ 提升文本遵循"]
\`\`\`

### 文本遵循度对比

\`\`\`echarts
return {
  title: { text: '图像生成模型 DrawBench 文本遵循度', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['SDXL', 'DALL-E 2', 'Midjourney 5', 'Imagen', 'DALL-E 3'] },
  yAxis: { type: 'value', min: 60, max: 95, name: '文本-图像对齐率 (%)' },
  series: [{
    type: 'bar',
    data: [65, 71, 75, 82, 90],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top', formatter: '{c}%' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### Imagen 风格的级联流水线（概念）

\`\`\`python
class ImagenPipeline:
    def __init__(self, t5_encoder, base_unet, sr_unet_256, sr_unet_1024):
        self.t5 = t5_encoder       # 冻结的 T5-XXL
        self.base = base_unet      # 64×64 → 64×64
        self.sr_256 = sr_unet_256  # 64→256
        self.sr_1024 = sr_unet_1024  # 256→1024

    def generate(self, prompt, steps=256):
        # 1. 文本编码 (T5, 幂等)
        text_emb = self.t5.encode(prompt)  # [B, T, 4096]

        # 2. 基础生成: 64×64
        z_64 = self._diffusion_sample(self.base, text_emb, (64, 64), steps)

        # 3. 超分阶段 1: 64→256
        z_256 = self._diffusion_sample(self.sr_256, text_emb, (256, 256), steps,
                                        low_res=z_64)

        # 4. 超分阶段 2: 256→1024
        z_1024 = self._diffusion_sample(self.sr_1024, text_emb, (1024, 1024), steps,
                                         low_res=z_256)
        return z_1024

    def _diffusion_sample(self, model, cond, size, steps, low_res=None):
        """DDIM 采样"""
        z = torch.randn(size)
        for t in self._timesteps(steps):
            noise_pred = model(z, t, cond, low_res=low_res)
            z = self._ddim_step(z, noise_pred, t)
        return z
\`\`\`

### DALL-E 3 风格的 Prompt 重写

\`\`\`python
def rewrite_prompt(user_prompt, llm):
    """用 LLM 将简短 prompt 展开为详细描述"""
    system_prompt = """You are a visual description expert. Given a brief image
description, expand it into a detailed caption covering: subject, action,
setting, lighting, colors, composition, style, and mood. Be specific and vivid."""

    response = llm.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Expand: {user_prompt}"}
    ])
    return response
\`\`\`
`;export{n as default};
