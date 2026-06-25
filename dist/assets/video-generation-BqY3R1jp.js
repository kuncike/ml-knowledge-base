const n=`# Video Generation (Sora / Runway / Pika / SVD)

## 核心思想

从图像生成到视频生成，核心挑战是**时间一致性**——帧与帧之间的人物身份、物体运动、光照变化必须物理合理。Sora 将视频视为**时空块**（spacetime patches），纯粹用 Diffusion Transformer 建模——没有任何显式的时间建模组件。Runway Gen-2/3 用**多模态条件**（文本 + 图像 + 结构）驱动视频生成。Pika 专注于**交互式视频编辑**（局部修改、风格转换）。SVD (Stable Video Diffusion) 则从 SD 权重初始化，加上时间层实现图像到视频的生成。

---

## 数学定义与原理解析

### Sora — 视频即时空块

将视频表示为 $T \\times H \\times W \\times C$ 的 4D 张量，直接 3D patchify：

$$
\\text{Video}(T, H, W, C) \\Rightarrow \\text{序列化的 Spacetime Patches}
$$

关键设计：压缩视频到 latent space，然后在 latent 上做 DiT：

$$
\\mathbf{z}_{video} = \\text{VAE}_{3D}(video) \\in \\mathbb{R}^{B \\times T' \\times C \\times H' \\times W'}
$$

与图像 DiT 的区别：注意力在全部 $T' \\times H' \\times W'$ 个 token 上进行——**全时空注意力**处理运动。

### Stable Video Diffusion (SVD)

从 SD 2.1 权重初始化，添加时间层：

- 在 UNet 的每个空间 block 中插入**时间卷积**和**时间注意力**
- 输入是单张图像 + 噪声 latent 序列
- 时间注意力在帧之间交换信息：

$$
\\text{attn}(Q_t, K_{t'}, V_{t'}) = \\text{Softmax}\\left(\\frac{Q_t K_{t'}^T}{\\sqrt{d}}\\right) V_{t'}
$$

### 级联生成策略

大多数视频生成系统使用级联：
1. 先生成低帧率、低分辨率的"关键帧"
2. 用插值模型生成中间帧
3. 用超分模型提升空间分辨率

---

## 可视化展示

### 视频生成架构对比

\`\`\`mermaid
graph TD
    subgraph Sora["Sora (OpenAI)"]
        S1["原始视频"] --> S2["3D VAE 压缩"] --> S3["Spacetime Patches"] --> S4["DiT (全时空注意力)"] --> S5["生成视频"]
    end
    subgraph SVD["SVD (Stability)"]
        V1["单张图像"] --> V2["UNet + 时间卷积/注意力层"] --> V3["视频 latent 序列"] --> V4["VAE Decoder"] --> V5["生成视频"]
    end
\`\`\`

### 视频生成模型对比

\`\`\`echarts
return {
  title: { text: '视频生成模型能力对比', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['时长', '分辨率', '运动质量', '文本遵循'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '相对得分' },
  legend: { data: ['SVD', 'Runway Gen-3', 'Pika 1.0', 'Sora'] },
  series: [
    { name: 'SVD', type: 'bar', data: [0.3, 0.5, 0.6, 0.3], itemStyle: { color: '#2980b9' } },
    { name: 'Runway Gen-3', type: 'bar', data: [0.5, 0.7, 0.7, 0.6], itemStyle: { color: '#2c3e50' } },
    { name: 'Pika 1.0', type: 'bar', data: [0.2, 0.6, 0.5, 0.5], itemStyle: { color: '#d35400' } },
    { name: 'Sora', type: 'bar', data: [1, 1, 0.9, 0.9], itemStyle: { color: '#16a085' } }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

---

## 核心代码实现

### SVD 风格的时间注意力

\`\`\`python
import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    """在帧之间交换信息的时间注意力"""
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        # 重塑: [B*H*W, T, C]
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C)

        qkv = self.qkv(x).view(B * H * W, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)  # [B*H*W, H, T, D]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = torch.softmax(q @ k.transpose(-2, -1) * (self.head_dim ** -0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B * H * W, T, C)
        out = self.proj(out)

        # 重塑回: [B, T, C, H, W]
        return out.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)
\`\`\`
`;export{n as default};
