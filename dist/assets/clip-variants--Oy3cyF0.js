const n=`# CLIP Variants (SigLIP / Florence / EVA-CLIP)

## 核心思想

CLIP 的对比学习损失（InfoNCE）需要超大 batch（32k+）来提供足够多的负样本。SigLIP 用一个简单而优雅的替代方案省去了大 batch 的需求——把 softmax 对比 loss 改成 per-sample 的 sigmoid loss，每个文本-图像对独立分类"是否匹配"。Florence 则将 CLIP 扩展到**统一视觉表示**——同一个编码器适用于分类、检测、分割、检索。EVA-CLIP 探索了 CLIP 的缩放极限——更大的 ViT + 更强的预训练策略。

---

## 数学定义与原理解析

### SigLIP — Sigmoid Loss 替代对比 Loss

CLIP 将 batch 内的所有对放在一起做 softmax：

$$
\\mathcal{L}_{CLIP} = -\\frac{1}{B} \\sum_i \\log \\frac{\\exp(\\tau \\cdot x_i^T y_i)}{\\sum_j \\exp(\\tau \\cdot x_i^T y_j)}
$$

SigLIP 将每个 $(x_i, y_j)$ 对独立处理——是正样本就希望 sigmoid 输出大，负样本就希望小：

$$
\\mathcal{L}_{SigLIP} = -\\frac{1}{B^2} \\sum_{i,j} \\log \\sigma(z_{ij} \\cdot \\tau \\cdot (x_i^T y_j + b))
$$

其中 $z_{ij} = 1$ 当 $i=j$（正样本），否则 $z_{ij} = -1$。关键：不再需要 batch 级别的 softmax 归一化，小 batch 也能训。

### Florence — 统一视觉表示

引入**动态头**（dynamic head）——不是为每个任务设计单独的输出头，而是用一个统一的表示层 + 任务特定的轻量适配：

$$
\\mathbf{v} = \\text{FlorenceEncoder}(x), \\quad y_{task} = \\text{TaskHead}_{task}(\\mathbf{v})
$$

Florence 在 900M 图像-文本对上预训练，覆盖分类/检测/分割/VQA/检索等多个任务。

### EVA-CLIP — 缩放策略

- **EVA-01**：用 1B 参数的 ViT-G（超过 ViT-L 10×）做 CLIP 训练
- **EVA-02**：引入 masked image modeling (MIM) 辅助任务 + 更长的训练
- 关键发现：CLIP 可以在更大模型上持续受益，未出现性能饱和

---

## 可视化展示

### CLIP vs SigLIP 损失结构

\`\`\`mermaid
graph TD
    subgraph CLIP["CLIP (InfoNCE)"]
        C1["Batch 内所有对"] --> C2["Softmax 归一化"] --> C3["对正样本求 -log"]
    end
    subgraph SigLIP["SigLIP"]
        S1["每个 (图,文) 对独立"] --> S2["Sigmoid: 正→1, 负→0"] --> S3["二元交叉熵"]
    end
    C1 -.->|"依赖大batch"| CLIP
    S1 -.->|"小batch可用"| SigLIP
\`\`\`

### CLIP 家族性能

\`\`\`echarts
return {
  title: { text: 'CLIP 变体 ImageNet Zero-Shot Top-1', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['CLIP ViT-B', 'CLIP ViT-L', 'SigLIP ViT-L', 'EVA-CLIP ViT-G', 'Florence-2'] },
  yAxis: { type: 'value', min: 65, max: 90, name: 'Zero-Shot Top-1 (%)' },
  series: [{
    type: 'bar',
    data: [68.3, 75.5, 80.5, 82.0, 85.8],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top', formatter: '{c}%' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — SigLIP Loss

\`\`\`python
import torch
import torch.nn.functional as F

def siglip_loss(img_emb, text_emb, temperature=1.0, bias=0.0):
    """
    img_emb: [B, D] — 图像编码 (已归一化)
    text_emb: [B, D] — 文本编码 (已归一化)
    """
    B = img_emb.shape[0]

    # 所有对 (i,j) 的 logits
    logits = temperature * (img_emb @ text_emb.T + bias)  # [B, B]

    # 正样本标签: 对角线为 1, 其余为 -1
    labels = 2 * torch.eye(B, device=img_emb.device) - 1  # [B, B]

    # Sigmoid loss: -log σ(z · l)
    loss = -F.logsigmoid(labels * logits)
    return loss.mean()
\`\`\`

### SigLIP 训练循环

\`\`\`python
class SigLIPModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, dim=768):
        super().__init__()
        self.img_enc = image_encoder
        self.txt_enc = text_encoder
        self.temperature = nn.Parameter(torch.ones(1) * 10.0)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, images, texts):
        img_emb = F.normalize(self.img_enc(images), dim=-1)
        txt_emb = F.normalize(self.txt_enc(texts), dim=-1)
        return siglip_loss(img_emb, txt_emb, self.temperature, self.bias)
\`\`\`
`;export{n as default};
