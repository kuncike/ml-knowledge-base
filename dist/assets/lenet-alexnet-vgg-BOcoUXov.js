const n=`# LeNet / AlexNet / VGG

## LeNet-5 (1998)

CNN 的鼻祖，用于手写数字识别 (MNIST)。

\`\`\`
Input(32×32) → Conv(6,5×5) → Pool(2×2) → Conv(16,5×5)
→ Pool(2×2) → FC(120) → FC(84) → FC(10)
\`\`\`

- 参数量约 60K
- 使用 Sigmoid 激活
- 奠定了 Conv → Pool → FC 的经典模式

## AlexNet (2012)

ImageNet 竞赛上一举将 top-5 错误率从 26% 降到 15%，开启了深度学习时代。

### 关键创新

1. **ReLU 激活**：首次在 CNN 中使用（替代 Sigmoid/Tanh）
2. **Dropout**：全连接层使用 p=0.5 的 Dropout 防过拟合
3. **双 GPU 训练**：模型分两半各跑一个 GPU（当时显存限制）
4. **Local Response Normalization (LRN)**：被后来的 BN 取代
5. **数据增强**：随机裁剪 + 水平翻转

### 架构

\`\`\`
Input(227×227×3) → Conv(96,11×11,S=4) → MaxPool
→ Conv(256,5×5,S=1,P=2) → MaxPool
→ Conv(384,3×3) → Conv(384,3×3) → Conv(256,3×3) → MaxPool
→ FC(4096) → Dropout → FC(4096) → Dropout → FC(1000)
\`\`\`

参数量约 60M。

## VGG (2014)

### 核心哲学：简单即美

VGG 只用 $3 \\times 3$ 的卷积核（最小的能捕获空间信息的核），通过堆叠深度来获得大感受野。

两个 $3 \\times 3$ 卷积的感受野 = 一个 $5 \\times 5$ 卷积。
三个 $3 \\times 3$ 卷积的感受野 = 一个 $7 \\times 7$ 卷积。

但参数量更少、非线性更多。

### VGG16 vs VGG19

| 版本 | 层数 | 参数量 |
|------|------|--------|
| VGG16 | 13 Conv + 3 FC | 138M |
| VGG19 | 16 Conv + 3 FC | 143M |

### 问题

FC 层参数量巨大（VGG16 中 $7 \\times 7 \\times 512 \\times 4096 \\approx 102M$，占总参数 74%），后来被 GAP + 1×1 Conv 替代。

## PyTorch 快速使用

\`\`\`python
import torchvision.models as models

lenet = ...  # 手写（torchvision 不内置）
alexnet = models.AlexNet(pretrained=False)
vgg16 = models.vgg16(pretrained=False)
\`\`\`
`;export{n as default};
