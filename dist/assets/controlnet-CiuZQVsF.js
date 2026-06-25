const n=`# ControlNet

## 核心思想

ControlNet 为文生图模型（如 Stable Diffusion）增加了**空间条件控制**——用户可以提供边缘图、姿态骨架、深度图等作为额外输入，精确控制生成图像的构图。

## 架构

### 双副本设计

\`\`\`
原始 SD 编码器块 (锁定, 不参与训练)
      ↓
      复制一份 → ControlNet 可训练副本
      ↓
零卷积层 (1×1 Conv, 权重和偏置初始化为 0)
      ↓
  +   ← 加回原始 SD 编码器块的输出
\`\`\`

**零卷积**使得训练开始时 ControlNet 不干扰原始 SD（恒等映射），训练过程中逐步学习条件信号。

## 支持的控制类型

| 控制类型 | 提取方法 | 控制内容 |
|----------|----------|----------|
| Canny Edge | Canny 边缘检测 | 轮廓/构图 |
| HED Boundary | HED 网络 | 柔和的边界 |
| Depth Map | MiDaS / ZoeDepth | 空间深度关系 |
| Normal Map | 表面法线估计 | 3D 表面朝向 |
| OpenPose | 姿态估计 | 人物姿态 |
| Scribble | 用户涂鸦 | 自由绘制 |
| Segmentation | SAM / 语义分割 | 区域语义 |
| Line Art | 线稿提取 | 动漫/素描风格 |

## 训练目标

$$L = \\mathbb{E}_{z_0, t, c_t, c_f, \\epsilon} \\left[ \\|\\epsilon - \\epsilon_\\theta(z_t, t, c_t, c_f)\\|^2 \\right]$$

- $c_t$：文本条件（prompt）
- $c_f$：ControlNet 的条件信号（边缘图等）

## 多条件组合

可以叠加多个 ControlNet（Multi-ControlNet），例如：

\`\`\`
OpenPose (控制人物姿态) + Depth Map (控制空间布局) + Canny Edge (控制细节轮廓)
\`\`\`

每个 ControlNet 独立处理一种条件，输出简单相加。

## 关键公式

第 $l$ 层的输出：

$$\\mathbf{y}_l = \\mathcal{F}_l(\\mathbf{x}) + \\mathcal{Z}(\\mathcal{F}'_l(\\mathbf{x} + \\mathcal{Z}(\\mathbf{c}_f)))$$

其中 $\\mathcal{F}_l$ 是原始 SD 块，$\\mathcal{F}'_l$ 是 ControlNet 副本，$\\mathcal{Z}$ 是零卷积。

## 应用场景

- **建筑设计**：线稿 → 效果图
- **游戏美术**：涂鸦 → 完整场景
- **电商**：产品草图 → 真实渲染
- **影视**：分镜 → 概念图

ControlNet 证明了**空间条件注入**比单纯的文本条件提供了更精确的生成控制。
`;export{n as default};
