const n=`# SORT / DeepSORT

## SORT (Simple Online and Realtime Tracking)

### 核心思想

将目标跟踪简化为检测+关联。用检测器在每帧找到物体，然后用**卡尔曼滤波**预测位置，用**匈牙利算法**做数据关联。

### 流程

1. **检测**：在当前帧中检测所有目标
2. **预测**：用卡尔曼滤波预测已有轨迹在当前帧的位置
3. **关联**：用匈牙利算法匹配检测框和预测框（IoU 作为代价矩阵）
4. **更新**：匹配成功 → 更新轨迹，未匹配 → 新轨迹或删除

### 状态向量

$$\\mathbf{x} = [u, v, s, r, \\dot{u}, \\dot{v}, \\dot{s}]$$

- $(u, v)$：边界框中心
- $s$：面积（尺度）
- $r$：长宽比
- 带点表示速度分量

### 速度

达到 260 Hz（不含检测器时间）。主要瓶颈在检测而非跟踪。

---

## DeepSORT

### 核心改进：深度外观特征

在 SORT 的基础上加入**外观信息**（ReID 特征），减少 ID Switch。

### 关联度量

结合运动信息和外观信息：

$$c_{i,j} = \\lambda \\cdot d_{motion}(i, j) + (1 - \\lambda) \\cdot d_{appearance}(i, j)$$

- **马氏距离**（运动）：基于卡尔曼滤波的不确定性

$$d_{motion} = (\\mathbf{d}_j - \\mathbf{y}_i)^T \\mathbf{S}_i^{-1} (\\mathbf{d}_j - \\mathbf{y}_i)$$

- **余弦距离**（外观）：ReID 网络提取特征的余弦相似度

$$d_{appearance} = \\min\\{1 - \\mathbf{r}_j^T \\mathbf{r}_k^{(i)}\\}$$

### 级联匹配

优先匹配活跃时间短的轨迹（刚被遮挡的轨迹优先匹配），减少 ID Switch。

### 外观特征提取

用一个小型 CNN（在行人 ReID 数据集上训练）为每个检测框提取 128 维特征向量。

| | SORT | DeepSORT |
|------|------|----------|
| 关联依据 | IoU | IoU + 外观 |
| ID Switch | 较多 | 少 |
| 遮挡处理 | 差 | 较好 |
| 速度 | 极快 | 较快 |
`;export{n as default};
