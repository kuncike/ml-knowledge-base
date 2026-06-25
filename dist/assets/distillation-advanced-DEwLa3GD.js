const t=`# Advanced Distillation (特征 KD / 关系 KD / 自蒸馏)

## 核心思想

Hinton 的 logit 蒸馏只从教师最后一层的"答案"中学习。特征蒸馏更进一步：**让学生中间层的特征图尽可能匹配教师的特征图**——相当于让学生模仿教师的"解题过程"而非仅"最终答案"。关系蒸馏走另一条路：不匹配单个样本的绝对值，而是保持样本之间的**相对关系**（样本 A 和 B 的距离 / 角度在师生模型中一致）。自蒸馏则颠覆了需要大教师的假设——**学生自己当自己的教师**。

---

## 数学定义与原理解析

### FitNet — 特征蒸馏

让学生第 $l$ 层的特征匹配教师第 $m$ 层的特征：

$$
\\mathcal{L}_{FitNet} = \\frac{1}{2} \\| \\mathbf{F}_s^{(l)} - \\text{Proj}(\\mathbf{F}_t^{(m)}) \\|^2
$$

中间的 Projection 层确保特征维度匹配。直觉：特征层蒸馏强迫学生学习与教师相似的中间表示（更高层次的结构知识）。

### RKD — 关系知识蒸馏

**距离关系**（Distance-wise）：保持样本对之间的欧氏距离：

$$
\\mathcal{L}_{RKD-D} = \\sum_{(x_i, x_j) \\in \\mathcal{X}^2} \\ell_\\delta(\\psi_D(t_i, t_j), \\psi_D(s_i, s_j))
$$

$$
\\psi_D(\\mathbf{t}_i, \\mathbf{t}_j) = \\frac{1}{\\mu_D} \\|\\mathbf{t}_i - \\mathbf{t}_j\\|_2
$$

**角度关系**（Angle-wise）：保持三个样本之间的夹角：

$$
\\mathcal{L}_{RKD-A} = \\sum_{(x_i, x_j, x_k)} \\ell_\\delta(\\psi_A(t_i, t_j, t_k), \\psi_A(s_i, s_j, s_k))
$$

$$
\\psi_A(\\mathbf{t}_i, \\mathbf{t}_j, \\mathbf{t}_k) = \\cos \\angle \\mathbf{t}_i \\mathbf{t}_j \\mathbf{t}_k = \\langle \\frac{\\mathbf{t}_i - \\mathbf{t}_j}{\\|\\mathbf{t}_i - \\mathbf{t}_j\\|}, \\frac{\\mathbf{t}_k - \\mathbf{t}_j}{\\|\\mathbf{t}_k - \\mathbf{t}_j\\|} \\rangle
$$

### 自蒸馏

教师和学生是**同一架构**，训练分为两轮或使用历史 EMA 参数：

1. **Born-Again**：Student = Teacher（相同架构），用 Teacher 的软标签重新训练 Student
2. **Self-Distillation (Online)**：训练过程中，EMA 版本的模型作为教师指导当前模型

自蒸馏的关键发现：即使教师不比学生强，蒸馏仍然有效——因为软标签提供了比 hard label 更丰富的监督信号（类别间的相对关系）。

### 总损失

$$
\\mathcal{L}_{total} = \\mathcal{L}_{CE}(y_{true}, y_{student}) + \\alpha \\mathcal{L}_{KD}(y_{teacher}, y_{student}) + \\beta \\mathcal{L}_{feat}
$$

---

## 可视化展示

### 蒸馏方法对比

\`\`\`mermaid
graph TD
    subgraph Logit["Logit KD (Hinton)"]
        L_T["教师 logits"] --> L_KL["KL Div"] --> L_LOSS["Total Loss"]
        L_S["学生 logits"] --> L_KL
    end
    subgraph Feat["Feature KD (FitNet)"]
        F_T["教师中间特征"] --> F_MSE["MSE"] --> F_LOSS["Total Loss"]
        F_S["学生中间特征"] --> F_MSE
    end
    subgraph Rel["Relation KD (RKD)"]
        R_T["教师样本对关系<br/>(距离+角度)"] --> R_LOSS["Distance Loss"] --> R_TOTAL["Total Loss"]
        R_S["学生样本对关系"] --> R_LOSS
    end
\`\`\`

### 蒸馏策略对比

\`\`\`echarts
return {
  title: { text: '蒸馏方法在 CIFAR-100 上的提升 (ResNet-18 学生)', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['No KD', 'Logit KD', 'FitNet', 'RKD', 'Self-Distill'] },
  yAxis: { type: 'value', min: 73, max: 80, name: 'Top-1 Accuracy (%)' },
  series: [{
    type: 'bar',
    data: [74.5, 76.2, 77.1, 77.8, 76.5],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top', formatter: '{c}%' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

---

## 核心代码实现

### PyTorch — RKD Loss (距离 + 角度)

\`\`\`python
import torch
import torch.nn.functional as F

def rkd_distance_loss(student_feat, teacher_feat):
    """
    student_feat / teacher_feat: [B, D]
    保持样本对之间的欧氏距离一致
    """
    # 成对距离矩阵
    with torch.no_grad():
        t_dist = torch.cdist(teacher_feat, teacher_feat, p=2)
        mu_t = t_dist.mean()
        t_dist = t_dist / mu_t

    s_dist = torch.cdist(student_feat, student_feat, p=2)
    mu_s = s_dist.mean()
    s_dist = s_dist / mu_s

    return F.smooth_l1_loss(s_dist, t_dist)


def rkd_angle_loss(student_feat, teacher_feat):
    """
    保持三个样本之间的角度关系
    Student 的局部结构与 Teacher 一致
    """
    B = student_feat.shape[0]
    # 随机选 triplets: (anchor, positive, negative)
    with torch.no_grad():
        t = F.normalize(teacher_feat, dim=1)  # [B, D]
        t_diff = t.unsqueeze(0) - t.unsqueeze(1)  # [B, B, D]
        t_angle = torch.einsum('bnd,bmd->bnm', t_diff, t_diff)  # cosine of angle

    s = F.normalize(student_feat, dim=1)
    s_diff = s.unsqueeze(0) - s.unsqueeze(1)
    s_angle = torch.einsum('bnd,bmd->bnm', s_diff, s_diff)

    return F.smooth_l1_loss(s_angle, t_angle)


def rkd_total_loss(student_feat, teacher_feat, lambda_d=1.0, lambda_a=2.0):
    return lambda_d * rkd_distance_loss(student_feat, teacher_feat) + \\
           lambda_a * rkd_angle_loss(student_feat, teacher_feat)
\`\`\`

### 自蒸馏 (Born-Again)

\`\`\`python
def born_again_train(student, X_train, y_train, teacher_logits, T=4.0):
    """
    Born-Again Network: teacher = student(同架构)
    第一阶段已经产生 teacher_logits
    """
    optimizer = torch.optim.Adam(student.parameters())
    for x, y, t_logits in zip(X_train, y_train, teacher_logits):
        s_logits = student(x)
        # CE + KL
        ce_loss = F.cross_entropy(s_logits, y)
        kd_loss = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction='batchmean') * (T * T)
        loss = ce_loss + kd_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
\`\`\`
`;export{t as default};
