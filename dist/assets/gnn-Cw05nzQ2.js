const n=`# GNN (图神经网络)

## 核心思想

CNN 处理网格数据（图像），RNN 处理序列数据，GNN 处理**图结构数据**——社交网络、分子结构、知识图谱。核心直觉是**消息传递**（Message Passing）：每个节点通过邻居节点的信息来更新自己的表示，多层堆叠后节点能感知更大范围的图结构。

GNN 家族的核心区别在于"怎么聚合邻居消息"——GCN 用归一化平均，GAT 用注意力加权，GraphSAGE 用可学习的聚合函数，GIN 则证明了"求和聚合"是表达能力最强的。

---

## 数学定义与原理解析

### 消息传递框架（统一形式）

对每一层 $l$：

$$
\\mathbf{h}_v^{(l+1)} = \\text{UPDATE}\\left( \\mathbf{h}_v^{(l)}, \\text{AGGREGATE}\\left( \\{\\mathbf{h}_u^{(l)} : u \\in \\mathcal{N}(v)\\} \\right) \\right)
$$

### GCN (Graph Convolutional Network)

使用对称归一化的邻接矩阵：

$$
\\mathbf{H}^{(l+1)} = \\sigma\\left( \\tilde{\\mathbf{D}}^{-\\frac12} \\tilde{\\mathbf{A}} \\tilde{\\mathbf{D}}^{-\\frac12} \\mathbf{H}^{(l)} \\mathbf{W}^{(l)} \\right)
$$

其中 $\\tilde{\\mathbf{A}} = \\mathbf{A} + \\mathbf{I}$（加自环），$\\tilde{D}_{ii} = \\sum_j \\tilde{A}_{ij}$。

### GAT (Graph Attention Network)

邻居聚合时的权重由注意力机制自动学习：

$$
\\alpha_{vu} = \\frac{\\exp(\\text{LeakyReLU}(\\mathbf{a}^T [\\mathbf{W}\\mathbf{h}_v \\| \\mathbf{W}\\mathbf{h}_u]))}{\\sum_{k \\in \\mathcal{N}(v)} \\exp(\\text{LeakyReLU}(\\mathbf{a}^T [\\mathbf{W}\\mathbf{h}_v \\| \\mathbf{W}\\mathbf{h}_k]))}
$$

$$
\\mathbf{h}_v' = \\sigma\\left( \\sum_{u \\in \\mathcal{N}(v)} \\alpha_{vu} \\mathbf{W} \\mathbf{h}_u \\right)
$$

多头注意力：$K$ 个头拼接（或平均）。

### GraphSAGE

不再依赖全图的邻接矩阵，而是对邻居**采样**+**聚合**：

$$
\\mathbf{h}_v^{(l+1)} = \\sigma\\left( \\mathbf{W}^{(l)} \\cdot \\text{CONCAT}\\left( \\mathbf{h}_v^{(l)}, \\text{AGG}\\left( \\{\\mathbf{h}_u^{(l)} : u \\in \\mathcal{N}(v)\\} \\right) \\right) \\right)
$$

AGG 可以是 Mean/Max/LSTM pooling。关键优势：可以处理未见过的节点（inductive learning）。

---

## 可视化展示

### 消息传递范式

\`\`\`mermaid
graph TD
    subgraph Layer_l["第 l 层"]
        A["节点 A"] -->|发送消息| B["节点 B"]
        B -->|发送消息| A
        B -->|发送消息| C["节点 C"]
        C -->|发送消息| B
        D["节点 D"] -->|发送消息| A
    end
    Layer_l --> Layer_l1["第 l+1 层"]
    A_msg["A: AGG(B,D)"] --> A_new["A'"]
    B_msg["B: AGG(A,C)"] --> B_new["B'"]
    C_msg["C: AGG(B)"] --> C_new["C'"]
    D_msg["D: AGG(A)"] --> D_new["D'"]
\`\`\`

### GCN vs GAT 聚合方式

\`\`\`echarts
return {
  title: { text: 'GNN 变体对比', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['GCN', 'GAT', 'GraphSAGE', 'GIN'] },
  yAxis: { type: 'value', min: 0, max: 1, name: '相对得分' },
  legend: { data: ['表达能力', '计算效率', 'Inductive'] },
  series: [
    { name: '表达能力', type: 'bar', data: [0.65, 0.85, 0.75, 0.95], itemStyle: { color: '#2c3e50' } },
    { name: '计算效率', type: 'bar', data: [0.9, 0.7, 0.8, 0.75], itemStyle: { color: '#16a085' } },
    { name: 'Inductive', type: 'bar', data: [0.0, 0.3, 1.0, 0.0], itemStyle: { color: '#2980b9' } }
  ],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

GIN 表达力最强（理论上和 WL test 一样强），GCN 最简单高效，GraphSAGE 支持 inductive 推理（可泛化到新节点）。

---

## 核心代码实现

### PyTorch — GCN Layer

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        """
        x: [N, in_dim]  节点特征
        adj: [N, N]      邻接矩阵（含自环）
        """
        # 对称归一化: D^(-1/2) * A * D^(-1/2)
        deg = adj.sum(dim=1)  # [N]
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        norm = deg_inv_sqrt.unsqueeze(1) * deg_inv_sqrt.unsqueeze(0)
        adj_norm = adj * norm

        return F.relu(self.W(adj_norm @ x))
\`\`\`

### PyTorch — GAT Layer

\`\`\`python
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.a = nn.Parameter(torch.randn(1, n_heads, 2 * out_dim))
        self.dropout = dropout

    def forward(self, x, adj):
        # x: [N, in_dim]
        N = x.shape[0]
        h = self.W(x).view(N, self.n_heads, self.out_dim)  # [N, H, D]

        # 计算注意力系数
        h_i = h.unsqueeze(1).repeat(1, N, 1, 1)  # [N, N, H, D]
        h_j = h.unsqueeze(0).repeat(N, 1, 1, 1)  # [N, N, H, D]
        cat = torch.cat([h_i, h_j], dim=-1)  # [N, N, H, 2D]
        e = torch.sum(self.a * cat, dim=-1)  # [N, N, H]
        e = F.leaky_relu(e, 0.2)

        # Mask 非邻居
        e = e.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))
        alpha = F.softmax(e, dim=1)  # [N, N, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.einsum('ijh,jhd->ihd', alpha, h)  # [N, H, D]
        return out.flatten(1)  # [N, H*D]
\`\`\`

### PyTorch — GraphSAGE

\`\`\`python
class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, aggr='mean'):
        super().__init__()
        self.aggr = aggr
        self.W = nn.Linear(2 * in_dim, out_dim)

    def forward(self, x, adj):
        # 邻居聚合
        if self.aggr == 'mean':
            neighbor_msg = adj @ x / adj.sum(dim=1, keepdim=True).clamp(min=1)
        elif self.aggr == 'max':
            neighbor_msg = torch.max(x.unsqueeze(0) * adj.unsqueeze(-1), dim=1)[0]
        # Concat + transform
        combined = torch.cat([x, neighbor_msg], dim=-1)
        return F.relu(self.W(combined))
\`\`\`
`;export{n as default};
