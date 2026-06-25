const n=`# GloVe / FastText

## GloVe (Global Vectors)

### 核心思想

GloVe 结合了矩阵分解（全局统计）和 Word2Vec（局部上下文窗口）的优点。基于**词共现矩阵**的非零元素进行分解。

### 共现概率比

GloVe 的核心洞察：**共现概率的比值**编码了语义关系：

$$\\frac{P(k|\\text{ice})}{P(k|\\text{steam})}$$

- 当 $k=$ "solid"，比值大（>1）
- 当 $k=$ "gas"，比值小（<1）
- 当 $k=$ "water" 或 "fashion"，比值接近 1

### 目标函数

$$J = \\sum_{i,j=1}^{V} f(X_{ij}) (\\mathbf{w}_i^T \\tilde{\\mathbf{w}}_j + b_i + \\tilde{b}_j - \\log X_{ij})^2$$

其中 $X_{ij}$ 是词 $j$ 在词 $i$ 的上下文中出现的次数。

**加权函数 $f$**：
- $f(0) = 0$
- 对高频词做截断（防止"的"、"是"等词主导损失）

$$f(x) = \\begin{cases} (x/x_{max})^\\alpha & x < x_{max} \\\\ 1 & \\text{otherwise} \\end{cases}$$

通常 $x_{max}=100, \\alpha=0.75$。

---

## FastText

### 核心思想

FastText 将每个词表示为 **字符 n-gram** 的向量和。例如 "where" 的 tri-gram：\`<wh, whe, her, ere, re>\`。

### 优势

1. **处理 OOV**：未见词可由已知字符 n-gram 组成
2. **利用词形信息**：前缀、后缀、词根
3. **少资源语言**：利用形态学（如英语、法语）

### 数学表示

词 $w$ 的向量：

$$\\mathbf{v}_w = \\sum_{g \\in G_w} \\mathbf{z}_g$$

其中 $G_w$ 是词 $w$ 的字符 n-gram 集合（包括完整的词本身作为一个特殊 n-gram）。

### 子词信息

| 语言 | 利用的形态信息 |
|------|---------------|
| 英语 | 前缀 (un-, re-) 后缀 (-ing, -ed, -ly) |
| 德语 | 复合词分解 |
| 中文 | 偏旁部首/笔画 |

## GloVe vs FastText vs Word2Vec

| | Word2Vec | GloVe | FastText |
|------|----------|-------|----------|
| 依据 | 局部上下文 | 全局共现 | 局部上下文 |
| OOV | 不支持 | 不支持 | 支持（字符 n-gram） |
| 形态学 | 不利用 | 不利用 | 利用 |
| 训练速度 | 快（负采样）| 中等 | 快 |
| 何时用 | 通用 | 需全局统计 | 多形态语言/OOV 多 |
`;export{n as default};
