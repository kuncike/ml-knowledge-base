const n=`# TF-IDF

## 核心思想

TF-IDF (Term Frequency-Inverse Document Frequency) 是信息检索中最经典的词权重方案。核心直觉：

- **TF（词频）**：一个词在文档中出现越频繁，越重要
- **IDF（逆文档频率）**：一个词在越多文档中出现，越不重要（如"的"、"是"）

## 数学定义

### 词频 (TF)

$$\\text{TF}(t, d) = \\frac{\\text{词 } t \\text{ 在文档 } d \\text{ 中出现的次数}}{\\text{文档 } d \\text{ 的总词数}}$$

### 逆文档频率 (IDF)

$$\\text{IDF}(t) = \\log \\frac{N}{|\\{d \\in D: t \\in d\\}|} + 1$$

其中 $N$ 是总文档数。加 1 平滑防止值为 0。

### TF-IDF

$$\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\text{IDF}(t)$$

## 变体与平滑

| 变体 | TF | IDF |
|------|-----|-----|
| 标准 | $f_{t,d} / \\sum_k f_{k,d}$ | $\\log(N/n_t)$ |
| 次线性 TF | $1 + \\log f_{t,d}$ | 同上 |
| 归一化 TF | $\\frac{0.5 + 0.5 \\cdot f_{t,d}}{\\max_k f_{k,d}}$ | $\\log\\frac{N-n_t+0.5}{n_t+0.5}$ |
| BM25 | $\\frac{f_{t,d}(k_1+1)}{f_{t,d} + k_1(1-b+b\\cdot dl/avgdl)}$ | $\\log\\frac{N-n_t+0.5}{n_t+0.5}$ |

## BM25 公式

$$\\text{BM25}(q, d) = \\sum_{t \\in q} \\text{IDF}(t) \\cdot \\frac{f_{t,d} (k_1+1)}{f_{t,d} + k_1(1-b+b \\cdot \\frac{|d|}{avgdl})}$$

其中 $k_1=1.5$（词频饱和度），$b=0.75$（长度归一化强度）。

## Python 实现

\`\`\`python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "机器学习是人工智能的子领域",
    "深度学习使用神经网络进行学习",
    "人工智能包括机器学习和深度学习",
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
# X.shape = (3, vocab_size)

# 获取词和权重
words = vectorizer.get_feature_names_out()
\`\`\`

## 局限性

- **词袋模型**：忽略词序（"不好" vs "好不"）
- **语义鸿沟**："汽车"和"轿车"在向量空间中不相关
- **维度灾难**：词汇量大时矩阵稀疏
- **OOV 问题**：无法处理未见词

这些问题催生了 Word2Vec 等稠密向量方法。
`;export{n as default};
