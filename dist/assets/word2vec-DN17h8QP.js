const n=`# Word2Vec (CBOW / Skip-gram)

## 核心思想

Word2Vec 通过**预测任务**学习词的稠密向量表示。核心假设（分布假说）：**上下文相似的词，语义也相似**。

## 两种架构

### CBOW (Continuous Bag of Words)

用**上下文词预测目标词**：

$$\\text{Input: } w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2} \\quad \\rightarrow \\quad \\text{Output: } w_t$$

- 上下文词向量 → 平均 → 预测目标词
- 较快，适合频繁词

### Skip-gram

用**目标词预测上下文词**：

$$\\text{Input: } w_t \\quad \\rightarrow \\quad \\text{Output: } w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}$$

- 对低频词效果更好
- 训练较慢（一个中心词预测多个上下文词）

## 目标函数

Skip-gram 的负对数似然（使用负采样）：

$$J = -\\sum_{(w, c) \\in D_+} \\log \\sigma(\\mathbf{v}_c^T \\mathbf{v}_w) - \\sum_{(w, c) \\in D_-} \\log \\sigma(-\\mathbf{v}_c^T \\mathbf{v}_w)$$

其中 $\\sigma(x) = 1/(1+e^{-x})$，$D_+$ 是正样本（真实上下文），$D_-$ 是负样本（随机采样）。

## 负采样

全量 Softmax 在百万级词汇上不可行。负采样将其转化为二分类问题：对每个正样本，采样 $k$ 个噪声词作为负样本（通常 $k=5\\sim 20$）。

噪声分布：$P_n(w) \\propto U(w)^{3/4}$（提升低频词被采样的概率）

## 层次 Softmax (Hierarchical Softmax)

将词汇组织成 Huffman 树，每个词的概率 = 从根到叶的路径上各节点的二分类概率之积，将复杂度从 $O(V)$ 降为 $O(\\log V)$。

## 著名的语义关系

$$\\text{vec}(\\text{king}) - \\text{vec}(\\text{man}) + \\text{vec}(\\text{woman}) \\approx \\text{vec}(\\text{queen})$$

$$\\text{vec}(\\text{Paris}) - \\text{vec}(\\text{France}) + \\text{vec}(\\text{Italy}) \\approx \\text{vec}(\\text{Rome})$$

## Gensim 使用

\`\`\`python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences,
    vector_size=300,
    window=5,
    min_count=5,
    sg=1,          # 1=Skip-gram, 0=CBOW
    negative=5,    # 负采样数
    epochs=10,
)
king_vec = model.wv['king']
similar = model.wv.most_similar('king', topn=5)
\`\`\`

## 局限性

- 每个词只有一个向量（无法表示多义词）
- 窗口限制（不能捕获长距离依赖）
- 不能生成上下文相关的表示（ELMo/BERT 的动机）
`;export{n as default};
