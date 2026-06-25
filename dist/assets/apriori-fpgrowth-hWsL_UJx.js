const n=`# Apriori / FP-Growth

## 核心概念

关联规则挖掘用于发现数据中项之间的有趣关系。经典案例：**啤酒与尿布**。

### 基本定义

- **项集 (Itemset)**：项的集合
- **支持度 (Support)**：$P(A \\cup B) = \\frac{\\text{count}(A \\cup B)}{n}$
- **置信度 (Confidence)**：$P(B \\mid A) = \\frac{\\text{support}(A \\cup B)}{\\text{support}(A)}$
- **提升度 (Lift)**：$\\frac{P(B \\mid A)}{P(B)} = \\frac{\\text{support}(A \\cup B)}{\\text{support}(A) \\cdot \\text{support}(B)}$
  - Lift > 1：正相关
  - Lift = 1：独立
  - Lift < 1：负相关

## Apriori 算法

### 核心原理 — 先验性质

> **如果一个项集是频繁的，那么它的所有子集也是频繁的。**
> 等价：如果一个项集是非频繁的，那么它的所有超集也是非频繁的。

### 算法流程

1. 扫描数据库，找出所有频繁 1-项集
2. 从频繁 $k$-项集生成候选 $(k+1)$-项集
3. 剪枝：剔除含有非频繁子集的候选项
4. 扫描数据库，计算候选项的支持度
5. 重复 2-4 直到无频繁项集产生

### 从频繁项集生成规则

对每个频繁项集 $L$，枚举所有非空真子集 $A \\subset L$，检查：

$$\\text{confidence}(A \\Rightarrow L \\setminus A) = \\frac{\\text{support}(L)}{\\text{support}(A)} \\geq \\text{min\\_conf}$$

---

## FP-Growth 算法

### 核心思想

Apriori 需要反复扫描数据库，效率低。FP-Growth 通过构建 **FP-Tree**（频繁模式树）压缩数据库，只需两次扫描。

### FP-Tree 构建

1. **第一次扫描**：计算所有项的频率，排序，剔除不频繁项
2. **第二次扫描**：对每条事务：
   - 按频率降序排列频繁项
   - 插入到 FP-Tree 中（共享公共前缀）

### 挖掘 FP-Tree

递归地构建**条件模式基**和**条件 FP-Tree**：

\`\`\`python
def mine_fp_tree(tree, header_table, min_support, prefix, frequent_itemsets):
    for item in header_table (from bottom to top):
        new_freq_set = prefix + [item]
        frequent_itemsets.append(new_freq_set)
        # 构建条件模式基
        conditional_pattern_base = get_conditional_paths(tree, item)
        # 构建条件 FP-Tree
        conditional_tree = build_fp_tree(conditional_pattern_base, min_support)
        if conditional_tree:
            mine_fp_tree(conditional_tree, ...)
\`\`\`

## Apriori vs FP-Growth

| | Apriori | FP-Growth |
|------|---------|-----------|
| 数据库扫描 | 每次迭代扫描 | 仅 2 次 |
| 候选生成 | 需要 | 不需要 |
| 内存使用 | 较小 | 较大（FP-Tree）|
| 适用场景 | 稀疏数据 | 稠密数据 |

## 应用场景

- 购物篮分析（推荐系统）
- 网页浏览路径分析
- 入侵检测（异常模式关联）
- 医疗诊断（症状-疾病关联）
`;export{n as default};
