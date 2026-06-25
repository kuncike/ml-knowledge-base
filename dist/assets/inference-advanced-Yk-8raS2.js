const n=`# Advanced Inference (FlashDecoding / Continuous Batching / Speculative Decoding)

## 核心思想

LLM 推理有三大瓶颈：**注意力计算量大**（每个新 token 要 attend 所有历史 token）、**内存带宽受限**（KV Cache 不断增长）、**无法并行化**（自回归解码每个 token 必须串行）。三个技术分别突破：FlashDecoding 解决长序列注意力的并行化，Continuous Batching 解决多请求时 GPU 利用率低，Speculative Decoding 用小型"草稿模型"预测多个 token 再让大模型验证——在数学等价的前提下提升吞吐。

---

## 数学定义与原理解析

### FlashDecoding — 序列维度并行

FlashAttention 已经将注意力分解为 tile 计算。FlashDecoding 进一步将长序列的 **softmax 计算沿 token 维度并行化**。

标准 FlashAttention 在 batch 和 head 维度并行。当 batch=1 且序列很长时，GPU 利用率低。FlashDecoding 添加了序列维度的并行：

$$
\\text{Attention}(Q_i, K, V) = \\frac{\\sum_j \\exp(Q_i K_j^T) V_j}{\\sum_j \\exp(Q_i K_j^T)}
$$

对于每个 query token $i$，将 key/value 拆分为多个 chunk 并行计算，然后用 **online softmax** 合并各 chunk 的分子/分母（与 FlashAttention 相同的技巧，但沿序列维度拆分）。

### Continuous Batching — 迭代级调度

传统 static batching：一次传入一批请求，等所有请求完成后再传下一批。如果某条请求提前结束（已生成 EOS），其 GPU 资源被浪费。

Continuous Batching 在**每步生成后**动态调整 batch：
- 完成的请求被移除
- 新请求可以随时加入
- 每个 iteration 的 batch 可能不同

\`\`\` 
Time →     Step1  Step2  Step3  Step4  Step5
Request1:  [gen]  [gen]  [gen]  [gen]  [DONE] → 移除
Request2:  [gen]  [gen]  [DONE] → 移除
Request3:          [new]  [gen]  [gen]  [gen]
Batch size:  2      3      2      2      1
\`\`\`

### Speculative Decoding — 草稿+验证

大模型一次生成 1 个 token，小模型也一次生成 1 个 token，但小模型快得多。Speculative Decoding 让小模型先生成 $k$ 个候选 token，大模型一次性验证（并行）：

1. **Draft**：小模型 $q$ 快速生成 $x_{t+1}, \\ldots, x_{t+k}$ 及其概率 $q(x_i)$
2. **Verify**：大模型 $p$ 并行计算 $p(x_{t+1}), \\ldots, p(x_{t+k})$
3. **Accept/Reject**：对每个位置，如果 $q(x_i) \\leq p(x_i)$ 则接受，否则以概率 $p(x_i)/q(x_i)$ 接受（rejection sampling）
4. 第一个被拒绝的位置用 $p$ 重新采样，之后丢弃

数学保证了输出分布严格等价于 $p$ 的采样。

---

## 可视化展示

### Continuous Batching

\`\`\`mermaid
graph TD
    subgraph Static["Static Batching"]
        S1["Batch(Req1, Req2)<br/>→ 等两者都完成 →"] --> S2["Batch(Req3)"]
    end
    subgraph Continuous["Continuous Batching"]
        C1["Iter1: Req1, Req2"] --> C2["Iter2: Req1, Req3 (Req2完成)"] --> C3["Iter3: Req1, Req3, Req4"] --> C4["Iter4: Req3, Req4 (Req1完成)"]
    end
\`\`\`

### 吞吐量对比

\`\`\`echarts
return {
  title: { text: '推理优化技术吞吐量提升 (LLaMA-7B)', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['Baseline', '+Continuous Batching', '+FlashAttention', '+Speculative Decode', '全部组合'] },
  yAxis: { type: 'value', min: 0, max: 50, name: 'Tokens/s (相对)' },
  series: [{
    type: 'bar',
    data: [1, 4, 7, 15, 45],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top', formatter: '{c}×' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 50 }
}
\`\`\`

---

## 核心代码实现

### Speculative Decoding

\`\`\`python
import torch
import torch.nn.functional as F

@torch.no_grad()
def speculative_decode(target_model, draft_model, prefix, k=5, T=1.0):
    """
    prefix: 已有 token [1, L]
    返回: 生成的 token 序列
    """
    generated = list(prefix[0].tolist())

    while True:
        draft_input = torch.tensor([generated])
        # 1. Draft: 小模型生成 k 个候选
        draft_tokens = []
        draft_probs = []
        for _ in range(k):
            logits = draft_model(draft_input)[0, -1] / T
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1).item()
            draft_tokens.append(token)
            draft_probs.append(probs[token].item())
            draft_input = torch.tensor([generated + draft_tokens])

        # 2. Verify: 大模型并行计算
        target_input = torch.tensor([generated + draft_tokens])
        target_logits = target_model(target_input)[0]  # [L, V]
        target_probs = F.softmax(target_logits[len(generated)-1:] / T, dim=-1)

        # 3. Accept/Reject
        for i in range(k):
            q_prob = draft_probs[i]
            p_prob = target_probs[i, draft_tokens[i]].item()

            if random.random() < min(1.0, p_prob / max(q_prob, 1e-10)):
                generated.append(draft_tokens[i])  # 接受
            else:
                # 拒绝: 从 p 重新采样 (减去 q 的贡献)
                adjusted_probs = F.relu(target_probs[i] - F.softmax(
                    torch.tensor(draft_probs[i]), dim=-1))
                adjusted_probs /= adjusted_probs.sum()
                new_token = torch.multinomial(adjusted_probs, 1).item()
                generated.append(new_token)
                break
        else:
            # 所有 k 个都被接受, 额外加一个来自 p 的 token
            last_probs = F.softmax(target_logits[-1] / T, dim=-1)
            generated.append(torch.multinomial(last_probs, 1).item())
            continue

        if generated[-1] == EOS_TOKEN:
            break

    return generated[len(prefix[0]):]
\`\`\`

### Continuous Batching 示意

\`\`\`python
class ContinuousBatchingScheduler:
    def __init__(self, model, max_batch=32):
        self.model = model
        self.max_batch = max_batch
        self.active_requests = []

    def step(self):
        """单步推理+调度"""
        if len(self.active_requests) < self.max_batch:
            # 尝试从队列拉新请求
            while self.pending_queue and len(self.active_requests) < self.max_batch:
                self.active_requests.append(self.pending_queue.pop(0))

        # 生成一步
        inputs = torch.stack([r.current_seq for r in self.active_requests])
        logits = self.model(inputs)[:, -1]
        next_tokens = logits.argmax(dim=-1)

        # 更新序列 + 移除已完成请求
        remaining = []
        for i, req in enumerate(self.active_requests):
            req.append_token(next_tokens[i].item())
            if not req.is_done():
                remaining.append(req)
        self.active_requests = remaining
\`\`\`
`;export{n as default};
