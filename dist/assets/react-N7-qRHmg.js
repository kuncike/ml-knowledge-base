const n=`# ReAct (Reasoning + Acting)

## 核心思想

ReAct 将**推理 (Reasoning)** 和**行动 (Acting)** 交织在一起，让 LLM 可以调用工具获取外部信息，根据反馈调整推理方向。

## 循环流程

\`\`\`
Thought → Action → Observation → Thought → Action → ... → Final Answer
\`\`\`

### 示例

\`\`\`
Question: 苹果公司的 CEO 是谁，他多大了？

Thought: 我需要知道苹果公司的 CEO 是谁。
Action: Search[苹果公司 CEO]
Observation: 蒂姆·库克 (Tim Cook)，2011 年起担任 CEO

Thought: 现在我需要知道蒂姆·库克的年龄。
Action: Search[蒂姆·库克 出生日期]
Observation: 蒂姆·库克出生于 1960 年 11 月 1 日

Thought: 我可以计算年龄了。
现在 2024 年，2024-1960=64。

Final Answer: 苹果公司的 CEO 是蒂姆·库克，他今年 64 岁（2024年）。
\`\`\`

## 为什么需要 ReAct？

单纯的推理（如 CoT）存在**幻觉**问题——模型可能编造事实。

单纯的动作（如直接搜索）缺乏**推理能力**——无法处理需要多步推理的复杂问题。

ReAct 结合两者：推理指导行动，行动验证推理。

## 常用工具 (Actions)

| 工具 | 功能 |
|------|------|
| Search | 搜索网络获取信息 |
| Lookup | 精确查找关键词 |
| Calculator | 数学计算 |
| Finish | 给出最终答案 |

## Prompt 模板

\`\`\`
Solve a question answering task with interleaving
Thought, Action, Observation steps.

你可以使用以下工具:
- Search[query]: 搜索互联网
- Lookup[keyword]: 在搜索结果中精确定位
- Finish[answer]: 返回最终答案

{examples}

Question: {question}
\`\`\`

实际实现中，根据模型的输出格式解析 Thought / Action / Observation 标签。

## ReAct vs CoT vs Agentic

| | CoT | ReAct | Agentic |
|------|-----|-------|---------|
| 与外界交互 | 无 | 有 (工具) | 有 (多工具 + 记忆) |
| 推理方式 | 线性链 | 循环 | 复杂工作流 |
| 错误恢复 | 差 | 中 | 好 |
| 适用场景 | 数学/逻辑 | QA/事实核查 | 复杂任务自动化 |

## 代码示例

\`\`\`python
def react_loop(question, max_steps=10):
    context = ""
    for step in range(max_steps):
        prompt = build_prompt(question, context)
        response = llm(prompt)
        thought, action = parse(response)

        if action.startswith("Finish"):
            return extract_answer(action)

        observation = execute_action(action)
        context += f"Thought: {thought}\\n"
        context += f"Action: {action}\\n"
        context += f"Observation: {observation}\\n"
\`\`\`
`;export{n as default};
