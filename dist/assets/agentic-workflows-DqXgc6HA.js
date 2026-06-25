const n=`# Agentic Workflows (智能体工作流)

## 核心思想

Agent 不是简单地"输入 → 输出"，而是可以**规划、使用工具、反思、迭代**的自主系统，能在复杂环境中完成多步任务。

## Agent 的核心组件

\`\`\`
Agent = LLM + Planning + Memory + Tools
\`\`\`

### 1. 规划 (Planning)

- **任务分解**：将大任务拆分为子任务
- **反思 (Reflection)**：检查中间结果，纠正错误

### 2. 记忆 (Memory)

- **短期记忆**：当前对话的上下文
- **长期记忆**：跨对话持久化的信息（向量数据库）
- **工作记忆**：当前任务的中间结果和状态

### 3. 工具 (Tools)

Agent 可以调用的外部能力：

| 工具类型 | 示例 |
|----------|------|
| 搜索 | 网页搜索、知识库检索 |
| 计算 | 代码执行、数学计算 |
| 数据 | 数据库查询、API 调用 |
| 生成 | 图片生成、音频合成 |
| 系统 | 文件操作、命令执行 |

## 关键设计模式

### ReAct 模式

循环交替推理 (Thought) 和行动 (Action)。

### Plan-and-Execute

先制定完整计划，再逐步执行。

### Reflexion

执行后评估结果，如果不满意就反思原因并重试。

### Multi-Agent 协作

多个专业 Agent 分工协作：

\`\`\`
Manager Agent
  ├── Researcher Agent (搜索信息)
  ├── Coder Agent (编写代码)
  ├── Reviewer Agent (审查代码)
  └── Writer Agent (撰写报告)
\`\`\`

## 工具使用 (Function Calling)

LLM 通过特殊格式指定要调用的函数：

\`\`\`json
{
  "tool_calls": [{
    "function": {
      "name": "search_web",
      "arguments": "{\\"query\\": \\"2024年诺贝尔物理学奖获得者\\"}"
    }
  }]
}
\`\`\`

执行后，将结果作为 Observation 返回给 LLM 继续推理。

## 可靠性挑战

| 问题 | 缓解措施 |
|------|----------|
| 工具选择错误 | 提供清晰的工具描述和使用示例 |
| 循环/死锁 | 设置最大步数限制 + 循环检测 |
| 幻觉事实 | 强制引用来源 + RAG |
| 复合错误 | Reflexion / Self-Correction |
| 上下文过长 | 摘要压缩 + 滑动窗口 |

## 框架

| 框架 | 特点 |
|------|------|
| LangChain / LangGraph | 图状态机，丰富的工具生态 |
| AutoGen | 多 Agent 对话 |
| CrewAI | 角色分配的多 Agent |
| OpenAI Swarm | 轻量级多 Agent 编排 |
| Anthropic MCP | 模型-上下文-协议 |

## 一个 Agent 循环的伪代码

\`\`\`python
class Agent:
    def run(self, task):
        memory = [{"role": "user", "content": task}]
        for step in range(max_steps):
            response = self.llm(memory, tools=self.tools)
            if response.is_final:
                return response.content
            if response.tool_call:
                result = self.execute(response.tool_call)
                memory.append({"role": "tool", "content": result})
\`\`\`
`;export{n as default};
