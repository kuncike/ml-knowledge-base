const n=`# SFT (监督微调 / Supervised Fine-Tuning)

## 核心思想

在预训练语言模型的基础上，使用**高质量的人工标注数据**（指令-回答对）进行监督学习，使模型学会遵循人类指令。

## 数据格式

典型的指令微调数据：

\`\`\`json
{
  "instruction": "将以下句子翻译为英文",
  "input": "今天天气很好",
  "output": "The weather is nice today."
}
\`\`\`

或对话格式：

\`\`\`json
{
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是..."}
  ]
}
\`\`\`

## 训练目标

标准的下一个 token 预测损失，但**仅在 assistant 回复部分计算损失**：

$$L = -\\sum_{t \\in \\text{assistant}} \\log P(y_t \\mid y_{<t}, \\text{context})$$

对 system/user prompt 部分使用 label mask（设为 -100），不参与损失计算。

## 数据质量 > 数据数量

LLaMA 论文的关键发现：

- 少量高质量数据（~10K 条）的 SFT 效果远超大量低质量数据
- 数据多样性很重要（涵盖推理、对话、写作、代码等）
- 格式一致性对模型输出风格影响大

## 训练技巧

### Packing

将多个短对话打包到一个序列中，提高 GPU 利用率：

\`\`\`
[对话1] [EOS] [对话2] [EOS] [对话3] [EOS]
\`\`\`

注意在序列边界位置正确处理 attention mask（每个对话独立）。

### 学习率与 Epoch

- 学习率：1e-5 ~ 5e-5（比预训练小一个数量级）
- Epoch：1-3（SFT 容易过拟合，通常 1-2 个 epoch 足够）
- Batch size：128 左右

## SFT 的局限性

- 无法让模型产生超出训练数据的上限
- 依赖标注质量（错误标注会植入不良行为）
- 不能教会模型"判断什么是好回答"（→ 需要 RLHF）

## 示例代码

\`\`\`python
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
tokenizer.pad_token = tokenizer.eos_token

# 使用 HuggingFace TRL 库
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=sft_dataset,
    max_seq_length=2048,
    packing=True,
)
trainer.train()
\`\`\`
`;export{n as default};
