const n=`# Audio & Speech (Whisper / VITS / Bark / MusicGen)

## 核心思想

语音和音乐处理涵盖三个核心任务：ASR（语音→文本）、TTS（文本→语音）、音乐生成。Whisper 用经典的 Seq2Seq Transformer 统一了多语言 ASR（微弱的 encoder-decoder 就够），VITS 独创性地用 VAE + Normalizing Flow + GAN 的混合架构做端到端 TTS（音质自然度极高），MusicGen 证明**单一语言模型**就能生成结构化音乐（音频离散化为 codebook token + Transformer 自回归）。

---

## 数学定义与原理解析

### Whisper — 统一的 Seq2Seq ASR

架构：标准的 encoder-decoder Transformer，输入 log-mel 频谱，输出文本。

$$
\\mathbf{H} = \\text{Encoder}(\\text{MelSpectrogram}(audio))
$$

$$
P(y_t | y_{<t}, \\mathbf{H}) = \\text{Decoder}(y_{<t}, \\mathbf{H})
$$

核心优势来自数据：68 万小时的多语言弱监督数据（"弱监督"指对齐不完美的 ASR 转录），而非仅靠架构创新。Whisper 支持 99 种语言 → 英语翻译（不限于英文转录）。

### VITS — 三合一 TTS

将 VAE + Normalizing Flow + GAN 整合为统一的端到端 TTS：

1. **Variational Autoencoder**：$\\log p(x|c) \\geq \\mathbb{E}_{q(z|x)}[\\log p(x|z)] - D_{KL}(q(z|x) \\| p(z|c))$
2. **Normalizing Flow**：将简单的先验 $p(z|c)$ 映射为复杂的后验 $q(z|x)$，通过可逆变换 $f$：

$$
z = f_K \\circ \\cdots \\circ f_1(z_0), \\quad \\log p(z) = \\log p(z_0) - \\sum_k \\log \\left| \\det \\frac{\\partial f_k}{\\partial z_{k-1}} \\right|
$$

3. **HiFi-GAN Vocoder** 将 mel-spectrogram 转为音频波形（作为 GAN 的判别器）

**单调对齐搜索（MAS）**找到文本到音频的最优对齐——基于 DP 的 $O(T_{text} \\times T_{mel})$ 搜索。

### MusicGen — 音频 Token 的自回归生成

1. **音频 → 离散 Token**：用 EnCodec（神经音频编解码器）将音频压缩为离散 token 序列
2. **音乐生成 = 语言建模**：Transformer decoder 逐 token 预测下一个音频 token，条件为文本/旋律
3. **延迟模式**：4 个并行的 codebook，交错排列以减少自回归步数

### Bark — GPT 风格的 TTS

与 MusicGen 类似：音频 → EnCodec token → 自回归生成。区别在于 Bark 专门设计了一个层次化的 token 生成策略，首先生成粗粒度的语义 token，再生成细粒度的声学 token。

---

## 可视化展示

### 语音模型架构对比

\`\`\`mermaid
graph TD
    subgraph Whisper["Whisper (ASR)"]
        W_IN["Log-Mel 频谱"] --> W_ENC["Encoder"] --> W_DEC["Decoder"] --> W_OUT["文本"]
    end
    subgraph VITS["VITS (TTS)"]
        V_IN["文本"] --> V_ENC["Text Encoder"] --> V_FLOW["Normalizing Flow"] --> V_DEC["Decoder + GAN"] --> V_OUT["波形"]
    end
    subgraph MusicGen["MusicGen (音乐)"]
        M_IN["文本/旋律"] --> M_ENC["Condition Encoder"] --> M_TRANS["Transformer Decoder"] --> M_TOK["EnCodec Token→波形"]
    end
\`\`\`

### TTS 自然度对比

\`\`\`echarts
return {
  title: { text: 'TTS 模型 MOS (Mean Opinion Score)', left: 'center', textStyle: { fontSize: 13 } },
  xAxis: { type: 'category', data: ['Tacotron2', 'FastSpeech2', 'VITS', 'Bark', 'Ground Truth'] },
  yAxis: { type: 'value', min: 3.0, max: 4.8, name: 'MOS' },
  series: [{
    type: 'bar',
    data: [3.7, 3.9, 4.3, 4.1, 4.5],
    itemStyle: { color: '#2c3e50' },
    label: { show: true, position: 'top' }
  }],
  grid: { left: 60, right: 20, top: 50, bottom: 40 }
}
\`\`\`

VITS 接近真人音质，且推理速度远超自回归方法。

---

## 核心代码实现

### Whisper 推理

\`\`\`python
import torch
import whisper

# 加载模型 (tiny/base/small/medium/large-v3)
model = whisper.load_model("large-v3")

# 英文转录
result = model.transcribe("audio.mp3", language="en")
print(result["text"])

# 多语言 → 英语翻译
result = model.transcribe("audio.mp3", task="translate")
print(result["text"])

# 低层 API
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(model.device)
options = whisper.DecodingOptions(language="zh", task="transcribe")
result = whisper.decode(model, mel, options)
\`\`\`

### MusicGen 使用

\`\`\`python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-medium")
model.set_generation_params(duration=8)  # 生成 8 秒

# 文本+旋律条件生成
wav = model.generate(
    descriptions=["upbeat electronic dance music with a driving beat"],
    progress=True)
torchaudio.save("output.wav", wav[0].cpu(), sample_rate=32000)
\`\`\`

### 简化的 VITS 单调对齐搜索

\`\`\`python
def monotonic_alignment_search(log_prob, text_len, mel_len):
    """
    log_prob: [T_text, T_mel] — 对数概率矩阵
    返回最优单调路径 (每个文本帧对应一个 mel 帧范围)
    """
    T_t, T_m = log_prob.shape
    # DP: dp[i][j] = 到达 (i,j) 的最大累积概率
    dp = torch.full((T_t, T_m), float('-inf'))
    path = torch.zeros((T_t, T_m), dtype=torch.long)

    dp[0, 0] = log_prob[0, 0]
    for j in range(1, T_m):
        dp[0, j] = dp[0, j-1] + log_prob[0, j]

    for i in range(1, T_t):
        for j in range(i, T_m):
            from_left = dp[i, j-1]
            from_diag = dp[i-1, j-1]
            if from_left >= from_diag:
                dp[i, j] = from_left + log_prob[i, j]
                path[i, j] = 0  # 横向移动
            else:
                dp[i, j] = from_diag + log_prob[i, j]
                path[i, j] = 1  # 对角线移动

    # 回溯
    alignment = []
    i, j = T_t - 1, T_m - 1
    while i > 0 or j > 0:
        alignment.append((i, j))
        if i > 0 and (j == 0 or path[i, j] == 1):
            i -= 1; j -= 1
        else:
            j -= 1
    alignment.append((0, 0))
    return alignment[::-1]
\`\`\`
`;export{n as default};
