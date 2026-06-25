### 一、传统机器学习算法

* **线性模型**：线性回归、逻辑回归、Ridge / Lasso / Elastic Net
* **支持向量机 (SVM)**：线性 SVM、非线性 SVM (RBF核/多项式核)
* **贝叶斯方法**：朴素贝叶斯、贝叶斯网络
* **树模型**：决策树 (ID3 / C4.5 / CART)
* **集成学习 (Ensemble)**：
  * Bagging：随机森林
  * Boosting：AdaBoost / GBDT / XGBoost / LightGBM / CatBoost
* **聚类算法**：K-Means / K-Medoids、层次聚类、DBSCAN / OPTICS、谱聚类 / GMM、SOM
* **降维算法**：PCA / SVD、ICA / LDA、t-SNE / UMAP
* **异常检测**：孤立森林 / LOF / One-Class SVM
* **关联规则**：Apriori / FP-Growth

### 二、神经网络基础与核心组件

* **经典网络架构**：FFNN / MLP、自编码器 / 稀疏AE / 去噪AE、VAE、RBM / DBN、Hopfield 网络、SNN (脉冲神经网络)
* **网络层 (Layers)**：全连接层、卷积层 (1D/2D/3D/空洞/深度可分离/转置)、池化层、RNN / LSTM / GRU、BiLSTM / BiGRU、ConvLSTM、嵌入层 (Embedding)、GNN (图神经网络)
* **注意力机制 (Attention)**：自注意力 (Self-Attention)、多头注意力 (Multi-Head Attention)、交叉注意力 (Cross-Attention)、FlashAttention、滑动窗口注意力
* **激活函数**：Sigmoid / Tanh、ReLU / LeakyReLU / PReLU / ELU、GELU / Swish / Mish、Softmax
* **损失函数**：MSE / MAE / Huber Loss、交叉熵、Focal Loss / Triplet Loss / 对比损失、KL 散度
* **正则化与归一化**：Dropout、L1 / L2 正则化、BN / LN / IN / GN
* **优化器**：SGD / Momentum / Nesterov、Adam / AdamW

### 三、计算机视觉 (CV)

* **卷积变体**：空洞卷积 / 深度可分离卷积 / 可变形卷积 / 转置卷积 / 分组卷积
* **图像分类**：LeNet / AlexNet / VGG、GoogLeNet / ResNet、ResNeXt / DenseNet / Wide ResNet、SENet / ShuffleNet / GhostNet、MobileNet / EfficientNet、ConvNeXt (v1/v2)、ViT / Swin Transformer、DeiT / PVT / CaiT
* **目标检测**：R-CNN / Fast R-CNN / Faster R-CNN、YOLO (v1-v11)、SSD / RetinaNet、CenterNet / FCOS / YOLOX (Anchor-Free)、DETR、Deformable DETR / DINO、Mask R-CNN
* **图像分割**：FCN / U-Net、UNet++ / V-Net / Mask2Former、DeepLab / PSPNet、SAM、FastSAM / MobileSAM
* **姿态估计与跟踪**：OpenPose / HRNet、SORT / DeepSORT
* **视频理解**：3D CNN / I3D / R(2+1)D / SlowFast

### 四、自然语言处理 (NLP) 与文本检索

* **传统词向量**：TF-IDF、Word2Vec (CBOW / Skip-gram)、GloVe / FastText
* **上下文词向量与预训练**：ELMo、Transformer、BERT / RoBERTa / ALBERT / DistilBERT、DeBERTa / ELECTRA / XLNet、T5 / BART、MASS / Pegasus (Seq2Seq 预训练)
* **密集检索与向量模型**：BM25 / DPR、Sentence-BERT / ColBERT、BGE / BCEmbedding / E5、Contriever / ANCE / PLAID、FAISS (向量索引与 ANN)

### 五、大语言模型 (LLM) 与提示工程

* **主流模型架构**：GPT 系列、LLaMA / Mistral / Mixtral (MoE)、Qwen / DeepSeek / ChatGLM、Gemma / Phi / Falcon (小型 LLM)
* **对齐与微调技术**：SFT (监督微调)、RLHF (基于人类反馈的强化学习)、DPO、KTO / ORPO / SimPO
* **参数高效微调 (PEFT)**：LoRA / QLoRA、AdaLoRA / DoRA / IA3、Prompt Tuning / Prefix Tuning / P-Tuning
* **提示工程与推理框架**：Few-Shot / CoT / ToT、ReAct、RAG / GraphRAG、Agentic Workflows

### 六、生成式人工智能 (AIGC) 与多模态

* **生成对抗网络 (GANs)**：DCGAN / WGAN、WGAN-GP / CGAN / InfoGAN / BigGAN、StyleGAN (v1-v3)、CycleGAN / Pix2Pix
* **扩散模型 (Diffusion)**：DDPM / DDIM、Stable Diffusion、Midjourney / DALL-E / Imagen、SGM / DiT / PixArt-α、ControlNet、IP-Adapter / GLIGEN / InstantID
* **其他生成模型**：VAE / VQ-VAE / 流模型
* **视频与音频生成**：Sora / Runway / Pika / SVD、Whisper / VITS / Bark / MusicGen
* **多模态大模型 (VLM/MLLM)**：CLIP / ALIGN、SigLIP / Florence / EVA-CLIP、BLIP / LLaVA / MiniGPT-4、Qwen-VL / CogVLM / InternVL

### 七、模型优化、压缩与部署

* **模型压缩**：剪枝 (Pruning)、彩票假设 / 结构化剪枝 / N:M 稀疏、量化 (PTQ / QAT / AWQ / GPTQ)、SmoothQuant / FP8 / NF4 / BitNet、知识蒸馏、特征KD / 关系KD / 自蒸馏
* **架构搜索 (NAS)**：DARTS / ENAS / OFA
* **高效推理技术与框架**：KV Cache / PagedAttention、FlashDecoding / Continuous Batching / Speculative Decoding、vLLM / SGLang / TGI / LMDeploy、TensorRT / ONNX / OpenVINO / llama.cpp、DeepSpeed / Megatron-LM
