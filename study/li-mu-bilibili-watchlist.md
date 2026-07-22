# 跟李沐学AI：入学前学习计划对应观看清单

更新时间：2026-06-23

## 使用原则

这个账号适合补：

- Transformer / Attention 基础
- GPT / LLaMA 等大模型背景
- 论文阅读方法
- 大模型评测和训练 infra 的宏观理解

它不适合作为主资料来补：

- Adana 低比特量化细节
- AxCore approximate GEMM
- FPGA LUT-based multiplier
- Verilog / 硬件综合 / 资源评估

这些仍然要以导师给的论文、代码和官方文档为主。

## 第一优先级：必须看

### 1. 如何读论文

链接：https://www.bilibili.com/video/BV1H44y1t75x/

用途：

- 先建立读论文方法。
- 后续读 Adana、AxCore、AM-LUT 时按这个方法做结构化笔记。

输出：

- 整理一个自己的论文阅读模板：问题、观察、方法、实验、局限、和我项目的关系。

### 2. 动手学深度学习：注意力机制相关

建议顺序：

1. 64 注意力机制：https://www.bilibili.com/video/BV1264y1i7R1/
2. 65 注意力分数：https://www.bilibili.com/video/BV1Tb4y167rb/
3. 67 自注意力：https://www.bilibili.com/video/BV19o4y1m7mo/
4. 68 Transformer：https://www.bilibili.com/video/BV1Kq4y1H7FL/

用途：

- 补 Attention、Self-Attention、Transformer 的最小数学和代码直觉。
- 对应入学前计划第 1 周。

输出：

- 画出 Q/K/V、Attention、Transformer block 的结构图。
- 写 5 句话解释 prefill/decode 为什么和矩阵计算、KV Cache 有关。

### 3. Transformer 论文逐段精读

链接：https://www.bilibili.com/video/BV1pu411o7BE/

用途：

- 把 Transformer 从课程直觉提升到论文级理解。
- 重点看 self-attention、multi-head attention、FFN、LayerNorm、position encoding。

输出：

- 整理一页：《Transformer block 中哪些部分会成为 LLM 推理计算热点》。

## 第二优先级：建议看

### 4. GPT, GPT-2, GPT-3 论文精读

链接：https://www.bilibili.com/video/BV1AF411b7xQ/

用途：

- 理解 decoder-only LM、scaling、prompt、生成式语言模型的演进。
- 对你理解 LLM inference 很有帮助。

输出：

- 写清楚 GPT 和原始 Transformer encoder-decoder 的区别。

### 5. Llama 3.1 论文精读：导言

链接：https://www.bilibili.com/video/BV1WM4m1y7Uh/

用途：

- 建立现代开源大模型整体视角。

输出：

- 记录 Llama 3.1 的模型、数据、训练、推理/部署中和你项目相关的关键词。

### 6. Llama 3.1 论文精读：模型

链接：https://www.bilibili.com/video/BV1Q4421Z7Tj/

用途：

- 重点看现代 LLM 结构。
- 对 Adana / AxCore 中选择 LLaMA、OPT、Qwen 等模型做评估有帮助。

输出：

- 整理 LLM block 中 GEMM 主要出现在哪里：QKV projection、attention output projection、FFN/MLP。

### 7. Llama 3.1 论文精读：训练 infra

链接：https://www.bilibili.com/video/BV1b4421f7fa/

用途：

- 不是为了做训练，而是了解大模型 infra 的系统视角。
- 可帮助你把项目语言从“硬件模块”升级到“AI Systems/Infra”。

输出：

- 记录训练/推理系统中和 memory、bandwidth、parallelism、hardware efficiency 相关的点。

## 第三优先级：按需看

### 8. HELM 全面语言模型评测

链接：https://www.bilibili.com/video/BV1z24y1B7uX/

用途：

- 如果你要理解 perplexity、zero-shot、模型评估为什么重要，可以看。
- 对 Adana/AxCore 的 accuracy evaluation 有辅助意义。

### 9. 大模型时代下做科研的四个思路

链接：https://www.bilibili.com/video/BV1oX4y1d7X6/

用途：

- 如果你要思考如何从论文中找研究切口，可以看。
- 不属于技术必修。

### 10. ViT 论文逐段精读

可在 UP 主空间或论文精读合集里搜索 “ViT论文逐段精读”。

用途：

- 如果你要读老师给的 ViT FPGA accelerator 文章，可以补。
- 当前不是主线，除非老师明确让你看 ViT 迁移到 LLM 的实现形式。

## 暂时不建议优先看

- ResNet、AlexNet、GAN、MoCo、CLIP、多模态串讲：和当前项目关系不大。
- GPipe、Megatron、ZeRO：偏训练并行，除非后续要做 training infra。
- InstructGPT、Anthropic LLM、GPT-4：偏对齐/能力/产品和评测，不是当前硬件量化主线。
- Sora、视频生成相关：暂时无关。

## 推荐观看顺序

### 2 天快速版

1. 如何读论文
2. 64 注意力机制
3. 67 自注意力
4. 68 Transformer
5. Transformer 论文逐段精读

### 1 周稳妥版

1. 如何读论文
2. 64 注意力机制
3. 65 注意力分数
4. 67 自注意力
5. 68 Transformer
6. Transformer 论文逐段精读
7. GPT, GPT-2, GPT-3 论文精读
8. Llama 3.1：模型

### 和导师项目最贴合版

1. 如何读论文
2. Transformer 相关 4 个动手学视频
3. Transformer 论文逐段精读
4. GPT, GPT-2, GPT-3 论文精读
5. Llama 3.1：模型
6. Llama 3.1：训练 infra
7. HELM 全面语言模型评测

## 看视频时的检查问题

每个视频结束后，不要只记“看过”，要回答：

1. 它和 Adana / AxCore / AM-LUT 哪一部分有关？
2. 它解释了 LLM 的结构、评估、系统，还是硬件瓶颈？
3. 我能否用它解释 GEMM 为什么重要？
4. 我能否把它转化成一张图或一个实验变量？

如果不能回答，就说明这个视频对当前项目帮助有限，可以先跳过。
