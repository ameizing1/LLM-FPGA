# TCASI24 中文精读稿

论文：Hardware-Efficient Multipliers With FPGA-Based Approximation for Error-Resilient Applications

来源：IEEE TCAS-I 2024, Vol. 71, No. 12

本文件是学习用中文精读稿，不是逐字全文翻译。它按论文原有结构整理主要内容、技术逻辑、公式含义和与当前 AM-LUT / LLM-FPGA 项目的关系。

## 0. 先把这篇文章放到我们项目里

这篇文章研究的是 FPGA 上的准确与近似整数乘法器设计，重点不是 LLM，也不是浮点量化，而是 FPGA 底层乘法单元。

它的核心问题是：

```text
很多 approximate multiplier 是按 ASIC 门级电路思路设计的。
但 FPGA 的基本资源是 LUT、LUT6_2、carry chain。
所以 ASIC 上省门数的设计，搬到 FPGA 上不一定省资源、功耗或延迟。
```

因此，作者提出了更贴合 FPGA 结构的 4x4、8x8、16x16 乘法器设计方法。

和我们当前项目的关系可以这样理解：

```text
AxCore:
  面向 LLM GEMM，用 FP16 activation × FP4 weight 的近似计算替代昂贵乘法。

TCASI24 / AM-LUT:
  面向 FPGA 乘法器底层，用 LUT-friendly 的近似整数乘法器替代精确乘法器。

我们关心的问题:
  能不能把 TCASI24 这类 approximate multiplier 用到量化 LLM 的整数 GEMM 或 mixed-precision 近似路径里。
```

## 1. 摘要意译

近似乘法器可以在一些容错应用中节省硬件资源。过去大量近似乘法器主要面向 ASIC 电路设计；但是这些设计直接放到 FPGA 加速器上时，不一定能获得类似的性能收益。

本文提出了一组面向 FPGA 的准确与近似 4x4 乘法器。关键方法有两个：

```text
LUT sharing:
  当两个 LUT 的输入相同或高度相似时，把它们合并到一个 LUT 里，提高 LUT 利用率。

carry switching:
  不直接把截断的 carry 丢掉，而是把它转移到相邻 bit 的计算中，让误差变小。
```

为了构造更高位宽的乘法器，作者还提出了三种近似加法器，用于累加子乘法器结果。通过组合 4x4 乘法器和不同加法器，作者构造了 140 种 8x8 近似乘法器，为不同精度/硬件开销需求提供选择。

实验结果显示，在 Xilinx exact multiplier 上，某些 8x8 近似乘法器最多可以获得明显的功耗、延迟和面积改善。作者还把 8x8 设计扩展到 16x16，并用图像处理任务验证这些近似乘法器的应用可行性。

一句话：

```text
这篇文章不是单个近似乘法器，而是一套 FPGA-aware approximate multiplier library 的设计方法。
```

## 2. Introduction：为什么要重新设计 FPGA 近似乘法器

### 2.1 近似计算的背景

很多应用天然有容错性，比如图像处理、多媒体、部分神经网络计算等。它们不一定要求每一次乘法都完全精确。

因此可以用近似计算牺牲一点数值精度，换取：

```text
更低功耗
更小面积
更短延迟
更高吞吐
```

乘法器在这些应用里通常非常频繁，所以乘法器的硬件开销会成为关键。

### 2.2 为什么 ASIC 方法不能直接搬到 FPGA

已有很多近似乘法器来自 ASIC 电路设计。ASIC 上常见优化目标是减少门数量、压缩部分积、使用近似 compressor 等。

但 FPGA 的实现基础不同。FPGA 不是任意门随便拼，而是由结构化资源组成，例如：

```text
6-input LUT
LUT6_2 双输出模式
carry chain
slice
CLB
```

所以，一个 ASIC 设计即使门数很少，在 FPGA 上可能仍然需要很多 LUT 层、很长 carry chain，甚至比精确乘法器还差。

作者用 Fig. 1 说明：一些 ASIC-based approximate multiplier 放到 FPGA 上，并没有获得预期的 power、latency、area 收益。

### 2.3 作者指出的四个动机

第一，准确乘法器本身的 LUT 层和 carry chain 可能太长。

传统 FPGA accurate multiplier 的 critical path 可能经过多个 LUT 层和多个 carry chain。近似乘法器通常是在准确乘法器基础上删减而来。如果基础准确乘法器本身不够好，近似版本也很难高效。

第二，简单删除 LUT 或 carry chain 会带来较大误差。

很多近似乘法器通过删逻辑来省资源，这确实能降低硬件开销，但 carry 被直接截断后，误差可能迅速变大。

第三，高位宽乘法器需要加法器累加子乘法器结果。

比如 8x8 可以由四个 4x4 子乘法器加起来得到。这个加法器如果设计不好，可能把子乘法器省下来的资源又花回去了。

第四，已有近似乘法器库的配置范围不够宽。

应用需求不同，有的更看重精度，有的更看重功耗。作者希望构造一组配置丰富的 multiplier library，而不是只给几个孤立设计。

## 3. 本文贡献

作者的主要贡献可以概括成四点。

第一，提出了一个优化后的 accurate 4x4 multiplier。它面向 LUT 结构优化，critical path 只有两个 LUT 和一个 carry chain。

第二，在这个 accurate 4x4 multiplier 基础上，提出四种 approximate 4x4 multiplier：

```text
LSAM1
LSAM2
CSAM1
CSAM2
```

其中 LSAM 代表 LUT Shared Approximate Multiplier，强调 LUT sharing；CSAM 代表 Carry Switched Approximate Multiplier，强调 carry switching。

第三，提出三种近似加法器，用于构造更高位宽乘法器：

```text
MODA
HSLP
NCCA
```

再加上准确加法器 ACCA，一共有四种 adder 选择。

第四，组合 4x4 乘法器和 adder，构造出 140 种 8x8 approximate multiplier，并进一步扩展到 16x16。

## 4. Section II：FPGA 基础资源

这篇文章主要面向 Xilinx 7-series FPGA，但作者认为方法也可以推广到其他有 6-input LUT 和 carry chain 的 FPGA。

### 4.1 CLB、slice、LUT

现代 FPGA 的基本计算资源是 CLB。一个 CLB 通常包含两个 slice，每个 slice 里有若干 LUT 和 carry chain。

在 Xilinx 7-series 中，一个 slice 里有：

```text
4 个 6-input LUT
1 条相关 carry chain
```

### 4.2 LUT6

LUT6 可以实现任意 6 输入 1 输出的组合逻辑。

可以理解成：

```text
输入: I0, I1, I2, I3, I4, I5
输出: O
功能: 任意 6-input boolean function
```

LUT 的函数由 INIT 值定义。INIT 本质上就是 truth table 的压缩表示。

### 4.3 LUT6_2

LUT6_2 是 Xilinx LUT 的一个重要模式。它可以把一个 6-input LUT 用成两个共享输入的 5-input 函数：

```text
O5: 一个 5-input function
O6: 另一个 5-input 或 6-input function
```

关键点是：如果两个逻辑函数共享相同的 5 个输入，就可以塞进同一个 LUT6_2，而不是用两个 LUT。

这就是后面 LUT sharing 的硬件基础。

### 4.4 Carry chain

FPGA 里有专门的 carry chain，用来高效实现加法器。一般 LUT 产生：

```text
carry-generate: Gen
carry-propagate: Prop
```

carry chain 根据这些信号快速传播进位。

在这篇文章里，carry chain 很关键，因为乘法器最终需要把部分积加起来。carry chain 用得好可以快，用得多也会带来资源、功耗和延迟开销。

所以作者后面会反复处理一个矛盾：

```text
carry chain 有利于准确加法，
但 carry chain 也是硬件开销来源。
```

## 5. Section III-A：优化 accurate 4x4 multiplier

作者先设计了一个准确的 4x4 乘法器。输入是两个 4-bit operand：

```text
A3 A2 A1 A0
B3 B2 B1 B0
```

输出是 8-bit product：

```text
P7 P6 P5 P4 P3 P2 P1 P0
```

### 5.1 为什么先设计 accurate 版本

因为 approximate multiplier 往往是在 accurate multiplier 的基础上删减或合并逻辑。如果 accurate baseline 本身 LUT 层太深，后面的 approximate 设计也很难漂亮。

作者的 accurate 4x4 multiplier 把乘法拆成两个 4x2 乘法：

```text
Layer 1: A3 A2 A1 A0 × B1 B0
Layer 2: A3 A2 A1 A0 × B3 B2
```

这两个 layer 可以并行计算。

### 5.2 两个 stage

Stage 1：用两组 LUT 计算两个 4x2 layer 的中间结果。

每一层的输入刚好适配 6-input LUT，因此 LUT 利用率较高。

Stage 2：把两个 layer 的结果继续组合，生成 carry chain 需要的 Prop 和 Gen 信号，最终由 carry chain 算出 P3 到 P7 一类的高位结果。

这使 critical path 缩短到：

```text
2 个 LUT + 1 条 carry chain
```

这是后续 LSAM / CSAM 设计的基础。

## 6. Section III-B：LSAM，LUT sharing 近似 4x4 乘法器

LSAM 的核心思想是：

```text
如果两个 LUT 的输入相同或高度相同，就尝试把它们合并。
```

这会省 LUT，但会引入近似误差。

### 6.1 LSAM1

作者观察 accurate multiplier 结构，发现 LUT10 和 LUT13 有相同的 5 个输入。

其中一个可以使用 LUT6_2，另一个像是单独的 LUT5。于是作者把 LUT13 删除，把它的相关逻辑合并到 LUT10 中。

LSAM1 和 accurate multiplier 的主要区别就是：

```text
删除 LUT13
保留 carry chain
设置 carry-in
```

结果是硬件更省，但只在少量输入组合下产生误差。

论文给出的错误发生概率约为：

$$
6 / 256 = 0.023
$$

这个误差较小的原因是：错误主要限制在 P3，没有继续传播到更高位。

### 6.2 LSAM2

LSAM2 在 LSAM1 基础上进一步共享 LUT。

作者发现某些 LUT 的输入元素相同，因此可以通过截断某个低位 carry，让两个输出共享一个 LUT。

这里问题变成：到底共享哪两个信号，误差最小？

作者分析了三种策略：

```text
Strategy 1: L13 和 L14 共享
Strategy 2: L13 和 L15 共享
Strategy 3: L14 和 L15 共享
```

然后用 error distance 比较。

论文中给出的三种误差距离为：

$$
ED_1 = 2^3 \times 8 + 2^4 \times 4 = 128
$$

$$
ED_2 = 2^3 \times 8 + 2^5 \times 4 = 192
$$

$$
ED_3 = (2^5 - 2^4) \times 4 = 64
$$

因为 ED3 最小，所以 LSAM2 采用 Strategy 3，也就是让 L14 和 L15 共享一个 LUT。

注意这里很重要：LSAM2 不是随便删逻辑，而是用误差距离来选择“哪个删法最划算”。

相比 LSAM1，LSAM2 的错误输入组合数量从 6 增加到 13，但换来了更多硬件节省。

## 7. Section III-C：CSAM，carry switching 近似 4x4 乘法器

CSAM 的目标比 LSAM 更激进：减少甚至去掉 carry chain。

carry chain 对准确加法有帮助，但它也是硬件开销较大的部分。如果完全去掉 carry chain，硬件会省很多，但误差会增加。

作者提出 carry switching 来缓解这个问题：

```text
不要把 carry 完全丢掉。
把本来要进入高位的 carry，转移到相邻低位的计算中。
这样虽然仍然近似，但 error distance 会变小。
```

### 7.1 CSAM1

CSAM1 基于 LSAM2，去掉 carry chain，用 LUT 直接计算输出。

问题出现在 P3 和 P4 附近。准确计算 P4 时需要一个来自低位的 carry term，但如果把这个 carry term 直接截断，P4 会在很多情况下错。

论文把这个 carry 写成：

$$
carry\ term\ 1 =
L13L21 + L13L12A0B2 + L21L12A0B2
$$

准确情况下：

$$
P3_{exact} =
L13 \oplus L21 \oplus (L12A0B2)
$$

$$
P4_{exact} =
L14 \oplus L22 \oplus carry\ term\ 1
$$

如果 P4 的 LUT 输入不够，就不能直接把所有项放进去。简单做法是丢掉 carry term 1，但作者认为这样误差大。

于是作者把 carry term 1 转移到 P3：

$$
P3_{CSAM1} =
L13 \oplus L21 \oplus (L12A0B2) + carry\ term\ 1
$$

$$
P4_{CSAM1} =
L14 \oplus L22
$$

直观理解：

```text
本来应该让 P4 加 1。
现在没有办法准确加到 P4，就尽量让 P3 变大一点。
这样比完全丢 carry 更接近原值。
```

这就是 carry switching 的味道。

### 7.2 CSAM2

CSAM2 在 CSAM1 基础上进一步删除 LUT10，并把 P7 设置成 L25。

删除 LUT10 会导致 P6 / P7 附近产生误差，因此作者再次使用 carry switching。

论文抽出：

$$
carry\ term\ 2 =
L24(L15L23 + L15L14L22 + L23L14L22)
$$

然后在 P6 里补偿它：

$$
P6_{CSAM2} =
L24 \oplus (L15L23 + L15L14L22 + L23L14L22)
+ carry\ term\ 2
$$

$$
P7_{CSAM2} =
L25
$$

相比 CSAM1，CSAM2 的错误输入组合数量从 55 增加到 60，但硬件进一步降低。

在我们的代码中，已经有 CSAM2 的行为模型：

[tcasi24.py](</D:/LiuZhiWei/Documents/STUDTORWORK/研0/SJTU relative project/LLM-FPGA/am_lut_tcasi24/tcasi24.py>)

对应函数是：

```python
mul4_csam2(a, b)
```

## 8. Section IV：从 4x4 构造更高位宽乘法器

### 8.1 2n x 2n 的递归构造

一个 2n x 2n 乘法器可以由四个 n x n 乘法器构造。

把输入拆成高低两半：

```text
A = AH · 2^n + AL
B = BH · 2^n + BL
```

则：

$$
A \times B =
HH \cdot 2^{2n}
+ (HL + LH) \cdot 2^n
+ LL
$$

其中：

```text
HH = AH × BH
HL = AH × BL
LH = AL × BH
LL = AL × BL
```

所以，高位乘法器需要：

```text
4 个低位乘法器
1 个加法器
```

这也是为什么作者不仅设计 4x4 multiplier，还要设计 approximate adder。

### 8.2 ACCA：准确加法器

ACCA 是传统准确加法器，用 LUT6_2 产生 Gen 和 Prop 信号，然后通过 carry chain 做准确进位。

优点：

```text
准确
```

缺点：

```text
需要较多 carry chain
很多操作串行发生
```

### 8.3 MODA：中等近似加法器

MODA 删除了部分低位 carry chain。

低位部分直接用 OR 操作近似求和，高位部分仍然保留 carry chain。它还在某个边界位置做 carry prediction，把预测 carry 作为后面 carry chain 的输入。

它的定位是：

```text
比 ACCA 省硬件
比更激进的近似加法器更准确
```

### 8.4 HSLP：高位保 carry，低位预测 carry

HSLP 的名字可以拆开看：

```text
High-bit Carry Save
Low-bit Carry Predict
```

它让最高位一段继续用 carry chain 保持准确，而中低位用 LUT 做近似求和，并用前一列信息预测 carry。

直观理解：

```text
高位错误代价大，所以高位尽量准。
低位错误代价小，所以低位可以更近似。
```

### 8.5 NCCA：无 carry chain 加法器

NCCA 更激进，完全使用 LUT-based approximate adder，不使用 carry chain。

优点：

```text
硬件开销最低
```

缺点：

```text
误差通常更大
```

这四种 adder 形成了一个精度/资源梯度：

```text
ACCA: 最准确，资源最多
MODA: 中等近似
HSLP: 更近似
NCCA: 最省资源，误差最大
```

### 8.6 8x8 multiplier 的组合规则

一个 8x8 乘法器由四个 4x4 乘法器加一个 adder 组成。

四个 4x4 子乘法器的位置重要性不同：

```text
HH 最重要
HL / LH 次之
LL 最不重要
```

因此，作者不是任意排列 LSAM1、LSAM2、CSAM1、CSAM2，而是遵循一个原则：

```text
更准确的子乘法器放在更重要的位置。
更近似的子乘法器放在更不重要的位置。
```

例如，允许：

```text
HH: LSAM1
HL: LSAM2
LH: CSAM1
LL: CSAM2
```

但不允许把更差的 LSAM2 放在比 LSAM1 更重要的位置。

这样，四种 4x4 multiplier 可以形成 35 种有效 8x8 组合。再乘以 4 种 adder：

$$
35 \times 4 = 140
$$

因此得到 140 种 approximate 8x8 multiplier。

命名方式例如：

```text
MODA_1234
```

含义是：

```text
adder = MODA
HH = LSAM1
HL = LSAM2
LH = CSAM1
LL = CSAM2
```

我们的本地代码现在实现的是一个第一阶段行为模型：用四个 4x4 block 拼成 unsigned 8x8，然后用 sign-magnitude 方式包装成 signed int8。

对应文件：

[tcasi24.py](</D:/LiuZhiWei/Documents/STUDTORWORK/研0/SJTU relative project/LLM-FPGA/am_lut_tcasi24/tcasi24.py>)

关键函数：

```python
mul8_unsigned(a, b, mode)
mul_int8_signed(a, b, mode)
build_int8_lut(mode)
```

### 8.7 16x16 multiplier

16x16 可以继续由四个 8x8 multiplier 加一个 adder 构造。

作者强调，这种方法可以递归扩展到更高位宽。

但对我们来说要注意一点：位宽越高，误差传播和分布会更复杂。不能只看单个 4x4 block 的误差，还要看 GEMM 累加后的统计误差。

这也是我们当前项目已经做了 int8 GEMM error report 的原因：

[tcasi24_int8_error_report.md](</D:/LiuZhiWei/Documents/STUDTORWORK/研0/SJTU relative project/LLM-FPGA/outputs/reports/tcasi24_int8_error_report.md>)

## 9. Section V：实验指标和结果

### 9.1 实验设置

作者用 Verilog 实现乘法器，并在 Vivado 2019.1 下综合到 Xilinx xc7vx330tffg1157。

比较对象包括：

```text
Xilinx default exact multiplier
Xilinx multiplier IP
已有 FPGA-based approximate multiplier
若干 ASIC-based approximate multiplier 搬到 FPGA 上的实现
```

### 9.2 误差指标

论文使用几类常见 approximate arithmetic 指标。

#### ED：Error Distance

ED 表示精确输出和近似输出之间的绝对差：

$$
ED = |M - M'|
$$

其中 M 是精确结果，M' 是近似结果。

#### ER：Error Rate

ER 表示发生非零误差的概率：

$$
ER =
\frac{\#(ED \ne 0)}
{\#(ED \ne 0) + \#(ED = 0)}
$$

#### MED：Mean Error Distance

MED 是所有输入组合上的平均 ED：

$$
MED =
\frac{1}{2^{2N}}
\sum_{i=1}^{2^{2N}} ED_i
$$

这里 N 是 N-bit multiplier 的位宽。

#### NED：Normalized Error Distance

NED 是归一化后的 MED，用于不同位宽乘法器之间比较：

$$
NED =
\frac{MED}{D}
$$

通常 D 可以取最大可能误差，例如：

$$
D = (2^N - 1)^2
$$

#### RED / MRED

RED 表示相对误差距离，也就是把 ED 除以精确结果：

$$
RED =
\sum_{i=1}^{2^{2N}}
\frac{ED_i}{M_i}
$$

MRED 是 RED 的平均值：

$$
MRED =
\frac{RED}{2^{2N}}
$$

### 9.3 4x4 结果

准确 4x4 multiplier 的比较中，作者的 accurate design 延迟接近 Xilinx default exact multiplier，并且低于其他一些准确设计。

原因是：

```text
作者设计的 critical path = 2 LUT + 1 carry chain
其他设计可能需要 3 LUT + 更多 carry chain
```

近似 4x4 multiplier 的比较中：

```text
LSAM1: 精度损失最小
CSAM2: 功耗、延迟、面积最低
```

CSAM2 之所以硬件开销低，是因为它减少 LUT，并且不用 carry chain。

论文强调：carry switching 能补偿一部分没有 carry chain 带来的精度损失。

### 9.4 8x8 结果

作者用 Pareto 图比较 8x8 multiplier 的精度/硬件折中。

重点结论：

```text
作者提出的 8x8 multiplier 在很多配置下位于 Pareto frontier。
在相似误差下，它们通常比已有 FPGA-based multiplier 更省硬件。
ASIC-based approximate multiplier 搬到 FPGA 上后表现较差。
```

这再次支撑本文核心观点：

```text
FPGA approximate multiplier 需要 FPGA-aware 设计。
```

一个重要细节是，carry chain 往往比 LUT 更影响功耗和延迟。比如两个设计 LUT 数接近，但一个 carry chain 更多，整体 power / latency 可能仍然更差。

### 9.5 16x16 结果

作者把部分 8x8 设计扩展到 16x16，并和其他 16-bit approximate multiplier 比较。

主要结论：

```text
作者的 16x16 approximate multiplier 在类似精度下，power 和 latency 更低。
低位乘法器中的优势可以延续到更高位宽。
```

另一个有意思的观察是，有些已有方法扩展到 16-bit 后 MRED 明显变差，而作者某些组合在 16-bit 下仍能保持较稳定的误差水平。

对我们项目的启发：

```text
如果要把 AM-LUT 用到 LLM GEMM，不能只看 4x4 或 8x8 单点误差。
必须看扩展后、矩阵累加后、真实数据分布下的误差。
```

## 10. Section VI：图像处理应用

作者用两个图像处理任务测试 approximate 8x8 multiplier：

```text
image sharpening
image multiplication
```

评价指标是 PSNR 和 SSIM。

### 10.1 PSNR

PSNR 用来衡量图像误差，通常越高越好。很多图像处理场景中，30 dB 以上通常被认为视觉上可接受。

论文给出的形式可以理解为：

$$
PSNR =
10 \log
\left(
\frac{m \times p \times MAX_I^2}
{\sum_{i=0}^{m-1}\sum_{j=0}^{p-1}(I(i,j)-k(i,j))^2}
\right)
$$

其中：

```text
I(i,j): 精确计算得到的像素
k(i,j): 近似计算得到的像素
MAX_I: 像素最大值
m, p: 图像尺寸
```

### 10.2 SSIM

SSIM 衡量结构相似度，不只看像素差，也看图像结构是否接近。

它比单纯 MSE / PSNR 更符合人眼感知。

### 10.3 图像锐化

图像锐化中，作者发现本文 multiplier 在相似 MRED 下可以得到更好的图像质量。

论文解释了一个原因：本文乘法器误差有不对称性，错误更多发生在第一个操作数小于第二个操作数的时候。而在图像锐化里，第一个操作数通常是像素值，第二个操作数通常是 kernel 值，像素值往往更大，因此实际误差影响较小。

这个点对我们做 LLM 很重要：

```text
近似乘法器好不好，不只看平均误差。
还要看误差和输入数据分布是否匹配。
```

LLM 的 activation / weight 分布和图像像素分布完全不同，所以 TCASI24 在图像上表现好，不代表直接在 LLM 上也一定好。

### 10.4 图像乘法

图像乘法是逐像素相乘，因此更直接测试 multiplier 本身。

作者发现，在相似精度损失下，本文提出的 multiplier 通常能得到更高 PSNR，趋势与图像锐化实验一致。

## 11. Section VII：结论

本文提出了一套 FPGA-based approximate multiplier 设计方法。

完整路径是：

```text
先优化 accurate 4x4 multiplier
再基于它设计 LSAM / CSAM approximate 4x4 multiplier
再设计 ACCA / MODA / HSLP / NCCA adders
再组合成 8x8 multiplier library
再扩展到 16x16
最后用图像处理验证应用效果
```

最重要的两个方法是：

```text
LUT sharing:
  利用 LUT6_2 共享输入的能力，合并相似 LUT，减少 LUT 数。

carry switching:
  不把 carry 完全截断，而是转移到相邻 bit，降低 error distance。
```

作者报告 CSAM2 这种最硬件高效的 4x4 approximate multiplier 相比 Xilinx multiplier IP core，在 power、latency、area 上有明显改善。

最终，作者强调：在相似精度损失下，本文设计比已有方法有更多硬件节省。

## 12. 和 AxCore 对照理解

现在可以把 AxCore 和 TCASI24 放在一张概念表里：

| 维度 | AxCore | TCASI24 |
|---|---|---|
| 目标 | LLM GEMM 近似计算 | FPGA 近似整数乘法器 |
| 数据类型 | FP16 activation × FP4 weight | unsigned integer 4x4 / 8x8 / 16x16 |
| 核心方法 | FPMA / mpFPMA / correction advancing | LUT sharing / carry switching |
| 硬件关注点 | PE 内乘加近似、修正提前 | LUT6_2 利用、carry chain 减少 |
| 误差来源 | 浮点近似、FP4 subnormal、mantissa/exponent 对齐 | LUT 合并、carry 截断/转移、近似加法 |
| 适用方向 | LLM 低精度 GEMM | error-resilient integer applications |

一个很重要的判断：

```text
AxCore 是 algorithm + arithmetic co-design，更贴近 LLM 数据格式。
TCASI24 是 FPGA arithmetic primitive design，更贴近底层硬件资源。
```

所以如果导师让我们考虑 AM-LUT 替换 AxCore，真正的问题不是“谁更省”，而是：

```text
TCASI24 这种 LUT-based approximate integer multiplier
能不能在 LLM 量化 GEMM 的数据分布下保持可接受误差？

它省下的 LUT / carry chain / delay
能不能抵消它在精度、累加误差、符号处理、量化格式适配上的代价？
```

## 13. 你读的时候可以优先关注的问题

第一，先读 Section II，把 LUT6、LUT6_2、carry chain 看懂。

如果不理解 FPGA 资源模型，后面的 LUT sharing 和 carry switching 会像纯布尔公式。

第二，读 Section III 时不要急着完全推每个 LUT 表达式。

先抓住：

```text
LSAM: 合并 LUT
CSAM: 处理 carry
```

第三，Section IV 很关键。

因为它告诉我们：单个 4x4 block 不是终点，真正应用需要扩展到 8x8 / 16x16。LLM GEMM 更是如此。

第四，Section V 的表格不用死记数字。

重点看趋势：

```text
carry chain 少，power/latency 往往更好
LUT 数少不等于一定更优
ASIC approximate design 搬到 FPGA 不一定有效
```

第五，Section VI 要带着怀疑读。

图像处理能证明它适用于 error-resilient application，但不能直接证明它适用于 LLM。

我们后续应该重点问：

```text
LLM activation / weight 的分布是否适合这种误差模式？
int8 GEMM 累加后误差是否会放大？
能否做 signed two's complement 版本，而不是简单 sign-magnitude wrapper？
能否和 quantization scale / zero point / group-wise quantization 对齐？
```

## 14. 当前项目中已经对应上的内容

本地已有 TCASI24 行为模型：

[tcasi24.py](</D:/LiuZhiWei/Documents/STUDTORWORK/研0/SJTU relative project/LLM-FPGA/am_lut_tcasi24/tcasi24.py>)

当前实现包括：

```text
mul4_lsam1
mul4_csam2
mul8_unsigned
mul_int8_signed
build_int8_lut
```

本地已有 GEMM helper：

[gemm.py](</D:/LiuZhiWei/Documents/STUDTORWORK/研0/SJTU relative project/LLM-FPGA/am_lut_tcasi24/gemm.py>)

当前实验报告：

[tcasi24_int8_error_report.md](</D:/LiuZhiWei/Documents/STUDTORWORK/研0/SJTU relative project/LLM-FPGA/outputs/reports/tcasi24_int8_error_report.md>)

报告里可以看到一个初步趋势：

```text
LSAM1 在当前 int8 wrapper 下误差明显小于 CSAM2。
不同输入分布下误差差异很大。
small_normal / sparse_small 的误差远低于 uniform_int8。
outlier_channels 分布下误差会变差。
```

这和论文 Section VI 的启发一致：

```text
approximate multiplier 的可用性强烈依赖输入分布。
```

