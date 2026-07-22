# AxCore Section 4.1 mpFPMA 讲解笔记

更新时间：2026-07-05

主题：Extending FPMA to mpFPMA

## 核心问题

普通 FPMA 主要处理同精度浮点乘法，例如 FP16 x FP16。AxCore 要处理的是 mixed-precision GEMM，例如 FP16 activation x FP4 weight。

因此，AxCore 必须解决两个问题：

1. FP16 和 FP4 的 mantissa 位宽不同，需要 alignment。
2. FP16 和 FP4 的 exponent bias 不同，需要 bias correction。

最终，AxCore 将 mixed-precision multiplication 近似为：

$$
R = A + \mathrm{Align}(W_q) - B_1 + C_1
$$

其中 `C1` 是后续 Section 4.3 引入的 compensation term。

## 直觉

FPMA 把浮点乘法近似成“浮点编码表示上的加法”。但 FP16 和 FP4 的编码长度、mantissa 精度、bias 都不一样，所以不能直接相加。

AxCore 的 4.1 做的就是：

- 先把 FP4 weight 的 mantissa 补零左移，让它对齐到 FP16 activation 的 fixed-point 表示。
- 再减去 mixed-precision 带来的 bias mismatch。
- 最后加上 compensation term，降低近似误差。

