# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v023。
阶段：平均池化手写反向，真实 GPU 执行及 CPU 中心差分验证；不是仅文档更新。尚未完成全网络反向、参数更新或训练。

## 问题范围与数据来源（事实）

用户报告分类头梯度 dW_cls `(32,10)`、db_cls `(10,)`、d_image_features `(32,)`，均在 GPU 0。本轮继续把图片特征的梯度反向通过位置均值池化，传至 encoder_output。只交付聊天代码；不修改用户主文件，不交付完整 Python 模型文件。最终目标仍为逐层手写全部反向与更新，并在 GPU 训练。

独立进程读取并执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前二十二步，再执行下面的聊天片段。未操作用户终端中的实时变量。主文件 SHA256 `6b5f19487a001474167783fe9ae76a4d0b6505bb666914037dd15c71a85ac4e9`，执行前后相同。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000 张、测试前 1000 张；像素转换 CuPy float32 并除以 255，标签 int64。本轮只用首张训练图，标签 5，没有额外清洗、下载、洗牌、数据增强或测试集评估。

## 假设、变量与公式

延续图片每行视为一个空间位置、单层编码器、十类单样本交叉熵设置。平均池化在位置轴 axis=0 上进行，不在特征轴取均值；无掩码或加权池化。上游梯度已经由当前分类头传回，所有参数保持本次前向值。

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| encoder_output=E | (28,32) | 28 个位置、每个位置 32 个特征 |
| image_features=f | (32,) | 图片的均值特征 |
| d_image_features=g | (32,) | 损失对图片特征的梯度 |
| d_encoder_output | (28,32) | 损失对每个位置特征的梯度 |

```text
f[k] = (1/L) * sum_i E[i,k], L=seq_len=28
∂f[k]/∂E[i,k] = 1/L
d_encoder_output[i,k] = d_image_features[k] / L
sum_i d_encoder_output[i,k] = d_image_features[k]
```

因均值使每行权重相同，该节点传回的各行梯度相同。这不意味着后续编码器内部各行或输入像素梯度相同。每个特征保持独立，32 个特征之间不求和。参数数量不增加，仍 9802。

## 方法选择理由（判断）

除以实际位置数，再用 CuPy broadcast_to 沿位置维展开，直接实现均值的链式法则。copy 分配独立存储，便于后续梯度运算，不让广播结果共享同一行数据。只使用基础数组操作，没有自动求导或现成模型层。保持小步教学，下一步单独推导第二次 LayerNorm 的反向。

## 本步代码

```python
# ==================================================
# 反向第三步：平均池化
# ==================================================

# 每行接收图片特征梯度的 1/28
# (32,) → (28, 32)
d_encoder_output = cp.broadcast_to(
    d_image_features / cp.float32(seq_len),
    encoder_output.shape
).copy()

print("d_encoder_output:", d_encoder_output.shape)
print("第 0 行的梯度形状:", d_encoder_output[0].shape)
print("首尾两行梯度是否相同:", bool(cp.all(
    d_encoder_output[0] == d_encoder_output[-1]
)))
print("各行梯度之和与上游梯度的最大差值:", float(cp.max(cp.abs(
    cp.sum(d_encoder_output, axis=0) - d_image_features
))))
print("d_encoder_output 的实际设备:", d_encoder_output.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0；可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实运行 CuPy 片段并同步 CUDA；未安装或修改依赖。CuPy 缓存放于工作区 work/cupy_cache，解释器使用 -B。

输出梯度为 `(28,32)`、float32、GPU 0，全 896 个分量有限。每行梯度一致；输出有独立连续存储。各行梯度求和恢复上游梯度，最大绝对差 `7.450580597e-09`，通过 rtol=1e-6、atol=2e-7 检查。GPU float32 累加有舍入误差，不要求逐位相等。所有 20 组参数、前向缓存、loss 和上游梯度保持不变。

独立 CPU float64 从实际 encoder_output 和分类参数重算 E→mean(axis=0)→f@W_cls+b_cls→稳定交叉熵，在 E 的全部 896 个分量分别做 ±1e-5 中心差分；rtol=1e-5、atol=1e-7 下全部通过。最大绝对误差 `5.221316721e-10`，相对 L2 误差 `6.573678622e-08`。CPU 双精度重算与 GPU 前向存在舍入差异。这是局部池化及后续分类图验证，不声称验证了编码器内部反向。

本步真实输出：

```text
d_encoder_output: (28, 32)
第 0 行的梯度形状: (32,)
首尾两行梯度是否相同: True
各行梯度之和与上游梯度的最大差值: 7.450580596923828e-09
d_encoder_output 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU 与差分证据](2026-10-07_transformer_mean_pool_backward_validation_v023.json)。JSON 保存逐字片段、输入来源、设备、实际梯度与差分值、误差和输出，代码 SHA256 `b43cd92c268d0a9edf65589957fca994acd253b35ea4d211805ec9c047c956d3`。相对链接已核验。本步无新图，数值与维度即可完整表示验证结果。

## 版本变化、限制与下一步

v022 完成分类头反向；本版新增 d_encoder_output，完成均值池化反向。无参数更新，历史记录保留，生成本版工作区索引并追加项目 modeling_records/index.md；同步记录仅复制和更新索引，不重跑模型。

尚未执行第二次 LayerNorm、前馈、第一层 LayerNorm、注意力、输入投影的反向，也未训练或测量准确率，不能据此得出模型学习效果或 GPU 性能结论。下一步由 d_encoder_output 计算第二次 LayerNorm 的 gamma_ffn、beta_ffn 与 ff_residual 梯度，再逐节点完成反向、统一更新参数、组织 GPU 训练。
