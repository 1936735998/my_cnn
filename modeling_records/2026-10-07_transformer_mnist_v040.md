# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v040。
阶段：固定位置编码相加反向。本轮真实执行与核验；未完成输入投影反向、参数更新或训练。

## 问题范围、来源与清洗（事实）

用户报告完整 d_encoder_input 为 `(28,32)`、GPU 0。本轮沿前向 encoder_input=embedding+position_encoding 将梯度传回输入映射的输出 embedding。逐步教学代码贴在聊天，不改用户主文件，不交付完整 Python 模型文件。

独立进程读取执行 `F:\PythonProjects\deep_learning\transformer_mnist.py` 已保存前三十九步，再执行本片段；未操作用户终端的实时内存。主文件 SHA256 `3bc11b9afc06f20cff1ad5da4902af7224e71becac21e63cc6697dfac67f39af`，执行前后未变。

数据沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。取前 5000 张训练图、前 1000 张测试图；像素 CuPy float32 除以 255，标签 int64。本轮仅首张训练图、标签 5。无新清洗、下载、数据划分、增强或测试集评估。

## 假设、变量与公式

延续图片 28 行作为序列、模型宽度 32、四头注意力、单层 post-LN 编码器、前馈宽度 64、位置均值池化与十类交叉熵。位置编码为固定正弦/余弦值，全部参数保持同一次前向时的值。

```text
E = B + P
E=encoder_input, B=embedding, P=position_encoding，均 (28,32)
dB = dE * 1 = d_encoder_input
```

固定 P 时，E 对 B 的逐元素导数为 1，因此不额外平均或缩放。固定位置编码表示不优化 P，而不表示对 P 的数学偏导为零；若 P 可训练，dP 同样为 dE。本版无需创建 d_position_encoding。embedding 是输入仿射输出，非词嵌入表。

## 方法选择理由（判断）

CuPy copy 生成独立 GPU 梯度数组，保留上游梯度便于后续检查。v039 已对完整下游的 896 个 E 分量做过差分；本轮只新增固定 P 后对 B 的三个有代表性的方向检查，不重复全部分量核验。不使用自动求导、现成神经网络层或 CPU 模型训练。

## 本步代码

```python
# ==================================================
# 反向第二十步：从 encoder_input 传回 embedding
# ==================================================

# encoder_input = embedding + position_encoding
# 位置编码固定，梯度直接传回输入映射的输出
d_embedding = d_encoder_input.copy()

print("d_embedding:", d_embedding.shape)
print("一行映射特征的梯度:", d_embedding[0].shape)
print("d_embedding 的实际设备:", d_embedding.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`，Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。本轮实际执行并同步 CUDA；未修改依赖。

d_embedding `(28,32)`、float32、GPU 0、全部有限，与上游逐元素相等且存储独立。所有 20 组参数、embedding、位置编码、encoder_input、上游梯度与 loss 均未改变。参数量仍 9802。

使用 v039 的独立 NumPy float64 完整编码器前向参考。固定真实 P，分别沿单个特征 (0,0)、第 0 行全特征单位方向、固定种子稠密单位方向扰动 B，重算 B+P 后全部注意力、两次 LayerNorm、前馈、池化和分类损失。采用中心差分，初始步长 1e-5；若跨 ReLU 折点则减半，实际全部最终扰动保持掩码。三个方向在 rtol=2e-5、atol=2e-7 下通过，最大绝对误差 `1.16058256e-09`；每方向实际数值、步长与折半次数见 JSON。该核验不代表全部参数整图梯度已检查。

CPU 基础损失 `1.97143177229` 与 GPU 损失 `1.97143173218` 的绝对差为 `4.011606669e-08`。CPU 重建 B+P 使用双精度，GPU 前向使用 float32，舍入差异在容差内。

实际输出：

```text
d_embedding: (28, 32)
一行映射特征的梯度: (32,)
d_embedding 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [实际 GPU 和方向差分证据](2026-10-07_transformer_position_add_backward_validation_v040.json)。JSON 包含准确聊天片段、参考前向源码、实际设备与误差；相对链接已检查，本步无新增图。

## 与前版变化、限制及下一步

v039 完成四路径梯度汇合，本版新增 d_embedding。新增参数 0、参数更新 0；固定位置编码未改变。保留历史记录，更新本版与项目记录索引；同步仅复制文档及证据，不重跑模型。

还未计算 W_in/b_in 或像素输入梯度，完整反向尚未结束；无训练或识别准确率结果。下一步对 embedding=X@W_in+b_in 手写反向，随后检查全部参数梯度，再统一更新参数并组织 GPU 训练。
