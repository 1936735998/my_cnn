# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v011。
阶段：注意力子层残差连接局部前向 GPU 验证。本轮真实执行验证，不是仅文档更新；未训练。

## 范围与证据来源

用户输出 W_O `(32,32)`、b_O `(32,)`、attention_output `(28,32)`，相关设备 GPU 0，符合前一步设置。本轮只将注意力子层输出与该子层输入相加；代码直接贴在聊天，不交付单独 Python 文件，不修改主文件。整体教学目标仍为手写前向、反向与参数更新并在 GPU 训练。

独立进程读取执行保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前十步，再执行本片段；不声称读取编辑器实时内存或操作用户终端。

## 输入来源与处理（事实）

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000 图、测试前 1000 图；像素转 CuPy float32 除以 255，标签 int64。本轮没有新增下载、清洗、划分、洗牌或增强。验证只用首张训练图的局部前向，没有测试集评估。

encoder_input 是图片行输入投影与固定位置编码之和；attention_output 是同一输入经 Q/K/V 投影、多头注意力、拼接与 W_O 输出投影所得。主文件已保存前十步，无需补充旧片段。

## 变量、假设与公式

T=28，d_model=32。X=encoder_input `(28,32)`，F(X)=attention_output `(28,32)`。

```text
R = X + F(X)
R[i,k] = encoder_input[i,k] + attention_output[i,k]
attention_residual: (28,32)
```

这是对应位置、对应特征的逐元素加法。跳连使用进入注意力子层的 encoder_input；该输入已包含位置编码。两个分支形状、float32、GPU 0 及有限数值前提本轮均实际核验。沿用图片行空间序列、无因果掩码的分类设置。本步无新增可训练参数。

## 方法选择理由（判断）

残差结构把子层输入直接加入处理结果，为后续反向保留一条直接的加法路径。教学采用先残差相加、再 LayerNorm 的顺序；本轮仅实现加法，下一步再实现归一化。先单独确认加法对象和形状，便于理解后续分支梯度。使用 CuPy 基础操作，不调用现成层或自动求导。本轮没有验证梯度或训练效果。

## 本步代码

```python
# ==================================================
# 注意力子层的残差连接
# ==================================================

# (28, 32) + (28, 32) → (28, 32)
attention_residual = encoder_input + attention_output

print("attention_residual:", attention_residual.shape)
print("一行残差相加后的特征:", attention_residual[0].shape)
print("attention_residual 的实际设备:", attention_residual.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，选 GPU 0，型号 NVIDIA GeForce RTX 5060。在独立 PowerShell 进程执行并完成 CUDA 同步；CuPy 缓存在工作区 work/cupy_cache，Python 使用 -B。本轮未安装或调整依赖。

两个输入和输出均为 `(28,32)`、有限 float32、GPU 0，单行输出 `(32,)`。GPU 加法与 CPU float32 逐元素加法参考逐元素一致，最大绝对误差 `0.0`。两个输入执行前后均逐元素不变。局部加法正确不能证明模型学会识别数字。

主文件执行前后 SHA256 一致：`77780b389255da5718c08a75085f61e27b4832d8d6710402b066177fd6a204f9`；片段 SHA256：`6f2aa8572f09becd28ddd647edf04c741556a49e561b734ed6f613f951e52a3a`。[本步代码](#本步代码) · [真实 GPU 验证证据](2026-10-07_transformer_attention_residual_validation_v011.json)。JSON 保存逐字代码、来源、环境、设备、形状与实际输出；相对文件链接已核对。

## 相比上一版的变化

v010 得到多头注意力投影输出；v011 与 encoder_input 相加，新增 attention_residual `(28,32)`，新增参数 0，参数更新次数 0。保留历史版本，生成本版工作区索引并按约定追加项目 modeling_records/index.md。

## 未解决限制与下一步

尚未 LayerNorm、前馈、分类头、损失或梯度，当前未完成整个编码器。没有训练损失、准确率、梯度验证或 GPU 提速结论。

下一步对 attention_residual 的每一行 32 个特征手写 LayerNorm，包括均值、方差、epsilon 与可训练缩放/偏移；随后继续补全前向和手写反向。
