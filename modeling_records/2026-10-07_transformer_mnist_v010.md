# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v010。
阶段：多头输出投影局部前向 GPU 验证。本轮真实运行验证，不是仅文档更新；未训练。

## 范围与证据来源

用户输出拼接中间数组 `(28,4,8)`、concatenated_heads `(28,32)`、单行 `(32,)`、GPU 0，与前一步一致。本轮只初始化输出投影并计算多头注意力输出。代码贴在聊天，不交付单独 Python 代码文件，不修改主文件。后续仍逐步手写反向、参数更新并在 GPU 训练。

独立进程读取执行保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前九步，再运行当前片段；不声称操作用户编辑器终端或实时内存。

## 输入来源与处理（事实）

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000 图、测试前 1000 图；像素转 CuPy float32 除以 255，标签 int64。本轮没有新增下载、清洗、划分、洗牌或增强。验证只用首张训练图的局部前向，不评估测试集。

拼接输入来源为图片行输入投影、固定位置编码、随机 Q/K/V 投影、多头拆分、缩放点积、稳定 softmax、加权 V 和按位置拼接。用户主文件已保存前九步；不用补充旧片段。

## 变量、假设与公式

T=28，d_model=32，H=4，d_head=8。C 为 concatenated_heads `(28,32)`，W_O `(32,32)`，b_O `(32,)`。

```text
Y = C @ W_O + b_O: (28,32)
Y[i,k] = sum_j C[i,j] * W_O[j,k] + b_O[k]
```

b_O 沿 28 个位置广播，共享同一组输出偏置。W_O 对每个位置的 32 个拼接特征做线性组合，本身不额外混合不同位置；位置间汇总已发生在注意力加权 V 步骤。新增可训练参数 32*32+32=1056，累计输入投影、Q/K/V 与输出投影参数 5152。

沿用 float32 标准正态随机数乘以 0.1、零偏置初始化，承接前面的随机状态，不在本片段重置种子。输入维度及有限数值假设本轮实际核验通过。继续使用图片行空间序列、无因果掩码设置。

## 方法选择理由（判断）

输出投影提供学习组合不同头特征的参数。沿用已有初始化方式便于教学连贯，当前没有比较不同初始化尺度或证明该尺度最优。使用基础二维 cp.dot 与偏置广播，不调用现成注意力层或自动求导。当前参数随机未训练，不能声称已学到有效的头间组合。

## 本步代码

```python
# ==================================================
# 多头注意力的输出投影
# ==================================================

# 沿用前面的随机数状态，这里不用重新设置种子
W_O = cp.random.randn(d_model, d_model).astype(cp.float32) * 0.1
b_O = cp.zeros(d_model, dtype=cp.float32)

# (28, 32) @ (32, 32) + (32,) → (28, 32)
attention_output = cp.dot(concatenated_heads, W_O) + b_O

print("W_O:", W_O.shape)
print("b_O:", b_O.shape)
print("attention_output:", attention_output.shape)
print("W_O 的实际设备:", W_O.device)
print("attention_output 的实际设备:", attention_output.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，选 GPU 0，型号 NVIDIA GeForce RTX 5060。在独立 PowerShell 进程执行、CUDA 同步完成。CuPy 缓存在工作区 work/cupy_cache，使用 -B；本轮未安装或调整依赖。

W_O `(32,32)`、b_O `(32,)`、attention_output `(28,32)`；输入和新参数、输出均为有限 float32、GPU 0。输入拼接结果未改变。CPU float64 矩阵乘法与偏置参考在 rtol=1e-5、atol=1e-6 下匹配，最大绝对误差 `1.168965079e-07`；另对第 3 行第 7 个输出特征逐项加权求和核验通过。实际按数组 size 累计得到新增参数 1056、总参数 5152。以上是局部计算核验，不是识别性能或训练验证。

主文件执行前后 SHA256 一致：`d0b73666a1f2a834c3f57b860d43844ebfd56780b362ec7bac902c17a525910f`；片段 SHA256：`9daeb36745388341b77dd5f9134ba087fea11bbf4d50291a13635bf9e8d26709`。[本步代码](#本步代码) · [真实 GPU 验证证据](2026-10-07_transformer_output_projection_validation_v010.json)。JSON 保存逐字代码、来源、环境、设备、误差与实际输出；相对文件链接已核对。

## 相比上一版的变化

v009 只拼接四头；v010 新增 W_O、b_O 和输出投影，产生同样 `(28,32)` 形状的新表示。参数增加 1056，更新次数仍为 0。保留历史版本，生成工作区本版索引并按约定追加项目 modeling_records/index.md。

## 未解决限制与下一步

尚未残差连接、LayerNorm、前馈、分类头、损失或梯度；没有训练损失、准确率或 GPU 提速结论。只验证首图局部前向不能证明完整模型与反向正确。

下一步将多头注意力输出与 encoder_input 相加形成残差连接，随后逐步手写 LayerNorm、前馈、分类与训练；后续梯度需要必要的数值核验。
