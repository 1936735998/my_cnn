# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v014。
阶段：逐位置前馈第二层局部前向 GPU 验证。本轮真实执行验证，不是仅文档更新；未训练。

## 范围与输入来源（事实）

用户输出前馈第一层参数 `(32,64)`/`(64,)`，线性和 ReLU 结果 `(28,64)`，GPU 0，与前版一致。本轮仅初始化第二层并将 64 维映射回 32 维，代码直接贴在聊天，不交付单独 Python 文件，不修改主文件。整体目标仍为手写 Transformer 前向、反向和更新，在 GPU 训练。

独立进程读取执行保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前十三步，再执行本片段；不声称操作用户终端或实时内存。主文件已保存前十三步，无需补充旧片段。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000 图、测试前 1000 图；像素转 CuPy float32 除以 255，标签 int64。本轮没有下载、新增清洗、划分、洗牌或增强。只验证首张训练图局部前向，不评估测试集。输入 ff_hidden 来自注意力后的 LayerNorm、32→64 第一层和 ReLU。

## 变量、假设与公式

T=28，d_model=32，d_ff=64；H=ff_hidden `(28,64)`，W_ff2 `(64,32)`，b_ff2 `(32,)`。

```text
F = H @ W_ff2 + b_ff2: (28,32)
FFN(A) = ReLU(A @ W_ff1 + b_ff1) @ W_ff2 + b_ff2
```

各行使用同一组参数和偏置、独立变换特征；位置间信息交换由前面的注意力完成。本步第二层为线性映射，输出可以为负，是特征表示而非概率。新增参数 64*32+32=2080，当前累计 9408。沿用 float32 randn*0.1、零偏置，承接现有随机状态，不重新设置种子。

输入形状、finite float32、GPU 0 假设本轮核验通过；输出与前馈子层输入 attention_norm 同为 `(28,32)`，为下一步残差相加准备。继续沿用图片行空间序列、无因果掩码设置。

## 方法选择理由（判断）

前一层扩展到 64 维并引入 ReLU，第二层把隐藏特征组合回模型宽度 32，使该子层具有非线性变换且兼容残差维度。采用小型教学模型的 32→64→32 配置，未比较其他宽度或初始化的性能。使用基础 CuPy dot 和偏置广播，不使用现成前馈层或自动求导。

## 本步代码

```python
# ==================================================
# 前馈网络第二层：64 → 32
# ==================================================

W_ff2 = cp.random.randn(d_ff, d_model).astype(cp.float32) * 0.1
b_ff2 = cp.zeros(d_model, dtype=cp.float32)

# 对全部 28 行使用同一组参数
# (28, 64) @ (64, 32) + (32,) → (28, 32)
ff_output = cp.dot(ff_hidden, W_ff2) + b_ff2

print("W_ff2:", W_ff2.shape)
print("b_ff2:", b_ff2.shape)
print("ff_output:", ff_output.shape)
print("ff_output 的实际设备:", ff_output.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，使用 GPU 0，型号 NVIDIA GeForce RTX 5060。独立 PowerShell 进程执行并完成 CUDA 同步；CuPy 缓存在工作区 work/cupy_cache，Python 使用 -B，本轮未安装或调整依赖。

参数实际 `(64,32)`/`(32,)`，输出 `(28,32)`；输入、参数和输出均为有限 float32、GPU 0。输出形状与 attention_norm 一致，输入未改变。CPU float64 参考在 rtol=1e-5、atol=2e-6 下匹配，最大绝对误差 `9.459899841e-08`。实际输出范围 `-0.82048070`～`0.95998847`，这些数值是局部前向特征，不是识别性能。按实际参数 size 核对新增 2080、累计 9408。

主文件执行前后 SHA256 一致：`869fce28d6a6f56dd9326078f74c156a98dcb293a54e071ed0fe4224420b2c5f`；片段 SHA256：`515223219ae98d0ae6891ad33d7228d4af286a679e1f568a7f04078fea7cba1d`。[本步代码](#本步代码) · [真实 GPU 验证证据](2026-10-07_transformer_ffn_output_validation_v014.json)。JSON 保存逐字代码、来源、设备、形状、实际误差和输出；相对链接已核对。

## 相比上一版的变化

v013 产生 ReLU 隐藏表示 `(28,64)`；v014 增加第二层、输出 `(28,32)`，完成前馈网络两层前向。新增参数 2080，更新次数仍为 0。保留历史记录，生成本版工作区索引并按约定追加项目 modeling_records/index.md。

## 未解决限制与下一步

尚未前馈子层残差和 LayerNorm、分类头、损失、梯度或更新；尚未完成整个编码器，没有训练损失、准确率或 GPU 提速结论。

下一步将 ff_output 与前馈子层输入 attention_norm 相加，然后对结果逐行 LayerNorm；之后补全分类与手写反向，执行必要的梯度数值核验。
