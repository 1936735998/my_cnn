# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v017。
阶段：平均池化得到图片级表示的 GPU 局部前向验证。本轮真实执行验证，不是仅文档更新；未分类或训练。

## 范围、来源与处理（事实）

用户输出 encoder_output `(28,32)`、第 0 行均值 1.4901161193847656e-08、方差 0.9999919533729553、GPU 0，与前版一致。本轮仅沿行位置平均汇总编码器输出。代码直接贴在聊天，不交付独立 Python 文件，不修改主文件；目标仍为手写完整 Transformer 图像分类前向、反向和更新，并用 GPU 训练。

独立进程读取执行保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前十六步，再执行片段，不声称操作终端或实时内存。主文件已保存前十六步，无需补充旧片段。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素 float32 除以 255、标签 int64。本轮没有下载、新增清洗、划分、洗牌或增强，仅验证首张训练图局部前向，不评估测试集。输入来源为简化单层编码器块：多头注意力、残差后 LayerNorm、前馈、残差后 LayerNorm。

## 变量、假设与公式

E=encoder_output `(28,32)`，轴 0 是图片行位置，轴 1 是表示特征。

```text
image_features[k] = sum_i encoder_output[i,k] / 28
i=0..27，k=0..31
image_features: (32,)
```

沿位置轴 0 分别汇总每个特征；不同于 LayerNorm 沿每行的特征轴归一化。每行特征均值接近 0 不意味着每个特征跨位置平均为 0；不据此声称池化向量全零。操作不新增参数，当前累计 9472。输入维度、finite float32、GPU 0 前提本轮核验。继续沿用图片行空间序列、含位置编码、无因果掩码的分类设置。

## 方法选择理由（判断）

选择均值池化作为小型教学模型的图片级汇总方式，容易手写前向与后续梯度，并将 28 行表示变成固定 32 维向量供分类。未比较其他汇总方式、未证明该选择最优；这里平均的是编码器处理后的表示，而非原始图片像素。

## 本步代码

```python
# ==================================================
# 平均池化：汇总整张图片的特征
# ==================================================

# 沿 28 个行位置取平均，保留 32 个特征
# (28, 32) → (32,)
image_features = cp.mean(encoder_output, axis=0)

print("encoder_output:", encoder_output.shape)
print("image_features:", image_features.shape)
print("image_features 的实际设备:", image_features.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，使用 GPU 0，型号 NVIDIA GeForce RTX 5060。独立 PowerShell 进程执行并同步 CUDA，CuPy 缓存在工作区 work/cupy_cache，Python 使用 -B。本轮未安装或调整依赖。

实际 `(28,32)` 输入转为 `(32,)` 输出，均为有限 float32、GPU 0，输入未改变。CPU float64 用位置求和除以 28 独立核对，rtol=1e-5、atol=1e-6 下匹配，最大绝对误差 `6.795328644e-08`。每个输出分量位于对应特征跨 28 个位置的最小值与最大值之间，允许尺度相关浮点容差。按实际参数 size 核对总参数仍 9472；JSON 保存全部 32 个实际池化值，但这些是未训练特征，不是识别结果。

主文件执行前后 SHA256 一致：`1fec26fc552d2c698ad785fe57142319dafbf0a0506899ce9b8e1b9a1ab3ae73`；片段 SHA256：`e8f7d13fa62e6884058bd5fbea27d59fa2928e1a82f85c42f830ee496587cfbc`。[本步代码](#本步代码) · [真实 GPU 验证证据](2026-10-07_transformer_mean_pool_validation_v017.json)。证据包括逐字代码、来源、设备、误差与实际输出，相对文件链接已核对。

## 版本变化、限制与下一步

相比 v016 的每个位置编码器表示，本版新增图片级 image_features `(32,)`。新增参数 0、更新次数 0，保留历史记录，生成本版工作区索引并按约定追加项目 modeling_records/index.md。

尚未分类头、损失、梯度或训练，没有识别准确率或 GPU 提速结论。均值池化压缩位置表示，未单独评估其信息损失。下一步将 32 维图片表示线性映射为数字 0..9 的 10 个分类分数，再添加稳定概率/损失计算，继续手写反向与更新并做必要数值核验。
