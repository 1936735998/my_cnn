# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v018。
阶段：十类线性分类头 GPU 前向。本轮真实执行验证，不是仅文档更新；未损失计算、反向或训练。

## 范围、来源与处理（事实）

用户输出 encoder_output `(28,32)`、image_features `(32,)` 与 GPU 0，与前版一致。本轮仅初始化分类头，产生数字 0..9 的 10 个原始分数。代码贴在聊天，不交付独立 Python 文件，不修改主文件；完整目标仍是手写 Transformer 图像分类前向、反向和更新，并用 GPU 训练。

独立进程读取执行保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前十七步，再执行本片段；不声称操作用户终端或实时内存。主文件已保存前十七步，无需补充旧片段。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素转 CuPy float32 除以 255、标签 int64。本轮没有下载、新增清洗、划分、洗牌或增强。仅验证首张训练图的局部前向，不评估测试集。分类输入来自单层编码器输出沿 28 个图片行位置平均池化的 32 维表示。

## 变量、假设与公式

f=image_features `(32,)`，W_cls `(32,10)`，b_cls `(10,)`，C=10。

```text
logits = f @ W_cls + b_cls: (10,)
logits[c] = sum_k f[k] * W_cls[k,c] + b_cls[c]
k=0..31，c=0..9，对应数字类别 0..9
```

一维向量乘二维权重收缩特征轴，保留类别轴。logits 是可正可负的原始分数，不是概率。沿用 float32 randn*0.1 与零偏置，承接当前随机状态，不重新设置种子。新增参数 32*10+10=330，当前累计 9802。输入维度、finite float32、GPU 0 前提本轮核验通过；继续沿用图片行空间序列、位置编码、无因果掩码的分类设置。

## 方法选择理由（判断）

采用单层线性分类头，将固定 32 维图片表示直接映射为十类分数，便于后续稳定 softmax、交叉熵及手写反向。未比较更深分类头、初始化尺度或性能；参数随机未训练，当前分数不能证明识别能力。使用基础 CuPy dot 与加法，不调用现成分类层或自动求导。

## 本步代码

```python
# ==================================================
# 分类头：32 个图片特征 → 10 个分类分数
# ==================================================

num_classes = 10

W_cls = cp.random.randn(d_model, num_classes).astype(cp.float32) * 0.1
b_cls = cp.zeros(num_classes, dtype=cp.float32)

# (32,) @ (32, 10) + (10,) → (10,)
logits = cp.dot(image_features, W_cls) + b_cls

print("W_cls:", W_cls.shape)
print("b_cls:", b_cls.shape)
print("logits:", logits.shape)
print("logits 的实际设备:", logits.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，使用 GPU 0，型号 NVIDIA GeForce RTX 5060。独立 PowerShell 进程执行并同步 CUDA，CuPy 缓存在工作区 work/cupy_cache，Python 使用 -B；本轮未安装或调整依赖。

输入 `(32,)`、权重 `(32,10)`、偏置/输出 `(10,)`，均为有限 float32、GPU 0。CPU float64 逐特征乘积求和加偏置参考在 rtol=1e-5、atol=1e-6 下匹配，最大绝对误差 `5.809948844e-08`。输入未改变，按参数 size 核对新增 330、累计 9802。JSON 保存 10 个实际原始分数，未把它们当作概率、准确率或训练结果。

主文件执行前后 SHA256 一致：`888be083fad798397d3fd3e0976458f32dbdb11bb291f3721dcbd62e4e5e42f8`；片段 SHA256：`b46a502ada0c1d6d2b6f1ce152ca31158f13dbe204175b99fefb923f8a4120f5`。[本步代码](#本步代码) · [真实 GPU 验证证据](2026-10-07_transformer_classifier_validation_v018.json)。JSON 包含来源、逐字代码、实际环境、设备、误差和输出；相对文件链接已核对。

## 版本变化、限制与下一步

相比 v017 的图片级特征，本版新增分类头与 logits `(10,)`，从图像输入到分类原始分数的单图前向已串联。新增参数 330、更新次数 0。保留历史版本，生成本版工作区索引并按约定追加项目 modeling_records/index.md。

尚未分类 softmax、交叉熵、预测展示、梯度或训练，没有识别准确率、梯度验证或 GPU 提速结论。下一步对 10 个分数做稳定 softmax，再计算真实标签的交叉熵，随后逐步手写反向、更新与 GPU 训练，并执行必要数值核验。
