# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v021。
阶段：softmax 与交叉熵对 logits 的手写联合梯度，GPU 计算与 CPU 中心差分验证。本轮真实执行验证，不是仅文档更新；未完整反向、参数更新或训练。

## 范围、来源与处理（事实）

用户贴出真实标签 5、当前预测 6、真实类概率 0.13925732672214508、损失 1.9714317321777344、损失零维与 GPU 0，符合前版结果。本轮只计算损失对十个原始分类分数的梯度。代码直接贴在聊天，不交付独立 Python 文件，不修改主文件。完整目标仍为逐层手写反向、更新并用 GPU 训练 Transformer 图像分类模型。

独立进程读取执行保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前二十步，再执行本片段；不声称操作用户终端或实时内存。主文件已保存前二十步，无需补充旧片段。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素转 CuPy float32 除以 255、标签 int64。本轮未下载、新增清洗、划分、洗牌或增强，仅核验首张训练图当前 logits 与真实标签。输入来自前向分类 softmax 和单图稳定交叉熵，不评估测试集。

## 变量、假设与推导

单图、无类别权重损失，十类原始分数 z、概率 p=softmax(z)、真实标签 y。

```text
L = log(sum_c exp(z[c])) - z[y]
dL/dz[c] = p[c] - 1[c=y]
d_logits: (10,)
```

因为对数指数和的导数是 p[c]，减去真实类分数的导数是真实类指示项，得到联合梯度。当前 loss 是单图损失，不额外平均类别或样本。直接复制概率后将真实类对应分量减 1；保存原概率供后续核验和反向使用。本步无新增参数，累计 9802。

梯度和数学上为 0，浮点允许误差。真实类梯度非正，其他类非负；其符号解释是在其他 logits 固定时，该分数对损失的局部变化方向。更新网络共享参数后，不据此保证每个 logit 单调改变。本轮不更新 logits 或网络参数。

沿用图片行空间序列、位置池化、十类分类、无因果掩码设置；输入概率与标签来源于已保存的单图前向。CuPy 梯度为 float32，差分核验使用 CPU float64 防止小扰动被 float32 精度掩盖。

## 方法选择理由（判断）

把 softmax 与交叉熵联合求导，使表达式简洁并保持稳定损失对应的准确梯度。先核验最靠近损失的一个反向节点，再逐层传回网络，便于定位梯度问题。使用 CuPy 基础复制与索引减法，没有自动求导或现成网络层。

## 本步代码

```python
# ==================================================
# 反向第一步：损失对 logits 的梯度
# ==================================================

# 单独保存梯度，保留原来的 probabilities
d_logits = probabilities.copy()
d_logits[target] -= cp.float32(1.0)

print("d_logits:", d_logits.shape)
print("10 个分数的梯度:", d_logits)
print("梯度之和:", float(cp.sum(d_logits)))
print("d_logits 的实际设备:", d_logits.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，使用 GPU 0，型号 NVIDIA GeForce RTX 5060。独立 PowerShell 进程执行并同步 CUDA，CuPy 缓存在工作区 work/cupy_cache，Python 使用 -B。本轮未安装或调整依赖。

梯度 `(10,)`、有限 float32、GPU 0，梯度和 `-2.980232239e-08`，通过绝对容差 1e-6。实际目标 5 的梯度非正，其他类非负。原 probabilities、logits 与 loss 未改变，梯度数组与概率数组为独立存储。

对 GPU 当前 logits 的固定数值，CPU float64 用稳定损失 logaddexp.reduce(z)-z[y]，分别扰动每个分类分数 ±1e-5，中心差分检查全部 10 个分量。rtol=1e-5、atol=1e-7 下匹配，最大绝对误差 `1.722504804e-08`，相对 L2 误差 `2.956685433e-08`。本核验只覆盖 dL/dlogits，尚未覆盖权重或整个网络的梯度。实际参数 size 核对累计仍 9802。

主文件执行前后 SHA256 一致：`d5790c81f47ab67ebf4722db830a503071ff117a21be552cc4c89a1ae5ed391a`；片段 SHA256：`6a9b54c0b76962459a2362dd980e8f8963a2c317a6eb3f5fb214a2fe7d4d4b08`。[本步代码](#本步代码) · [真实 GPU 与差分证据](2026-10-07_transformer_logits_gradient_validation_v021.json)。JSON 保存逐字代码、来源、设备、全部梯度和差分值、误差及真实输出；相对链接已核对。

## 版本变化、限制与下一步

v020 只计算损失和预测；本版新增 d_logits，开始手写反向传播。新增参数 0、更新次数 0，保留历史记录，生成本版工作区索引并按约定追加项目 modeling_records/index.md。

尚未分类头参数梯度、池化反向、编码器与输入投影反向或训练，没有准确率、训练收益或 GPU 提速结论。下一步手写分类头的 dW_cls、db_cls 和 d_image_features，再逐层反向，并进行对应的必要数值核验。
