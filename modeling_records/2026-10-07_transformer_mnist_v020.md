# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v020。
阶段：首图稳定交叉熵与预测展示 GPU 验证。本轮真实执行验证，不是仅文档更新；未反向、更新或训练。

## 范围、来源与处理（事实）

用户贴出十类概率 `(10,)`、概率和 1.0 与 GPU 0，与前版一致。本轮读取首图真实标签，计算稳定交叉熵并展示当前预测。代码直接贴在聊天，不交付独立 Python 文件，不修改主文件。目标仍为手写 Transformer 图像分类前向、反向和更新，在 GPU 训练。

独立进程读取执行保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前十九步，再执行本片段；不声称操作终端或实时内存。主文件已保存前十九步，无需补充旧片段。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素转 CuPy float32 除以 255、标签 int64。本轮没有下载、新增清洗、划分、洗牌或增强。只核验首张训练图，不评估测试集。实际核对 X 与 x_train[0] 一致，标签从 y_train[0] 读取，没有硬编码标签。

## 变量、假设与公式

z=logits `(10,)`，m=max_c z[c]，s=shifted_logits=z-m，e=exp_logits=exp(s)，p=probabilities，y=target。

```text
L = -log(p[y])
  = log(sum_c exp(s[c])) - s[y]
prediction = argmax_c p[c]
loss: CuPy float32 零维数组，shape ()
```

使用自然对数、单图无类别加权交叉熵，标签范围 0..9。log_normalizer 是平移后指数和的对数。无需向概率加 epsilon 或裁剪概率，即保持损失定义并避免先将极小概率舍入到零后取对数。本步无新增参数，累计 9802。继续沿用图片行空间序列、单层编码器、位置均值池化、无因果掩码设置。

int 读取标签/预测供索引和显示，float 只用于打印和记录；loss 变量仍保留为 GPU 数组。损失不是错误率或准确率，一张图的预测也不构成数据集评估。

## 方法选择理由（判断）

交叉熵以真实类别获得的概率度量预测分布与标签的偏差，便于后续推导 softmax 与交叉熵联合梯度。使用已保留的 shifted_logits/exp_logits 做稳定计算，避免极小概率下溢造成 log(0)。当前参数随机未训练，不据局部预测推断性能。采用 CuPy 基础操作，不调用现成损失层或自动求导。

## 本步代码

```python
# ==================================================
# 真实标签、稳定交叉熵与当前预测
# ==================================================

# 当前输入是 x_train[0]，读取对应的真实标签
target = int(y_train[0])

# 等价于 -log(probabilities[target]) 的稳定计算
log_normalizer = cp.log(cp.sum(exp_logits))
loss = log_normalizer - shifted_logits[target]

predicted_class = int(cp.argmax(probabilities))

print("真实标签:", target)
print("当前预测:", predicted_class)
print("真实类别的概率:", float(probabilities[target]))
print("交叉熵损失:", float(loss))
print("loss 的形状:", loss.shape)
print("loss 的实际设备:", loss.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，使用 GPU 0，型号 NVIDIA GeForce RTX 5060。独立 PowerShell 进程执行并同步 CUDA，CuPy 缓存在工作区 work/cupy_cache，Python 使用 -B；本轮未安装或调整依赖。

实际真实标签 5，当前预测 6，首图预测是否正确为 False，真实类别概率 `0.1392573267`，交叉熵 `1.971431732`。这些是未训练模型对首图的局部输出，不是训练效果或准确率。

loss 为有限、非负 float32、零维、GPU 0。CPU float64 的 logaddexp.reduce(z)-z[y] 参考在 rtol=1e-5、atol=1e-6 下匹配，绝对误差 `4.002862131e-08`；当前真实类概率为正，与 -log(p[y]) 交叉核对也通过。softmax 中间量输入均未改变，参数按实际 size 核对仍 9802。

新增损失数值边界样例：十类合成分数，第 0 类 1000、其余 -1000，目标为第 1 类。float32 目标概率下溢为 0，但同一稳定损失片段输出有限的 2000，符合理论值。此合成样例不是 MNIST 数据或训练结果。

主文件执行前后 SHA256 一致：`71758a123d5c695ee641d7069d42a9da892bf0de35552e52499af9c7a0dd4687`；片段 SHA256：`43d171da8286b0092652789e31327f328539dd2098a4fd654eac19e22f75e0e1`。[本步代码](#本步代码) · [真实 GPU 验证证据](2026-10-07_transformer_cross_entropy_validation_v020.json)。JSON 保存来源、逐字代码、实际环境、设备、结果、误差和边界样例，相对链接已核对。

## 版本变化、限制与下一步

相比 v019 的分类概率，本版新增真实标签、稳定交叉熵与单图预测展示，从首图输入到损失的前向计算已串联。新增参数 0、更新次数 0。保留历史记录，生成本版工作区索引并按约定追加项目 modeling_records/index.md。

尚未梯度、参数更新或训练，没有识别准确率、梯度验证或 GPU 提速结论。下一步从损失推导并手写对 logits 的联合梯度 p-one_hot(y)，再逐层反向传回分类头、平均池化、编码器和输入投影，进行必要的梯度数值核验后组织 GPU 训练。
