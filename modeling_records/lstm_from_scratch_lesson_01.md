# 手写 MNIST LSTM：第 1 步
日期：2026-10-05；版本：v001。

沿用此前教学标准：MNIST、NumPy 基础运算、普通函数、手写前向/梯度/更新、分段注释、用户在 VS Code 逐段输入运行。tensorflow.keras.datasets.mnist 仅用于数据入口；网络计算计划使用 NumPy CPU。助手没有创建或修改用户主 Python 文件。

## 本步代码
在 F:\PythonProjects\deep_learning 中新建 lstm_from_scratch.py，输入：

```python
import numpy as np
from tensorflow.keras.datasets import mnist


# ==================================================
# MNIST 数据
# ==================================================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:5000] / 255.0
y_train = y_train[:5000]

x_test = x_test[:1000] / 255.0
y_test = y_test[:1000]


# ==================================================
# 检查序列形状
# ==================================================

sequence = x_train[0]
label = y_train[0]

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

print("一张图片的序列形状:", sequence.shape)
print("第一个时间步的输入形状:", sequence[0].shape)
print("像素范围:", x_train.min(), x_train.max())
print("第一张图片的标签:", label)
```

## 输入解释
每张 28×28 图片按从上到下的行顺序构成序列，共 28 个时间步，每步 28 个像素；这是人为采用的空间扫描序列表示，MNIST 本身是图片数据。保留 (样本数,28,28)，本步不展开为 784 维。sequence[t] 对应第 t 行，label 是整张图片的类别，不是每一行的类别。

LSTM 逐步读取输入，并传递隐藏状态与细胞状态，其门控和递推公式已核对 [PyTorch 官方 LSTM 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html)；文档仅作为数学公式参考，教学实现不会调用其 LSTM 层。

## 实际验证
助手通过项目 .venv/Scripts/python.exe，直接读取同一本地 Keras 缓存进行等价数据准备检查。训练输入 (5000,28,28)，标签 (5000,)；测试输入 (1000,28,28)，标签 (1000,)；首图 (28,28)，首时间步 (28,)；像素范围 0～1；首图标签 5。输入有限，实际为 CPU NumPy 数组。未初始化参数、执行 LSTM 前向或训练。

下一步讲解并初始化遗忘门、输入门、候选记忆、输出门及两种状态，之后逐步实现一个时间步、28 步前向、损失、时间反向传播、梯度检查、训练与测试。

[本步建模记录](2026-10-05_lstm_mnist_v001.md) · [实际验证结果](2026-10-05_lstm_data_validation_v001.json)
