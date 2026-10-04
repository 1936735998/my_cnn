# 手写 MNIST DNN：第 1 步
日期：2026-10-04；版本：v001。按现有 cnn_mnist.py 的 NumPy、普通函数和分段注释风格教学。

## 学习路线
数据准备 → 784→64→10 参数初始化 → 隐藏层 ReLU 与线性输出的前向传播 → one-hot 与均方误差 → 手写反向传播 → 参数更新 → 测试集准确率。此结构及损失是教学方案，还没有训练验证；后续可以再学习 Softmax 与交叉熵。

## 本步代码
在 F:\PythonProjects\deep_learning 中自行新建 dnn_from_scratch.py，选择项目 .venv 解释器，输入下方代码。助手未新建或修改该 Python 文件。
```python
import numpy as np
from tensorflow.keras.datasets import mnist


# ==================================================
# MNIST 数据
# ==================================================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:5000]
y_train = y_train[:5000]

x_test = x_test[:1000]
y_test = y_test[:1000]


# ==================================================
# 展开图片与归一化
# ==================================================

x_train = x_train.reshape(5000, 784) / 255.0
x_test = x_test.reshape(1000, 784) / 255.0

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

print("像素范围:", x_train.min(), x_train.max())
print("前 10 个标签:", y_train[:10])
```

## 解释
每张图片有 28×28=784 个像素。reshape 保留像素顺序，改为每行一张图片；除以 255 将像素从 0～255 缩放到 0～1。标签仍为 0～9 的整数，后续才编码为 one-hot。先选前 5000 个训练样本与前 1000 个测试样本方便学习；该固定子集的结果不能直接视为完整 MNIST 基准成绩。测试集不用于参数更新。

## 已验证输出
```text
x_train: (5000, 784)
y_train: (5000,)
x_test: (1000, 784)
y_test: (1000,)
像素范围: 0.0 1.0
前 10 个标签: [5 0 4 1 9 2 1 3 1 4]
```
助手已在项目解释器中执行等价数据准备代码；尚未训练网络。TensorFlow 仅提供数据加载入口，后续网络使用 NumPy CPU 运算。未将 TensorFlow 导入日志当作 GPU 训练证据。

[建模记录](2026-10-04_dnn_mnist_v001.md) · [验证结果](2026-10-04_dnn_data_validation_v001.json)
