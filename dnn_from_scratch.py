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


# ==================================================
# 参数初始化
# ==================================================

np.random.seed(42)

W1 = np.random.randn(784, 64) * 0.01
b1 = np.zeros(64)

W2 = np.random.randn(64, 10) * 0.01
b2 = np.zeros(10)

print("W1:", W1.shape)
print("b1:", b1.shape)
print("W2:", W2.shape)
print("b2:", b2.shape)
# ==================================================
# ReLU
# ==================================================

# ==================================================
# 基础函数
# ==================================================

def relu(x):
    return np.maximum(0, x)


def one_hot(label):
    target = np.zeros(10)
    target[label] = 1
    return target


# ==================================================
# 训练
# ==================================================

import sys
import platform

print("Python:", sys.version.split()[0])
print("计算框架: NumPy", np.__version__)
print("参数计算设备: CPU（NumPy 数组）")
print("CPU 标识:", platform.processor())

learning_rate = 0.01
epochs = 5

for epoch in range(epochs):

    total_loss = 0

    for i in range(len(x_train)):

        x = x_train[i]
        label = y_train[i]

        # 前向传播
        z1 = np.dot(x, W1) + b1
        a1 = relu(z1)
        output = np.dot(a1, W2) + b2

        # 目标与损失
        target = one_hot(label)
        loss = np.mean((output - target) ** 2)
        total_loss += loss

        # 输出层反向传播
        d_output = 2 * (output - target) / 10
        d_W2 = np.outer(a1, d_output)
        d_b2 = d_output

        # 隐藏层反向传播
        d_a1 = np.dot(d_output, W2.T)
        d_z1 = d_a1 * (z1 > 0)
        d_W1 = np.outer(x, d_z1)
        d_b1 = d_z1

        # 全部梯度算完，再更新参数
        W1 = W1 - learning_rate * d_W1
        b1 = b1 - learning_rate * d_b1
        W2 = W2 - learning_rate * d_W2
        b2 = b2 - learning_rate * d_b2

    print(
        "epoch:", epoch + 1,
        "loss:", total_loss / len(x_train)
    )

# ==================================================
# 测试集评价
# ==================================================

correct = 0

for i in range(len(x_test)):

    x = x_test[i]
    label = y_test[i]

    # 只做前向传播
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)
    output = np.dot(a1, W2) + b2

    prediction = np.argmax(output)

    if prediction == label:
        correct += 1

    # 展示前 10 张测试图片的预测
    if i < 10:
        print(
            "测试图片:", i,
            "真实标签:", label,
            "预测类别:", prediction
        )

accuracy = correct / len(x_test)

print("测试图片数:", len(x_test))
print("预测正确数:", correct)
print(f"测试准确率: {accuracy:.2%}")