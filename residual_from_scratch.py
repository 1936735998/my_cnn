import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:5000].reshape(5000, 784) / 255.0
y_train = y_train[:5000]

x_test = x_test[:1000].reshape(1000, 784) / 255.0
y_test = y_test[:1000]

# ==================================================
# 参数初始化
# ==================================================

np.random.seed(42)

# 输入层：784 → 64
W1 = np.random.randn(784, 64) * 0.01
b1 = np.zeros(64)

# 残差分支第一层：64 → 64
W2 = np.random.randn(64, 64) * 0.01
b2 = np.zeros(64)

# 残差分支第二层：64 → 64
W3 = np.random.randn(64, 64) * 0.01
b3 = np.zeros(64)

# 输出层：64 → 10
W4 = np.random.randn(64, 10) * 0.01
b4 = np.zeros(10)

print("W1:", W1.shape, "b1:", b1.shape)
print("W2:", W2.shape, "b2:", b2.shape)
print("W3:", W3.shape, "b3:", b3.shape)
print("W4:", W4.shape, "b4:", b4.shape)

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

        # 前向传播：输入映射
        z1 = np.dot(x, W1) + b1
        h = relu(z1)

        # 前向传播：残差分支
        z2 = np.dot(h, W2) + b2
        u = relu(z2)
        F_h = np.dot(u, W3) + b3

        # 前向传播：相加与输出
        s = h + F_h
        r = relu(s)
        output = np.dot(r, W4) + b4

        # 目标与损失
        target = one_hot(label)
        loss = np.mean((output - target) ** 2)
        total_loss += loss

        # 输出层与相加节点反向传播
        d_output = 2 * (output - target) / 10
        d_W4 = np.outer(r, d_output)
        d_b4 = d_output

        d_r = np.dot(d_output, W4.T)
        d_s = d_r * (s > 0)

        d_F_h = d_s.copy()
        d_h_skip = d_s.copy()

        # 残差分支反向传播
        d_W3 = np.outer(u, d_F_h)
        d_b3 = d_F_h

        d_u = np.dot(d_F_h, W3.T)
        d_z2 = d_u * (z2 > 0)

        d_W2 = np.outer(h, d_z2)
        d_b2 = d_z2
        d_h_main = np.dot(d_z2, W2.T)

        # 汇总两条路径，再传回输入映射层
        d_h = d_h_skip + d_h_main
        d_z1 = d_h * (z1 > 0)
        d_W1 = np.outer(x, d_z1)
        d_b1 = d_z1

        # 全部梯度算完，再更新八组参数
        W1 = W1 - learning_rate * d_W1
        b1 = b1 - learning_rate * d_b1
        W2 = W2 - learning_rate * d_W2
        b2 = b2 - learning_rate * d_b2
        W3 = W3 - learning_rate * d_W3
        b3 = b3 - learning_rate * d_b3
        W4 = W4 - learning_rate * d_W4
        b4 = b4 - learning_rate * d_b4

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

    # 输入映射
    z1 = np.dot(x, W1) + b1
    h = relu(z1)

    # 残差分支
    z2 = np.dot(h, W2) + b2
    u = relu(z2)
    F_h = np.dot(u, W3) + b3

    # 相加与输出
    s = h + F_h
    r = relu(s)
    output = np.dot(r, W4) + b4

    prediction = np.argmax(output)

    if prediction == label:
        correct += 1

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