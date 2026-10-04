# 手写 MNIST 残差网络：第 8 步
日期：2026-10-04；版本：v008。

保留用户主文件中的导入、数据准备、八组参数初始化及形状打印。从原来的 ReLU 注释段开始到文件末尾，替换为下面代码；其中重新给出 relu 与 one_hot，移除单样本演示及其额外更新。参数初始化只执行一次，位于训练循环外。助手没有修改用户主 Python 文件。

```python
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
```

外层为 5 个 epoch，内层为每轮 5000 个训练样本，共 25000 次学习更新。每张图片重新做全部前向、目标、损失、反向和八组更新。total_loss 每轮清零；打印平均值是训练过程中各样本更新前损失的平均，参数在过程中不断变化，不是固定轮末参数的全样本重新评价。

实际独立执行使用同一本地 Keras 缓存，直接通过 np.load 读取，与用户 mnist.load_data 的来源一致。5 轮损失为 0.07203154156436087、0.043807667717091936、0.03240254914629507、0.026596944269749123、0.022964733526452744；参数与损失有限。未评价测试集，不能据此宣称比先前 DNN 更准确。八组训练后参数已保存，后续验证可复用而不重训。

[本步建模记录](2026-10-04_residual_mnist_v008.md) · [实际训练结果](2026-10-04_residual_training_validation_v008.json) · [训练后参数](2026-10-04_residual_trained_parameters_v008.npz)
