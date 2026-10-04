# 手写 MNIST DNN：第 8 步
日期：2026-10-04；版本：v008。

保留现有文件的导入、数据准备与参数初始化；从原来的 ReLU 注释开始到文件末尾，替换为下面代码。这样同时保留两个函数、移除单样本演示更新，并避免训练前额外学习一次。助手没有修改用户 Python 文件。

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
```

外层循环控制 epoch，内层循环逐张学习 5000 个训练样本；一轮就是每张图片学习一次，5 轮共 25000 次参数更新。每张图片必须重新执行前向、目标、损失、反向和更新。参数初始化只放在训练循环之前；total_loss 每轮开始清零。打印的 loss 是本轮每张图片在更新之前的损失平均，期间参数持续变化，因此不是固定轮末参数在所有训练样本上的重新评价。此学习方式是逐样本随机梯度下降的基础形式；当前按固定顺序遍历，没有洗牌。

助手已在项目解释器中独立执行同样逻辑（直接读取同一本地 Keras 缓存，而不通过 TensorFlow 加载入口），5 轮损失：0.0717452499336227、0.043550414432283316、0.0321534218824488、0.025845543993940935、0.022735846683943576。参数与损失有限，使用 NumPy CPU。训练损失下降不能代替测试准确率；本步未评价测试集。

[本步建模记录](2026-10-04_dnn_mnist_v008.md) · [实际训练结果](2026-10-04_dnn_training_validation_v008.json) · [验证运行的训练后参数](2026-10-04_dnn_trained_parameters_v008.npz)
