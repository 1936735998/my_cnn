# 手写 LSTM 第 15 步：一张图片的完整训练函数
日期：2026-10-06；版本：v015。

接续[第14步](lstm_from_scratch_lesson_14.md)。用户已保存并运行backward，十组参数梯度中的打印形状与b_g前5项吻合。

将以下代码追加到F:\PythonProjects\deep_learning\lstm_from_scratch.py末尾，由用户继续逐步编写。函数完成一张图片的前向、MSE、完整28步反向与一次SGD。助手未编辑主源。

```python
# ==================================================
# 将一张图片的完整训练过程整理成函数
# ==================================================

def train_one(sequence, label, learning_rate):
    global W_f, b_f, W_i, b_i, W_g, b_g
    global W_o, b_o, W_y, b_y

    # 1. 前向计算
    output, cache = forward(sequence)
    target = one_hot(label)
    loss = np.mean((output - target) ** 2)

    # 2. 算完全部参数梯度
    gradients = backward(output, target, cache)

    # 3. 更新参数
    W_f = W_f - learning_rate * gradients["W_f"]
    b_f = b_f - learning_rate * gradients["b_f"]

    W_i = W_i - learning_rate * gradients["W_i"]
    b_i = b_i - learning_rate * gradients["b_i"]

    W_g = W_g - learning_rate * gradients["W_g"]
    b_g = b_g - learning_rate * gradients["b_g"]

    W_o = W_o - learning_rate * gradients["W_o"]
    b_o = b_o - learning_rate * gradients["b_o"]

    W_y = W_y - learning_rate * gradients["W_y"]
    b_y = b_y - learning_rate * gradients["b_y"]

    # 返回这次更新前的损失
    return loss


loss_before_train = train_one(x_train[0], y_train[0], 0.01)

# 用更新后的参数重新前向计算
output_after_train, _ = forward(x_train[0])
target_after_train = one_hot(y_train[0])
loss_after_train = np.mean((output_after_train - target_after_train) ** 2)

print("本次更新前 loss:", loss_before_train)
print("本次更新后 loss:", loss_after_train)
print("loss 减少量:", loss_before_train - loss_after_train)
```

## 关键理解
1. global声明的是函数要重新赋值的外部权重/偏置。forward/backward仅读取这些参数，所以无需global；train_one改变绑定，须声明，否则Python会将这些名称视为局部变量。
2. 每次调用forward重置该图片的h/c，重新生成output/cache；backward使用同次结果，先算完十组梯度，再按θ←θ−η∂L/∂θ更新参数。学习率是函数局部传入值。
3. return loss返回本次更新前的MSE。更新后的误差须重新前向计算；output_after_train, _中下划线接收这次暂不用的缓存，缓存仍会生成。
4. 从头运行当前完整脚本时，第12步已有一次更新，本步再更新一次，共2次。这里沿用当前权重，未在train_one重新初始化参数。
5. 原output_check/cache_check/gradients_check在本步更新后对应旧权重，后续训练须重新前向、求导。

## 实际验证与预期输出
项目Python3.13.15、NumPy2.5.3，NumPy float64 CPU执行。独立命名空间重放已有1次SGD恢复起点，验证函数额外1次更新；为核查全局赋值与顺序，先观测函数调用，再将隔离权重恢复至同一起点执行原样教学片段，两个验证调用不连续累加。

所有十组全局权重符合更新公式，最大逐元素误差0；前向→反向调用顺序正确，求导前参数未变，更新后参数有限。首图标签5，28行输入；新前向h/c从零开始，cache长度28。

```text
本次更新前 loss: 0.0881374096811086
本次更新后 loss: 0.08695209627476536
loss 减少量: 0.0011853134063432441
```

这只是首张训练图的一次更新检查，不能推出识别准确率或泛化性能。下一步把train_one放入遍历训练图片的循环，统计epoch平均损失。

[建模记录](2026-10-06_lstm_mnist_v015.md) · [验证JSON](2026-10-06_lstm_single_train_function_validation_v015.json)
