# 手写 LSTM 第 13 步：前向函数与独立样本缓存

日期：2026-10-06；版本：v013。
前置：已完成首图一次SGD更新；目前全局参数是更新后的值。本步整理前向函数，不新增学习步骤。

## 追加代码
```python
# ==================================================
# 将前向计算整理成函数
# ==================================================

def forward(sequence):
    h = np.zeros(hidden_size)
    c = np.zeros(hidden_size)
    cache = []

    for t in range(len(sequence)):
        x = sequence[t]

        h_prev = h
        c_prev = c
        combined = np.concatenate([x, h_prev])

        f = sigmoid(np.dot(combined, W_f) + b_f)
        i_gate = sigmoid(np.dot(combined, W_i) + b_i)
        g = np.tanh(np.dot(combined, W_g) + b_g)
        o = sigmoid(np.dot(combined, W_o) + b_o)

        c = f * c_prev + i_gate * g
        h = o * np.tanh(c)

        cache.append({
            "combined": combined.copy(),
            "c_prev": c_prev.copy(),
            "f": f.copy(),
            "i_gate": i_gate.copy(),
            "g": g.copy(),
            "o": o.copy(),
            "c": c.copy(),
            "h": h.copy()
        })

    output = np.dot(h, W_y) + b_y
    return output, cache


output_check, cache_check = forward(x_train[0])
target_check = one_hot(y_train[0])
loss_check = np.mean((output_check - target_check) ** 2)

print("函数输出形状:", output_check.shape)
print("函数缓存步数:", len(cache_check))
print("函数计算的 loss:", loss_check)
```

## 调用与返回
forward 接收一张图片的(28,28)序列，读取当前全局W/b。函数内部每次重新建立零h/c和新cache，逐行递推，输出最后h的十类线性分数。
return output,cache 返回两个对象，output_check/cache_check分别接收。loss在函数外用真实label构造one-hot计算；forward本身不读取标签、不求梯度、不更新参数。
内部h/c/cache都是本次调用的局部变量，不修改主脚本旧状态或旧缓存。新返回缓存对应当前更新后参数，下一步反向必须使用output_check和cache_check配套输入，不能使用旧全局cache。

## 实际预期
```text
函数输出形状: (10,)
函数缓存步数: 28
函数计算的 loss: 0.0881374096811086
```
与刚才loss_after相等，因为函数重现了同图/同参数的前向；没有额外更新。
已实际检查首图、第二训练图、首图三次调用：两次首图输出逐元素一致，每次28步、首步旧h/c全零、缓存实例和数组互相独立，输出及各字段形状/有限值通过。主脚本旧h/cache与参数不变。
Python3.13.15、NumPy2.5.3、CPU float64；独立验证重放前版已有单步更新以建立同样参数状态，之后本课无新增优化更新。未编辑用户主文件，也未训练全数据或评价测试集。

[建模记录](2026-10-06_lstm_mnist_v013.md) · [真实函数验证JSON](2026-10-06_lstm_forward_function_validation_v013.json) · [上一课](lstm_from_scratch_lesson_12.md)。

下一步将完整BPTT整理成backward函数，使用同次前向返回的输出和缓存计算梯度，再组成训练循环。
