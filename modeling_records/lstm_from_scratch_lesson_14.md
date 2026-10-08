# 手写 LSTM 第 14 步：把完整反向传播整理成函数
日期：2026-10-06；版本：v014。

接续[第13步](lstm_from_scratch_lesson_13.md)。用户已运行前向函数，output_check=(10,)、cache_check有28步，loss_check=0.0881374096811086。

本步把已经写过的分类层梯度、28步BPTT装入自定义函数，返回十组参数梯度，不进行参数更新。请追加到 F:\PythonProjects\deep_learning\lstm_from_scratch.py 的前向函数检查后；主Python文件由用户自己逐步编写，助手未修改。

```python
# ==================================================
# 将完整反向传播整理成函数
# ==================================================

def backward(output, target, cache):
    # 分类层的梯度
    d_output = 2 * (output - target) / output_size
    h_final = cache[-1]["h"]

    gradients = {
        "W_f": np.zeros_like(W_f),
        "b_f": np.zeros_like(b_f),
        "W_i": np.zeros_like(W_i),
        "b_i": np.zeros_like(b_i),
        "W_g": np.zeros_like(W_g),
        "b_g": np.zeros_like(b_g),
        "W_o": np.zeros_like(W_o),
        "b_o": np.zeros_like(b_o),
        "W_y": np.outer(h_final, d_output),
        "b_y": d_output.copy()
    }

    # 从最后一个时间步开始
    d_h_next = np.dot(d_output, W_y.T)
    d_c_next = np.zeros(hidden_size)

    for t in range(len(cache) - 1, -1, -1):
        step = cache[t]

        combined_t = step["combined"]
        c_prev_t = step["c_prev"]
        f_t = step["f"]
        i_t = step["i_gate"]
        g_t = step["g"]
        o_t = step["o"]
        c_t = step["c"]

        tanh_c = np.tanh(c_t)

        d_o = d_h_next * tanh_c
        d_c = d_c_next + d_h_next * o_t * (1 - tanh_c ** 2)

        d_f = d_c * c_prev_t
        d_i = d_c * g_t
        d_g = d_c * i_t

        d_z_f = d_f * f_t * (1 - f_t)
        d_z_i = d_i * i_t * (1 - i_t)
        d_z_g = d_g * (1 - g_t ** 2)
        d_z_o = d_o * o_t * (1 - o_t)

        # 累加各时间步对共享参数的贡献
        gradients["W_f"] += np.outer(combined_t, d_z_f)
        gradients["b_f"] += d_z_f

        gradients["W_i"] += np.outer(combined_t, d_z_i)
        gradients["b_i"] += d_z_i

        gradients["W_g"] += np.outer(combined_t, d_z_g)
        gradients["b_g"] += d_z_g

        gradients["W_o"] += np.outer(combined_t, d_z_o)
        gradients["b_o"] += d_z_o

        d_combined = (
            np.dot(d_z_f, W_f.T)
            + np.dot(d_z_i, W_i.T)
            + np.dot(d_z_g, W_g.T)
            + np.dot(d_z_o, W_o.T)
        )

        d_h_next = d_combined[input_size:]
        d_c_next = d_c * f_t

    return gradients


gradients_check = backward(output_check, target_check, cache_check)

print("W_y 梯度:", gradients_check["W_y"].shape)
print("W_f 梯度:", gradients_check["W_f"].shape)
print("b_g 梯度:", gradients_check["b_g"].shape)
print("b_g 梯度前 5 个值:", gradients_check["b_g"][:5])
```

## 如何读这个函数

- output、target、cache都是这次调用传入的局部变量，使用同一次前向结果。分类层用cache[-1]["h"]获取最后一步隐藏状态，不读取此前演示留下的全局h。
- gradients是字典，存4组循环层W/b与1组分类层W/b，共10个数组。gradients["W_f"]就是这张图片对W_f的总梯度，与W_f同形。
- 每次调用，四组循环层梯度从零开始；分类层只计算一次。28步共享相同循环层参数，所以每一步的贡献用+=累加。
- d_h_next、d_c_next把后一时间步传来的梯度带到前一时间步；反向公式与前面的完整BPTT相同。
- return gradients把字典交给调用处。此函数读取当前全局权重以计算梯度，不修改权重；读全局变量不需要global声明。
- 前向→反向期间参数须保持不变。当前脚本原来的全局output/cache与旧d_*来自更新前，本次调用使用新output_check/target_check/cache_check，不能混用。

## 本轮实际执行

在项目Python3.13.15、NumPy2.5.3环境，CPU float64，隔离命名空间重放已保存前置代码的那一次SGD，再验证函数；函数阶段新增参数更新0次。十组梯度与对当前参数/新缓存重新执行的原完整BPTT逐元素一致；72处完整序列中心差分最大绝对误差为2.2317063670153348e-11，阈值1e-8。重复调用得到相同梯度但独立数组；权重、传入output/target/cache均不变。

预期本轮输出：
```text
W_y 梯度: (64, 10)
W_f 梯度: (92, 64)
b_g 梯度: (64,)
b_g 梯度前 5 个值: [ 0.01042833  0.01863533  0.02344139 -0.00394581 -0.00202638]
```

这里b_g梯度与更新前略有不同，反映参数已变化。未执行全数据训练或测试评价。下一步才把前向、反向、参数更新按样本连接起来。

[建模记录](2026-10-06_lstm_mnist_v014.md) · [验证结果](2026-10-06_lstm_backward_function_validation_v014.json)
