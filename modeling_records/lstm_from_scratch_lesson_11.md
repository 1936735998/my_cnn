# 手写 LSTM 第 11 步：完整时间反向传播

日期：2026-10-06；版本：v011。
前置：已完成末时间步局部反向。本步在文件末尾追加倒序循环，计算一张图片总梯度，不更新参数。

```python
# ==================================================
# 完整时间反向传播：累加 28 步的参数梯度
# ==================================================

d_W_f = np.zeros_like(W_f)
d_b_f = np.zeros_like(b_f)
d_W_i = np.zeros_like(W_i)
d_b_i = np.zeros_like(b_i)
d_W_g = np.zeros_like(W_g)
d_b_g = np.zeros_like(b_g)
d_W_o = np.zeros_like(W_o)
d_b_o = np.zeros_like(b_o)

# 分类层只接在最后一个时间步
d_h_next = np.dot(d_output, W_y.T)
d_c_next = np.zeros(hidden_size)
backward_steps = 0

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

    # 来自后一时间步的状态梯度
    d_o = d_h_next * tanh_c
    d_c = d_c_next + d_h_next * o_t * (1 - tanh_c ** 2)

    d_f = d_c * c_prev_t
    d_i = d_c * g_t
    d_g = d_c * i_t

    d_z_f = d_f * f_t * (1 - f_t)
    d_z_i = d_i * i_t * (1 - i_t)
    d_z_g = d_g * (1 - g_t ** 2)
    d_z_o = d_o * o_t * (1 - o_t)

    # 累加本步对共享参数的贡献
    d_W_f += np.outer(combined_t, d_z_f)
    d_b_f += d_z_f

    d_W_i += np.outer(combined_t, d_z_i)
    d_b_i += d_z_i

    d_W_g += np.outer(combined_t, d_z_g)
    d_b_g += d_z_g

    d_W_o += np.outer(combined_t, d_z_o)
    d_b_o += d_z_o

    d_combined = (
        np.dot(d_z_f, W_f.T)
        + np.dot(d_z_i, W_i.T)
        + np.dot(d_z_g, W_g.T)
        + np.dot(d_z_o, W_o.T)
    )

    # 传给前一个时间步
    d_h_next = d_combined[input_size:]
    d_c_next = d_c * f_t
    backward_steps += 1

print("反向处理的时间步数:", backward_steps)
print("d_W_f:", d_W_f.shape)
print("d_W_i:", d_W_i.shape)
print("d_W_g:", d_W_g.shape)
print("d_W_o:", d_W_o.shape)
print("d_b_g:", d_b_g.shape)
print("d_b_g 前 5 个值:", d_b_g[:5])
```

## 关键含义
zeros_like 创建与参数同形状的零数组。range(len(cache)-1,-1,-1) 依次取索引27、26、…、0，覆盖28行。
d_h_next/d_c_next 在逆序循环中接收后一时间步传回的梯度，不在循环内清零。分类仅接末h，因此分类梯度只在循环前注入一次；更早步没有额外分类损失。
+= 累加当前步对共享参数的贡献。总梯度从零开始，不额外加入之前的 _step，避免末步贡献重复计入。分类层 d_W_y/d_b_y 继续使用之前结果；期间保持所有权重原值。
循环后两个 next 变量是进入第一行前的初始状态梯度，与 cache[0] 第一行处理后的状态不同；当前初始状态固定为零，不更新。

## 预期输出
```text
反向处理的时间步数: 28
d_W_f: (92, 64)
d_W_i: (92, 64)
d_W_g: (92, 64)
d_W_o: (92, 64)
d_b_g: (64,)
d_b_g 前 5 个值: [ 0.01051590  0.01904307  0.02379184 -0.00399105 -0.00202667]
```

## 实际验证
Python3.13.15、NumPy2.5.3、CPU float64，项目解释器独立执行；未修改用户主文件。倒序轨迹实际为27～0，28步，所有参数梯度形状和有限值通过；原参数和全部缓存保持原值，loss仍0.08934437001590566。
中央差分epsilon=1e-6、容差1e-8，每次扰动后从初始状态重算全部28步，共享参数用于每一步，不复用旧缓存。门权重每组抽6项、门偏置每组8项、分类矩阵6项、分类偏置全10项，初始h/c各64项通过，最大绝对误差约1.99e-11。未逐个检验全部24458个参数。
没有优化更新、训练或测试评价。下一步单次更新并重算同图损失，然后整理训练循环。

[建模记录](2026-10-06_lstm_mnist_v011.md) · [真实BPTT验证](2026-10-06_lstm_bptt_validation_v011.json) · [上一课](lstm_from_scratch_lesson_10.md)。
