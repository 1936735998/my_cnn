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

# ==================================================
# LSTM 参数初始化
# ==================================================

np.random.seed(42)

input_size = 28
hidden_size = 64
output_size = 10

combined_size = input_size + hidden_size

# 遗忘门：控制旧记忆保留多少
W_f = np.random.randn(combined_size, hidden_size) * 0.1
b_f = np.ones(hidden_size)

# 输入门：控制新信息写入多少
W_i = np.random.randn(combined_size, hidden_size) * 0.1
b_i = np.zeros(hidden_size)

# 候选记忆：准备写入的新内容
W_g = np.random.randn(combined_size, hidden_size) * 0.1
b_g = np.zeros(hidden_size)

# 输出门：控制记忆向隐藏状态输出多少
W_o = np.random.randn(combined_size, hidden_size) * 0.1
b_o = np.zeros(hidden_size)

# 分类层：把最后一步的隐藏状态映射到 10 类
W_y = np.random.randn(hidden_size, output_size) * 0.1
b_y = np.zeros(output_size)

# 一张图片开始时的状态
h = np.zeros(hidden_size)
c = np.zeros(hidden_size)

print("W_f:", W_f.shape)
print("W_y:", W_y.shape)
print("h:", h.shape)
print("c:", c.shape)
# ==================================================
# LSTM：一个时间步的前向计算
# ==================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 第一张图片的第一行
x = sequence[0]

# 保存上一步的状态
h_prev = h
c_prev = c

# 拼接当前输入与上一步隐藏状态
combined = np.concatenate([x, h_prev])

# 遗忘门
f = sigmoid(np.dot(combined, W_f) + b_f)

# 输入门
i_gate = sigmoid(np.dot(combined, W_i) + b_i)

# 候选记忆
g = np.tanh(np.dot(combined, W_g) + b_g)

# 输出门
o = sigmoid(np.dot(combined, W_o) + b_o)

# 更新细胞状态
c = f * c_prev + i_gate * g

# 更新隐藏状态
h = o * np.tanh(c)

print("当前行的非零像素数:", np.count_nonzero(x))
print("combined:", combined.shape)
print("c:", c.shape)
print("h:", h.shape)

print("f 前 5 个值:", f[:5])
print("g 前 5 个值:", g[:5])
print("c 前 5 个值:", c[:5])
print("h 前 5 个值:", h[:5])
# ==================================================
# LSTM：读完整张图片并计算类别分数
# ==================================================

sequence = x_train[0]
label = y_train[0]

# 从这张图片的第一行重新开始
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

# 循环结束后，h 是最后一个时间步的隐藏状态
output = np.dot(h, W_y) + b_y
prediction = int(np.argmax(output))

print("处理的时间步数:", len(sequence))
print("最终 c:", c.shape)
print("最终 h:", h.shape)
print("最终 h 前 5 个值:", h[:5])

print("output:", output.shape)
print("10 个类别分数:", output)
print("真实标签:", label)
print("未训练时的预测类别:", prediction)
# ==================================================
# one-hot 标签与均方误差
# ==================================================

def one_hot(label):
    target = np.zeros(output_size)
    target[label] = 1.0
    return target


target = one_hot(label)

squared_errors = (output - target) ** 2
loss = np.mean(squared_errors)

print("目标向量:", target)
print("各类别的平方误差:", squared_errors)
print("均方误差 loss:", loss)
# ==================================================
# 输出层反向传播
# ==================================================

# 损失对十个输出分数的梯度
d_output = 2 * (output - target) / output_size

# 分类层参数的梯度
d_W_y = np.outer(h, d_output)
d_b_y = d_output.copy()

# 分类层传回最后一个时间步 h 的梯度
d_h = np.dot(d_output, W_y.T)

print("d_output:", d_output.shape)
print("d_W_y:", d_W_y.shape)
print("d_b_y:", d_b_y.shape)
print("d_h:", d_h.shape)

print("d_output 的值:", d_output)
print("d_h 前 5 个值:", d_h[:5])
print("缓存的时间步数:", len(cache))
print("第一步 combined:", cache[0]["combined"].shape)
print("最后一步 c:", cache[-1]["c"].shape)
print("最后一步 h:", cache[-1]["h"].shape)
# ==================================================
# 最后一个时间步：输出门与细胞状态的梯度
# ==================================================

step = cache[-1]

c_t = step["c"]
o_t = step["o"]
tanh_c = np.tanh(c_t)

# 损失对输出门值的梯度
d_o = d_h * tanh_c

# 最后一步没有来自后一时间步的细胞状态梯度
d_c_next = np.zeros(hidden_size)

# 损失对当前细胞状态的梯度
d_c = d_c_next + d_h * o_t * (1 - tanh_c ** 2)

# 继续传过输出门的 sigmoid
d_z_o = d_o * o_t * (1 - o_t)

print("d_o:", d_o.shape)
print("d_c:", d_c.shape)
print("d_z_o:", d_z_o.shape)

print("d_o 前 5 个值:", d_o[:5])
print("d_c 前 5 个值:", d_c[:5])
print("d_z_o 前 5 个值:", d_z_o[:5])
# ==================================================
# 最后一个时间步：遗忘门、输入门与候选记忆的梯度
# ==================================================

f_t = step["f"]
i_t = step["i_gate"]
g_t = step["g"]
c_prev_t = step["c_prev"]

# 传回三个分支
d_f = d_c * c_prev_t
d_i = d_c * g_t
d_g = d_c * i_t

# 继续传过 sigmoid 和 tanh
d_z_f = d_f * f_t * (1 - f_t)
d_z_i = d_i * i_t * (1 - i_t)
d_z_g = d_g * (1 - g_t ** 2)

print("d_f:", d_f.shape)
print("d_i:", d_i.shape)
print("d_g:", d_g.shape)

print("d_z_f:", d_z_f.shape)
print("d_z_i:", d_z_i.shape)
print("d_z_g:", d_z_g.shape)

print("d_z_f 前 5 个值:", d_z_f[:5])
print("d_z_i 前 5 个值:", d_z_i[:5])
print("d_z_g 前 5 个值:", d_z_g[:5])
# ==================================================
# 最后一个时间步：参数梯度与上一时刻状态的梯度
# ==================================================

combined_t = step["combined"]

# 四组参数在本时间步的梯度贡献
d_W_f_step = np.outer(combined_t, d_z_f)
d_b_f_step = d_z_f.copy()

d_W_i_step = np.outer(combined_t, d_z_i)
d_b_i_step = d_z_i.copy()

d_W_g_step = np.outer(combined_t, d_z_g)
d_b_g_step = d_z_g.copy()

d_W_o_step = np.outer(combined_t, d_z_o)
d_b_o_step = d_z_o.copy()

# 四条路径共同传回 combined
d_combined = (
    np.dot(d_z_f, W_f.T)
    + np.dot(d_z_i, W_i.T)
    + np.dot(d_z_g, W_g.T)
    + np.dot(d_z_o, W_o.T)
)

# combined = [当前输入, 上一步 h]
d_h_prev = d_combined[input_size:]

# 旧记忆通过 f * c_prev 传到当前 c
d_c_prev = d_c * f_t

print("d_W_f_step:", d_W_f_step.shape)
print("d_W_i_step:", d_W_i_step.shape)
print("d_W_g_step:", d_W_g_step.shape)
print("d_W_o_step:", d_W_o_step.shape)
print("d_b_f_step:", d_b_f_step.shape)

print("d_combined:", d_combined.shape)
print("d_h_prev:", d_h_prev.shape)
print("d_c_prev:", d_c_prev.shape)

print("d_h_prev 前 5 个值:", d_h_prev[:5])
print("d_c_prev 前 5 个值:", d_c_prev[:5])
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
# ==================================================
# 一次参数更新，并比较同一张图片的损失
# ==================================================

import os
import sys

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("计算设备: CPU（NumPy ndarray）")
print("CPU 标识:", os.environ.get("PROCESSOR_IDENTIFIER", "未取得"))
print("GPU 设备枚举: NumPy 未提供此接口")

learning_rate = 0.01
loss_before = loss

# 使用整张图片的总梯度更新全部参数
W_f = W_f - learning_rate * d_W_f
b_f = b_f - learning_rate * d_b_f

W_i = W_i - learning_rate * d_W_i
b_i = b_i - learning_rate * d_b_i

W_g = W_g - learning_rate * d_W_g
b_g = b_g - learning_rate * d_b_g

W_o = W_o - learning_rate * d_W_o
b_o = b_o - learning_rate * d_b_o

W_y = W_y - learning_rate * d_W_y
b_y = b_y - learning_rate * d_b_y

# 从零状态重新读取同一张图片
h_after = np.zeros(hidden_size)
c_after = np.zeros(hidden_size)

for t in range(len(sequence)):
    combined_after = np.concatenate([sequence[t], h_after])

    f_after = sigmoid(np.dot(combined_after, W_f) + b_f)
    i_after = sigmoid(np.dot(combined_after, W_i) + b_i)
    g_after = np.tanh(np.dot(combined_after, W_g) + b_g)
    o_after = sigmoid(np.dot(combined_after, W_o) + b_o)

    c_after = f_after * c_after + i_after * g_after
    h_after = o_after * np.tanh(c_after)

output_after = np.dot(h_after, W_y) + b_y
loss_after = np.mean((output_after - target) ** 2)

print("更新前 loss:", loss_before)
print("更新后 loss:", loss_after)
print("loss 减少量:", loss_before - loss_after)
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

# ==================================================
# 遍历训练图片，统计每一轮的平均损失
# ==================================================

epochs = 5
learning_rate = 0.01
loss_history = []

for epoch in range(epochs):
    total_loss = 0.0

    for j in range(len(x_train)):
        loss = train_one(x_train[j], y_train[j], learning_rate)
        total_loss += loss

    average_loss = total_loss / len(x_train)
    loss_history.append(average_loss)

    print(f"epoch: {epoch + 1} loss: {average_loss:.6f}")
