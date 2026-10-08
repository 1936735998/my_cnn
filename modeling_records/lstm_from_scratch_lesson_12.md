# 手写 LSTM 第 12 步：一次参数更新与同图损失比较

日期：2026-10-06；版本：v012。
前置：已经求得一张图片的完整共享参数梯度。本步执行一次单样本SGD步骤，尚未训练全部5000张图片。

## 追加代码
```python
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
```

## 更新规则与损失重算
每个可训练参数使用 P_new=P_old-learning_rate*d_P，learning_rate=0.01。所有梯度已在更新前计算完成，更新十组门/候选/分类参数。d_W_f/i/g/o 是28步贡献之和，更新时不再除以28。
负号表示沿损失梯度的反方向移动；步长影响一次移动幅度，不保证任意步长都使损失下降。
参数改变后，旧loss不会自动改变。h_after/c_after从零开始，用新参数重新读取同一图片全部28行，再计算output_after/loss_after；真实target不变。
原h/output/loss/cache仍保存更新前值，after变量避免混淆。新参数与旧缓存、旧梯度不再匹配；下一次更新必须重新前向、创建新缓存、反向求新梯度。本步重算仅用于比较损失，没有创建下一次反向所需的新缓存。

## 实际输出
```text
更新前 loss: 0.08934437001590566
更新后 loss: 0.0881374096811086
loss 减少量: 0.0012069603347970581
```
本次仅首训练图一个SGD步骤，同图损失实际下降，不等同于测试准确率提高或完成数据集训练。

## 运行环境与验证
Python3.13.15、NumPy2.5.3，实际CPU float64，CPU系统标识AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD，不是完整商品型号。NumPy不提供GPU枚举接口，未取得物理GPU清单，不推断机器没有GPU。
项目解释器实际独立执行保存的前置代码和本课更新段；十种参数逐元素等于原值减0.01乘总梯度，参数和新输出有限，零状态重算同图，旧h/output/loss和全部缓存保持原值。助手没有编辑用户主文件。
没有完整数据集训练或测试评价。下一步整理基础前向/反向函数，使每个样本都生成新缓存和新梯度，再建立训练循环。

[建模记录](2026-10-06_lstm_mnist_v012.md) · [真实单步更新JSON](2026-10-06_lstm_single_update_validation_v012.json) · [上一课](lstm_from_scratch_lesson_11.md)。
