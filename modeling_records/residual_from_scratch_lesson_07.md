# 手写 MNIST 残差网络：第 7 步
日期：2026-10-04；版本：v007。

当前保存文件还缺少 v006 输入映射层梯度。若上一段尚未加入，先补齐：

```python
d_z1 = d_h * (z1 > 0)
d_W1 = np.outer(x, d_z1)
d_b1 = d_z1
```

再追加本段；助手未修改用户主 Python 文件。

```python
# ==================================================
# 参数更新
# ==================================================

learning_rate = 0.01
loss_before = loss

W1 = W1 - learning_rate * d_W1
b1 = b1 - learning_rate * d_b1

W2 = W2 - learning_rate * d_W2
b2 = b2 - learning_rate * d_b2

W3 = W3 - learning_rate * d_W3
b3 = b3 - learning_rate * d_b3

W4 = W4 - learning_rate * d_W4
b4 = b4 - learning_rate * d_b4


# ==================================================
# 更新后重新计算前向传播与损失
# ==================================================

z1_new = np.dot(x, W1) + b1
h_new = relu(z1_new)

z2_new = np.dot(h_new, W2) + b2
u_new = relu(z2_new)
F_h_new = np.dot(u_new, W3) + b3

s_new = h_new + F_h_new
r_new = relu(s_new)

output_new = np.dot(r_new, W4) + b4
loss_after = np.mean((output_new - target) ** 2)

import sys
import platform

print("Python:", sys.version.split()[0])
print("计算框架: NumPy", np.__version__)
print("参数计算设备: CPU（NumPy 数组）")
print("CPU 标识:", platform.processor())

print("更新前损失:", loss_before)
print("更新后损失:", loss_after)
```

梯度下降规则仍为新参数=旧参数-学习率×梯度。所有梯度先由同一次旧参数前向计算，再更新八组参数。恒等跳跃路径本身没有额外可学习参数；其梯度贡献已经进入 d_h，从而影响 W1/b1。

参数变化后必须重算完整前向图，包含新 h、新残差分支 F_h_new 和相加结果，不能把旧 output 或 loss 当作新结果。首样本 target 不变，仍为类别 5 的 one-hot。

实际独立执行一次学习更新，loss_before=0.10042891272387644，loss_after=0.09976679365138812；本样本损失下降，参数均为有限值。预测类别仍为 0，不是正确标签 5，一次更新不足以完成学习。使用 CPU NumPy，尚未多样本训练或测试评价。

[本步建模记录](2026-10-04_residual_mnist_v007.md) · [实际执行结果](2026-10-04_residual_update_validation_v007.json)
