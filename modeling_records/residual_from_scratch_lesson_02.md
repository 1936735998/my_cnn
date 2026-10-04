# 手写 MNIST 残差网络：第 2 步
日期：2026-10-04；版本：v002。

读取当前已保存 residual_from_scratch.py，发现只有参数初始化，缺少 NumPy 导入及数据准备。先在文件最上方补齐下段（编辑器若已补齐则保存即可，不要重复粘贴）：

```python
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:5000].reshape(5000, 784) / 255.0
y_train = y_train[:5000]

x_test = x_test[:1000].reshape(1000, 784) / 255.0
y_test = y_test[:1000]
```

保留上一段参数初始化，再在末尾追加：

```python
# ==================================================
# ReLU
# ==================================================

def relu(x):
    return np.maximum(0, x)


# ==================================================
# 单张图片的前向传播
# ==================================================

x = x_train[0]
label = y_train[0]

# 输入映射：784 → 64
z1 = np.dot(x, W1) + b1
h = relu(z1)

# 残差分支：64 → 64 → 64
z2 = np.dot(h, W2) + b2
u = relu(z2)
F_h = np.dot(u, W3) + b3

# 两条路径相加
s = h + F_h
r = relu(s)

# 输出层：64 → 10
output = np.dot(r, W4) + b4

print("h:", h.shape)
print("F_h:", F_h.shape)
print("s:", s.shape)
print("r:", r.shape)
print("output:", output.shape)

print("真实标签:", label)
print("预测类别:", np.argmax(output))
```

h 是残差块的输入，u 是残差分支第一层激活，F_h 是分支两层计算得到的修正量；s=h+F_h 是跳跃路径与分支相加的结果；r 是块末 ReLU 输出；output 是十个类别分数。

核心是 s=h+F_h：h 直接传递一份，同时送入分支，最后逐元素相加。h 和 F_h 都为 (64,)。F_h 可为正或负，表示相加时增大或减小对应特征，本实现分支末层使用线性输出，相加后再做 ReLU。

助手独立执行首个样本前向：h、F_h、s、r 均为 (64,)，output 为 (10,)；全部中间值有限；检查加法关系，且 F_h 置零的独立边界检查中块输出保留 h。真实标签 5、随机初始化预测 0；没有训练，不能用单样本预测判断性能。助手未修改用户主文件。

[本步建模记录](2026-10-04_residual_mnist_v002.md) · [实际验证结果](2026-10-04_residual_forward_validation_v002.json)
