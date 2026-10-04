# 手写 MNIST 残差网络：第 1 步
日期：2026-10-04；版本：v001。

沿用之前 NumPy、普通函数、手写梯度、用户亲自在 VS Code 逐段输入的标准。先学习全连接残差网络，通过一个两层全连接残差块理解跳跃相加；它是保留残差结构的教学模型，不是原始卷积 ResNet-18 的实现。助手没有创建或修改用户主 Python 文件。

## 新文件与数据复用
在 F:\PythonProjects\deep_learning 中新建 residual_from_scratch.py。将 dnn_from_scratch.py 从 import numpy as np 开始，到 x_test = x_test.reshape(1000, 784) / 255.0 为止的代码复制过去；暂不复制旧网络参数、函数、训练和评价。数据设置沿用首 5000 张训练图片、首 1000 张测试图片、展开与归一化。

## 参数代码
在数据准备代码末尾追加：

```python
# ==================================================
# 参数初始化
# ==================================================

np.random.seed(42)

# 输入层：784 → 64
W1 = np.random.randn(784, 64) * 0.01
b1 = np.zeros(64)

# 残差分支第一层：64 → 64
W2 = np.random.randn(64, 64) * 0.01
b2 = np.zeros(64)

# 残差分支第二层：64 → 64
W3 = np.random.randn(64, 64) * 0.01
b3 = np.zeros(64)

# 输出层：64 → 10
W4 = np.random.randn(64, 10) * 0.01
b4 = np.zeros(10)

print("W1:", W1.shape, "b1:", b1.shape)
print("W2:", W2.shape, "b2:", b2.shape)
print("W3:", W3.shape, "b3:", b3.shape)
print("W4:", W4.shape, "b4:", b4.shape)
```

## 计划结构
x(784)→W1+ReLU→h(64)。残差分支：u=ReLU(hW2+b2)，F(h)=uW3+b3，F(h) 为 64 维。跳跃路径直接传递 h；s=h+F(h)，r=ReLU(s)，output=rW4+b4，共 10 个分数。

h 和 F(h) 必须有相同形状，本例都是 (64,)，才能逐元素相加。不能直接将 784 维原始图片加到 64 维残差分支上。本例先用 W1 将输入映射到 64 维，再执行恒等跳接。若 F(h)=0，和 s 就等于 h；h 已非负，块末 ReLU 也保留 h。

## 已执行验证
助手通过项目解释器直接读取同一 Keras 缓存，完成原有数据处理和八组参数初始化检查；Python 3.13.15、NumPy 2.5.3、CPU NumPy 数组。形状符合上面四组 W/b，参数总数 59210，参数均为有限值。残差前向、梯度、训练和测试都尚未执行，不能据此推断性能。

下一步用户运行初始化后，写出残差块的前向传播并观察相加路径。

[本步建模记录](2026-10-04_residual_mnist_v001.md) · [实际验证结果](2026-10-04_residual_init_validation_v001.json)
