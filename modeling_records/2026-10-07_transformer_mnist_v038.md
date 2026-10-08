# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v038。
阶段：QKV 三个投影手写 GPU 反向与独立 CPU 局部差分验证。本轮真实执行，不是仅文档更新；未汇合 encoder_input 路径，未完成全网络反向、参数更新或训练。

## 问题范围、输入来源与处理（事实）

用户报告 d_Q、d_K、d_V 均 `(28,32)`、GPU 0。本版反向三个投影，得到六份权重/偏置梯度和三份返回 encoder_input 的路径梯度。目标仍为逐节点手写 Transformer 全部反向与更新，最终在 GPU 训练。代码贴在聊天，不改用户主文件、不交付完整 Python 模型文件。

独立进程读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前三十七步，再执行本片段；未操作用户终端实时变量。主文件 SHA256 `967b5112567545ea8e043d4084b71dcdacaf53aff1bb7b005e928d97ed6fe752`，执行前后不变。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素 CuPy float32 除以 255、标签 int64。本轮只用首张训练图、标签 5，无新增清洗、下载、划分、洗牌、增强或测试评估。输入为实际 encoder_input、三组独立投影参数和对应上游梯度。

## 假设、变量与公式

延续图片行空间序列、单层编码器、位置均值池化、十类单图交叉熵。Q/K/V 投影各自有独立权重和偏置，每组参数在 28 行之间共享。所有反向保持同次前向的权重，不中途更新。

```text
E=encoder_input: (28,32)
X=Q/K/V=E@W_X+b_X: (28,32)
W_X: (32,32), b_X: (32,), G_X=d_X: (28,32)
dW_X[j,k]=sum_i E[i,j]*G_X[i,k] = (E.T@G_X)[j,k]: (32,32)
db_X[k]=sum_i G_X[i,k]: (32,)
dE_from_X[i,j]=sum_k G_X[i,k]*W_X[j,k] = (G_X@W_X.T)[i,j]: (28,32)
```

参数梯度累加行贡献，不额外平均。三份 d_encoder_input_from_Q/K/V 分别仅代表对应投影路径的贡献，本版保持 d_encoder_input_skip 并暂不汇合。新增参数 0，当前参数仍 9802。

## 方法选择理由（判断）

显式写出三组相同链式法则，分别使用对应上游梯度和前向权重，便于检查路径与参数配对。基础 CuPy dot/sum 在 GPU 完成位置累加和梯度回传，无自动求导或现成层。保留输入路径的独立变量，下一步显式汇合四条路径。

## 本步代码

```python
# ==================================================
# 反向第十八步：Q、K、V 的线性投影
# ==================================================

# Q 投影：共享参数梯度，以及 Q 路径返回的输入梯度
dW_Q = cp.dot(encoder_input.T, d_Q)
db_Q = cp.sum(d_Q, axis=0)
d_encoder_input_from_Q = cp.dot(d_Q, W_Q.T)

# K 投影
dW_K = cp.dot(encoder_input.T, d_K)
db_K = cp.sum(d_K, axis=0)
d_encoder_input_from_K = cp.dot(d_K, W_K.T)

# V 投影
dW_V = cp.dot(encoder_input.T, d_V)
db_V = cp.sum(d_V, axis=0)
d_encoder_input_from_V = cp.dot(d_V, W_V.T)

print("dW_Q:", dW_Q.shape)
print("db_Q:", db_Q.shape)
print("d_encoder_input_from_Q:", d_encoder_input_from_Q.shape)
print("dW_K:", dW_K.shape)
print("db_K:", db_K.shape)
print("d_encoder_input_from_K:", d_encoder_input_from_K.shape)
print("dW_V:", dW_V.shape)
print("db_V:", db_V.shape)
print("d_encoder_input_from_V:", d_encoder_input_from_V.shape)
print("Q 路径的实际设备:", d_encoder_input_from_Q.device)
print("K 路径的实际设备:", d_encoder_input_from_K.device)
print("V 路径的实际设备:", d_encoder_input_from_V.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，解释器带 -B、缓存在工作区 work/cupy_cache，本轮未调整依赖。

九份梯度均为有限 float32、GPU 0，权重 `(32,32)`、偏置 `(32,)`、输入路径 `(28,32)`。三个投影的权重和偏置存储分别独立。所有 20 组参数、相关前向缓存、loss、三份上游梯度与直连梯度保持不变。

CPU float64 为每条投影独立重算 E@W_X+b_X，构造 sum(X*固定 G_X)，不复用 GPU 投影输出或反向公式。每组检查 W 的 1024、b 的 32、E 的 896 分量，共 1952 项；三组共 5856 项做 ±1e-5 中心差分，rtol=1e-5、atol=2e-8 下全部通过，九组最大绝对误差 `1.965869934e-08`。各组误差、参考范数及实际梯度记录在 JSON。

参考梯度 L2 范数不超过 1e-12 时，相对误差记录为 null；接近零的梯度主要以绝对误差判断，避免除零或不稳定比值。CPU 双精度与 GPU float32 矩阵累加存在舍入差异。本核验验证三个线性节点的局部雅可比向量积，不代表已验证 encoder_input 总梯度或全网络反向。

实际输出：

```text
dW_Q: (32, 32)
db_Q: (32,)
d_encoder_input_from_Q: (28, 32)
dW_K: (32, 32)
db_K: (32,)
d_encoder_input_from_K: (28, 32)
dW_V: (32, 32)
db_V: (32,)
d_encoder_input_from_V: (28, 32)
Q 路径的实际设备: <CUDA Device 0>
K 路径的实际设备: <CUDA Device 0>
V 路径的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU 与中心差分证据](2026-10-07_transformer_qkv_projection_backward_validation_v038.json)。JSON 保存逐字代码、来源、设备、全部实际和差分梯度、误差及输出，片段 SHA256 `0ecaa5e8735a3b923dfbe32594494ad8f3882ba8a5e012af3905f12fdcede17f`。相对链接已核验，本步无新增图。

## 与前版变化、限制与下一步

v037 还原 Q/K/V 二维梯度；本版新增 dW_Q/db_Q、dW_K/db_K、dW_V/db_V 和三份 encoder_input 路径梯度，完成三个投影反向。新增参数 0、更新次数 0。保留历史记录，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录并更新索引，不重跑模型。

尚未汇合 encoder_input 的四条路径，未反向通过位置相加与输入投影，未训练或测量识别准确率。下一步将 Q/K/V 三份返回梯度与 d_encoder_input_skip 相加，得到 d_encoder_input，再传回 embedding，完成 W_in/b_in 反向，统一更新参数并组织 GPU 训练。
