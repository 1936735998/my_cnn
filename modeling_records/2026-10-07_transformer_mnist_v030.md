# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v030。
阶段：第一次 LayerNorm 手写 GPU 反向及独立 CPU 差分验证。本轮真实执行，不是仅文档更新；未完成注意力与全网络反向、参数更新或训练。

## 范围、来源与处理（事实）

用户报告两路径汇合后的 d_attention_norm `(28,32)`、GPU 0。本版由该总梯度反向经过第一次 LayerNorm，计算 dgamma_attn、dbeta_attn 与 d_attention_residual。目标仍为逐节点手写全部 Transformer 反向与更新，并在 GPU 训练。代码贴在聊天，不改用户主文件、不交付完整 Python 模型文件。

独立进程读取并执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前二十九步，再执行本片段；未操作用户终端实时变量。主文件 SHA256 `bf7df98d3181689ae06c81c6ac21ae3f8b0822b46d0aa7b63345d5f184e29813`，执行前后不变。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图；像素 CuPy float32 除以 255、标签 int64。本轮只用首张训练图、标签 5；没有新增清洗、下载、划分、洗牌、增强或测试评估。输入为第一次 LayerNorm 的独立前向缓存、参数和两路径汇合的梯度。

## 假设、变量与公式

延续图片行空间序列、单层编码器、位置均值池化与十类单图交叉熵。LayerNorm 逐行在 32 特征上使用总体方差，epsilon 固定不训练；gamma_attn/beta_attn 在 28 行共享，与第二次 LayerNorm 独立。反向使用本次前向参数和缓存。

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| X=attention_residual | (28,32) | 第一次 LayerNorm 输入 |
| z=normalized_attn | (28,32) | 第一次标准化缓存 |
| r=inv_std_attn | (28,1) | 1/sqrt(var_attn+epsilon) |
| gamma_attn、beta_attn | (32,) | 逐特征共享参数 |
| G=d_attention_norm | (28,32) | 输出的总上游梯度 |
| H=d_normalized_attn | (28,32) | G*gamma_attn |
| dgamma_attn、dbeta_attn | (32,) | 参数梯度 |
| d_attention_residual | (28,32) | 输入梯度 |

```text
前向：z=(X-mean_features(X))*r, Y=z*gamma+beta
dgamma[k]=sum_i G[i,k]*z[i,k]
dbeta[k]=sum_i G[i,k]
H=G*gamma
dX=r*(H-mean_features(H)-z*mean_features(H*z))
```

参数梯度沿 axis=0 累加位置贡献；输入反向的均值沿 axis=-1 在每行 32 特征内计算，keepdims=True。epsilon 包含在前向 r 和 z 中，合并公式保持正确，不假定标准化后方差严格等于 1。新增参数 0，当前参数仍 9802。

## 方法选择理由（判断）

沿用 v024 的 LayerNorm 反向规则，但使用第一次的独立参数、缓存与当前总上游梯度。先反向 affine，再应用含均值/方差依赖的合并链式法则，避免构建显式雅可比。只使用基础 CuPy 操作，没有自动求导或现成模型层；全网络反向完成后再统一更新参数。

## 本步代码

```python
# ==================================================
# 反向第十步：第一次 LayerNorm
# ==================================================

# gamma、beta 在 28 行之间共享，参数梯度沿位置轴求和
dgamma_attn = cp.sum(d_attention_norm * normalized_attn, axis=0)
dbeta_attn = cp.sum(d_attention_norm, axis=0)

# 反向通过缩放：attention_norm = normalized_attn * gamma_attn + beta_attn
d_normalized_attn = d_attention_norm * gamma_attn

# 每行在 32 个特征内取平均，保留 (28, 1) 便于广播
mean_d_normalized_attn = cp.mean(
    d_normalized_attn, axis=-1, keepdims=True
)
mean_d_normalized_times_x_attn = cp.mean(
    d_normalized_attn * normalized_attn, axis=-1, keepdims=True
)

# 反向通过标准化，得到对 attention_residual 的梯度
d_attention_residual = inv_std_attn * (
    d_normalized_attn
    - mean_d_normalized_attn
    - normalized_attn * mean_d_normalized_times_x_attn
)

print("dgamma_attn:", dgamma_attn.shape)
print("dbeta_attn:", dbeta_attn.shape)
print("d_normalized_attn:", d_normalized_attn.shape)
print("mean_d_normalized_attn:", mean_d_normalized_attn.shape)
print("mean_d_normalized_times_x_attn:", mean_d_normalized_times_x_attn.shape)
print("d_attention_residual:", d_attention_residual.shape)
print("dgamma_attn 的实际设备:", dgamma_attn.device)
print("dbeta_attn 的实际设备:", dbeta_attn.device)
print("d_attention_residual 的实际设备:", d_attention_residual.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，解释器带 -B、缓存放在工作区 work/cupy_cache，本轮未调整依赖。

全部参数梯度、输入梯度与中间数组的形状符合公式，均为有限 float32、GPU 0。所有 20 组参数、第一次 LayerNorm 前向缓存、loss 及上游梯度未改变；两次 LayerNorm 的 gamma、beta 存储地址分别独立。

CPU float64 独立重新计算第一次 LayerNorm 的均值、方差与 affine 输出，构造标量 sum(LN(X,gamma,beta)*固定 G)，保持 epsilon 的实际值 `9.999999747378752e-06`。对输入 896、gamma 32、beta 32 个分量分别做 ±1e-5 中心差分，共 960 分量；rtol=2e-5、atol=2e-7 下全部通过。最大绝对误差：d_attention_residual `1.457385046e-09`，dgamma_attn `6.413942724e-09`，dbeta_attn `6.984202855e-09`。

本轮验证关注第一次 LayerNorm 当前输入、缓存和上游梯度的局部雅可比向量积；CPU 双精度与 GPU float32 存在舍入差异，不把局部验证解释为注意力或全网络反向已经正确。公式的非单位 affine 与零方差行为此前在 v024 有单独验证记录。

实际输出：

```text
dgamma_attn: (32,)
dbeta_attn: (32,)
d_normalized_attn: (28, 32)
mean_d_normalized_attn: (28, 1)
mean_d_normalized_times_x_attn: (28, 1)
d_attention_residual: (28, 32)
dgamma_attn 的实际设备: <CUDA Device 0>
dbeta_attn 的实际设备: <CUDA Device 0>
d_attention_residual 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU 与中心差分证据](2026-10-07_transformer_attention_layernorm_backward_validation_v030.json)。JSON 保存逐字代码、来源、设备、全部实际与差分梯度、误差及输出，片段 SHA256 `bf3acf43c73c76027270ca986f88820a5cfcaf57edae4c8e47bcec2d0dde3b1c`。相对链接已核验，本步无新增图。

## 与前版变化、限制与下一步

v029 得到 d_attention_norm；本版新增 dgamma_attn、dbeta_attn 与 d_attention_residual，完成第一次 LayerNorm 反向。新增参数 0、更新次数 0。保留历史记录，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录并更新索引，不重跑模型。

尚未反向经过注意力残差相加、注意力内部或输入投影，未训练或测量准确率。下一步从 attention_residual=encoder_input+attention_output，将 d_attention_residual 传入直连 encoder_input 和注意力输出两条路径，继续反向注意力；之后汇合输入路径并完成输入投影反向，统一更新参数并组织 GPU 训练。
