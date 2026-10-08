# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v024。
阶段：第二次 LayerNorm 手写反向，真实 GPU 执行及 CPU 中心差分验证。本轮不是仅文档更新，尚未完成全网络反向、参数更新或训练。

## 问题范围、来源与处理（事实）

用户报告平均池化梯度 `(28,32)`、首尾行梯度相同、各行梯度和与上游的最大差值 7.450580596923828e-09。本版沿第二次 LayerNorm 反向，计算 gamma_ffn、beta_ffn 参数梯度与 ff_residual 输入梯度。交付聊天代码，不改用户主文件，不交付完整 Python 模型。最终目标仍是逐节点手写反向与更新，并在 GPU 训练。

独立进程读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前二十三步，再执行本片段。未操作用户终端中的实时变量。主文件 SHA256 `8f7d8a4a8caab5f53b4c3b2d326b6017cc520fb350a52d17fd08172058efbdfd`，执行前后相同。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 张，像素 CuPy float32 除以 255，标签 int64。本轮只用首张训练图、标签 5；无额外下载、清洗、重划分、洗牌、增强或测试评估。输入为第二次 LayerNorm 的实际缓存和池化返回的梯度。额外数值测试用实际 ff_residual 的副本与明确生成的 affine/上游值，不作为训练数据。

## 假设、变量与公式

延续每行是空间位置、单层编码器、位置均值池化与十类单图交叉熵。LayerNorm 逐行在 32 特征上计算总体方差，epsilon 固定、不训练；gamma_ffn/beta_ffn 在 28 行共享，且与第一次 LayerNorm 的参数独立。反向使用本次前向缓存与参数，不在中途更新。

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| X=ff_residual | (28,32) | 第二次 LayerNorm 输入 |
| z=normalized_ffn | (28,32) | 已保存的标准化输入 |
| r=inv_std_ffn | (28,1) | 1/sqrt(var+epsilon) |
| gamma_ffn、beta_ffn | (32,) | 逐特征共享缩放、偏移 |
| G=d_encoder_output | (28,32) | 本节点输出的上游梯度 |
| H=d_normalized_ffn | (28,32) | G*gamma_ffn |
| dgamma_ffn、dbeta_ffn | (32,) | 参数梯度 |
| d_ff_residual | (28,32) | 本节点输入梯度 |

```text
前向：z=(X-mean_features(X))*r, Y=z*gamma+beta
dgamma[k]=sum_i G[i,k]*z[i,k]
dbeta[k]=sum_i G[i,k]
H=G*gamma
dX=r*(H-mean_features(H)-z*mean_features(H*z))
```

参数梯度在位置轴 axis=0 求和；输入梯度中的均值在每行特征轴 axis=-1 计算，keepdims=True。公式包含均值和方差依赖，epsilon 已包含在 r 与 z 中；不假定 z 的方差严格等于 1，不额外添加 epsilon，也不把前向统计量作为常数直接乘 r。输入梯度形状恢复 `(28,32)`。新增参数 0、当前参数总数 9802。

## 方法选择理由（判断）

先反向 affine，再使用 LayerNorm 链式法则的合并公式，避免显式构建每行 32×32 雅可比。清楚保留两种求和轴，便于学习和后续批处理扩展。使用基础 CuPy 数组操作，无自动求导或现成模型层。局部差分固定上游 G 的标量目标 sum(LN(X)*G)，独立重算 CPU 均值、方差与输出；直接检查该节点的雅可比向量积，不声称重跑全图梯度。

## 本步代码

```python
# ==================================================
# 反向第四步：第二次 LayerNorm
# ==================================================

# gamma、beta 在 28 行之间共享，参数梯度沿位置轴求和
dgamma_ffn = cp.sum(d_encoder_output * normalized_ffn, axis=0)
dbeta_ffn = cp.sum(d_encoder_output, axis=0)

# 先反向通过缩放：encoder_output = normalized_ffn * gamma_ffn + beta_ffn
d_normalized_ffn = d_encoder_output * gamma_ffn

# 每一行分别在 32 个特征内取平均，保留 (28, 1) 便于广播
mean_d_normalized_ffn = cp.mean(
    d_normalized_ffn, axis=-1, keepdims=True
)
mean_d_normalized_times_x_ffn = cp.mean(
    d_normalized_ffn * normalized_ffn, axis=-1, keepdims=True
)

# 继续反向通过标准化，得到对 ff_residual 的梯度
d_ff_residual = inv_std_ffn * (
    d_normalized_ffn
    - mean_d_normalized_ffn
    - normalized_ffn * mean_d_normalized_times_x_ffn
)

print("dgamma_ffn:", dgamma_ffn.shape)
print("dbeta_ffn:", dbeta_ffn.shape)
print("d_normalized_ffn:", d_normalized_ffn.shape)
print("mean_d_normalized_ffn:", mean_d_normalized_ffn.shape)
print("mean_d_normalized_times_x_ffn:", mean_d_normalized_times_x_ffn.shape)
print("d_ff_residual:", d_ff_residual.shape)
print("dgamma_ffn 的实际设备:", dgamma_ffn.device)
print("dbeta_ffn 的实际设备:", dbeta_ffn.device)
print("d_ff_residual 的实际设备:", d_ff_residual.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实 CuPy 执行并同步 CUDA，未修改依赖；缓存使用工作区 work/cupy_cache，解释器带 -B。

全部参数梯度、输入梯度及中间数组均为有限 float32、GPU 0，形状与公式一致。所有 20 组参数、本节点前向缓存、上游梯度及 loss 不变。

CPU float64 独立重算 LayerNorm 的局部标量目标，保持 GPU float32 epsilon 的实际值 `9.999999747378752e-06`。对输入 896、gamma 32、beta 32 个分量分别做 ±1e-5 中心差分，共 960 个分量；rtol=2e-5、atol=2e-7 下全部通过。最大绝对误差：d_ff_residual `8.978027211e-10`，dgamma_ffn `5.78525404e-09`，dbeta_ffn `4.659419517e-09`。

额外两个独立案例各检查 960 分量：gamma 在 0.5～1.5、beta 在 -0.3～0.3 的逐特征值，上游 G[i,k]=0.02*cos(0.17*(32*i+k))；第二个案例将首行 X 设为 float32 精确可表示常数 0.25。两次执行完全相同聊天片段，独立重算缓存，测试不同位置梯度和零方差边界；合计额外 1920 个分量均通过，最大绝对误差 `1.351132771e-06`。恒定首行方差为 0、z 为 0，输入梯度满足 r*(H-mean(H)) 且有限。

实际案例加额外案例总计 2880 个局部分量检查。本验证覆盖第二次 LayerNorm 及其 affine 参数，不代表全网络反向验证，也未执行参数更新。实际输出：

```text
dgamma_ffn: (32,)
dbeta_ffn: (32,)
d_normalized_ffn: (28, 32)
mean_d_normalized_ffn: (28, 1)
mean_d_normalized_times_x_ffn: (28, 1)
d_ff_residual: (28, 32)
dgamma_ffn 的实际设备: <CUDA Device 0>
dbeta_ffn 的实际设备: <CUDA Device 0>
d_ff_residual 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU 与中心差分证据](2026-10-07_transformer_ffn_layernorm_backward_validation_v024.json)。JSON 保存逐字片段、来源、全部案例输入和梯度、差分结果、误差、设备与输出，片段 SHA256 `42cbf4e6112c98ef69cd2894ebebfc8746f373487f15b73cff5c38ce306f334d`。相对链接已核验；本步无新图，维度和数值验证足以表达结果。

## 与前版变化、限制与下一步

v023 得到 d_encoder_output；本版新增 dgamma_ffn、dbeta_ffn 和 d_ff_residual，完成第二次 LayerNorm 反向，新增参数与参数更新次数均 0。保留历史记录，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录和更新索引，不再次运行模型。

尚未执行前馈残差、前馈网络、第一次 LayerNorm、注意力与输入投影反向，没有训练或识别准确率结果。下一步通过 ff_residual=attention_norm+ff_output，把 d_ff_residual 分到直连支路和前馈支路，再逐节点反向，汇合支路后统一更新参数，组织 GPU 训练。
