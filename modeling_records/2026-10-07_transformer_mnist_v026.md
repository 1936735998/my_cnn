# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v026。
阶段：前馈第二层线性映射手写反向，真实 GPU 执行及 CPU 差分验证。本轮不是仅文档更新，未经过 ReLU、第一层前馈或全网络反向，也未更新参数或训练。

## 问题范围、数据来源与处理（事实）

用户报告 d_attention_norm_skip 与 d_ff_output 均 `(28,32)`、GPU 0。本版继续由 d_ff_output 计算 W_ff2、b_ff2 和 ff_hidden 的梯度，保留直连支路梯度。最终目标仍为逐节点手写 Transformer 全部反向与参数更新，并在 GPU 训练。代码贴于聊天，不修改用户主文件、不交付完整 Python 模型文件。

独立进程读取并执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前二十五步，再执行本片段；未操作用户终端中的实时变量。主文件 SHA256 `31052a61f23c1256375ebe518bd9ecae2c40f1d712111c1c823f3d8d5e87a9e5`，执行前后不变。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 张，像素 CuPy float32 除以 255、标签 int64。本轮仅用首张训练图，标签 5；无额外下载、清洗、洗牌、增强、重划分或测试评估。局部输入是实际 ff_hidden、W_ff2、b_ff2 与 d_ff_output。

## 假设、变量与公式

延续图片行空间序列、单层编码器、位置均值池化、十类单图交叉熵。第二层为线性映射，无其后的额外激活；28 行共享一组权重和偏置。反向使用本次前向的参数，统一更新留到全网络反向完成后。

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| H=ff_hidden | (28,64) | 第一层 ReLU 的输出 |
| W=W_ff2 | (64,32) | 第二层共享权重 |
| b=b_ff2 | (32,) | 第二层共享偏置 |
| G=d_ff_output | (28,32) | 第二层输出的上游梯度 |
| dW_ff2 | (64,32) | 损失对权重的梯度 |
| db_ff2 | (32,) | 损失对偏置的梯度 |
| d_ff_hidden | (28,64) | 损失对 ReLU 输出的梯度 |

```text
前向：ff_output = H@W+b
dW[j,k] = sum_i H[i,j]*G[i,k] = (H.T@G)[j,k]
db[k] = sum_i G[i,k]
dH[i,j] = sum_k G[i,k]*W[j,k] = (G@W.T)[i,j]
```

共享参数梯度累加 28 个位置；不额外平均，均值池化的 1/28 已包含在上游梯度。d_ff_hidden 只到 ReLU 的输出端，尚未乘 ReLU 掩码。新增参数 0，当前参数数目仍 9802。

## 方法选择理由（判断）

矩阵乘法一次实现每行外积的累加，明确权重、偏置与输入梯度的不同收缩轴，避免 Python 循环逐行求梯度。基础 CuPy dot/sum 均在 GPU 执行，无自动求导或现成层。使用当前 W_ff2 保持链式法则与前向一致，保留 skip 梯度供后续汇合。

## 本步代码

```python
# ==================================================
# 反向第六步：前馈网络第二层
# ==================================================

# (64, 28) @ (28, 32) → (64, 32)
# 累加 28 行对共享权重的梯度贡献
dW_ff2 = cp.dot(ff_hidden.T, d_ff_output)

# 每个输出特征的偏置梯度，沿 28 行求和
db_ff2 = cp.sum(d_ff_output, axis=0)

# (28, 32) @ (32, 64) → (28, 64)
# 使用本次前向计算时的 W_ff2
d_ff_hidden = cp.dot(d_ff_output, W_ff2.T)

print("dW_ff2:", dW_ff2.shape)
print("db_ff2:", db_ff2.shape)
print("d_ff_hidden:", d_ff_hidden.shape)
print("dW_ff2 的实际设备:", dW_ff2.device)
print("db_ff2 的实际设备:", db_ff2.device)
print("d_ff_hidden 的实际设备:", d_ff_hidden.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，解释器使用 -B，缓存在工作区 work/cupy_cache。本轮未修改依赖。

三组梯度 `(64,32)`、`(32,)`、`(28,64)` 均为有限 float32、GPU 0。所有 20 组参数、相关前向输入/缓存、loss、上游梯度和 d_attention_norm_skip 均未改变。

独立 CPU float64 重建 H@W+b→加固定 attention_norm→第二次 LayerNorm→位置均值池化→分类器→稳定交叉熵。对 W 的 2048 个、b 的 32 个与 H 的 1792 个分量做 ±1e-5 中心差分，共 3872 个分量；rtol=2e-5、atol=2e-7 下全部通过。最大绝对误差：dW_ff2 `1.235907732e-08`，db_ff2 `1.448544396e-08`，d_ff_hidden `9.286580394e-10`。

H 被视为本线性节点的独立连续输入，即使某项前向由 ReLU 得到 0，其局部仿射函数也可用正负扰动验证。这不检查 ReLU 在 0 处的导数约定，也不代表第一层前馈或全网络反向正确。CPU 独立双精度重算与 GPU float32 存在舍入差异。

实际输出：

```text
dW_ff2: (64, 32)
db_ff2: (32,)
d_ff_hidden: (28, 64)
dW_ff2 的实际设备: <CUDA Device 0>
db_ff2 的实际设备: <CUDA Device 0>
d_ff_hidden 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU 与中心差分证据](2026-10-07_transformer_ffn_second_backward_validation_v026.json)。JSON 保存逐字代码、来源、设备、全部实际与差分梯度、误差及输出，片段 SHA256 `0b5a91b39c4f14f31a50cad8884892baf65139a65622f108437021e776d03cc6`。相对链接已核验。本步无新增图。

## 与前版变化、限制与下一步

v025 分出前馈残差的直连与前馈支路梯度；本版新增 dW_ff2、db_ff2、d_ff_hidden，完成第二层线性反向。新增参数 0、参数更新次数 0。保留历史记录，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录并更新索引，不重跑模型。

尚未反向经过 ReLU、第一层前馈、第一层 LayerNorm、注意力和输入投影；未训练或测量准确率。下一步使用 ff_hidden_linear>0 的掩码，把 d_ff_hidden 传回 d_ff_hidden_linear；零点导数约定为 0。之后反向第一层前馈并汇合 skip 梯度，逐节点完成后统一更新参数并组织 GPU 训练。
