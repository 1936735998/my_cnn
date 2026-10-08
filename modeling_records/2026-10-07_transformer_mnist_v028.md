# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v028。
阶段：前馈第一层手写 GPU 反向与独立 CPU 差分验证。本轮真实执行，不是仅文档更新；未汇合 skip 梯度，未完成全网络反向、参数更新或训练。

## 问题范围、来源与处理（事实）

用户报告 d_ff_hidden_linear `(28,64)`、非正输入处梯度为零、正输入处梯度保持上游值。本版反向经过第一层线性映射，计算 dW_ff1、db_ff1 和 d_attention_norm_from_ffn。按小步继续手写全部 Transformer 反向和更新，最终 GPU 训练；交付聊天代码，不修改用户主文件、不交付完整 Python 模型文件。

独立进程读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前二十七步，再执行本片段；未操作用户终端实时内存。主文件 SHA256 `29529201e66ca1b35fafff69d384b23970a85c40507031f9834fccd454906b4f`，执行前后不变。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素 CuPy float32 除以 255、标签 int64。只用首张训练图、标签 5，没有新增清洗、下载、重划分、洗牌、增强或测试评估。本节点输入是实际 attention_norm、W_ff1、b_ff1 与 ReLU 返回的 d_ff_hidden_linear。

## 假设、变量与公式

延续图片行空间序列、单层编码器、位置均值池化、十类单图交叉熵。第一层线性后使用 ReLU，掩码已在前一步应用。28 行共享 W_ff1 与 b_ff1，所有反向使用本次前向权重，不中途更新。

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| A=attention_norm | (28,32) | 第一层前馈输入，同时进入直连支路 |
| W=W_ff1 | (32,64) | 第一层共享权重 |
| b=b_ff1 | (64,) | 第一层共享偏置 |
| G=d_ff_hidden_linear | (28,64) | 第一层线性输出的梯度 |
| dW_ff1 | (32,64) | 权重梯度 |
| db_ff1 | (64,) | 偏置梯度 |
| d_attention_norm_from_ffn | (28,32) | 前馈路径对 A 的梯度贡献 |

```text
前向：ff_hidden_linear = A@W+b
dW[j,k] = sum_i A[i,j]*G[i,k] = (A.T@G)[j,k]
db[k] = sum_i G[i,k]
dA_from_ffn[i,j] = sum_k G[i,k]*W[j,k] = (G@W.T)[i,j]
后续：dA_total = d_attention_norm_skip + d_attention_norm_from_ffn
```

共享参数梯度沿位置轴累加，不额外平均。前一步的 G 已经过 ReLU，这里不重复添加掩码。d_attention_norm_from_ffn 只是前馈路径的贡献，本版暂不汇合直连支路，不称为 A 的总梯度。新增参数 0，累计参数 9802。

## 方法选择理由（判断）

沿用线性层链式法则，用基础 CuPy dot/sum 一次完成共享参数贡献的累加和输入梯度传递，避免逐行外积循环。使用 _from_ffn 标明路径来源，下一步显式汇合两条路径便于学习。没有自动求导或现成模型层，参数保持前向值。

## 本步代码

```python
# ==================================================
# 反向第八步：前馈网络第一层
# ==================================================

# (32, 28) @ (28, 64) → (32, 64)
# 累加 28 行对共享权重的梯度贡献
dW_ff1 = cp.dot(attention_norm.T, d_ff_hidden_linear)

# 每个隐藏特征的偏置梯度，沿 28 行求和
db_ff1 = cp.sum(d_ff_hidden_linear, axis=0)

# (28, 64) @ (64, 32) → (28, 32)
# 使用本次前向的 W_ff1，得到前馈支路返回的梯度
d_attention_norm_from_ffn = cp.dot(d_ff_hidden_linear, W_ff1.T)

print("dW_ff1:", dW_ff1.shape)
print("db_ff1:", db_ff1.shape)
print("d_attention_norm_from_ffn:", d_attention_norm_from_ffn.shape)
print("dW_ff1 的实际设备:", dW_ff1.device)
print("db_ff1 的实际设备:", db_ff1.device)
print("d_attention_norm_from_ffn 的实际设备:", d_attention_norm_from_ffn.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，解释器带 -B、缓存位于工作区 work/cupy_cache，本轮未调整依赖。

三组梯度 `(32,64)`、`(64,)`、`(28,32)` 均为有限 float32、GPU 0。所有 20 组参数、相关前向缓存、上游梯度、loss 与 d_attention_norm_skip 保持不变。

CPU float64 独立重建 A@W+b→ReLU→第二层线性→加固定 skip=A0→第二次 LayerNorm→位置均值池化→分类器→稳定交叉熵。A 只在 FFN 路径变化，skip 固定，故检查的是前馈路径贡献，而非总输入梯度。分别对 2048 权重、64 偏置、896 输入分量做中心差分，共 3008 分量；rtol=2e-5、atol=2e-7 下全部通过。

CPU 基础预激活最小绝对值 `0.0001828443665`，全部非零，正负掩码与 GPU 完全一致。每项扰动步长取 min(1e-5, 受影响预激活的 abs(Z)/(4*abs(变化系数)))，忽略系数为零的项，并逐次检查 ReLU 正负不变。变化系数分别为权重扰动对应的 A 列、偏置扰动的 1、输入扰动对应的 W 行，避免跨不可导点。

最大绝对误差：dW_ff1 `1.858933629e-08`，db_ff1 `9.589562908e-09`，d_attention_norm_from_ffn `5.529159353e-10`。各组实际步长最小/最大值已写入 JSON。CPU 独立双精度重算与 GPU float32 存在舍入差异，此验证不代表全网络反向正确。

实际输出：

```text
dW_ff1: (32, 64)
db_ff1: (64,)
d_attention_norm_from_ffn: (28, 32)
dW_ff1 的实际设备: <CUDA Device 0>
db_ff1 的实际设备: <CUDA Device 0>
d_attention_norm_from_ffn 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU 与中心差分证据](2026-10-07_transformer_ffn_first_backward_validation_v028.json)。JSON 保存逐字代码、来源、设备、全部实际和差分梯度、步长、误差及输出，片段 SHA256 `996f63482f15fc2783543bb6cd607e66b735a98c5d40153194c8b0b4332c9656`。相对链接已核验，本步无新增图。

## 与前版变化、限制与下一步

v027 完成 ReLU 反向；本版新增 dW_ff1、db_ff1、d_attention_norm_from_ffn，完成前馈第一层反向。新增参数 0、更新次数 0。保留历史记录，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录并更新索引，不重跑模型。

尚未汇合两条路径，也未反向经过第一层 LayerNorm、注意力或输入投影，没有训练或识别准确率结果。下一步将 d_attention_norm_skip 与 d_attention_norm_from_ffn 相加，得到 d_attention_norm，再继续第一层 LayerNorm 和注意力反向，逐节点完成后统一更新参数并组织 GPU 训练。
