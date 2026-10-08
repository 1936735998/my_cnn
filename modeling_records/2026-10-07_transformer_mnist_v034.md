# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v034。
阶段：注意力加权求和手写 GPU 反向及独立 CPU 局部差分验证。本轮真实执行，不是仅文档更新；未完成 softmax、QKV 投影、全网络反向、参数更新或训练。

## 问题范围、来源与处理（事实）

用户报告 d_head_output `(4,28,8)`、第一个头 `(28,8)`、GPU 0。本版由每头输出梯度反向经过 attention_weights@V_heads，得到 d_attention_weights 与 d_V_heads。目标仍为逐节点手写全部 Transformer 反向和更新，并在 GPU 训练。代码贴在聊天，不修改用户主文件、不交付完整 Python 模型文件。

独立进程读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前三十三步，再执行本片段；未操作用户终端实时内存。主文件 SHA256 `cf26739833de94c946b793f115dd13847022f0fbd0017af1001e3147bacd411c`，执行前后不变。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素 CuPy float32 除以 255、标签 int64。本轮只用首张训练图、标签 5；无新增清洗、下载、划分、洗牌、增强或测试评估。输入为实际注意力权重、V_heads 和上游 d_head_output。

## 假设、变量与公式

延续图片行空间序列、单层编码器、位置均值池化与十类单图交叉熵。头数 4、位置数 28、每头宽度 8，每个头独立计算加权求和。注意力权重来自按键位置归一化的 softmax，V 来自输入投影；当前节点反向将两者视作独立输入，参数保持本次前向值。

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| P=attention_weights | (4,28,28) | 每头查询位置对键位置的权重 |
| V=V_heads | (4,28,8) | 每头各键位置的信息特征 |
| H=head_output | (4,28,8) | 每头查询位置的加权输出 |
| G=d_head_output | (4,28,8) | 输出的上游梯度 |
| d_attention_weights | (4,28,28) | 损失对 softmax 输出权重的梯度 |
| d_V_heads | (4,28,8) | 损失对 V 特征的梯度 |

```text
H[h,i,k] = sum_j P[h,i,j]*V[h,j,k]
dP[h,i,j] = sum_k G[h,i,k]*V[h,j,k] = (G@V.T)[h,i,j]
dV[h,j,k] = sum_i P[h,i,j]*G[h,i,k] = (P.T@G)[h,j,k]
```

transpose(0,2,1) 只交换每头内部最后两个轴，保留头轴。dV 已累加同一键位置被各查询使用的贡献，不需要额外平均。d_attention_weights 是对权重的梯度，尚未传回注意力分数；d_V_heads 尚未传回 W_V/b_V。两者是中间激活梯度，不新增参数，当前参数仍 9802。

## 方法选择理由（判断）

矩阵乘法链式法则分别产生两个输入梯度，基础 CuPy matmul 将四个头视为独立批次，避免头间混合和 Python 循环。明确最后两轴的转置顺序，直接保留键、查询和特征的含义。没有自动求导或现成层，直连梯度继续保存，全部反向完成后再统一更新参数。

## 本步代码

```python
# ==================================================
# 反向第十四步：注意力加权求和
# ==================================================

# 每个头分别计算：(28, 8) @ (8, 28) → (28, 28)
# 头轴保持不变，只交换 V 的位置轴和特征轴
d_attention_weights = cp.matmul(
    d_head_output,
    V_heads.transpose(0, 2, 1)
)

# 每个头分别计算：(28, 28) @ (28, 8) → (28, 8)
# 累加所有查询位置对 V 的梯度贡献
d_V_heads = cp.matmul(
    attention_weights.transpose(0, 2, 1),
    d_head_output
)

print("d_attention_weights:", d_attention_weights.shape)
print("d_V_heads:", d_V_heads.shape)
print("d_attention_weights 的实际设备:", d_attention_weights.device)
print("d_V_heads 的实际设备:", d_V_heads.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，解释器带 -B、缓存位于工作区 work/cupy_cache，本轮未调整依赖。

两组梯度 `(4,28,28)`、`(4,28,8)` 均为有限 float32、GPU 0。所有 20 组参数、当前节点前向缓存、loss、上游梯度和 d_encoder_input_skip 保持不变。

独立 CPU float64 通过 einsum('hij,hjk->hik',P,V) 重建每头加权前向，构造 sum(H*固定 G)。分别对 P 的 3136 个、V 的 896 个分量做 ±1e-5 中心差分，共 4032 分量；rtol=1e-5、atol=1e-8 下全部通过。最大绝对误差：d_attention_weights `1.460297439e-09`，d_V_heads `1.074184074e-09`。

差分中 P 作为当前矩阵乘法节点的独立输入，不强制行和为 1；这对应普通局部偏导，并不改变实际模型前向概率或直接更新权重。CPU 双精度与 GPU float32 存在舍入差异。本核验只覆盖加权求和的局部雅可比向量积，不能据此声称 softmax、注意力分数或全网络反向正确。

实际输出：

```text
d_attention_weights: (4, 28, 28)
d_V_heads: (4, 28, 8)
d_attention_weights 的实际设备: <CUDA Device 0>
d_V_heads 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU 与中心差分证据](2026-10-07_transformer_attention_weighted_backward_validation_v034.json)。JSON 保存逐字代码、来源、设备、全部实际与差分梯度、误差和输出，片段 SHA256 `a72efa862292553efcb8e7694a648390b67cdf56bdb0b2ea7f97afd32ba1dff0`。相对链接已核验，本步无新增图。

## 与前版变化、限制与下一步

v033 得到 d_head_output；本版新增 d_attention_weights 和 d_V_heads，完成注意力加权求和反向。新增参数 0、更新次数 0。保留历史记录，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录并更新索引，不重跑模型。

尚未反向通过 softmax、QK 分数、QKV 投影与输入投影，未训练或测量准确率。下一步用 softmax 的向量雅可比公式，从 d_attention_weights 计算 d_attention_scores；之后反向缩放 QK 矩阵乘法，恢复 Q/K/V 的二维布局，反向各投影并汇合输入梯度，完成输入投影后统一更新参数、组织 GPU 训练。
