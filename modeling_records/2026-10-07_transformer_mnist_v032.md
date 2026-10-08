# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v032。
阶段：注意力输出投影手写 GPU 反向与独立 CPU 局部差分验证。本轮真实执行，不是仅文档更新；未完成多头内部、全网络反向、参数更新或训练。

## 问题范围、来源与处理（事实）

用户报告 d_encoder_input_skip 和 d_attention_output 均 `(28,32)`、GPU 0。本版从注意力输出梯度反向通过输出投影，计算 dW_O、db_O 与 d_concatenated_heads。目标仍为逐节点手写全部 Transformer 反向和更新，最终 GPU 训练。代码贴在聊天，不改用户主文件、不交付完整 Python 模型文件。

独立进程读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前三十一步，再执行本片段；未操作用户终端实时内存。主文件 SHA256 `8388717adcbc87d34c00b8dae36849f2289d8abfbe8002ade6570a56e7a01f59`，执行前后不变。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 张，像素 CuPy float32 除以 255、标签 int64。只用首张训练图，标签 5；无新增下载、清洗、划分、洗牌、增强或测试评估。输入为实际拼接多头特征、输出投影参数和 d_attention_output。

## 假设、变量与公式

延续图片行空间序列、单层编码器、位置均值池化与十类单图交叉熵。输出投影为线性映射，28 行共享参数，随后与 encoder_input 相加。反向保持本次前向权重与缓存。

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| C=concatenated_heads | (28,32) | 四个头拼接后的特征 |
| W=W_O | (32,32) | 输出投影共享权重 |
| b=b_O | (32,) | 输出投影共享偏置 |
| G=d_attention_output | (28,32) | 投影输出的上游梯度 |
| dW_O | (32,32) | 权重梯度 |
| db_O | (32,) | 偏置梯度 |
| d_concatenated_heads | (28,32) | 拼接特征梯度 |

```text
前向：attention_output = C@W+b
dW[j,k] = sum_i C[i,j]*G[i,k] = (C.T@G)[j,k]
db[k] = sum_i G[i,k]
dC[i,j] = sum_k G[i,k]*W[j,k] = (G@W.T)[i,j]
```

共享参数梯度累加位置贡献，不额外平均；输入梯度使用当前前向 W_O。直连支路 d_encoder_input_skip 继续保存，尚未汇合注意力返回的梯度。新增参数 0，当前参数仍 9802。

## 方法选择理由（判断）

沿用线性层链式法则，用基础 CuPy dot/sum 实现权重、偏置和输入梯度。显式写出转置及维度，即使输入与输出维度同为 32，也能分清各轴含义。没有自动求导或现成层，保持参数不变直到全网络反向完成。

## 本步代码

```python
# ==================================================
# 反向第十二步：注意力输出投影
# ==================================================

# (32, 28) @ (28, 32) → (32, 32)
# 累加 28 行对共享权重的梯度贡献
dW_O = cp.dot(concatenated_heads.T, d_attention_output)

# 输出偏置在 28 行之间共享，沿位置轴求和
db_O = cp.sum(d_attention_output, axis=0)

# (28, 32) @ (32, 32) → (28, 32)
# 使用本次前向的 W_O，把梯度传回拼接后的多头特征
d_concatenated_heads = cp.dot(d_attention_output, W_O.T)

print("dW_O:", dW_O.shape)
print("db_O:", db_O.shape)
print("d_concatenated_heads:", d_concatenated_heads.shape)
print("dW_O 的实际设备:", dW_O.device)
print("db_O 的实际设备:", db_O.device)
print("d_concatenated_heads 的实际设备:", d_concatenated_heads.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，解释器带 -B、缓存在工作区 work/cupy_cache，本轮未调整依赖。

三组梯度 `(32,32)`、`(32,)`、`(28,32)` 均为有限 float32、GPU 0。所有 20 组参数、相关前向输入、loss、上游梯度和直连支路梯度均未改变。

CPU float64 独立重建 C@W+b→加固定 encoder_input→第一次 LayerNorm→前馈及残差→第二次 LayerNorm→位置均值池化→分类器→稳定交叉熵。对 1024 个权重、32 个偏置、896 个输入分量做中心差分，共 1952 分量；rtol=2e-5、atol=2e-7 下全部通过。

CPU/GPU 基础 ReLU 掩码一致。差分从 ±1e-5 开始，若任一方向改变掩码则折半，最多 20 次；所有使用的扰动均保持掩码。实际步长及折半次数保存于 JSON。最大绝对误差：dW_O `2.411074362e-08`，db_O `2.779740171e-08`，d_concatenated_heads `2.721417758e-09`。

CPU 独立双精度与 GPU float32 存在舍入差异。本核验只覆盖当前输出投影三组梯度，拼接特征被视为独立输入，不代表已经反向通过多头合并、注意力加权或 QKV 路径。

实际输出：

```text
dW_O: (32, 32)
db_O: (32,)
d_concatenated_heads: (28, 32)
dW_O 的实际设备: <CUDA Device 0>
db_O 的实际设备: <CUDA Device 0>
d_concatenated_heads 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU 与中心差分证据](2026-10-07_transformer_attention_projection_backward_validation_v032.json)。JSON 保存逐字代码、来源、设备、全部实际与差分梯度、步长、误差及输出，片段 SHA256 `745a634cdb4a15d1071a9a00c9532cb9d56947da607d2f773737a7c5f1643451`。相对链接已核验，本步无新增图。

## 与前版变化、限制与下一步

v031 得到 d_attention_output；本版新增 dW_O、db_O、d_concatenated_heads，完成注意力输出投影反向。新增参数 0、更新次数 0。保留历史记录，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录并更新索引，不重跑模型。

尚未反向通过多头合并、加权求和、softmax、QK 分数、QKV 投影与输入投影，未训练或测量识别准确率。下一步逆转多头拼接，按前向顺序把 `(28,32)` 梯度还原为 `(4,28,8)` 的 d_head_output；继续注意力内部反向，汇合输入梯度并完成输入投影反向后统一更新参数、组织 GPU 训练。
