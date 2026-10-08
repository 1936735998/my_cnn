# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v025。
阶段：前馈残差相加节点手写反向，真实 GPU 执行与 CPU 差分验证。本轮不是仅文档更新，未完成前馈支路、全网络反向或参数更新。

## 范围、来源与处理（事实）

用户报告第二次 LayerNorm 的 dgamma_ffn、dbeta_ffn 为 `(32,)`、d_ff_residual 为 `(28,32)`，均在 GPU 0。本版只将残差相加的输出梯度传入两条支路，保存直连支路对 attention_norm 的贡献与 ff_output 的梯度。目标仍是按小步手写 Transformer 的全部反向与参数更新，最终在 GPU 训练。交付代码贴在聊天，不改用户主文件，不交付完整 Python 模型。

独立进程读取并执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前二十四步，然后执行本片段；未操作用户终端中的实时变量。主文件 SHA256 `58c35b84bf1c96babadb61f777e60545b8491dbb25deddb892520b41db2fe2e6`，执行前后一致。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 张，像素 CuPy float32 除以 255、标签 int64。本轮只用首张训练图，标签 5，没有额外清洗、下载、洗牌、增强、数据重划分或测试评估。输入为实际 attention_norm、ff_output 及已保存的 d_ff_residual。

## 假设、变量与公式

延续图片行作为空间位置、单层编码器、位置均值池化与十类单样本交叉熵。前馈残差两项形状均 `(28,32)`，相加不含缩放、广播或额外权重。所有参数保持本次前向值。

```text
A = attention_norm, B = ff_output, R = ff_residual = A+B
G = d_ff_residual: (28,32)
∂R[i,k]/∂A[i,k] = 1, ∂R[i,k]/∂B[i,k] = 1
d_attention_norm_skip = G: (28,32)
d_ff_output = G: (28,32)
后续 d_attention_norm_total = d_attention_norm_skip + d_attention_norm_from_ffn
```

两支路各接收完整上游梯度，不除以 2、不取平均。d_attention_norm_skip 只代表直连路径的贡献；ff_output 在完整前向中也依赖 attention_norm，须在前馈网络反向完成后把该路径贡献累加，才能得到 attention_norm 的总梯度。本版未生成或声称生成这个总梯度。

## 方法选择理由（判断）

基础加法的偏导为 1，因此直接复制上游梯度。两次 copy 为各支路分配独立存储，后续支路运算可独立进行而不影响上游或另一支路。用 skip 命名标明部分梯度，避免误把直连贡献当成最终总梯度。暂不更新参数，保证后续反向使用与前向一致的权重。没有自动求导或现成模型层。

## 本步代码

```python
# ==================================================
# 反向第五步：前馈残差连接
# ==================================================

# 直连支路：先保存传给 attention_norm 的这部分梯度
d_attention_norm_skip = d_ff_residual.copy()

# 前馈支路：传给 ff_output，接下来继续反向经过前馈网络
d_ff_output = d_ff_residual.copy()

print("d_attention_norm_skip:", d_attention_norm_skip.shape)
print("d_ff_output:", d_ff_output.shape)
print("d_attention_norm_skip 的实际设备:", d_attention_norm_skip.device)
print("d_ff_output 的实际设备:", d_ff_output.device)
```

## 实际执行与验证（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，缓存位于工作区 work/cupy_cache，解释器带 -B；本轮未调整依赖。

两组梯度均 `(28,32)`、float32、GPU 0、全部有限，并逐元素等于上游 d_ff_residual。两支路与上游共有三个不同的连续存储地址。所有 20 组参数、本步相关前向缓存、loss 和上游梯度未改变；参数数目仍 9802。

独立 CPU float64 重建局部图 A+B→第二次 LayerNorm→位置均值池化→分类器→稳定交叉熵。把 A 与 B 当成相加节点的两个独立输入：扰动 A 时保持 B 不变，扰动 B 时保持 A 不变。分别对 A、B 各 896 个分量做 ±1e-5 中心差分，总 1792 个分量；rtol=2e-5、atol=1e-7 下全部通过。直连支路最大绝对误差 `8.759499788e-10`，ff_output 支路 `8.759499788e-10`。

CPU 双精度重新计算相加与后续图，和 GPU float32 存在舍入差异。本核验检查相加节点的两组局部输入偏导；固定 B 的 A 差分不包含原网络 B=FFN(A) 的依赖，不能据此声称验证了 attention_norm 总梯度或 FFN 内部反向。

实际输出：

```text
d_attention_norm_skip: (28, 32)
d_ff_output: (28, 32)
d_attention_norm_skip 的实际设备: <CUDA Device 0>
d_ff_output 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU 与差分证据](2026-10-07_transformer_ffn_residual_backward_validation_v025.json)。JSON 保存逐字代码、来源、设备、全部实际与差分梯度、误差与输出，片段 SHA256 `88aa71efe064bd7eb42cb8c709bac7a95c5af43fed9c7aa01872a59069cf47c9`。相对链接已核验，本步无新增图。

## 版本变化、限制与下一步

v024 完成第二次 LayerNorm 反向；本版新增 d_attention_norm_skip 与 d_ff_output，完成前馈残差相加节点反向。新增参数 0、参数更新 0。保留历史记录，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录和更新索引，不再次执行模型。

尚未进行前馈网络内部、第一次 LayerNorm、注意力与输入投影反向，未训练或测量准确率。下一步从 d_ff_output 反向经过 ff_output=ff_hidden@W_ff2+b_ff2，计算第二层前馈权重、偏置和 ff_hidden 梯度，逐节点反向后汇合直连支路，最终统一参数更新并组织 GPU 训练。
