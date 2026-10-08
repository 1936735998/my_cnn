# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v029。
阶段：前馈与直连路径返回梯度汇合，真实 GPU 执行与独立 CPU 两路径差分验证。本轮不是仅文档更新，未完成第一次 LayerNorm 或全网络反向、参数更新、训练。

## 范围、输入来源与处理（事实）

用户报告 dW_ff1 `(32,64)`、db_ff1 `(64,)`、d_attention_norm_from_ffn `(28,32)`，均在 GPU 0。本版将前馈路径与直连路径梯度相加，得到 d_attention_norm。目标仍为逐层手写全部 Transformer 反向、参数更新，并在 GPU 训练。代码贴在聊天，不改用户主文件、不交付完整 Python 模型文件。

独立进程读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前二十八步，再执行本片段；未操作用户终端实时变量。主文件 SHA256 `7b91895443fa1eaf6013d790a9cbfd89f8892175a2404117772e4e926914cecf`，执行前后不变。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素 CuPy float32 除以 255，标签 int64。本轮只用首张训练图、标签 5，没有新清洗、下载、划分、洗牌、增强或测试评估。本步输入为两份已计算的 `(28,32)` 路径梯度。

## 假设、变量与公式

延续图片行空间序列、单层编码器、位置均值池化和十类单图交叉熵。A=attention_norm 同时进入恒等直连与 FFN 两条路径，R=ff_residual=A+FFN(A)，下游沿第二次 LayerNorm、均值池化、分类器计算损失。所有反向保持本次前向参数值。

```text
d_attention_norm_skip: (28,32), 直连路径贡献
d_attention_norm_from_ffn: (28,32), 前馈路径贡献
d_attention_norm = d_attention_norm_skip + d_attention_norm_from_ffn: (28,32)
```

同一变量沿多条路径影响损失时，链式法则累加各路径梯度。对应元素相加，不新增位置或特征维，不进行额外平均。本步得到 A 的完整下游梯度，可作为第一次 LayerNorm 的上游梯度。新增参数 0，累计参数 9802。

## 方法选择理由（判断）

直接使用基础 CuPy 加法表达多路径链式法则，创建新数组并保留两份分支梯度，便于学习和检查。此时两条路径均已反向完成，才汇合后继续前一层；没有自动求导或现成层，参数仍不更新。

## 本步代码

```python
# ==================================================
# 反向第九步：汇合前馈与直连路径的梯度
# ==================================================

# attention_norm 同时进入两条路径，总梯度是两份贡献之和
d_attention_norm = d_attention_norm_skip + d_attention_norm_from_ffn

print("d_attention_norm_skip:", d_attention_norm_skip.shape)
print("d_attention_norm_from_ffn:", d_attention_norm_from_ffn.shape)
print("d_attention_norm:", d_attention_norm.shape)
print("d_attention_norm 的实际设备:", d_attention_norm.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，缓存使用工作区 work/cupy_cache、解释器带 -B，本轮未调整依赖。

汇合梯度 `(28,32)`、float32、GPU 0、全部有限，有独立连续存储；两份分支梯度、所有 20 组参数、相关前向缓存和 loss 均未改变。

CPU float64 独立重建 A+FFN(A)→第二次 LayerNorm→位置均值池化→分类器→稳定交叉熵，与 v028 不同，本版扰动 A 时同时改变直连与前馈路径。对 A 的全部 896 个分量做中心差分，rtol=2e-5、atol=1e-7 下全部通过。最大绝对误差 `1.179849107e-09`，相对 L2 误差 `1.002384809e-07`。

为避免 ReLU 折点，步长 min(1e-5,受影响 abs(Z)/(4*abs(W_ff1 对应系数)))，忽略零系数，实际最小/最大步长 `1e-05`/`1e-05`。基础 CPU/GPU 掩码一致，每次正负扰动均保持 ReLU 正负不变。CPU 双精度与 GPU float32 存在舍入差异。本验证覆盖 A 之后包含两条路径的局部图，不声称已经验证第一次 LayerNorm、注意力或全网络反向。

实际输出：

```text
d_attention_norm_skip: (28, 32)
d_attention_norm_from_ffn: (28, 32)
d_attention_norm: (28, 32)
d_attention_norm 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU 与双路径差分证据](2026-10-07_transformer_ffn_gradient_merge_validation_v029.json)。JSON 保存逐字片段、来源、设备、全部实际和差分梯度、步长、误差及输出，片段 SHA256 `6fc2c921a5f5824fdc9d21819fa16a94db9d7c48f1f60b4a068e1b64815c1f10`。相对链接已核验，本步无新增图。

## 与前版变化、限制与下一步

v028 得到前馈路径输入梯度；本版新增 d_attention_norm，完成前馈与直连两条路径的汇合。新增参数 0、更新次数 0。保留历史记录，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录并更新索引，不重跑模型。

尚未反向经过第一次 LayerNorm、注意力与输入投影，未训练或测量识别准确率。下一步由 d_attention_norm 反向经过第一次 LayerNorm，计算 dgamma_attn、dbeta_attn 和 d_attention_residual；继续注意力、输入投影反向，逐节点完成后统一更新参数并组织 GPU 训练。
