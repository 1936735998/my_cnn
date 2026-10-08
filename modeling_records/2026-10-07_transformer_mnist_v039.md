# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v039。
阶段：编码器输入四路径梯度汇合与独立 CPU 完整下游图差分。本轮真实执行，不是仅文档更新；未完成输入投影反向、全部参数整图差分、参数更新或训练。

## 范围、来源与处理（事实）

用户报告 Q/K/V 三个投影的权重 `(32,32)`、偏置 `(32,)`、路径输入梯度 `(28,32)`，均在 GPU 0。本版将这三份输入梯度与直连梯度相加，得到完整的 d_encoder_input。目标仍为逐节点手写 Transformer 全部反向和更新，最终 GPU 训练。模型代码贴在聊天，不改用户主文件、不交付完整 Python 模型文件。

独立进程读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前三十八步，再执行本片段；未操作用户终端实时内存。主文件 SHA256 `7e7ab8a1fffd93de85e303a2be086804b504e7dbb7b48fe1677468ff63704937`，执行前后不变。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素 CuPy float32 除以 255、标签 int64。本轮仅首张训练图、标签 5，无新增清洗、下载、划分、洗牌、增强或测试评估。输入为实际 encoder_input 和四份路径梯度。

## 假设、变量与公式

延续图片行空间序列、单层编码器、位置均值池化与十类单图交叉熵。encoder_input=E 在当前图中进入 Q/K/V 三个投影和注意力残差直连共四条路径，全部参数保持同次前向值。

```text
d_encoder_input_skip: (28,32)
d_encoder_input_from_Q/K/V: 各 (28,32)
d_encoder_input = d_encoder_input_skip
                  + d_encoder_input_from_Q
                  + d_encoder_input_from_K
                  + d_encoder_input_from_V: (28,32)
```

多路径链式法则将同一输入的各路径贡献逐元素累加，位置维和特征维都保留，不额外平均或缩放。注意力内部跨位置贡献已在各矩阵乘法中累加。新增参数 0，累计参数仍 9802。

## 方法选择理由（判断）

显式相加四份已完成的路径梯度，产生新数组并保留各分支，便于审查贡献是否完整。使用基础 CuPy 加法，无自动求导或现成层。独立 CPU 参考重新计算 E 之后的完整前向，检查汇合及各层链式传播在当前样本的一致性。

## 本步代码

```python
# ==================================================
# 反向第十九步：汇合 encoder_input 的四条路径
# ==================================================

# 同一个输入通过直连、Q、K、V 四条路径影响损失
d_encoder_input = (
    d_encoder_input_skip
    + d_encoder_input_from_Q
    + d_encoder_input_from_K
    + d_encoder_input_from_V
)

print("d_encoder_input:", d_encoder_input.shape)
print("一行输入特征的梯度:", d_encoder_input[0].shape)
print("d_encoder_input 的实际设备:", d_encoder_input.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，解释器带 -B、缓存在工作区 work/cupy_cache，本轮未调整依赖。

汇合梯度 `(28,32)`、float32、GPU 0、全部有限，拥有独立连续存储。所有 20 组参数、前向输入、loss 及四份分支梯度均未改变。

新增工作区验证用 NumPy float64 前向参考，从 E 重算三投影、显式特征切块、einsum 缩放 QK、logaddexp softmax、einsum 加权、逐头 concatenate、输出投影与残差、两次 LayerNorm、前馈、池化和分类交叉熵。该参考无自动求导、无训练、无文件副作用，不修改用户模型；源代码与 SHA256 `ca6dd807884196dcaedfd5550f297daa67b95b94d57ec634d2a92e38e1770eea` 全部保存到 JSON，便于复核和后续输入映射检查。

CPU 基础损失 `1.97143177282`，GPU float32 损失 `1.97143173218`，绝对差 `4.06468752e-08`，通过前向一致性检查。CPU/GPU 基础 ReLU 掩码一致，当前无 CPU 零预激活。对 E 的 896 个分量逐项做中心差分，从 ±1e-5 开始，如扰动跨 ReLU 折点则折半，全部最终扰动保持掩码。实际步长范围 `1e-05`～`1e-05`，折半次数 0。

rtol=2e-5、atol=2e-7 下全部 896 项通过，最大绝对误差 `2.904702257e-09`，相对 L2 误差 `1.342988341e-07`。CPU 独立双精度与 GPU float32 存在舍入差异。此核验覆盖当前样本 E 之后整个图的输入梯度，不等同于全部参数整图差分，也未包括输入仿射及位置相加反向。

实际输出：

```text
d_encoder_input: (28, 32)
一行输入特征的梯度: (32,)
d_encoder_input 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU、完整参考源代码与差分证据](2026-10-07_transformer_encoder_input_merge_validation_v039.json)。JSON 保存逐字片段、来源、设备、全部实际与差分梯度、步长、误差及输出，片段 SHA256 `c0a5ec143dcf800c0c26cc81d5de5aa18c0dc63646adb926ba3210681611859e`。相对链接已核验，本步无新增图。

## 与前版变化、限制与下一步

v038 完成三个 QKV 投影反向；本版新增 d_encoder_input，完成四路径汇合，并核验从 E 到分类损失的完整下游输入梯度。新增参数 0、更新次数 0。历史记录保留，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录并更新索引，不重跑模型。

尚未反向经过 encoder_input=embedding+position_encoding 或输入投影，未训练或测量识别准确率。下一步把 d_encoder_input 传给 embedding，位置编码固定；然后计算 W_in/b_in 及像素输入梯度，完成全部反向后统一更新参数并组织 GPU 训练。
