# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v033。
阶段：多头合并操作手写 GPU 反向、独立块映射与 CPU 局部差分验证。本轮真实执行，不是仅文档更新；未完成注意力加权、全网络反向、参数更新或训练。

## 范围、来源与处理（事实）

用户报告 dW_O `(32,32)`、db_O `(32,)`、d_concatenated_heads `(28,32)`，均在 GPU 0。本版逆转前向多头合并，将拼接特征梯度还原到 d_head_output `(4,28,8)`。目标仍为小步手写 Transformer 的全部反向与更新，并在 GPU 训练。代码贴于聊天，不改用户主文件、不交付完整 Python 模型文件。

独立进程读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前三十二步，再执行本片段；未操作用户终端实时内存。主文件 SHA256 `3afd4578e10f6cb3a731a6c82f7893a68aaa5441fa8cb507134eeac24e8c7e4e`，执行前后不变。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素 CuPy float32 除以 255、标签 int64。本轮仅首张训练图、标签 5，无新增清洗、下载、划分、洗牌、增强或测试评估。输入为实际多头前向布局和 d_concatenated_heads。

## 假设、变量与公式

延续图片行空间序列、单层编码器、位置均值池化、十类单图交叉熵。头数 4、每头宽 8、位置数 28、模型宽 32。前向每个位置的特征按头 0、1、2、3 顺序拼接，元素没有额外缩放。

```text
前向：head_output (4,28,8) -> transpose(1,0,2) (28,4,8) -> reshape (28,32)
C[i,h*8+k] = H[h,i,k]
反向：G (28,32) -> reshape (28,4,8) -> transpose(1,0,2) (4,28,8)
d_head_output[h,i,k] = d_concatenated_heads[i,h*8+k]
```

中间 d_head_output_by_position 的轴是位置、头、每头特征；最终 d_head_output 的轴是头、位置、每头特征。反向仅重新排列已有梯度，不求和、不缩放。新增参数 0，累计参数仍 9802。

## 方法选择理由（判断）

按前向操作的逆序先恢复每个位置内的头分组，再交换头和位置轴，直接体现排列操作的反向规则。最终 copy 物化为独立连续数组，便于后续矩阵乘法和梯度操作；中间 reshape 视图只用于读取。使用基础 CuPy 操作，没有自动求导或现成层，参数保持前向值。

## 本步代码

```python
# ==================================================
# 反向第十三步：逆转多头拼接
# ==================================================

# 每个位置的 32 个特征，重新分成 4 个头，每个头 8 个特征
# (28, 32) → (28, 4, 8)
d_head_output_by_position = d_concatenated_heads.reshape(
    seq_len, num_heads, head_dim
)

# 把头的轴移到最前面：(28, 4, 8) → (4, 28, 8)
d_head_output = d_head_output_by_position.transpose(1, 0, 2).copy()

print("d_head_output_by_position:", d_head_output_by_position.shape)
print("d_head_output:", d_head_output.shape)
print("第 0 个头的输出梯度:", d_head_output[0].shape)
print("d_head_output 的实际设备:", d_head_output.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，解释器带 -B、缓存在工作区 work/cupy_cache，本轮未调整依赖。

中间梯度 `(28,4,8)`，最终 `(4,28,8)`，均为 float32、GPU 0；最终全部有限并具有独立连续存储。所有 20 组参数、相关前向缓存、loss、上游梯度及 d_encoder_input_skip 未改变。

独立 CPU 参考从 G 每行的 0:8、8:16、16:24、24:32 四个特征块逐头切片并 stack，全部元素与 GPU 最终结果精确相等。再用逐头 concatenate 前向重组梯度，与原 G 精确相等，确认头顺序和位置对应关系。

CPU float64 使用实际 H 与固定 G 构造 sum(concatenate([H[h] for h],axis=-1)*G)，逐头重建前向而不复用被测 reshape/transpose。对全部 896 个 H 分量做 ±1e-5 中心差分，rtol=1e-6、atol=1e-9 下全部通过；最大绝对误差 `6.612488335e-13`，相对 L2 误差 `9.192070752e-11`。这是多头合并节点的局部雅可比向量积验证，不代表注意力加权或全网络反向已经正确。

实际输出：

```text
d_head_output_by_position: (28, 4, 8)
d_head_output: (4, 28, 8)
第 0 个头的输出梯度: (28, 8)
d_head_output 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU、块映射与差分证据](2026-10-07_transformer_multihead_unmerge_backward_validation_v033.json)。JSON 保存逐字代码、来源、设备、全部实际与差分梯度、误差及输出，片段 SHA256 `580c7b065d3ee32f3a48f670723aacb45e6d7aaa14cdcbd254ec7db2d55f33c7`。相对链接已核验，本步无新增图。

## 与前版变化、限制与下一步

v032 得到 d_concatenated_heads；本版新增 d_head_output_by_position 与 d_head_output，完成多头合并反向。新增参数 0、更新次数 0。保留历史记录，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录并更新索引，不重跑模型。

尚未反向经过 attention_weights@V_heads、softmax、QK 分数、QKV 投影或输入投影，未训练或测量准确率。下一步从 head_output=attention_weights@V_heads 计算注意力权重和 V_heads 的梯度，继续反向 softmax 和 QK 分数，再汇合 QKV 输入梯度与直连支路，完成输入投影后统一更新参数并组织 GPU 训练。
