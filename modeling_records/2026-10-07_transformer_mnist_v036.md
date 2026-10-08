# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v036。
阶段：缩放 QK 分数手写 GPU 反向与独立 CPU 局部差分验证。本轮真实执行，不是仅文档更新；未完成 QKV 投影、全网络反向、参数更新或训练。

## 范围、来源与处理（事实）

用户报告 d_attention_scores `(4,28,28)`、GPU 0、行梯度和约 1e-10。本版反向缩放和 QK 矩阵乘法，得到 d_Q_heads 与 d_K_heads，保留此前的 d_V_heads。目标仍为逐节点手写 Transformer 全部反向与更新，最终在 GPU 训练。代码贴于聊天，不改用户主文件、不交付完整 Python 模型文件。

独立进程读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前三十五步，再执行本片段；未操作用户终端实时变量。主文件 SHA256 `c07e6e0e7489e2ffffd417baa4648d339952aed924084c0e13f6b41e6715096d`，执行前后不变。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素 CuPy float32 除以 255、标签 int64。本轮只用首张训练图、标签 5，无新增清洗、下载、划分、洗牌、增强或测试评估。输入是实际 Q_heads、K_heads 和分数梯度。

## 假设、变量与公式

延续图片行空间序列、单层编码器、位置均值池化、十类单图交叉熵。头数 4，每头特征数 8，位置数 28，各头独立计算。缩放常数 a=sqrt(head_dim)，不训练，所有参数保持本次前向值。

```text
Q=Q_heads, K=K_heads: (4,28,8)
S=attention_scores=Q@K.T/a: (4,28,28)
G=d_attention_scores: (4,28,28)
D=d_qk_product=G/a: (4,28,28)
dQ[h,i,k]=sum_j D[h,i,j]*K[h,j,k] = (D@K)[h,i,k]
dK[h,j,k]=sum_i D[h,i,j]*Q[h,i,k] = (D.T@Q)[h,j,k]
d_Q_heads, d_K_heads: (4,28,8)
```

先除以缩放因子一次，再计算两个输入梯度。D.transpose(0,2,1) 交换查询/键轴，保留头轴；dK 已对应原始 K_heads，不再转置输出。Q/K/V 是中间特征，参数 W_Q/W_K/W_V 的梯度留到后续投影反向。新增参数 0，当前参数仍 9802。

## 方法选择理由（判断）

将常数缩放与矩阵乘法两部分按逆序处理，明确缩放只应用一次。基础 CuPy matmul 将头轴作为独立批次，用矩阵乘法完成各位置贡献的累加，无自动求导或现成层。d_V_heads 与直连梯度继续保存，待所有输入路径反向后汇合。

## 本步代码

```python
# ==================================================
# 反向第十六步：缩放 QK 分数
# ==================================================

# 前向除以 sqrt(head_dim)，反向也先除以同一个常数
attention_scale = cp.sqrt(cp.float32(head_dim))
d_qk_product = d_attention_scores / attention_scale

# 每个头分别计算：(28, 28) @ (28, 8) → (28, 8)
d_Q_heads = cp.matmul(d_qk_product, K_heads)

# 交换分数梯度的查询轴和键轴，累加各查询对 K 的贡献
# 每个头分别计算：(28, 28) @ (28, 8) → (28, 8)
d_K_heads = cp.matmul(
    d_qk_product.transpose(0, 2, 1),
    Q_heads
)

print("attention_scale:", float(attention_scale))
print("d_qk_product:", d_qk_product.shape)
print("d_Q_heads:", d_Q_heads.shape)
print("d_K_heads:", d_K_heads.shape)
print("d_Q_heads 的实际设备:", d_Q_heads.device)
print("d_K_heads 的实际设备:", d_K_heads.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，解释器带 -B、缓存位于工作区 work/cupy_cache，本轮未调整依赖。

缩放常数为 float32、GPU 0、标量，值 `2.8284270763397217`；中间 D `(4,28,28)`，Q/K 梯度 `(4,28,8)`，均为有限 float32、GPU 0。所有 20 组参数、相关前向缓存、loss、上游梯度、V 梯度和直连梯度均未改变。

独立 CPU float64 使用 einsum('hik,hjk->hij',Q,K)/sqrt(8) 重建缩放前向，构造 sum(S*固定 G)，不复用反向公式。分别对 Q、K 各 896 项做 ±1e-5 中心差分，共 1792 项；rtol=1e-5、atol=1e-8 下全部通过。最大绝对误差：d_Q_heads `9.512154955e-11`，d_K_heads `1.161914003e-10`。

CPU sqrt(8) 为 `2.8284271247461903`，与 GPU float32 缩放有微小舍入差，矩阵累加精度也不同。本验证覆盖缩放 QK 节点的局部雅可比向量积，不代表 QKV 投影或全网络反向已经正确。

实际输出：

```text
attention_scale: 2.8284270763397217
d_qk_product: (4, 28, 28)
d_Q_heads: (4, 28, 8)
d_K_heads: (4, 28, 8)
d_Q_heads 的实际设备: <CUDA Device 0>
d_K_heads 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU 与中心差分证据](2026-10-07_transformer_scaled_qk_backward_validation_v036.json)。JSON 保存逐字代码、来源、设备、全部实际和差分梯度、误差、缩放值及输出，片段 SHA256 `bbcffa857556e822eb0908b84fd6078799c4fee004daab372fcf8ea7639ed999`。相对链接已核验，本步无新增图。

## 与前版变化、限制与下一步

v035 完成注意力 softmax 反向；本版新增 attention_scale、d_qk_product、d_Q_heads、d_K_heads，完成缩放 QK 分数反向。新增参数 0、更新次数 0。保留历史记录，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录并更新索引，不重跑模型。

尚未反向通过 QKV 拆头、投影与输入投影，未训练或测量识别准确率。下一步将 Q/K/V 三份 `(4,28,8)` 梯度逆转拆头，还原成 d_Q、d_K、d_V `(28,32)`；反向三个投影并汇合 encoder_input 直连梯度，完成输入投影后统一更新参数、组织 GPU 训练。
