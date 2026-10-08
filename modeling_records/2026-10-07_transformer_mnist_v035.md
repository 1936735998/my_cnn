# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v035。
阶段：注意力 softmax 手写 GPU 反向、独立 CPU 差分及少量稳定性核验。本轮真实执行，不是仅文档更新；未完成 QK 分数、全网络反向、参数更新或训练。

## 范围、来源与处理（事实）

用户报告 d_attention_weights `(4,28,28)`、d_V_heads `(4,28,8)`，均在 GPU 0。本版从权重梯度反向 softmax，计算 d_attention_scores；V 梯度继续保存。目标仍为逐节点手写 Transformer 全部反向和更新，并在 GPU 训练。代码贴在聊天，不改用户主文件、不交付完整 Python 模型文件。

独立进程读取执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前三十四步，再执行本片段；未操作用户终端实时变量。主文件 SHA256 `afbabeb364b2b7a3c5e6ca37984aebf93f9e214a26f4a71b4bdcba620acf4797`，执行前后不变。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素 CuPy float32 除以 255、标签 int64。本轮只用首张训练图、标签 5；无新增清洗、下载、划分、洗牌、增强或测试评估。输入为实际注意力分数、权重和上游梯度；额外三条小测试行仅用于数值核验。

## 假设、变量与公式

延续图片行空间序列、单层编码器、位置均值池化与十类单图交叉熵。每头每个查询行沿最后的键位置轴计算无掩码 softmax，使用实际前向权重缓存，参数保持前向值。

```text
S=attention_scores: (4,28,28)
P=attention_weights=softmax(S,keys): (4,28,28)
G=d_attention_weights: (4,28,28)
c[h,i,0]=sum_j P[h,i,j]*G[h,i,j]: (4,28,1)
d_attention_scores[h,i,j]=P[h,i,j]*(G[h,i,j]-c[h,i,0]): (4,28,28)
```

softmax 雅可比含同一行内不同键位置的交叉项，使用上式直接计算反向向量积，避免构造每行 28×28 雅可比。这里是加权求和而非均值，不额外除以 28。整行共同平移不改变输出，因此分数梯度在每行的和应约为零。没有增加 epsilon、概率裁剪或额外缩放；1/sqrt(head_dim) 属于下一步 QK 分数反向。新增参数 0，当前参数仍 9802。

## 方法选择理由（判断）

用基础 CuPy sum 和逐元素乘法直接实现一般 softmax 链式法则，keepdims=True 保留行维便于广播，头和查询行分别计算。前向减最大值是稳定 softmax 的等价表达，本公式依据实际输出概率，不必显式反向 max 中间量。没有自动求导或现成层。

## 本步代码

```python
# ==================================================
# 反向第十五步：注意力 softmax
# ==================================================

# 对每个头、每个查询行，沿 28 个键位置计算加权和
# (4, 28, 28) → (4, 28, 1)
weighted_gradient_sum = cp.sum(
    d_attention_weights * attention_weights,
    axis=-1,
    keepdims=True
)

# softmax 反向：每个权重乘以对应上游梯度与该行加权和之差
d_attention_scores = attention_weights * (
    d_attention_weights - weighted_gradient_sum
)

print("weighted_gradient_sum:", weighted_gradient_sum.shape)
print("d_attention_scores:", d_attention_scores.shape)
print("第 0 个头前 5 行的梯度和:", cp.sum(
    d_attention_scores, axis=-1
)[0, :5])
print("d_attention_scores 的实际设备:", d_attention_scores.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，解释器带 -B、缓存在工作区 work/cupy_cache，本轮未调整依赖。

weighted_gradient_sum `(4,28,1)`，d_attention_scores `(4,28,28)`，均为有限 float32、GPU 0。各行梯度和最大绝对值 `5.820766091e-10`，通过绝对容差 2e-8 检查。所有 20 组参数、相关前向缓存、上游梯度、loss、d_V_heads 与直连梯度均未改变。

CPU float64 独立使用 logaddexp.reduce 沿键轴重算 softmax，构造 sum(softmax(S)*固定 G)，不复用 GPU 概率或反向公式。对 S 全部 3136 分量做 ±1e-5 中心差分，rtol=2e-5、atol=1e-8 下全部通过；最大绝对误差 `4.635158923e-11`，相对 L2 误差 `1.119618002e-07`。CPU 双精度重算与 GPU float32 缓存存在舍入差异。这是 softmax 节点的局部雅可比向量积核验，不是完整网络梯度核验。

额外三条小测试行使用同一聊天片段。整数分数 [-2,-1,0,1]、[-2,-1,0,1]、[1000,-1000,-1000,-1000]，上游分别 [0.2,-0.3,0.7,-0.1]、[1,1,1,1]、[0.2,-0.3,0.7,-0.1]。整批加 10000 后，概率和反向值精确相同；这些整数分数可精确表示，避免一般 float32 小数平移造成的低位丢失。常数上游行梯度在容差内为零，极端分数行概率为 [1,0,0,0]、梯度全零且有限，验证稳定性与数值饱和情况。

实际输出：

```text
weighted_gradient_sum: (4, 28, 1)
d_attention_scores: (4, 28, 28)
第 0 个头前 5 行的梯度和: [-1.7462298e-10  5.8207661e-10 -3.4924597e-10  4.6566129e-10
 -4.0745363e-10]
d_attention_scores 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU、差分与稳定性证据](2026-10-07_transformer_attention_softmax_backward_validation_v035.json)。JSON 保存逐字代码、来源、设备、全部实际和差分梯度、误差、行和及小测试输出，片段 SHA256 `9f37e6e27765bf4f9497b74db257d13055d79a880ce06bdb37c7128d5fdbd0c0`。相对链接已核验，本步无新增图。

## 与前版变化、限制与下一步

v034 得到权重与 V 梯度；本版新增 weighted_gradient_sum 与 d_attention_scores，完成注意力 softmax 反向。新增参数 0、更新次数 0。保留历史记录，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录并更新索引，不重跑模型。

尚未反向经过 scaled QK 分数、QKV 投影或输入投影，未训练或测量准确率。下一步从 attention_scores=Q_heads@K_heads.T/sqrt(head_dim) 计算 d_Q_heads 与 d_K_heads，连同已保存的 d_V_heads 还原二维布局，反向 QKV 投影并汇合 encoder_input 直连梯度，完成输入投影后统一更新参数、组织 GPU 训练。
