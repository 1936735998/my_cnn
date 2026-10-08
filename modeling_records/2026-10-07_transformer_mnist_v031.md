# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v031。
阶段：注意力残差相加节点手写 GPU 反向与独立 CPU 下游图差分验证。本轮真实执行，不是仅文档更新；未完成注意力内部、全网络反向、参数更新或训练。

## 问题范围、来源与处理（事实）

用户报告第一次 LayerNorm 的参数梯度 `(32,)`、d_attention_residual `(28,32)`，均为 GPU 0。本版将 d_attention_residual 分别传给 encoder_input 的直连支路和 attention_output，继续小步手写 Transformer 的全部反向与更新，最终在 GPU 训练。代码贴在聊天，不改用户主文件，不交付完整 Python 模型文件。

独立进程读取并执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前三十步，再执行本片段；未操作用户终端实时变量。主文件 SHA256 `0f72ddfd3241ef69e45e8f1cecbd9528ae3f01f4f90d401173db94b7697b05cb`，执行前后不变。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 张，像素 CuPy float32 除以 255、标签 int64。本轮只用首张训练图、标签 5；无新增下载、清洗、划分、洗牌、增强或测试评估。输入为相加节点的实际两项及输出梯度。

## 假设、变量与公式

延续图片行空间序列、单层编码器、位置均值池化与十类单图交叉熵。注意力残差两项同为 `(28,32)`，直接相加，不含缩放、加权或额外广播。所有反向继续使用本次前向参数。

```text
A=encoder_input, B=attention_output, R=attention_residual=A+B
G=d_attention_residual: (28,32)
∂R[i,k]/∂A[i,k]=1, ∂R[i,k]/∂B[i,k]=1
d_encoder_input_skip=G: (28,32)
d_attention_output=G: (28,32)
```

两支路各接收完整上游梯度。d_encoder_input_skip 只是恒等直连贡献；原网络 attention_output 还通过 Q/K/V 等路径依赖 encoder_input，待完整注意力反向后再累加该部分，才能得到 encoder_input 总梯度。本版未生成或宣称生成总梯度。新增参数 0，累计参数 9802。

## 方法选择理由（判断）

基础相加节点的局部导数为恒等映射，使用两次 CuPy copy 保存独立支路梯度，方便后续各自计算，避免共享存储影响上游或另一支路。skip 命名明确标出直连贡献，暂不汇合尚未计算的注意力路径。没有自动求导或现成层，参数保持前向值。

## 本步代码

```python
# ==================================================
# 反向第十一步：注意力残差连接
# ==================================================

# 直连支路：先保存传给 encoder_input 的这部分梯度
d_encoder_input_skip = d_attention_residual.copy()

# 注意力支路：传给 attention_output，接下来继续反向经过注意力
d_attention_output = d_attention_residual.copy()

print("d_encoder_input_skip:", d_encoder_input_skip.shape)
print("d_attention_output:", d_attention_output.shape)
print("d_encoder_input_skip 的实际设备:", d_encoder_input_skip.device)
print("d_attention_output 的实际设备:", d_attention_output.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，解释器带 -B、缓存在工作区 work/cupy_cache，本轮未修改依赖。

两份支路梯度均 `(28,32)`、float32、GPU 0、全有限，逐元素等于上游 d_attention_residual。两支路及上游有三个不同的连续存储地址。所有 20 组参数、相关前向缓存、loss 与上游梯度均未改变。

CPU float64 独立重建 A+B→第一次 LayerNorm→前馈及残差→第二次 LayerNorm→位置均值池化→分类器→稳定交叉熵，将 A、B 看作相加节点的两个独立输入。扰动 A 时固定 B，扰动 B 时固定 A，对每组各 896 项做中心差分，共 1792 项；rtol=2e-5、atol=1e-7 下全部通过。

CPU/GPU 基础 ReLU 掩码一致；差分从 ±1e-5 开始，如果任一方向改变基础掩码则折半，最多 20 次，所有最终使用的扰动均保持掩码。各组实际步长、折半次数记录在 JSON。直连支路最大绝对误差 `2.14172502e-09`，attention_output 支路 `2.14172502e-09`。

CPU 双精度与 GPU float32 存在舍入差异。固定 B 的 A 差分只检查恒等直连贡献，不包含原网络 B=Attention(A) 的依赖；本核验不是 encoder_input 总梯度或注意力内部反向验证。

实际输出：

```text
d_encoder_input_skip: (28, 32)
d_attention_output: (28, 32)
d_encoder_input_skip 的实际设备: <CUDA Device 0>
d_attention_output 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU 与中心差分证据](2026-10-07_transformer_attention_residual_backward_validation_v031.json)。JSON 保存逐字代码、来源、设备、全部实际与差分梯度、步长、误差及输出，片段 SHA256 `0b6b83ebf820f36029a5afc9a163d00cb4ce8f695f35a9d6a785068989e4264c`。相对链接已核验，本步无新增图。

## 与前版变化、限制与下一步

v030 完成第一次 LayerNorm 反向；本版新增 d_encoder_input_skip 与 d_attention_output，完成注意力残差相加节点反向。新增参数 0、更新次数 0。保留历史记录，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录并更新索引，不重跑模型。

尚未进行注意力输出投影、多头合并、加权求和、softmax、QK 分数、QKV 投影或输入投影反向，没有训练或准确率结果。下一步由 d_attention_output 反向经过 attention_output=concatenated_heads@W_O+b_O，计算 dW_O、db_O 与 d_concatenated_heads，逐节点继续注意力反向，汇合输入梯度后完成输入投影反向，统一参数更新并组织 GPU 训练。
