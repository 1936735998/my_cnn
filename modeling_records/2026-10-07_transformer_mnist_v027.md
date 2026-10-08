# MNIST 手写 Transformer 建模记录

日期：2026-10-07（Asia/Shanghai）；版本：v027。
阶段：ReLU 手写 GPU 反向、CPU 局部差分和零点约定核验。本轮真实执行，不是仅文档更新；未完成第一层前馈或全网络反向，未更新参数或训练。

## 问题范围、输入来源与处理（事实）

用户报告 dW_ff2 `(64,32)`、db_ff2 `(32,)`、d_ff_hidden `(28,64)`，均在 GPU 0。本版只反向通过第一层后的 ReLU，将 d_ff_hidden 传回 d_ff_hidden_linear。目标仍为逐节点手写 Transformer 的全部反向与参数更新，并在 GPU 训练。交付聊天代码，不改主文件，不交付完整 Python 模型。

独立进程读取并执行已保存的 `F:\PythonProjects\deep_learning\transformer_mnist.py` 前二十六步，再运行本片段；未操作用户终端实时变量。主文件 SHA256 `33815808fe8e5178c4ba31651b737263bf6fd2e9f66c4e04479a80a080fbdfc2`，执行前后不变。

沿用本地 MNIST `C:\Users\19367\.keras\datasets\mnist.npz`，SHA256 `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。训练前 5000、测试前 1000 图，像素 CuPy float32 除以 255，标签 int64。本轮只用首张训练图、标签 5，无新增下载、清洗、划分、洗牌、增强或测试评估。输入为实际 ff_hidden_linear 与 d_ff_hidden，额外三点测试仅用于数值核验。

## 假设、变量与公式

延续图片行空间序列、单层编码器、位置均值池化与十类单图交叉熵。Z=ff_hidden_linear `(28,64)`，H=ff_hidden=ReLU(Z)，G=d_ff_hidden `(28,64)`。

```text
ReLU(z) = max(z,0)
z>0 时导数为 1，z<0 时导数为 0
z=0 时经典导数不存在；本实现选择 0 作为反向值
relu_mask = Z>0: (28,64), bool
d_ff_hidden_linear = G*relu_mask: (28,64), float32
```

掩码依据激活前的 Z，逐元素乘法保留正输入处的上游梯度，负数和零输入处为 0。没有额外求和或平均，也不依赖上游梯度本身的正负。零点选 0 是实现约定，不是差分证明出的唯一导数。新增参数 0，累计仍 9802。

## 方法选择理由（判断）

布尔掩码直接实现 ReLU 分段导数，使用基础 CuPy 比较和逐元素乘法，全部在 GPU 执行，无自动求导或现成层。保存前向输入与上游梯度，下一节点继续反向第一层线性映射。

## 本步代码

```python
# ==================================================
# 反向第七步：ReLU
# ==================================================

# 正输入处为 True，负输入和零输入处为 False
relu_mask = ff_hidden_linear > cp.float32(0.0)

# True 相当于乘 1，False 相当于乘 0
d_ff_hidden_linear = d_ff_hidden * relu_mask

print("relu_mask:", relu_mask.shape)
print("d_ff_hidden_linear:", d_ff_hidden_linear.shape)
print("负数或零处的梯度是否全为 0:", bool(cp.all(
    d_ff_hidden_linear[~relu_mask] == cp.float32(0.0)
)))
print("正数处的梯度是否保持不变:", bool(cp.all(
    d_ff_hidden_linear[relu_mask] == d_ff_hidden[relu_mask]
)))
print("d_ff_hidden_linear 的实际设备:", d_ff_hidden_linear.device)
```

## 实际执行、验证与结果（事实）

解释器 `F:\PythonProjects\deep_learning\.venv\Scripts\python.exe`；Python 3.13.15、NumPy 2.5.3、CuPy 14.2.0。可见 GPU 1 张，GPU 0 为 NVIDIA GeForce RTX 5060。真实执行并同步 CUDA，缓存使用工作区 work/cupy_cache、解释器带 -B；没有调整依赖。

mask 与输出梯度形状均 `(28,64)`、GPU 0；mask 为 bool，梯度为有限 float32。负数/零处梯度全为 0，正数处逐元素保持上游值。实际输入正数 879 项、负数 913 项、零 0 项；所有 20 组参数、相关前向值、上游梯度、loss 与直连支路梯度保持不变。

CPU float64 独立重建 Z→ReLU→第二层线性→加固定 attention_norm→第二次 LayerNorm→位置均值池化→分类器→稳定交叉熵。对每个非零 Z 分量做中心差分，步长 min(1e-5,abs(Z)/4)，实际步长范围 `1e-05`～`1e-05`，保证扰动不跨过 ReLU 折点。检查 1792 项，rtol=2e-5、atol=1e-7 下全部通过；最大绝对误差 `8.595044676e-10`、相对 L2 误差 `1.365034123e-07`。CPU 重算与 GPU float32 前向存在舍入差异。实际零点不参加经典中心差分，选零值单独核验。

额外在 GPU 上执行同一聊天片段，输入 [-1,0,1]、上游 [0.3,-0.7,1.1]，得到 [0,0,1.1]，两项非零输入通过 CPU 中心差分。在零输入位置，标量 sum(ReLU(Z)*G) 的左导数 `0`、右导数 `-0.6999999881` 不相等；验证了折点不可导及代码选 0 的约定，未声称零点梯度通过中心差分。

实际输出：

```text
relu_mask: (28, 64)
d_ff_hidden_linear: (28, 64)
负数或零处的梯度是否全为 0: True
正数处的梯度是否保持不变: True
d_ff_hidden_linear 的实际设备: <CUDA Device 0>
```

[本步代码](#本步代码) · [真实 GPU、差分与零点证据](2026-10-07_transformer_relu_backward_validation_v027.json)。JSON 保存逐字片段、来源、设备、输入符号统计、梯度与差分值、误差和三点测试结果，片段 SHA256 `e741b5a18c0502834000cb2d9107b5f6eaa8a5386d13f5e2511a30f3a7f515d1`。相对链接已核验，本步无新增图。

## 与前版变化、限制与下一步

v026 得到 d_ff_hidden；本版新增 relu_mask 与 d_ff_hidden_linear，完成 ReLU 反向。新增参数 0、更新次数 0。历史记录保留，生成本版工作区索引并追加项目 modeling_records/index.md；同步仅复制记录和更新索引，不重跑模型。

尚未完成第一层前馈、第一层 LayerNorm、注意力及输入投影反向，未训练或测量识别准确率。下一步反向 ff_hidden_linear=attention_norm@W_ff1+b_ff1，计算 dW_ff1、db_ff1 与前馈路径返回 attention_norm 的梯度；再与 d_attention_norm_skip 汇合，逐节点完成反向后统一更新参数并组织 GPU 训练。
